"""C1 measurement study: what the fixed rhythm costs, from the step logs and the per-frame statistics.

Per episode:
  wait_share         time in the after-action waiter (sleep for fixed methods) / loop wall clock
  decomposition      obs / vlm / vlm_wasted / check / exec / wait / other (other = loop - all segments)
  unstable_obs_rate  observations taken while the screen was changing (stage-1 change within ±window)
  stale_rate         target actions whose target region changed between the agent's screenshot and dispatch
  obs_to_dispatch    seconds from the screenshot the action was based on to dispatch (TOCTOU gap)
  step types         normal / no_effect / stale / repeat / wait / no_action
Timing failures need a human: failed episodes are listed in timing_failure_sheet.csv with the flags
above; fill the `timing_failure` column (yes / no / unsure) and rerun to get the share.
"""

from __future__ import annotations

import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from nightwatch.analysis.common import episodes
from nightwatch.utils import jdump, read_jsonl

SEG_KINDS = ["obs", "vlm", "vlm_wasted", "check", "exec", "wait"]


def episode_metrics(ep: Path, window: float = 0.3, stale_thr: float = 0.05) -> dict | None:
    ep = Path(ep)
    if not (ep / "result.json").exists():
        return None
    res = json.loads((ep / "result.json").read_text())
    steps = read_jsonl(ep / "steps.jsonl")
    fs = json.loads((ep / "frame_stats.json").read_text())["rows"] if (ep / "frame_stats.json").exists() else []
    loop = float(res.get("loop_wall_s") or 0.0)
    dec = Counter()
    obs_times = []
    for s in steps:
        obs_times.append(s["t_obs"])
        for kind, t0, t1 in s["segments"]:
            dec[kind] += t1 - t0
            if kind == "obs":
                obs_times.append(t1)
    obs_times = sorted(set(round(t, 3) for t in obs_times))
    dec["other"] = max(0.0, loop - sum(dec.values()))
    sig_t = np.array([r[0] for r in fs if r[2]]) if fs else np.array([])
    unstable = [bool(((sig_t >= t - window) & (sig_t <= t + window)).any()) for t in obs_times] if fs else []
    stale = [s["stale_px"] >= stale_thr for s in steps if s.get("stale_px") is not None]
    gaps = [s["t_dispatch"] - s["t_obs_final"] for s in steps if s.get("t_dispatch") is not None]
    types = Counter(s["step_type"] for s in steps if s.get("t_dispatch") is not None)
    rel = ep.parts
    return {
        "dir": str(ep), "method": res.get("method"), "task": res.get("task_id"), "domain": res.get("domain"),
        "repeat": res.get("repeat", 0), "success": bool(res.get("success")), "score": res.get("score"),
        "error": res.get("error", ""), "loop_s": loop, "steps": res.get("steps", 0),
        "vlm_calls": res.get("vlm_calls", 0), "interrupts": res.get("interrupted", 0),
        "invalidations": res.get("invalidated", 0), "vetoes": res.get("vetoes", 0),
        "wait_s": float(res.get("wait_s", 0.0)), "wait_share": float(res.get("wait_s", 0.0)) / loop if loop else 0.0,
        "decomposition": {k: round(dec.get(k, 0.0), 3) for k in SEG_KINDS + ["other"]},
        "unstable_obs_rate": float(np.mean(unstable)) if unstable else None, "n_obs": len(obs_times),
        "stale_rate": float(np.mean(stale)) if stale else None, "n_target_actions": len(stale),
        "obs_to_dispatch_s": float(np.median(gaps)) if gaps else None,
        "step_types": dict(types), "invalid_steps": sum(v for k, v in types.items() if k != "normal"),
        "wake_reasons": res.get("wake_reasons", {}), "interrupt_wasted_s": res.get("interrupt_wasted_s", 0.0),
        "wasted_tokens": res.get("wasted_tokens", 0.0), "input_tokens": res.get("input_tokens", 0.0),
        "output_tokens": res.get("output_tokens", 0.0), "jev_calls": res.get("jev_calls", 0),
        "jev_ms_total": res.get("jev_ms_total", 0.0), "jev_ms_critical": res.get("jev_ms_critical", 0.0),
        "fps": res.get("frames", {}).get("fps"), "end_reason": res.get("end_reason"), "_parts": rel[-5:],
    }


def collect(root: Path) -> list[dict]:
    return [m for m in (episode_metrics(ep) for ep in episodes(root)) if m]


def measure(root: Path) -> str:
    root = Path(root)
    eps = collect(root)
    if not eps:
        return f"no episodes under {root}"
    out = root / "measure"
    out.mkdir(exist_ok=True)
    by = defaultdict(list)
    for e in eps:
        by[e["method"]].append(e)
    lines = [f"# Measurement study: {root.name}", "",
             "| method | episodes | success | loop s (median) | wait share | unstable obs | stale actions | "
             "obs->dispatch s | invalid steps / step |", "|---|---|---|---|---|---|---|---|---|"]
    summary = {}
    for m, es in sorted(by.items()):
        def mean(k):
            v = [e[k] for e in es if e.get(k) is not None]
            return float(np.mean(v)) if v else None

        steps = sum(e["steps"] for e in es) or 1
        s = {"n": len(es), "success": mean("success"), "loop_median": float(np.median([e["loop_s"] for e in es])),
             "wait_share": mean("wait_share"), "unstable_obs": mean("unstable_obs_rate"), "stale": mean("stale_rate"),
             "obs_to_dispatch": mean("obs_to_dispatch_s"), "invalid_per_step": sum(e["invalid_steps"] for e in es) / steps,
             "decomposition": {k: float(np.mean([e["decomposition"][k] for e in es])) for k in SEG_KINDS + ["other"]}}
        summary[m] = s
        f = lambda v, d=3: "–" if v is None else f"{v:.{d}f}"  # noqa: E731
        lines.append(f"| {m} | {s['n']} | {f(s['success'])} | {f(s['loop_median'], 1)} | {f(s['wait_share'])} | "
                     f"{f(s['unstable_obs'])} | {f(s['stale'])} | {f(s['obs_to_dispatch'], 2)} | {f(s['invalid_per_step'])} |")
    # timing-failure annotation sheet
    sheet = out / "timing_failure_sheet.csv"
    labelled = {}
    if sheet.exists():
        with open(sheet, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                if r.get("timing_failure"):
                    labelled[r["dir"]] = r["timing_failure"].strip().lower()
    with open(sheet, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["dir", "method", "task", "unstable_obs_rate", "stale_rate", "no_effect_steps", "recording",
                    "timing_failure", "note"])
        for e in eps:
            if not e["success"] and not e["error"]:
                w.writerow([e["dir"], e["method"], e["task"], e["unstable_obs_rate"], e["stale_rate"],
                            e["step_types"].get("no_effect", 0), str(Path(e["dir"]) / "label_sheet.html"),
                            labelled.get(e["dir"], ""), ""])
    if labelled:
        yes = sum(v == "yes" for v in labelled.values())
        lines += ["", f"Timing failures (human-labelled): {yes} / {len(labelled)} failed episodes"]
        summary["_timing_failures"] = {"yes": yes, "labelled": len(labelled)}
    lines += ["", f"Failed episodes to annotate: {sheet}"]
    with open(out / "per_episode.csv", "w", newline="", encoding="utf-8") as fh:
        keys = [k for k in eps[0] if not k.startswith("_") and not isinstance(eps[0][k], dict)]
        w = csv.DictWriter(fh, keys, extrasaction="ignore")
        w.writeheader()
        w.writerows(eps)
    jdump(summary, out / "summary.json")
    (out / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return "\n".join(lines)
