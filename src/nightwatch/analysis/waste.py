"""Wasted-time ledger: where the wall clock of an episode goes, judged against what the screen was actually doing.

Loop time and step counts depend on the trajectory the agent happens to take; this ledger does not. It uses the
frame stream (recorded for every method, the baseline included) as the screen's timeline and splits the time of
every executed action, from dispatch to the agent's next screenshot, into

  react   dispatch -> the screen's last significant change before that screenshot (+ one frame to confirm):
          the application doing the work. Necessary.
  idle    the screen had reached its final state, nobody was looking. Waste, whatever sat there: the harness's
          sleep, the waiter, or a time.sleep the agent wrote into its own action.

and counts, per step,

  premature   the screenshot was taken while the screen was still changing (a significant change in the half
              second before it) and it changed again before the agent acted: the agent decided on a screen that
              was not final. `premature_s` = how much earlier than the last of those changes it looked.
  stale       the screen changed (for whatever reason) between the screenshot the next action was decided on (the
              last one before its dispatch) and that dispatch.
  no_effect   no significant change between dispatch and the next screenshot (WAIT excluded). The whole step is
              ineffective; `timing_no_effect` are those whose screenshot was premature or stale, i.e. the ones a
              better rhythm could have avoided.
  wasted generation   interrupted / invalidated VLM time (vlm_wasted segments).

Per method, over the tasks every method ran: seconds per executed step and shares of the loop time, which do not
grow with the number of steps. `waste` = idle + wasted generation + the non-idle time of timing-caused no-effect
steps; `oracle_loop` = loop - idle (perfect wake-up timing, same trajectory).

    python -m nightwatch waste --exps osworld_20_vanilla,osworld_20_jev
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from nightwatch.analysis.common import episodes, exp_root
from nightwatch.utils import jdump, read_jsonl

PRE_WINDOW = 0.5  # a change this close before the screenshot means the screen was still moving
STALE_PX = 0.05  # configs/base.yaml step_types.stale_px


def _changes(ep: Path) -> tuple[np.ndarray, float]:
    """Instants of significant frame-to-frame changes (mid-grab), and the frame period."""
    p = ep / "frame_stats.json"
    if not p.exists():
        return np.array([]), 0.0
    rows = json.loads(p.read_text())["rows"]
    t = np.array([r[0] - r[3] / 2 for r in rows])
    period = float(np.median(np.diff(t))) if len(t) > 2 else 0.0
    return np.array([ti for ti, r in zip(t, rows) if r[2]]), period


def step_ledger(ep: Path) -> tuple[dict, list[dict]]:
    ep = Path(ep)
    res = json.loads((ep / "result.json").read_text())
    steps = read_jsonl(ep / "steps.jsonl")
    sig, period = _changes(ep)
    rows = []
    for i, s in enumerate(steps):
        seg = s["segments"]
        ex = [k for k, (kind, *_x) in enumerate(seg) if kind == "exec"]
        row = {"step": s["step"], "vlm": sum(t1 - t0 for k, t0, t1 in seg if k == "vlm"),
               "vlm_wasted": sum(t1 - t0 for k, t0, t1 in seg if k == "vlm_wasted"),
               "check": sum(t1 - t0 for k, t0, t1 in seg if k == "check"),
               "obs": sum(t1 - t0 for k, t0, t1 in seg if k == "obs"), "react": 0.0, "idle": 0.0,
               "idle_exec": 0.0, "idle_wait": 0.0,
               "effect": None, "premature": False, "premature_s": 0.0, "stale": False, "wait_action": False,
               "step_s": 0.0, "dispatch": None, "ready_t": None, "settle_to_act": None,
               # the action's target area changed between the agent's screenshot and dispatch (loop.py stale_px)
               "mismatch": s.get("stale_px") is not None and float(s["stale_px"]) >= STALE_PX,
               "targeted": s.get("stale_px") is not None, "repeat": s.get("step_type") == "repeat"}
        if not ex:
            rows.append(row)
            continue
        dispatch = seg[ex[0]][1]
        row["dispatch"] = dispatch
        after = [g for g in seg[ex[-1]:] if g[0] == "obs"]
        if not after:  # the episode ended with this action
            rows.append(row)
            continue
        capture = after[0][1]
        win = sig[(sig > dispatch) & (sig <= capture)]
        ready = min(capture, (win.max() if len(win) else dispatch) + period)
        row["react"], row["idle"] = ready - dispatch, capture - ready
        row["ready_t"] = ready
        for kind, t0, t1 in seg:  # where the idle time sat: still inside the action code, or in the harness's wait
            if kind in ("exec", "wait"):
                row["idle_" + kind] += max(0.0, min(t1, capture) - max(t0, ready))
        row["wait_action"] = (s.get("action") or "").strip() == "WAIT" or s.get("act_kind") == "wait"
        row["effect"] = bool(len(win)) or row["wait_action"]
        nxt_seg = steps[i + 1]["segments"] if i + 1 < len(steps) else []
        nxt = next((g[1] for g in nxt_seg if g[0] == "exec"), None)
        horizon = nxt if nxt is not None else capture + 2.0
        # premature / stale are judged on the screenshot the next action was decided on: the last one before its
        # dispatch, a new one when the agent observed again after an interruption or a voided action
        acted = max([capture] + [g[1] for g in nxt_seg if g[0] == "obs" and g[1] < horizon])
        later = sig[(sig > acted) & (sig < horizon)]
        moving = bool(((sig > acted - PRE_WINDOW) & (sig <= acted)).any())
        row["stale"] = bool(len(later))
        row["premature"] = moving and row["stale"]
        if row["premature"]:  # the screen settled at the last change of the burst that was going on at capture
            burst, last = later[later <= acted + 3.0], acted
            for t in burst:
                if t - last <= 1.0:
                    last = t
            row["premature_s"] = last - acted
        row["step_s"] = capture - s["t_obs"]
        rows.append(row)
    for i, r in enumerate(rows):  # a no-effect step is timing-caused when the screenshot it acted on was not final
        prev = rows[i - 1] if i else None
        r["timing_no_effect"] = r["effect"] is False and prev is not None and (prev["premature"] or prev["stale"])
        # screen settled after the previous action -> this action dispatched, without this step's LLM generation
        # (the part no harness controls): looking late, observing, checks, discarded generations
        if prev is not None and prev["ready_t"] is not None and r["dispatch"] is not None:
            r["settle_to_act"] = max(0.0, r["dispatch"] - prev["ready_t"] - r["vlm"])
    loop = float(res.get("loop_wall_s") or 0.0)
    ex_rows = [r for r in rows if r["effect"] is not None]
    tot = lambda k: float(sum(r[k] for r in rows))  # noqa: E731
    eff = [r for r in ex_rows if r["effect"]]
    dead = [r for r in ex_rows if r["effect"] is False]
    timing_ne = sum(r["step_s"] - r["idle"] for r in rows if r["timing_no_effect"])
    ep_sum = {
        "dir": str(ep), "method": res.get("method"), "task": f"{res.get('domain')}/{res.get('task_id')}",
        "success": bool(res.get("success")), "error": res.get("error", ""), "loop_s": loop,
        "steps": len(ex_rows), "frame_period_s": round(period, 3),
        "react_s": tot("react"), "idle_s": tot("idle"), "vlm_s": tot("vlm"), "vlm_wasted_s": tot("vlm_wasted"),
        "check_s": tot("check"), "obs_s": tot("obs"),
        "premature": sum(r["premature"] for r in ex_rows), "premature_s": tot("premature_s"),
        "stale": sum(r["stale"] for r in ex_rows),
        "no_effect": sum(r["effect"] is False for r in ex_rows),
        "timing_no_effect": sum(r["timing_no_effect"] for r in ex_rows), "timing_no_effect_s": timing_ne,
        "idle_steps": [round(r["idle"], 3) for r in ex_rows],
        "effect_steps": len(eff), "idle_effect_s": float(sum(r["idle"] for r in eff)),
        "idle_effect_exec_s": float(sum(r["idle_exec"] for r in eff)), "idle_effect_wait_s": float(sum(r["idle_wait"] for r in eff)),
        "idle_no_effect_s": float(sum(r["idle"] for r in dead)), "react_effect_s": float(sum(r["react"] for r in eff)),
        "targeted": sum(r["targeted"] for r in ex_rows), "mismatch": sum(r["mismatch"] for r in ex_rows),
        "mismatch_no_effect": sum(r["mismatch"] and r["effect"] is False for r in ex_rows),
        "repeat": sum(r["repeat"] for r in ex_rows),
        "settle_to_act": [round(r["settle_to_act"], 3) for r in rows if r["settle_to_act"] is not None],
    }
    ep_sum["waste_s"] = ep_sum["idle_s"] + ep_sum["vlm_wasted_s"] + timing_ne
    return ep_sum, rows


def ledger(exps: list[str]) -> tuple[str, dict]:
    found = defaultdict(dict)  # (experiment, method) -> task -> episode summary
    for exp in exps:
        for ep in episodes(exp_root(exp)):
            if not (ep / "result.json").exists() or "_setup_failed" in ep.parts:
                continue
            e, _ = step_ledger(ep)
            if not e["error"]:
                found[(Path(exp).name, e["method"])][e["task"]] = e
    names = defaultdict(list)
    for exp, m in found:
        names[m].append(exp)
    # a method run in several experiments (e.g. before / after a change) is labelled with its experiment
    by = {(m if len(names[m]) == 1 else f"{m} @ {exp}"): eps for (exp, m), eps in found.items()}
    methods = sorted(by, key=lambda m: (m.startswith("nightwatch"), m))
    tasks = sorted(set.intersection(*[set(by[m]) for m in methods])) if methods else []
    out, lines = {}, [
        f"# Wasted-time ledger ({len(tasks)} tasks run cleanly by every method: {', '.join(methods)})", "",
        "Per executed step (s) and share of loop time; trajectory-independent. See nightwatch/analysis/waste.py.", "",
        "| method | success | steps | loop s | react /effect step | **idle /effect step** (in action code + in waiter) | "
        "idle /no-effect step | idle share | wasted gen /step | premature | stale | no-effect | timing no-effect | "
        "**waste share** | oracle loop |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for m in methods:
        E = [by[m][t] for t in tasks]
        st = sum(e["steps"] for e in E) or 1
        loop = sum(e["loop_s"] for e in E) or 1e-9
        idle = [x for e in E for x in e["idle_steps"]]
        s = {k: sum(e[k] for e in E) for k in ("react_s", "idle_s", "vlm_wasted_s", "premature", "stale", "no_effect",
                                                "timing_no_effect", "timing_no_effect_s", "waste_s", "premature_s",
                                                "effect_steps", "idle_effect_s", "idle_effect_exec_s", "idle_effect_wait_s",
                                                "idle_no_effect_s", "react_effect_s")}
        s.update(n=len(E), success=sum(e["success"] for e in E), steps=st, loop_s=loop,
                 idle_p50=float(np.median(idle)) if idle else 0.0, idle_p90=float(np.percentile(idle, 90)) if idle else 0.0)
        out[m] = s
        ne, ef = max(s["no_effect"], 1), max(s["effect_steps"], 1)
        lines.append(
            f"| {m} | {s['success']}/{s['n']} | {st} | {loop:.0f} | {s['react_effect_s'] / ef:.2f} | "
            f"**{s['idle_effect_s'] / ef:.2f}** ({s['idle_effect_exec_s'] / ef:.2f} + {s['idle_effect_wait_s'] / ef:.2f}) | "
            f"{s['idle_no_effect_s'] / ne:.2f} | {s['idle_s'] / loop:.0%} | {s['vlm_wasted_s'] / st:.2f} | "
            f"{s['premature'] / st:.0%} | {s['stale'] / st:.0%} | {s['no_effect'] / st:.0%} | {s['timing_no_effect'] / st:.0%} | "
            f"**{s['waste_s'] / loop:.0%}** | {loop - s['idle_s']:.0f} |")
    return "\n".join(lines), {"tasks": tasks, "methods": out, "per_episode": {m: by[m] for m in methods}}


def waste(exps: list[str], out_dir: Path | None = None) -> str:
    text, data = ledger(exps)
    out_dir = Path(out_dir or exp_root(exps[-1]) / "waste")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.md").write_text(text + "\n", encoding="utf-8")
    jdump({"tasks": data["tasks"], "methods": data["methods"]}, out_dir / "summary.json")
    with open(out_dir / "per_episode.csv", "w", newline="", encoding="utf-8") as fh:
        w = None
        for m, eps in data["per_episode"].items():
            for e in eps.values():
                row = {k: v for k, v in e.items() if k != "idle_steps"}
                if w is None:
                    w = csv.DictWriter(fh, list(row))
                    w.writeheader()
                w.writerow(row)
    return text + f"\n\nwritten to {out_dir}"
