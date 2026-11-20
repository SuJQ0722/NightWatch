"""Tables and figures across experiments -> runs/report/report.md (+ PNGs).

Table 1 (per experiment x method): success (mean over repeats), loop wall clock (median, 95% CI by a
task-cluster bootstrap), wait share, steps, invalid steps, VLM calls incl. interrupted, interrupts,
invalidations (+ implicit vetoes), wasted generation seconds / tokens, System-1 calls and critical-path ms.
Paired comparison with the baseline (same task, same repeat): success difference with a bootstrap CI
and a non-inferiority verdict (lower bound > -margin), wall-clock ratio on all tasks and on tasks both
methods solved.
Figures: success vs wall clock (the fixed-sleep sweep as a curve), wall-clock decomposition, wake reasons.
"""

from __future__ import annotations

from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from nightwatch.analysis.measure import SEG_KINDS, collect
from nightwatch.config import RUNS_DIR
from nightwatch.utils import jdump

RNG = np.random.default_rng(0)


def _boot(values_by_task: dict, stat, n: int = 2000) -> tuple[float, float, float]:
    keys = list(values_by_task)
    if not keys:
        return (float("nan"),) * 3
    point = stat([v for k in keys for v in values_by_task[k]])
    sims = []
    for _ in range(n):
        pick = RNG.choice(len(keys), len(keys), replace=True)
        sims.append(stat([v for i in pick for v in values_by_task[keys[i]]]))
    lo, hi = np.percentile(sims, [2.5, 97.5])
    return float(point), float(lo), float(hi)


def _paired(base: list[dict], other: list[dict], margin: float) -> dict:
    b = {(e["task"], e["repeat"]): e for e in base}
    pairs = [(b[(e["task"], e["repeat"])], e) for e in other if (e["task"], e["repeat"]) in b]
    if not pairs:
        return {}
    by_task_d, by_task_r, both = defaultdict(list), defaultdict(list), defaultdict(list)
    for x, y in pairs:
        by_task_d[x["task"]].append(float(y["success"]) - float(x["success"]))
        if x["loop_s"] > 0:
            by_task_r[x["task"]].append(y["loop_s"] / x["loop_s"])
            if x["success"] and y["success"]:
                both[x["task"]].append(y["loop_s"] / x["loop_s"])
    d = _boot(by_task_d, np.mean)
    r = _boot(by_task_r, np.median) if by_task_r else (np.nan,) * 3
    rb = _boot(both, np.median) if both else (np.nan,) * 3
    return {"pairs": len(pairs), "success_diff": d, "noninferior": bool(d[1] > -margin),
            "time_ratio": r, "time_ratio_both_success": rb, "both_success_pairs": sum(len(v) for v in both.values())}


def summarize(eps: list[dict]) -> dict:
    by_task = defaultdict(list)
    for e in eps:
        by_task[e["task"]].append(e)
    steps = sum(e["steps"] for e in eps) or 1
    tot = lambda k: float(sum(e.get(k) or 0 for e in eps))  # noqa: E731
    return {
        "n": len(eps), "tasks": len(by_task),
        "success": _boot({t: [float(e["success"]) for e in v] for t, v in by_task.items()}, np.mean),
        "loop_median": _boot({t: [e["loop_s"] for e in v] for t, v in by_task.items()}, np.median),
        "wait_share": float(np.mean([e["wait_share"] for e in eps])),
        "steps_mean": float(np.mean([e["steps"] for e in eps])),
        "invalid_per_step": sum(e["invalid_steps"] for e in eps) / steps,
        "step_types": dict(sum((Counter(e["step_types"]) for e in eps), Counter())),
        "vlm_calls_mean": float(np.mean([e["vlm_calls"] for e in eps])),
        "interrupts": tot("interrupts"), "invalidations": tot("invalidations"), "vetoes": tot("vetoes"),
        "wasted_s_mean": float(np.mean([e["interrupt_wasted_s"] or 0 for e in eps])),
        "wasted_tokens": tot("wasted_tokens"), "output_tokens": tot("output_tokens"),
        "jev_calls_mean": float(np.mean([e["jev_calls"] for e in eps])),
        "jev_ms_critical_mean": float(np.mean([e["jev_ms_critical"] or 0 for e in eps])),
        "decomposition": {k: float(np.mean([e["decomposition"][k] for e in eps])) for k in SEG_KINDS + ["other"]},
        "wake_reasons": dict(sum((Counter(e["wake_reasons"]) for e in eps), Counter())),
        "unstable_obs": float(np.mean([e["unstable_obs_rate"] for e in eps if e["unstable_obs_rate"] is not None] or [np.nan])),
        "stale": float(np.mean([e["stale_rate"] for e in eps if e["stale_rate"] is not None] or [np.nan])),
        "errors": sum(bool(e["error"]) for e in eps),
    }


def _fmt_ci(t, d=3) -> str:
    p, lo, hi = t
    return "–" if np.isnan(p) else f"{p:.{d}f} [{lo:.{d}f}, {hi:.{d}f}]"


def build_report(exps: list[str] | None, baseline: str = "fixed_default", margin: float = 0.02) -> str:
    roots = [RUNS_DIR / e for e in exps] if exps else sorted(p for p in RUNS_DIR.iterdir()
                                                            if p.is_dir() and p.name != "report")
    out = RUNS_DIR / "report"
    out.mkdir(parents=True, exist_ok=True)
    lines = ["# NightWatch report", ""]
    data = {}
    for root in roots:
        eps = collect(root)
        if not eps:
            continue
        by = defaultdict(list)
        for e in eps:
            if not e["error"]:
                by[e["method"]].append(e)
        if root.name == "smoke":
            lines += ["> **smoke**: simulated desktop, scripted agent, simulated System-1. Pipeline check only, "
                      "not a result.", ""]
        sums = {m: summarize(v) for m, v in by.items()}
        base = baseline if baseline in by else None
        lines += [f"## {root.name}", "",
                  "| method | n | success [95% CI] | loop s median [CI] | wait share | steps | invalid/step | VLM calls | "
                  "interrupts | invalidations (vetoes) | wasted s / ep | S1 calls / ep | S1 critical ms / ep |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for m, s in sorted(sums.items()):
            lines.append(f"| {m} | {s['n']} | {_fmt_ci(s['success'])} | {_fmt_ci(s['loop_median'], 1)} | "
                         f"{s['wait_share']:.3f} | {s['steps_mean']:.1f} | {s['invalid_per_step']:.3f} | "
                         f"{s['vlm_calls_mean']:.1f} | {s['interrupts']:.0f} | {s['invalidations']:.0f} ({s['vetoes']:.0f}) | "
                         f"{s['wasted_s_mean']:.2f} | {s['jev_calls_mean']:.0f} | {s['jev_ms_critical_mean']:.0f} |")
        if base:
            lines += ["", f"Paired with **{base}** (non-inferiority margin {margin:.0%} on success):", "",
                      "| method | pairs | Δ success [CI] | non-inferior | time ratio (all) [CI] | time ratio (both solved) [CI] |",
                      "|---|---|---|---|---|---|"]
            for m in sorted(by):
                if m == base:
                    continue
                pr = _paired(by[base], by[m], margin)
                if pr:
                    sums[m]["paired"] = pr
                    lines.append(f"| {m} | {pr['pairs']} | {_fmt_ci(pr['success_diff'])} | {'yes' if pr['noninferior'] else 'no'} | "
                                 f"{_fmt_ci(pr['time_ratio'], 2)} | {_fmt_ci(pr['time_ratio_both_success'], 2)} |")
        lines += ["", "Wall-clock decomposition (mean seconds per episode): " + "; ".join(
            f"{m}: " + ", ".join(f"{k} {v:.1f}" for k, v in s["decomposition"].items()) for m, s in sorted(sums.items())), ""]
        data[root.name] = sums
        _figures(root.name, sums, out)
        lines += [f"![success vs time]({root.name}_success_time.png) ![decomposition]({root.name}_decomposition.png)", ""]
    jdump(data, out / "report.json")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return "\n".join(lines) + f"\n-> {out / 'report.md'}"


def _figures(name: str, sums: dict, out: Path) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(5.5, 4))
    fixed = sorted(((float(m.split("_", 1)[1]), s) for m, s in sums.items()
                    if m.startswith("fixed_") and m.split("_", 1)[1].replace(".", "").isdigit()), key=lambda t: t[0])
    if fixed:
        ax.plot([s["loop_median"][0] for _, s in fixed], [s["success"][0] for _, s in fixed], "-o", color="#888", label="fixed sleep")
        for sec, s in fixed:
            ax.annotate(f"{sec:g}s", (s["loop_median"][0], s["success"][0]), fontsize=8, xytext=(3, 3), textcoords="offset points")
    for m, s in sums.items():
        if not m.startswith("fixed_") or m == "fixed_default":
            ax.errorbar(s["loop_median"][0], s["success"][0], fmt="s", label=m,
                        xerr=[[s["loop_median"][0] - s["loop_median"][1]], [s["loop_median"][2] - s["loop_median"][0]]])
    ax.set_xlabel("loop wall clock per task, median (s)")
    ax.set_ylabel("success rate")
    ax.legend(fontsize=7)
    ax.set_title(name)
    fig.tight_layout()
    fig.savefig(out / f"{name}_success_time.png", dpi=150)
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(6.5, 0.45 * len(sums) + 1.2))
    ms = sorted(sums)
    left = np.zeros(len(ms))
    for k in SEG_KINDS + ["other"]:
        v = np.array([sums[m]["decomposition"][k] for m in ms])
        ax.barh(ms, v, left=left, label=k)
        left += v
    ax.set_xlabel("seconds per episode (mean)")
    ax.legend(fontsize=7, ncol=4)
    fig.tight_layout()
    fig.savefig(out / f"{name}_decomposition.png", dpi=150)
    plt.close(fig)
