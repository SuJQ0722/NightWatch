"""Per-method metrics for a set of experiments (e.g. the OSWorld full-set runs), on the tasks every method finished.

    python -m nightwatch metrics --exps osworld_full_fixed_0,osworld_full_nightwatch_obs_jev,osworld_full_aoi_visual,osworld_full_pusv

score     success rate, mean score, per-domain success (per_domain.csv)
steps     executed steps; effective steps = actions after which the screen visibly changed (frame stream, see waste.py)
latency   loop seconds per task (mean / median), seconds per step, and per step: VLM generation, time to first token,
          action execution, waiting, observation, pre-execution checks; mean VLM call
waste     the wasted-time ledger (waste.py): idle after the screen was ready, premature / stale observations,
          timing-caused no-effect steps, waste share of the loop
events    interrupts, invalidations, early wakes (none counted as steps), vetoes; System-1 calls and latency;
          AOI keyframes
cost      input / output tokens and USD at AGENT_PRICE_IN_PER_MTOK / AGENT_PRICE_OUT_PER_MTOK
no-LLM    (second table) the loop minus every agent LLM call (executed, interrupted and failed generations alike):
          the time the harness itself spends per task / step
controllable  (second table) the loop without the LLM generations that produced executed actions (the model's own
          latency, which no harness controls; discarded generations stay in, they are a cost of the method), and
          settle->act: from the screen settling after an action to the next action's dispatch, minus that step's
          LLM generation (looking late + observing + checks + discarded generations)
invalid   no-effect steps; mismatched dispatches = targeted actions whose target area changed between the agent's
          screenshot and dispatch (stale_px >= 5 %), and how many of them had no effect; timing-caused no-effect
          steps; repeats; actions prevented before dispatch (invalidations) and plans cancelled (interrupts)
Outputs summary.md, summary.json, per_episode.csv, per_domain.csv.
"""

from __future__ import annotations

import csv
import json
import statistics as S
from collections import Counter, defaultdict
from pathlib import Path

from nightwatch.analysis.common import episodes, exp_root
from nightwatch.analysis.waste import step_ledger
from nightwatch.config import env
from nightwatch.utils import jdump, read_jsonl


def _mean(v: list) -> float:
    return float(sum(v) / len(v)) if v else 0.0


def _median(v: list) -> float:
    return float(S.median(v)) if v else 0.0


def episode_row(ep: Path) -> dict | None:
    ep = Path(ep)
    if not (ep / "result.json").exists() or "_setup_failed" in ep.parts:
        return None
    r = json.loads((ep / "result.json").read_text())
    if r.get("protocol") == "osworld_qwen35_native_v1":
        raise ValueError("Native Qwen trajectories do not contain NIGHTWATCH timing/waste measurements. "
                         "Read their scores from result.json.")
    if r.get("error"):
        return None
    if "observation_mode" in r and not (ep / "steps.jsonl").exists():
        return _native_aoi_row(ep, r)
    w, rows = step_ledger(ep)
    steps = read_jsonl(ep / "steps.jsonl")
    dec = Counter({k: float(v) for k, v in (r.get("decomposition") or {}).items()})
    ttft = [s["t_first_token"] for s in steps if s.get("t_first_token") is not None]
    s1 = r.get("system1_calls") or {}
    ev = r.get("events") or {}
    return {
        "method": r.get("method"), "domain": r.get("domain"), "task": f"{r.get('domain')}/{r.get('task_id')}",
        "success": bool(r.get("success")), "score": float(r.get("score") or 0.0), "end_reason": r.get("end_reason"),
        "loop_s": float(r.get("loop_wall_s") or 0.0), "steps": int(r.get("steps") or 0), "exec_steps": int(w["steps"]),
        "effective_steps": int(w["effect_steps"]), "no_effect_steps": int(w["no_effect"]),
        "vlm_calls": int(r.get("vlm_calls") or 0), "vlm_s": dec["vlm"], "vlm_wasted_s": dec["vlm_wasted"],
        "exec_s": dec["exec"], "wait_s": dec["wait"], "obs_s": dec["obs"], "check_s": dec["check"],
        "ttft_sum": float(sum(ttft)), "ttft_n": len(ttft),
        "idle_s": w["idle_s"], "idle_effect_s": w["idle_effect_s"], "premature": w["premature"], "stale": w["stale"],
        "timing_no_effect": w["timing_no_effect"], "waste_s": w["waste_s"],
        "interrupts": int(ev.get("interrupt", r.get("interrupted") or 0)),
        "invalidations": int(ev.get("invalidate", r.get("invalidated") or 0)), "early_wakes": int(ev.get("wake", 0)),
        "vetoes": int(r.get("vetoes") or 0), "keyframes": int(r.get("keyframes") or 0),
        "system1_calls": int(sum(s1.values())), "system1_ms": float(sum((r.get("system1_ms") or {}).values())),
        "input_tokens": float(r.get("input_tokens") or 0.0), "output_tokens": float(r.get("output_tokens") or 0.0),
        "wasted_tokens": float(r.get("wasted_tokens") or 0.0),
        "stripped_sleep_s": float(r.get("stripped_sleep_s") or 0.0),
        "targeted": int(w["targeted"]), "mismatch": int(w["mismatch"]), "mismatch_no_effect": int(w["mismatch_no_effect"]),
        "repeat": int(w["repeat"]), "settle_to_act": w["settle_to_act"], "dir": str(ep),
    }


def _native_aoi_row(ep: Path, r: dict) -> dict:
    """An episode of AOI's own harness (DynaCU Vanilla / AOI arms): success, loop time and model time only.

    AOI logs no per-step timestamps, so the waste ledger (idle, premature, stale) is zero, not measured."""
    zero = ("score", "exec_steps", "effective_steps", "no_effect_steps", "vlm_calls", "vlm_wasted_s", "exec_s", "wait_s",
            "obs_s", "check_s", "ttft_sum", "ttft_n", "idle_s", "idle_effect_s", "premature", "stale",
            "timing_no_effect", "waste_s", "interrupts", "invalidations", "early_wakes", "vetoes", "keyframes",
            "system1_calls", "system1_ms", "input_tokens", "output_tokens", "wasted_tokens", "stripped_sleep_s", "targeted", "mismatch",
            "mismatch_no_effect", "repeat")
    row = {k: 0 for k in zero}
    steps = int(r.get("steps_taken") or 0)
    row.update(method=r.get("condition") or r.get("observation_mode"), domain=r.get("category"),
               task=f"{r.get('category')}/{r.get('task_id')}", success=bool(r.get("success")),
               score=float(bool(r.get("success"))), end_reason="native", loop_s=float(r.get("total_time_s") or 0.0),
               steps=steps, exec_steps=steps, vlm_s=float(r.get("total_model_latency_ms") or 0.0) / 1000,
               obs_s=float(r.get("total_obs_overhead_ms") or 0.0) / 1000, settle_to_act=[], dir=str(ep), native=True)
    return row


def _rep(ep: Path) -> str:
    return next((p for p in reversed(Path(ep).parts) if len(p) > 1 and p[0] == "r" and p[1:].isdigit()), "r0")


def metrics(exps: list[str], out_dir: Path | None = None) -> str:
    found = defaultdict(dict)  # (exp, method, repeat) -> task -> row
    names = defaultdict(set)
    for exp in exps:
        root = exp_root(exp)
        native = [r.parent for r in Path(root).rglob("result.json")
                  if not (r.parent / "steps.jsonl").exists() and "_setup_failed" not in r.parts
                  and '"observation_mode"' in r.read_text()]
        for ep in episodes(root) + native:
            row = episode_row(ep)
            if row:
                found[(Path(exp).name, row["method"], _rep(ep))][row["task"]] = row
                names[row["method"]].add((Path(exp).name, _rep(ep)))
    by = {(m if len(names[m]) == 1 else f"{m} @ {e}/{r}"): rows for (e, m, r), rows in found.items()}
    run_of = {(m if len(names[m]) == 1 else f"{m} @ {e}/{r}"): m for (e, m, r) in found}
    labels = sorted(by, key=lambda m: (m.startswith("nightwatch"), m))
    tasks = sorted(set.intersection(*[set(by[m]) for m in labels])) if labels else []
    p_in, p_out = float(env("AGENT_PRICE_IN_PER_MTOK", "0") or 0), float(env("AGENT_PRICE_OUT_PER_MTOK", "0") or 0)
    summary, per_domain = {}, defaultdict(dict)
    for m in labels:
        E = [by[m][t] for t in tasks]
        n, st = len(E), sum(e["steps"] for e in E) or 1
        ex_st = sum(e["exec_steps"] for e in E) or 1
        tot = lambda k: float(sum(e[k] for e in E))  # noqa: E731
        loop = tot("loop_s") or 1e-9
        summary[m] = {
            "tasks": n, "finished_any": len(by[m]), "success_rate": tot("success") / n, "mean_score": tot("score") / n,
            "steps": int(st), "steps_per_task": st / n, "effective_steps": int(tot("effective_steps")),
            "effective_share": tot("effective_steps") / st,
            "loop_mean_s": loop / n, "loop_median_s": S.median(e["loop_s"] for e in E), "s_per_step": loop / st,
            "vlm_per_step": tot("vlm_s") / st, "ttft_mean_s": tot("ttft_sum") / max(tot("ttft_n"), 1),
            "vlm_call_mean_s": (tot("vlm_s") + tot("vlm_wasted_s")) / max(tot("vlm_calls"), 1),
            "exec_per_step": tot("exec_s") / st, "wait_per_step": tot("wait_s") / st, "obs_per_step": tot("obs_s") / st,
            "check_per_step": tot("check_s") / st, "idle_per_step": tot("idle_s") / st,
            # shares of executed steps (steps without an action, e.g. narration only, cannot be premature or stale)
            "premature_rate": tot("premature") / ex_st, "stale_rate": tot("stale") / ex_st,
            "timing_no_effect_rate": tot("timing_no_effect") / ex_st, "waste_share": tot("waste_s") / loop,
            "interrupts": int(tot("interrupts")), "invalidations": int(tot("invalidations")),
            "early_wakes": int(tot("early_wakes")), "vetoes": int(tot("vetoes")), "keyframes": int(tot("keyframes")),
            "system1_calls_per_step": tot("system1_calls") / st,
            "system1_ms_mean": tot("system1_ms") / max(tot("system1_calls"), 1),
            "tokens_in_per_task": tot("input_tokens") / n, "tokens_out_per_task": tot("output_tokens") / n,
            "usd_per_task": (tot("input_tokens") * p_in + tot("output_tokens") * p_out) / 1e6 / n,
            "llm_all_per_task": (tot("vlm_s") + tot("vlm_wasted_s")) / n,
            "nollm_per_task": (loop - tot("vlm_s") - tot("vlm_wasted_s")) / n,
            "nollm_median": S.median(e["loop_s"] - e["vlm_s"] - e["vlm_wasted_s"] for e in E),
            "nollm_per_step": (loop - tot("vlm_s") - tot("vlm_wasted_s")) / st,
            "llm_per_task": tot("vlm_s") / n, "ctl_per_task": (loop - tot("vlm_s")) / n,
            "ctl_median": S.median(e["loop_s"] - e["vlm_s"] for e in E), "ctl_per_step": (loop - tot("vlm_s")) / st,
            "settle_to_act_mean": _mean([x for e in E for x in e["settle_to_act"]]),
            "settle_to_act_median": _median([x for e in E for x in e["settle_to_act"]]),
            "no_effect_rate": tot("no_effect_steps") / st, "targeted": int(tot("targeted")),
            "mismatch": int(tot("mismatch")), "mismatch_rate": tot("mismatch") / max(tot("targeted"), 1),
            "mismatch_no_effect": int(tot("mismatch_no_effect")), "timing_no_effect": int(tot("timing_no_effect")),
            "repeat_rate": tot("repeat") / st,
        }
        for d in sorted({e["domain"] for e in E}):
            D = [e for e in E if e["domain"] == d]
            per_domain[d][m] = (sum(e["success"] for e in D), len(D))
    f = lambda v, d=2: f"{v:.{d}f}"  # noqa: E731
    pct = lambda v: f"{100 * v:.1f}%"  # noqa: E731
    lines = [f"# Metrics on {len(tasks)} tasks finished by every method", "",
             "| method | success | score | steps/task | effective steps | loop s/task (mean / median) | s/step | "
             "VLM s/step | TTFT s | exec s/step | wait s/step | obs s/step | check s/step | idle s/step | premature | "
             "stale | timing no-effect | waste share | interrupt / invalidate / wake | S1 calls/step (ms) | "
             "tokens in/out per task | USD/task |",
             "|" + "---|" * 22]
    for m in labels:
        s = summary[m]
        lines.append(
            f"| {m} | {pct(s['success_rate'])} | {f(s['mean_score'], 3)} | {f(s['steps_per_task'], 1)} | "
            f"{s['effective_steps']} ({pct(s['effective_share'])}) | {f(s['loop_mean_s'], 0)} / {f(s['loop_median_s'], 0)} | "
            f"{f(s['s_per_step'])} | {f(s['vlm_per_step'])} | {f(s['ttft_mean_s'])} | {f(s['exec_per_step'])} | "
            f"{f(s['wait_per_step'])} | {f(s['obs_per_step'])} | {f(s['check_per_step'])} | {f(s['idle_per_step'])} | "
            f"{pct(s['premature_rate'])} | {pct(s['stale_rate'])} | {pct(s['timing_no_effect_rate'])} | "
            f"{pct(s['waste_share'])} | {s['interrupts']} / {s['invalidations']} / {s['early_wakes']} | "
            f"{f(s['system1_calls_per_step'])} ({f(s['system1_ms_mean'], 0)}) | "
            f"{f(s['tokens_in_per_task'] / 1000, 1)}k / {f(s['tokens_out_per_task'] / 1000, 1)}k | {f(s['usd_per_task'], 4)} |")
    lines += ["", "## Time without LLM calls", "",
              "| method | all LLM calls s/task (removed) | no-LLM s/task (mean / median) | no-LLM s/step | "
              "steps/task | success |", "|" + "---|" * 6]
    for m in labels:
        s = summary[m]
        lines.append(f"| {m} | {f(s['llm_all_per_task'], 0)} | {f(s['nollm_per_task'], 0)} / {f(s['nollm_median'], 0)} | "
                     f"{f(s['nollm_per_step'])} | {f(s['steps_per_task'], 1)} | {pct(s['success_rate'])} |")
    lines += ["", "## Controllable time (LLM generation removed) and invalid steps", "",
              "| method | LLM s/task (removed) | controllable s/task (mean / median) | controllable s/step | "
              "settle->act s/step (mean / median) | no-effect steps | mismatched dispatches (of targeted) | "
              "mismatched + no effect | timing-caused no-effect | repeats | prevented: invalidations / interrupts |",
              "|" + "---|" * 11]
    for m in labels:
        s = summary[m]
        lines.append(
            f"| {m} | {f(s['llm_per_task'], 0)} | {f(s['ctl_per_task'], 0)} / {f(s['ctl_median'], 0)} | "
            f"{f(s['ctl_per_step'])} | {f(s['settle_to_act_mean'])} / {f(s['settle_to_act_median'])} | "
            f"{pct(s['no_effect_rate'])} | {s['mismatch']} / {s['targeted']} ({pct(s['mismatch_rate'])}) | "
            f"{s['mismatch_no_effect']} | {s['timing_no_effect']} | {pct(s['repeat_rate'])} | "
            f"{s['invalidations']} / {s['interrupts']} |")
    groups = defaultdict(list)  # method -> its runs (avg@k over repeats / experiments)
    for m in labels:
        groups[run_of[m]].append(summary[m])
    if any(len(v) > 1 for v in groups.values()):
        sd = lambda v: S.stdev(v) if len(v) > 1 else 0.0  # noqa: E731
        pm = lambda k, R, sc=1.0, d=1: f"{sc * S.mean(r[k] for r in R):.{d}f} ± {sc * sd([r[k] for r in R]):.{d}f}"  # noqa: E731
        lines += ["", "## avg@k over runs (mean ± std across runs)", "",
                  "| method | runs | success % | steps/task | loop s/task | no-LLM s/task | no-LLM s/step | waste share % |",
                  "|" + "---|" * 8]
        for g, R in sorted(groups.items(), key=lambda x: (x[0].startswith("nightwatch"), x[0])):
            lines.append(f"| {g} | {len(R)} | {pm('success_rate', R, 100)} | {pm('steps_per_task', R)} | "
                         f"{pm('loop_mean_s', R, d=0)} | {pm('nollm_per_task', R, d=0)} | {pm('nollm_per_step', R, d=2)} | "
                         f"{pm('waste_share', R, 100)} |")
        summary["_avg"] = {g: {k: S.mean(r[k] for r in R) for k in R[0] if isinstance(R[0][k], (int, float))}
                           for g, R in groups.items()}
    lines += ["", "## Success by domain", "", "| domain | " + " | ".join(labels) + " |", "|---|" + "---|" * len(labels)]
    for d in sorted(per_domain):
        lines.append(f"| {d} | " + " | ".join(
            (f"{per_domain[d][m][0]}/{per_domain[d][m][1]}" if m in per_domain[d] else "-") for m in labels) + " |")
    out_dir = Path(out_dir or exp_root(exps[-1]) / "metrics")
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    jdump({"tasks": tasks, "methods": summary}, out_dir / "summary.json")
    rows = [row for m in labels for row in by[m].values()]
    if rows:
        with open(out_dir / "per_episode.csv", "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, list(rows[0]))
            w.writeheader()
            w.writerows(rows)
    with open(out_dir / "per_domain.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["domain"] + [f"{m} passed" for m in labels] + [f"{m} tasks" for m in labels])
        for d in sorted(per_domain):
            w.writerow([d] + [per_domain[d].get(m, (0, 0))[0] for m in labels] + [per_domain[d].get(m, (0, 0))[1] for m in labels])
    return "\n".join(lines) + f"\n\nwritten to {out_dir}"
