"""Per-question threshold calibration with Learn-then-Test (Angelopoulos et al., 2110.01052).

For each question q with labelled samples (p_i = fire probability, y_i = event present):
  * candidate thresholds λ on a grid; FP(λ) = #negatives with p >= λ, FN(λ) = #positives with p < λ;
  * exact binomial p-values for H0: FPR(λ) > α and H0: FNR(λ) > β; the joint null "FPR > α or FNR > β"
    is rejected only if both are (intersection-union: p = max of the two);
  * Bonferroni over the grid: λ is certified if p(λ) <= δ / m. Among certified λ we take the one with the
    lowest empirical miss rate. Then P(FPR <= α and FNR <= β) >= 1 - δ for the chosen λ (finite sample).
  * If no λ is certified jointly, fall back to FPR-only control by fixed-sequence testing from the most
    conservative λ downwards (valid because FPR is monotone in λ) and mark the guarantee `fpr_only`.
α is the per-query false-alarm rate; with r System-1 queries per minute on event-free screens it means
≈ α·r false wakes per minute, so `--false-per-min` is converted with the query rate seen in the data.
Held-out domains (OSWorld apps) give the test-set FPR / FNR reported next to the guarantee.
"""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binom

from nightwatch.config import DATA_DIR
from nightwatch.utils import jdump, read_jsonl

GRID = np.round(np.arange(0.02, 0.99, 0.01), 2)


def ltt_threshold(p: np.ndarray, y: np.ndarray, alpha: float, beta: float, delta: float) -> dict:
    p, y = np.asarray(p, float), np.asarray(y, bool)
    neg, pos = p[~y], p[y]
    n0, n1 = len(neg), len(pos)
    out = {"n_neg": n0, "n_pos": n1, "alpha": alpha, "beta": beta, "delta": delta}
    if n0 == 0:
        return {**out, "tau": None, "guarantee": "none", "why": "no negatives"}
    fp = np.array([(neg >= lam).sum() for lam in GRID])
    fn = np.array([(pos < lam).sum() for lam in GRID]) if n1 else np.zeros(len(GRID), int)
    p_fpr = binom.cdf(fp, n0, alpha)
    p_fnr = binom.cdf(fn, n1, beta) if n1 else np.ones(len(GRID))
    joint = np.maximum(p_fpr, p_fnr)
    ok = joint <= delta / len(GRID)
    if ok.any():
        idx = np.where(ok)[0]
        j = idx[np.argmin(fn[idx] / max(n1, 1))]
        g = "fpr_and_miss"
    else:
        j = None
        for i in range(len(GRID) - 1, -1, -1):  # fixed sequence from the most conservative threshold
            if p_fpr[i] <= delta:
                j = i
            else:
                break
        g = "fpr_only" if j is not None else "none"
    if j is None:
        return {**out, "tau": None, "guarantee": "none", "why": "too few negatives to certify the false-alarm rate"}
    return {**out, "tau": float(GRID[j]), "guarantee": g, "fpr_cal": float(fp[j] / n0),
            "fnr_cal": float(fn[j] / n1) if n1 else None}


def _rates(p, y, tau) -> dict:
    p, y = np.asarray(p, float), np.asarray(y, bool)
    if tau is None or len(p) == 0:
        return {}
    return {"fpr": float(((p >= tau) & ~y).sum() / max((~y).sum(), 1)),
            "fnr": float(((p < tau) & y).sum() / max(y.sum(), 1)) if y.any() else None,
            "n": int(len(p)), "n_pos": int(y.sum())}


def calibrate(root: Path, false_per_min: float = 1.0, miss: float = 0.2, delta: float = 0.1,
              test_domains: list[str] | None = None, out: str | None = None) -> str:
    rows = read_jsonl(Path(root) / "pilot" / "labelled.jsonl")
    if not rows:
        return f"no {root}/pilot/labelled.jsonl: run `python -m nightwatch pilot --exp {Path(root).name}` first"
    span = defaultdict(float)
    for r in rows:
        span[r["episode"]] = max(span[r["episode"]], r["t"])
    minutes = sum(span.values()) / 60 or 1.0
    by_q: dict = defaultdict(lambda: {"cal": ([], []), "test": ([], [])})
    queries_per_min = len({(r["episode"], r["t"]) for r in rows}) / minutes
    for r in rows:
        part = "test" if test_domains and r.get("domain") in test_domains else "cal"
        for qid, (pv, yv) in r["samples"].items():
            by_q[qid][part][0].append(pv)
            by_q[qid][part][1].append(yv)
    alpha = min(0.5, false_per_min / max(queries_per_min, 1e-9))
    result = {"source": str(root), "queries_per_min": round(queries_per_min, 2), "false_per_min": false_per_min,
              "alpha_per_query": alpha, "miss": miss, "delta": delta, "test_domains": test_domains, "thresholds": {}}
    lines = [f"calibration from {root} ({len(rows)} labelled frames, {queries_per_min:.1f} queries/min "
             f"-> per-query alpha {alpha:.4f}; miss <= {miss}; 1 - delta = {1 - delta})"]
    for qid, parts in sorted(by_q.items()):
        pc, yc = parts["cal"]
        row = ltt_threshold(np.array(pc), np.array(yc), alpha, miss, delta)
        if parts["test"][0]:
            row["test"] = _rates(*parts["test"], row.get("tau"))
        result["thresholds"][qid] = row
        lines.append(f"  {qid:16s} tau={row.get('tau')} [{row['guarantee']}] n={row['n_neg']}-/{row['n_pos']}+ "
                     f"cal fpr={row.get('fpr_cal')} fnr={row.get('fnr_cal')} test={row.get('test', {})}")
    dest = Path(out) if out else DATA_DIR / "calibration" / "thresholds.json"
    jdump(result, dest)
    jdump(result, Path(root) / "pilot" / "thresholds.json")
    lines.append(f"-> {dest} (used by methods with rules.calibration pointing to it)")
    return "\n".join(lines)


def load(path: Path) -> dict:
    return json.loads(Path(path).read_text())
