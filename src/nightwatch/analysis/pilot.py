"""Pilot 4.1(1): can System-1 tell dialogs, transient toasts, errors, loading and ordinary updates apart
on recorded GUI frames, how well calibrated is it, and how fast?

For every recorded episode (frames.jsonl + frames/) we sample a frame every `every` seconds and ask
the two event requests the loop uses (`reasoning`: B group + change kind; `post_exec`: A + D groups)
with `seen` / `before` = the frame `seen_lag` seconds earlier (a typical VLM thinking time) and
`recent` = 0.5 s earlier. Answers go to pilot/answers.jsonl; joined with labels (analysis/labels.py)
they go to pilot/labelled.jsonl, which `nightwatch calibrate` reads. Summary: AUROC, accuracy, Brier, ECE
per question; latency percentiles; a go / no-go line against the thresholds below.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from nightwatch.analysis.common import episodes
from nightwatch.analysis.labels import load_labels, question_label
from nightwatch.diff import ChangeDetector
from nightwatch.questions import BANK, build_request
from nightwatch.system1 import make_system1
from nightwatch.utils import crop, decode_image, jdump, jline, read_jsonl, small_gray

GO = {"dialog": 0.85, "error": 0.85, "toast": 0.85, "loading": 0.8, "settled": 0.8}


def auroc(p, y) -> float | None:
    p, y = np.asarray(p, float), np.asarray(y, bool)
    if y.all() or not y.any():
        return None
    order = np.argsort(p)
    ranks = np.empty(len(p))
    ranks[order] = np.arange(1, len(p) + 1)
    for v in np.unique(p):  # average ranks for ties
        m = p == v
        ranks[m] = ranks[m].mean()
    n1, n0 = y.sum(), (~y).sum()
    return float((ranks[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def ece(p, y, bins: int = 10) -> float:
    p, y = np.asarray(p, float), np.asarray(y, float)
    idx = np.minimum((p * bins).astype(int), bins - 1)
    return float(sum(abs(p[idx == b].mean() - y[idx == b].mean()) * (idx == b).mean() for b in range(bins) if (idx == b).any()))


def pilot(root: Path, every: float = 0.5, backend: str | None = None, limit: int | None = None,
          seen_lag: float = 3.0) -> str:
    root = Path(root)
    out = root / "pilot"
    out.mkdir(exist_ok=True)
    s1 = make_system1(backend)
    s1.warmup()
    ans_fh = open(out / "answers.jsonl", "w", encoding="utf-8")
    lab_fh = open(out / "labelled.jsonl", "w", encoding="utf-8")
    lat: dict = {"reasoning": [], "post_exec": []}
    n_ep = 0
    for ep in episodes(root):
        frames = read_jsonl(ep / "frames.jsonl")
        if not frames:
            continue
        n_ep += 1
        labels, src = load_labels(ep)
        domain = ep.parent.name
        task = json.loads((ep / "result.json").read_text()).get("instruction", "") if (ep / "result.json").exists() else ""
        cache: dict = {}

        def img(i):
            if i not in cache:
                cache[i] = decode_image((ep / "frames" / frames[i]["file"]).read_bytes())
                if len(cache) > 64:
                    cache.pop(next(iter(cache)))
            return cache[i]

        ts = np.array([f["t"] for f in frames])

        def at(t):
            return int(max(0, np.searchsorted(ts, t, side="right") - 1))

        det = None
        next_t, n = ts[0] + seen_lag, 0
        for i, f in enumerate(frames):
            if f["t"] < next_t:
                continue
            next_t = f["t"] + every
            n += 1
            if limit and n > limit:
                break
            now = img(i)
            if det is None:
                det = ChangeDetector((now.shape[1], now.shape[0]))
            j_seen, j_rec = at(f["t"] - seen_lag), at(f["t"] - 0.5)
            seen, recent = img(j_seen), img(j_rec)
            ch = det.compare(small_gray(recent), small_gray(now))
            if not ch.significant:
                ch = det.compare(small_gray(seen), small_gray(now))
            cimg = crop(now, ch.union, 24, 160) if ch.union else now
            g = {"seen": frames[j_seen].get("gt"), "now": f.get("gt"), "crop": f.get("gt"),
                 "before": frames[j_seen].get("gt"), "recent": frames[j_rec].get("gt")}
            row = {"episode": str(ep), "domain": domain, "t": f["t"], "frame": f["idx"], "t_seen": frames[j_seen]["t"]}
            samples = {}
            for cp, images in (("reasoning", {"seen": seen, "now": now, "crop": cimg}),
                               ("post_exec", {"before": seen, "recent": recent, "now": now, "crop": cimg})):
                req = build_request(cp, images, BANK, task, "", g, {"crop_box": ch.union})
                a = s1.ask(req)
                lat[cp].append(a.ms)
                row[cp] = a.to_dict()
                for qid, p in a.fire.items():
                    y = question_label(qid, labels, frames[j_seen]["t"], f["t"])
                    if y is not None:
                        samples[qid] = [round(p, 4), bool(y)]
            jline(ans_fh, row)
            jline(lab_fh, {"episode": row["episode"], "domain": domain, "t": f["t"], "samples": samples, "label_src": src})
    ans_fh.close()
    lab_fh.close()
    return summarize(root, lat, n_ep)


def summarize(root: Path, lat: dict, n_ep: int) -> str:
    rows = read_jsonl(Path(root) / "pilot" / "labelled.jsonl")
    by: dict = {}
    for r in rows:
        for qid, (p, y) in r["samples"].items():
            by.setdefault(qid, ([], []))
            by[qid][0].append(p)
            by[qid][1].append(y)
    table = {}
    lines = [f"# Pilot: System-1 event detection ({n_ep} episodes, {len(rows)} frames)", "",
             "| question | n | positives | AUROC | acc@0.5 | Brier | ECE |", "|---|---|---|---|---|---|---|"]
    for qid, (p, y) in sorted(by.items()):
        p, y = np.array(p), np.array(y, bool)
        a = auroc(p, y)
        table[qid] = {"n": int(len(p)), "pos": int(y.sum()), "auroc": a, "acc": float(((p >= 0.5) == y).mean()),
                      "brier": float(((p - y) ** 2).mean()), "ece": ece(p, y)}
        t = table[qid]
        lines.append(f"| {qid} | {t['n']} | {t['pos']} | {a if a is None else round(a, 3)} | {t['acc']:.3f} | "
                     f"{t['brier']:.3f} | {t['ece']:.3f} |")
    lat_s = {cp: {"p50": float(np.percentile(v, 50)), "p95": float(np.percentile(v, 95)), "n": len(v)}
             for cp, v in lat.items() if v}
    lines += ["", "Latency (ms): " + ", ".join(f"{cp} p50 {v['p50']:.0f} / p95 {v['p95']:.0f}" for cp, v in lat_s.items())]
    verdict = []
    for q, need in GO.items():
        a = table.get(q, {}).get("auroc")
        verdict.append(f"{q}: {'n/a' if a is None else f'{a:.3f}'} (need {need}) {'ok' if a is not None and a >= need else 'NOT MET'}")
    lines += ["", "Go / no-go (AUROC):", *[f"- {v}" for v in verdict]]
    jdump({"questions": table, "latency_ms": lat_s, "go": verdict}, Path(root) / "pilot" / "summary.json")
    (Path(root) / "pilot" / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return "\n".join(lines)
