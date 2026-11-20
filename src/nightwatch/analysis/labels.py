"""Event labels for recorded episodes, and labelling sheets for humans.

Label sources, merged per episode (all as intervals `class, t_start, t_end` on the episode clock):
  labels.csv     human labels (the sheet exported by `nightwatch label`; classes below)
  noise.jsonl    injected dialogs / notifications (a `noise:` block in the config): exact start, approximate end
  frames.jsonl   simulated desktop ground truth (gt per frame): exact
  weak pixel     `changing` from the stage-1 statistics when nobody labelled it (marked weak)

Classes: dialog, error, toast, focus, navigation (instant: t_start = t_end), loading, changing,
long_operation.  Question -> label semantics live in `question_label()`.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

from nightwatch.analysis.common import episodes
from nightwatch.utils import read_jsonl

CLASSES = ["dialog", "error", "toast", "focus", "navigation", "loading", "changing", "long_operation"]


def _from_gt(rows: list[dict]) -> list[tuple]:
    out: list = []
    open_: dict = {}
    keymap = {"popup": "dialog", "error": "error", "toast": "toast", "loading": "loading", "animating": "changing",
              "long_op": "long_operation"}
    last_page = None
    for r in rows:
        gt, t = r.get("gt"), r["t"]
        if not gt:
            continue
        for k, cls in keymap.items():
            on = bool(gt.get(k))
            if on and cls not in open_:
                open_[cls] = t
            elif not on and cls in open_:
                out.append((cls, open_.pop(cls), t))
        if last_page is not None and gt.get("page") != last_page:
            out.append(("navigation", t, t))
        last_page = gt.get("page")
    if rows:
        out.extend((cls, t0, rows[-1]["t"]) for cls, t0 in open_.items())
    return out


def load_labels(ep: Path) -> tuple[list[tuple], dict]:
    ep = Path(ep)
    labels, src = [], {"human": False, "noise": False, "gt": False, "weak_changing": False}
    if (ep / "labels.csv").exists():
        with open(ep / "labels.csv", newline="", encoding="utf-8") as fh:
            for row in csv.DictReader(fh):
                if row.get("class") in CLASSES and row.get("t_start"):
                    t0 = float(row["t_start"])
                    t1 = float(row["t_end"]) if row.get("t_end") else t0
                    labels.append((row["class"], t0, t1))
        src["human"] = bool(labels)
    for r in read_jsonl(ep / "noise.jsonl"):
        if r.get("ok"):
            dur = 15.0 if r["kind"] == "dialog" else 4.0
            labels.append(("dialog" if r["kind"] == "dialog" else "toast", r["t_done"] + 0.3, r["t_done"] + dur))
            src["noise"] = True
    frames = read_jsonl(ep / "frames.jsonl")
    if frames and frames[0].get("gt"):
        labels.extend(_from_gt(frames))
        src["gt"] = True
    if not any(c == "changing" for c, _, _ in labels):
        stats = _frame_stats(ep)
        run = None
        for t, _frac, sig, *_ in stats:
            if sig and run is None:
                run = [t, t]
            elif sig:
                run[1] = t
            elif run is not None:
                labels.append(("changing", run[0] - 0.1, run[1]))
                run = None
        src["weak_changing"] = True
    return labels, src


def _frame_stats(ep: Path) -> list:
    p = Path(ep) / "frame_stats.json"
    return json.loads(p.read_text())["rows"] if p.exists() else []


def active(labels: list[tuple], cls: str, t: float, tol: float = 0.0) -> bool:
    return any(c == cls and t0 - tol <= t <= t1 + tol for c, t0, t1 in labels)


def happened(labels: list[tuple], cls: str, ta: float, tb: float) -> bool:
    return any(c == cls and ta < t0 <= tb for c, t0, _ in labels)


def question_label(qid: str, labels: list[tuple], t_seen: float, t_now: float) -> bool | None:
    """Ground truth of P(yes) for a question asked at t_now about a screenshot seen at t_seen (None = unknown)."""
    new = lambda c: active(labels, c, t_now) and not active(labels, c, t_seen)  # noqa: E731
    if qid in ("dialog", "error", "toast"):
        return new(qid)
    if qid == "focus_lost":
        return new("dialog") or new("focus")
    if qid == "page_changed":
        return happened(labels, "navigation", t_seen, t_now)
    if qid == "still_changing":
        return active(labels, "changing", t_now)
    if qid == "loading":
        return active(labels, "loading", t_now)
    if qid == "settled":
        return not (active(labels, "loading", t_now) or active(labels, "changing", t_now))
    if qid == "long_operation":
        return active(labels, "long_operation", t_now)
    return None  # action-dependent questions (C group, effect, unexpected) need online labels


def export_sheets(root: Path, every: float = 0.5) -> str:
    """Per recorded episode: labels.csv template + label_sheet.html (thumbnails every `every` s)."""
    n = 0
    for ep in episodes(root):
        frames = read_jsonl(ep / "frames.jsonl")
        if not frames:
            continue
        if not (ep / "labels.csv").exists():
            (ep / "labels.csv").write_text("class,t_start,t_end,note\n", encoding="utf-8")
        steps = read_jsonl(ep / "steps.jsonl")
        marks = {round(s["t_obs"], 1): f"obs {s['step']}" for s in steps}
        marks.update({round(s["t_dispatch"], 1): f"act {s['step']}: {s['act_kind']}" for s in steps if s.get("t_dispatch")})
        cells, next_t = [], 0.0
        for f in frames:
            if f["t"] >= next_t:
                tag = marks.get(round(f["t"], 1), "")
                cells.append(f'<figure><img loading="lazy" src="frames/{f["file"]}"><figcaption>t={f["t"]:.1f}'
                             f'{" · " + tag if tag else ""}</figcaption></figure>')
                next_t = f["t"] + every
        html = ("<!doctype html><meta charset=utf-8><title>label sheet</title><style>body{font:13px sans-serif;margin:16px}"
                "main{display:grid;grid-template-columns:repeat(auto-fill,minmax(240px,1fr));gap:8px}"
                "img{width:100%}figure{margin:0}</style>"
                f"<h3>{ep}</h3><p>Fill labels.csv: class in {', '.join(CLASSES)}; times are the captions.</p>"
                f"<main>{''.join(cells)}</main>")
        (ep / "label_sheet.html").write_text(html, encoding="utf-8")
        n += 1
    return f"label sheets written for {n} recorded episode(s) under {root}"
