"""Build System-1 training examples from SynthWeb recordings (exact labels from page ground truth).

Every example is one checkpoint request exactly as the loop sends it (questions.build_request: same image
roles, same question text, same state text), with a label for each question of that checkpoint:

  reasoning  seen = frame at t - lag (lag ~ VLM latency), now = frame t, crop = changed region
  pre_exec   seen, now (+ red marker at the point), crops around a point that held an element at `seen`
  post_exec  before = frame right before a recorded action, recent = now - 0.5 s, now = action + offset

Half of the `now` frames are drawn where the ground truth changed since `seen` (events, transitions), half
uniformly (mostly benign: ambient motion, clock, nothing), so negatives include the hard benign changes.
Split by page seed: seed % 10 in 0-7 train, 8 val, 9 test; `--holdout-families` moves every page that uses
one of those families to a separate `holdout` split (generalisation to unseen event types).

  python -m training.data.build --rec data/synthweb/rec --out data/synthweb/ds
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
from PIL import Image

from nightwatch.diff import ChangeDetector
from nightwatch.questions import BANK, CHANGE_KINDS, build_request
from nightwatch.utils import box_around, crop, draw_marker, small_gray

TASKS = ["Update the {x} settings and save them.", "Find the {x} report and export it.", "Submit the {x} form.",
         "Open {x} and check whether the latest change was applied.", "Accept or dismiss any prompt, then open {x}.",
         "Wait for {x} to finish and confirm the result."]


def _el_at(gt, x, y):
    if not gt:
        return None
    for e in gt.get("elements", []):
        if e[1] <= x <= e[3] and e[2] <= y <= e[4]:
            return e
    return None


def _inside(b, x, y):
    return bool(b) and b[0] <= x <= b[2] and b[1] <= y <= b[3]


def _overlaps(a, b, pad=4):
    return bool(a and b) and not (a[2] < b[0] - pad or b[2] < a[0] - pad or a[3] < b[1] - pad or b[3] < a[1] - pad)


def label(qid: str, g: dict, ctx: dict):
    """Ground truth of P(yes) (bool) or the choice key; None = undefined for this example."""
    seen = g.get("seen") or g.get("before") or {}
    now = g.get("now") or g.get("now_marked") or {}
    before = g.get("before") or {}
    x, y = ctx.get("point") or (None, None)
    if qid == "still_changing":
        return bool(now["animating"])
    if qid == "loading":
        return bool(now["loading"])
    if qid == "dialog":
        return bool(now["dialog"]) and not seen["dialog"]
    if qid == "error":
        return bool(now["error"]) and not seen["error"]
    if qid == "toast":
        return bool(now["toast"]) and not seen["toast"]
    if qid == "page_changed":
        return now["page"] != seen["page"]
    if qid == "focus_lost":
        return now["focus"] != seen["focus"]
    if qid in ("target_present", "target_changed", "target_occluded", "target_disabled"):
        if x is None:
            return None
        a, b = _el_at(seen, x, y), _el_at(now, x, y)
        occl = _inside(now.get("dialog_box"), x, y) and not (b and b[0].startswith("dlg"))
        if qid == "target_occluded":
            return bool(occl)
        if qid == "target_disabled":
            return bool(b) and not b[5]
        same = a is not None and b is not None and a[0] == b[0] and not occl
        moved = same and max(abs(a[i] - b[i]) for i in range(1, 5)) > 8
        return same if qid == "target_present" else (not same) or moved
    if qid == "effect":
        return now["version"] != before["version"]
    if qid == "settled":
        return not now["loading"] and not now["animating"]
    if qid == "unexpected":
        return bool(now["error"]) and not before.get("error")
    if qid == "long_operation":
        return bool(now["long_op"])
    if qid == "change_kind":
        cb = ctx.get("crop_box")
        if now["dialog"] and not seen["dialog"]:
            return "dialog"
        if now["error"] and not seen["error"]:
            return "error"
        if now["toast"] and not seen["toast"]:
            return "toast"
        if now["page"] != seen["page"]:
            return "navigation"
        if cb is None:
            return None
        amb = now.get("ambient_boxes") or []
        if any(_overlaps(cb, a, 0) and cb[0] >= a[0] - 8 and cb[1] >= a[1] - 8 and cb[2] <= a[2] + 8 and cb[3] <= a[3] + 8
               for a in amb):
            return "trivial" if any(a[3] - a[1] < 40 and a[2] - a[0] < 200 for a in amb if _overlaps(cb, a, 0)) else "ambient"
        return "content" if now["version"] != seen["version"] else None
    return None


class Builder:
    def __init__(self, out: Path, holdout: set[str], seed: int = 0):
        self.out = out
        self.img = out / "img"
        self.img.mkdir(parents=True, exist_ok=True)
        self.holdout = holdout
        self.rng = random.Random(seed)
        self.counts: dict = {}

    def _save(self, arr: np.ndarray, name: str) -> str:
        p = self.img / name
        Image.fromarray(arr).save(p, quality=90)
        return str(p.relative_to(self.out))

    def episode(self, ep: Path, n_reason: int, n_pre: int, n_post: int, fh) -> None:
        spec = json.loads((ep / "spec.json").read_text())
        frames = [json.loads(l) for l in open(ep / "frames.jsonl")]
        actions = [json.loads(l) for l in open(ep / "actions.jsonl")]
        if len(frames) < 30:
            return
        seed = spec["seed"]
        split = "holdout" if self.holdout & set(spec["families"]) else ("train" if seed % 10 < 8 else "val" if seed % 10 == 8 else "test")
        ts = np.array([f["t"] for f in frames])
        W, H = spec["width"], spec["height"]
        det = ChangeDetector((W, H))
        task = self.rng.choice(TASKS).format(x=self.rng.choice(spec["nav"] + [spec["title"]]))
        cache: dict = {}

        def img(i):
            if i not in cache:
                cache[i] = np.asarray(Image.open(ep / "frames" / frames[i]["file"]).convert("RGB"))
                if len(cache) > 48:
                    cache.pop(next(iter(cache)))
            return cache[i]

        def at(t):
            return int(np.clip(np.searchsorted(ts, t, side="right") - 1, 0, len(frames) - 1))

        def pick_now(i_seen, lo, hi):
            cand = [j for j in range(at(ts[i_seen] + lo), at(ts[i_seen] + hi) + 1)]
            changed = [j for j in cand if _gt_key(frames[j]["gt"]) != _gt_key(frames[i_seen]["gt"])]
            return self.rng.choice(changed if changed and self.rng.random() < 0.5 else cand)

        def emit(cp, images, gts, ctx, action, meta):
            qs = [q for q in BANK if cp in q.sets]
            req = build_request(cp, images, qs, task, action, gts, ctx)
            labels = {}
            for q, _ in req.questions:
                v = label(q.id, {r: g for r, g in gts.items()}, ctx)
                if v is not None:
                    labels[q.id] = v
            if not labels:
                return
            k = f"{seed:07d}_{cp}_{meta['i']:06d}_{self.rng.randrange(1 << 30):08x}"
            paths = []
            for r_i, a in enumerate(req.images):
                if isinstance(a, str):
                    paths.append(a)
                else:
                    paths.append(self._save(a, f"{k}_{r_i}.jpg"))
            fh.write(json.dumps({"id": k, "split": split, "seed": seed, "families": spec["families"], "checkpoint": cp,
                                 "state": req.state, "questions": [[q.id, text] for q, text in req.questions],
                                 "images": paths, "labels": labels, "ctx": ctx, "meta": meta}) + "\n")
            self.counts[(split, cp)] = self.counts.get((split, cp), 0) + 1

        rel = lambda i: str((ep / "frames" / frames[i]["file"]).resolve())  # noqa: E731
        for _ in range(n_reason):
            i_s = self.rng.randrange(0, len(frames) - 25)
            i_n = pick_now(i_s, 0.5, self.rng.uniform(1.5, 7.0))
            ch = det.compare(small_gray(img(i_s)), small_gray(img(i_n)))
            emit("reasoning", {"seen": rel(i_s), "now": rel(i_n), "crop": crop(img(i_n), ch.union, 24, 160) if ch.union else rel(i_n)},
                 {"seen": frames[i_s]["gt"], "now": frames[i_n]["gt"], "crop": frames[i_n]["gt"]},
                 {"crop_box": ch.union}, "", {"i": i_n, "seen": i_s})
        for _ in range(n_pre):
            i_s = self.rng.randrange(0, len(frames) - 25)
            els = frames[i_s]["gt"]["elements"]
            if not els:
                continue
            e = self.rng.choice(els)
            x, y = (e[1] + e[3]) / 2 + self.rng.uniform(-5, 5), (e[2] + e[4]) / 2 + self.rng.uniform(-3, 3)
            i_n = pick_now(i_s, 0.3, self.rng.uniform(1.0, 7.0))
            tb = box_around(x, y, 80)
            emit("pre_exec", {"seen": rel(i_s), "now_marked": draw_marker(img(i_n), x, y),
                              "target_seen": crop(img(i_s), tb, 0, 160), "target_now": crop(img(i_n), tb, 0, 160)},
                 {"seen": frames[i_s]["gt"], "now_marked": frames[i_n]["gt"], "target_seen": frames[i_s]["gt"], "target_now": frames[i_n]["gt"]},
                 {"point": [x, y]}, f"click at ({x:.0f}, {y:.0f})", {"i": i_n, "seen": i_s})
        acts = [a for a in actions if a["idx_before"] + 5 < len(frames)]
        for a in self.rng.sample(acts, min(len(acts), max(1, n_post // 3))):
            i_b = a["idx_before"]
            lag = self.rng.uniform(0.3, 5.0)  # simulated VLM thinking time between the agent's screenshot and the action
            for off in sorted(self.rng.sample([0.1, 0.2, 0.3, 0.5, 0.8, 1.2, 2.0, 3.0, 5.0], 3)):
                i_n = at(a["t"] + off)
                if i_n <= i_b:
                    continue
                prev = at(ts[i_n] - 0.12)
                ch_mov = det.compare(small_gray(img(prev)), small_gray(img(i_n)))
                ch_pre = det.compare(small_gray(img(i_b)), small_gray(img(i_n)))
                cb = ch_mov.union if ch_mov.significant else ch_pre.union
                i_r = at(ts[i_n] - 0.5)
                i_s = at(a["t"] - lag)  # the screenshot the agent saw (as in the loop: before = obs_frame)
                desc = f"type the text '{a['text']}'" if a["kind"] == "type" else f"click at ({a['x']:.0f}, {a['y']:.0f})"
                emit("post_exec", {"before": rel(i_s), "recent": rel(i_r), "now": rel(i_n),
                                   "crop": crop(img(i_n), cb, 24, 160) if cb else rel(i_n)},
                     {"before": frames[i_s]["gt"], "recent": frames[i_r]["gt"], "now": frames[i_n]["gt"], "crop": frames[i_n]["gt"]},
                     {"crop_box": cb, "action": a}, desc, {"i": i_n, "before": i_s, "pre_action": i_b, "offset": off})


def _gt_key(g):
    return (g["version"], g["dialog"], g["toast"], g["error"], g["loading"], g["animating"], g["page"])


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rec", default="data/synthweb/rec")
    ap.add_argument("--out", default="data/synthweb/ds")
    ap.add_argument("--reasoning", type=int, default=12)
    ap.add_argument("--pre", type=int, default=12)
    ap.add_argument("--post", type=int, default=18)
    ap.add_argument("--holdout-families", default="")
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    b = Builder(out, set(f for f in a.holdout_families.split(",") if f))
    eps = sorted(p for p in Path(a.rec).iterdir() if (p / "done").exists())[: a.limit]
    with open(out / "examples.jsonl", "w") as fh:
        for i, ep in enumerate(eps):
            b.episode(ep, a.reasoning, a.pre, a.post, fh)
            if i % 100 == 0:
                print(f"{i}/{len(eps)} episodes, {sum(b.counts.values())} examples", flush=True)
    stats = {f"{s}/{c}": n for (s, c), n in sorted(b.counts.items())}
    (out / "stats.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
