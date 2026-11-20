"""Post-exec "settled" examples labelled by hindsight only: is the screen FINAL, i.e. will it not change again in the
next `--horizon` seconds? Built from the same recorded training episodes as cua_build (training task pool, never
benchmark tasks), at the moments the waiter has to decide:

  after a change   the first and second quiet frame after each significant change that follows the action (k = 1, 2)
  no change        action end + {0.3, 0.6} s while nothing has changed since the dispatch

cua_build's `settled` label needs 0.4 s of quiet before the frame, so a screen that has just stopped moving is "not
settled" by construction and the model cannot wake earlier than the pixel rule; here only the future decides.
Labels: settled = no significant change in (t, t + horizon] (gold, no teacher). No still_changing label: on these
moments (right after a change) it fights cua_build's still_changing and the runtime check that uses it, and the
fast tier does not ask it. Same row format and task split as cua_build (train / val by task id).

    python -m training.data.settle_build --runs runs/cua_collect --out data/cua_train/settle_ds
"""

from __future__ import annotations

import argparse
import hashlib
import json
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from nightwatch.actions import parse_action
from nightwatch.diff import ChangeDetector
from nightwatch.questions import BANK, build_request
from nightwatch.utils import crop, small_gray
from training.data.cua_build import CLEAR, NONE, Episode

QUESTIONS = ["settled", "loading"]  # the fast tier; loading is asked but has no pixel label


def candidates(ep: Episode, det: ChangeDetector, horizon: float) -> list[dict]:
    out = []
    sig = np.array([bool(f.get("sig")) for f in ep.frames])
    for i, s in enumerate(ep.steps):
        ex = [g for g in s.get("segments", []) if g[0] == "exec"]
        if not s.get("obs_file") or not ex:
            continue
        t_disp, t_done = ex[0][1], ex[-1][2]
        nxt = next((g[1] for g in (ep.steps[i + 1].get("segments", []) if i + 1 < len(ep.steps) else []) if g[0] == "exec"),
                   None)
        t_end = (nxt if nxt is not None else ep.t[-1]) - horizon  # the future must not hold the next action's effect
        j_pre = ep.at(t_disp)
        if j_pre < 0:
            continue
        seen, pre = ep.obs(s["obs_file"]), ep.frame(j_pre)
        act = parse_action(s.get("action") or "", "pyautogui")
        desc = act.describe() if act else (s.get("action") or "")[:200]
        picks = []  # (frame index, kind, k / offset)
        changed = [j for j in range(j_pre + 1, len(ep.frames)) if sig[j] and ep.t[j] <= t_done + 3.0]
        for off in (0.3, 0.6):  # nothing changed yet at the decision moment (whether or not it changes later)
            j = int(np.searchsorted(ep.t, t_done + off))
            if j < len(ep.frames) and not sig[j_pre + 1: j + 1].any():
                picks.append((j, "no_change", off))
        for c in changed:
            for k in (1, 2):  # k-th quiet frame after this change (no change in between)
                j = c + k
                if j < len(ep.frames) and ep.t[j] > t_done and not sig[c + 1: j + 1].any():
                    picks.append((j, "after_change", k))
        for j, kind, kk in picks:
            t = ep.t[j]
            if t > t_end:
                continue
            fut = ep.motion(t, t + horizon)
            if NONE < fut < CLEAR:
                continue  # neither clearly final nor clearly about to change
            now, recent = ep.frame(j), ep.frame(max(ep.at(t - 0.5), 0))
            ch = det.compare(small_gray(pre, 480), small_gray(now, 480))
            cr = crop(now, ch.union, pad=24, min_side=160) if ch.significant else now
            gold = {"settled": fut == 0.0}
            out.append({"images": {"before": seen, "recent": recent, "now": now, "crop": cr}, "action": desc,
                        "gold": gold, "meta": {"step": s["step"], "t": round(float(t), 3), "kind": kind, "k": kk,
                                               "since_done": round(float(t - t_done), 3)}})
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default="runs/cua_collect")
    ap.add_argument("--out", default="data/cua_train/settle_ds")
    ap.add_argument("--horizon", type=float, default=1.0)
    ap.add_argument("--per-episode", type=int, default=20, help="no-change moments kept per episode")
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--max-side", type=int, default=768)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args(argv)
    eps = sorted({p.parent for r in a.runs.split(",") for p in Path(r.strip()).rglob("frames.jsonl")
                  if (p.parent / "obs").is_dir() and "_setup_failed" not in p.parts})
    eps = eps[: a.limit] if a.limit else eps
    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    det = ChangeDetector((1920, 1080), small_width=480)
    lock, stats = threading.Lock(), Counter()
    fh = open(out / "examples.jsonl", "w", encoding="utf-8")

    def save(arr: np.ndarray, key: str) -> str:
        name = f"images/{key}.jpg"
        if not (out / name).exists():
            im = Image.fromarray(arr)
            if max(im.size) > a.max_side:
                sc = a.max_side / max(im.size)
                im = im.resize((max(28, round(im.width * sc)), max(28, round(im.height * sc))))
            im.save(out / name, quality=90)
        return name

    def one(ep_dir: Path) -> None:
        res = json.loads((ep_dir / "result.json").read_text()) if (ep_dir / "result.json").exists() else {}
        task_id = ep_dir.name
        split = "val" if int(hashlib.md5(task_id.encode()).hexdigest(), 16) % 1000 < a.val * 1000 else "train"
        ep = Episode(ep_dir)
        cands = candidates(ep, det, a.horizon)
        rng = np.random.default_rng(int(hashlib.md5(task_id.encode()).hexdigest()[:8], 16))
        # every after-change moment (the decision that matters, and "about to change" is rare); no-change moments
        # capped per episode, keeping every one whose screen does change late
        nc = [i for i, c in enumerate(cands) if c["meta"]["kind"] == "no_change" and c["gold"]["settled"]]
        if len(nc) > a.per_episode:
            drop = set(rng.choice(nc, len(nc) - a.per_episode, replace=False).tolist())
            cands = [c for i, c in enumerate(cands) if i not in drop]
        for n, c in enumerate(cands):
            req = build_request("post_exec", c["images"], BANK, res.get("instruction", ""), c["action"], only=QUESTIONS)
            key = f"{hashlib.md5(str(ep_dir).encode()).hexdigest()[:10]}_s{c['meta']['step']}"
            paths = [save(img, f"{key}_before" if role == "before" else f"{key}_{n}_{role}")
                     for role, img in zip(c["images"], req.images)]
            labels = {q: v for q, v in c["gold"].items()}
            row = {"id": f"{key}_{n}", "split": split, "task": task_id, "domain": ep_dir.parent.name,
                   "checkpoint": "post_exec", "state": req.state, "questions": [[q.id, text] for q, text in req.questions],
                   "images": paths, "labels": labels, "soft": {q: [float(not v), float(v)] for q, v in labels.items()},
                   "label_source": {q: "gold" for q in labels}, "meta": c["meta"]}
            with lock:
                fh.write(json.dumps(row) + "\n")
                stats[f"{split}_{c['meta']['kind']}_{'final' if labels['settled'] else 'changes'}"] += 1
        with lock:
            stats["episodes"] += 1
            if stats["episodes"] % 40 == 0:
                print(f"[settle_build] {stats['episodes']}/{len(eps)} episodes", flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(one, eps))
    fh.close()
    (out / "stats.json").write_text(json.dumps(dict(sorted(stats.items())), indent=2))
    print(json.dumps(dict(sorted(stats.items())), indent=2))


if __name__ == "__main__":
    main()
