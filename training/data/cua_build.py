"""Desktop (CUA) System-1 examples from recorded agent episodes, labelled by a teacher (hosted Jev) mixed with
hindsight gold from the frame stream: a "CUA Jev" distillation set.

Input: episodes run with `frames.record=true frames.save_obs=true` (steps.jsonl, frames.jsonl + frames/, obs/),
on the generated training task pool (training/data/cua_tasks.py), never on benchmark tasks.

Decision points, built with questions.build_request exactly as the loop sends them:
  reasoning  seen = the agent's screenshot, now = a frame while it reasoned (frames that changed first), crop =
             the changed region
  pre_exec   seen, now_marked (red circle at the click), target crops, now = last frame before dispatch
  post_exec  before = the agent's screenshot, recent = now - 0.5 s, now = action end + {0.3, 0.7, 1.2, 2, 3} s
Labels per question, over its options in canonical order (noul [no, yes], choice CHANGE_KINDS):
  teacher  Jev's distribution for the same request (OPENROUTER_API_KEY)
  gold     where the frame stream decides it: reasoning with no visible change -> no event; pre_exec with the
           target area unchanged (< 5 % of pixels) -> target present, not occluded, not changed; post_exec
           still_changing / settled (pixel motion around t; settled not used when Jev sees loading) / effect
  mixed    q = alpha * gold + (1 - alpha) * teacher; gold one-hot when the two disagree on the answer
           (JevAgentHarness data recipe, alpha 0.3)
Filtering: failed teacher calls dropped; near-duplicate requests within an episode dropped; at most
--per-episode examples per episode; split by task id (val share --val).

  python -m training.data.cua_build --runs runs/cua_collect --out data/cua_train/ds --workers 12
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image

from nightwatch.actions import parse_action
from nightwatch.diff import ChangeDetector
from nightwatch.questions import BANK, CHANGE_KINDS, build_request
from nightwatch.utils import box_around, crop, draw_marker, small_gray

POST_OFFSETS = (0.3, 0.7, 1.2, 2.0, 3.0)
CLEAR, NONE = 0.001, 0.0002  # changed share of the screen: a clear change / no change worth the name
NO_EVENT = ("dialog", "error", "page_changed", "toast", "focus_lost")


def _rgb(p: Path) -> np.ndarray:
    return np.asarray(Image.open(p).convert("RGB"))


def _fit(img: np.ndarray, w: int, h: int) -> np.ndarray:
    return img if img.shape[1] == w and img.shape[0] == h else np.asarray(Image.fromarray(img).resize((w, h)))


class Episode:
    def __init__(self, d: Path):
        self.d = d
        self.steps = [json.loads(l) for l in open(d / "steps.jsonl")]
        self.frames = [json.loads(l) for l in open(d / "frames.jsonl")]
        self.t = np.array([f["t"] for f in self.frames])
        self.frac = np.array([float(f.get("frac") or 0.0) if f.get("sig") else 0.0 for f in self.frames])
        first = _rgb(d / "frames" / self.frames[0]["file"]) if self.frames else None
        self.h, self.w = (first.shape[:2] if first is not None else (0, 0))
        self._cache: dict = {}

    def frame(self, i: int) -> np.ndarray:
        if i not in self._cache:
            if len(self._cache) > 64:
                self._cache.clear()
            self._cache[i] = _rgb(self.d / "frames" / self.frames[i]["file"])
        return self._cache[i]

    def at(self, t: float) -> int:
        """Index of the last frame captured at or before t (-1 if none)."""
        return int(np.searchsorted(self.t, t, side="right")) - 1

    def obs(self, name: str) -> np.ndarray:
        return _fit(_rgb(self.d / "obs" / name), self.w, self.h)

    def motion(self, t0: float, t1: float) -> float:
        """Largest frame-to-frame changed fraction (significant changes only) in (t0, t1]."""
        m = (self.t > t0) & (self.t <= t1)
        return float(self.frac[m].max()) if m.any() else 0.0


def _raw_to_dist(qid: str, raw: dict) -> list[float] | None:
    q = next(q for q in BANK if q.id == qid)
    r = raw.get(qid) or {}
    if q.kind == "choice":
        pr = r.get("probs") or {}
        v = [max(0.0, float(pr.get(k, 0.0))) for k in CHANGE_KINDS]
        s = sum(v)
        return [x / s for x in v] if s > 0 else None
    if "yes" not in r:
        return None
    p = min(max(float(r["yes"]), 0.0), 1.0)
    return [1 - p, p]


# gold that is a fact about the pixels (nothing changed at all) overrides the teacher; post-exec pixel motion is not
# readiness (a tooltip fading or a focus ring moving is motion on a ready screen), so it only sharpens agreement
OVERRIDE = {("reasoning", q) for q in ("dialog", "error", "page_changed", "toast", "focus_lost")} | \
           {("pre_exec", q) for q in ("target_present", "target_occluded", "target_changed")}


def _mix(teacher: list[float], gold: list[float] | None, alpha: float, override: bool) -> tuple[list[float], str]:
    if gold is None:
        return teacher, "teacher"
    if int(np.argmax(teacher)) != int(np.argmax(gold)):
        return (gold, "gold") if override else (teacher, "teacher_conflict")
    return [alpha * g + (1 - alpha) * t for g, t in zip(gold, teacher)], "mixed"


def _label_value(qid: str, dist: list[float]):
    q = next(q for q in BANK if q.id == qid)
    return list(CHANGE_KINDS)[int(np.argmax(dist))] if q.kind == "choice" else bool(dist[1] >= 0.5)


def candidates(ep: Episode, rng: random.Random, det: ChangeDetector, instruction: str) -> list[dict]:
    """Decision-point requests of one episode (images as arrays, gold where the frame stream decides it)."""
    out = []
    for i, s in enumerate(ep.steps):
        if not s.get("obs_file") or s.get("t_dispatch") is None:
            continue
        seen = ep.obs(s["obs_file"])
        g_seen = small_gray(seen, 480)
        act = parse_action(s.get("action") or "", "pyautogui")  # the whole snippet: its first pointer action
        desc = act.describe()
        t_obs, t_disp, t_done = s["t_obs_final"], s["t_dispatch"], s.get("t_action_done") or s["t_dispatch"]
        nxt = ep.steps[i + 1]["t_obs_final"] if i + 1 < len(ep.steps) else t_done + 4.0
        # ---- reasoning: frames while the agent reasoned
        idx = [j for j in range(max(0, ep.at(t_obs) + 1), max(0, ep.at(t_disp) + 1))]
        changed = [j for j in idx if det.compare(g_seen, small_gray(ep.frame(j), 480)).significant]
        picks = rng.sample(changed, min(2, len(changed)))
        quiet = [j for j in idx if j not in changed]
        if quiet and rng.random() < 0.3:
            picks.append(rng.choice(quiet))
        for j in picks:
            now = ep.frame(j)
            ch = det.compare(g_seen, small_gray(now, 480))
            box = ch.union if ch.significant else None
            cr = crop(now, box, pad=24, min_side=160) if box else now
            gold = {q: [1.0, 0.0] for q in NO_EVENT} if not ch.significant else {}
            out.append({"cp": "reasoning", "images": {"seen": seen, "now": now, "crop": cr}, "action": "",
                        "gold": gold, "meta": {"step": s["step"], "t": ep.frames[j]["t"], "changed": ch.significant}})
        # ---- pre_exec: the click the agent dispatched
        if act.has_target and act.kind in ("click", "double_click", "right_click"):
            j = ep.at(t_disp)
            if j >= 0:
                now = ep.frame(j)
                sx, sy = ep.w / 1920.0, ep.h / 1080.0
                x, y = act.x * sx, act.y * sy
                half = int(80 * sx)
                tb = box_around(x, y, half)
                ts, tn = crop(seen, tb, pad=0, min_side=2 * half), crop(now, tb, pad=0, min_side=2 * half)
                same = float((np.abs(ts.astype(np.int16).mean(-1) - tn.astype(np.int16).mean(-1)) > 16).mean()) < 0.05
                gold = {"target_present": [0.0, 1.0], "target_occluded": [1.0, 0.0], "target_changed": [1.0, 0.0]} if same else {}
                out.append({"cp": "pre_exec", "images": {"seen": seen, "now_marked": draw_marker(now, x, y, max(8, int(14 * sx))),
                                                         "target_seen": ts, "target_now": tn},
                            "action": desc, "gold": gold, "meta": {"step": s["step"], "t": ep.frames[j]["t"], "target_same": same}})
        # ---- post_exec: after the action, before the next observation
        j_pre = ep.at(t_disp)
        pre = ep.frame(j_pre) if j_pre >= 0 else seen
        for off in POST_OFFSETS:
            t = t_done + off
            if t >= nxt + 1.0:
                break
            j, jr = ep.at(t), ep.at(t - 0.5)
            if j < 0:
                continue
            now, recent = ep.frame(j), ep.frame(max(jr, 0))
            ch_pre = det.compare(small_gray(pre, 480), small_gray(now, 480))
            box = ch_pre.union if ch_pre.significant else None
            cr = crop(now, box, pad=24, min_side=160) if box else now
            # gold only where the pixels are unambiguous (CLEAR / NONE); in between the teacher alone decides
            gold = {}
            recent_m, around_m = ep.motion(t - 0.5, t), ep.motion(t - 0.4, t + 0.6)
            if recent_m >= CLEAR:
                gold["still_changing"] = [0.0, 1.0]
            elif recent_m == 0.0:
                gold["still_changing"] = [1.0, 0.0]
            if around_m == 0.0:
                gold["settled"] = [0.0, 1.0]  # hindsight: nothing moved from 0.4 s before to 0.6 s after
            elif around_m >= CLEAR:
                gold["settled"] = [1.0, 0.0]
            if ch_pre.significant and ch_pre.frac >= CLEAR:
                gold["effect"] = [0.0, 1.0]
            elif ch_pre.frac < NONE:
                gold["effect"] = [1.0, 0.0]
            out.append({"cp": "post_exec", "images": {"before": seen, "recent": recent, "now": now, "crop": cr},
                        "action": desc, "gold": gold, "meta": {"step": s["step"], "t": ep.frames[j]["t"], "offset": off}})
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True, help="run folder(s) with recorded episodes, comma separated")
    ap.add_argument("--out", default="data/cua_train/ds")
    ap.add_argument("--per-episode", type=int, default=60)
    ap.add_argument("--alpha", type=float, default=0.3)
    ap.add_argument("--val", type=float, default=0.1)
    ap.add_argument("--workers", type=int, default=12)
    ap.add_argument("--max-side", type=int, default=1280, help="stored image size (the trainer resizes again)")
    ap.add_argument("--limit", type=int, default=None, help="episodes (smoke tests)")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    from nightwatch.system1.remote import JevS1

    eps = sorted({p.parent for r in a.runs.split(",") for p in Path(r.strip()).rglob("frames.jsonl")
                  if (p.parent / "obs").is_dir() and "_setup_failed" not in p.parts})
    eps = eps[: a.limit] if a.limit else eps
    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    det = ChangeDetector((1920, 1080), small_width=480)
    teacher = JevS1()
    lock = threading.Lock()
    stats = Counter()
    fh = open(out / "examples.jsonl", "a", encoding="utf-8")

    def save(arr: np.ndarray, key: str) -> str:
        im = Image.fromarray(arr)
        if max(im.size) > a.max_side:
            s = a.max_side / max(im.size)
            im = im.resize((max(28, round(im.width * s)), max(28, round(im.height * s))))
        name = f"images/{key}.jpg"
        im.save(out / name, quality=90)
        return name

    def one(ep_dir: Path) -> None:
        res = json.loads((ep_dir / "result.json").read_text()) if (ep_dir / "result.json").exists() else {}
        if res.get("error") and "evaluate" not in res.get("error", ""):
            with lock:
                stats["episodes_skipped_error"] += 1
            return
        task_id = ep_dir.name
        split = "val" if int(hashlib.md5(task_id.encode()).hexdigest(), 16) % 1000 < a.val * 1000 else "train"
        rng = random.Random(f"{a.seed}-{task_id}")
        ep = Episode(ep_dir)
        cands = candidates(ep, rng, det, res.get("instruction", ""))
        rng.shuffle(cands)
        kept, seen_hash = 0, set()
        for k, c in enumerate(cands):
            if kept >= a.per_episode:
                break
            h = hashlib.md5(b"".join(small_gray(v, 64).tobytes() for v in c["images"].values()) + c["cp"].encode()).hexdigest()
            if h in seen_hash:
                with lock:
                    stats["dropped_duplicate"] += 1
                continue
            seen_hash.add(h)
            req = build_request(c["cp"], c["images"], BANK, res.get("instruction", ""), c["action"])
            try:
                raw = teacher._raw(req)
            except Exception as e:
                with lock:
                    stats["dropped_teacher_error"] += 1
                    if stats["dropped_teacher_error"] <= 3:
                        print("teacher failed:", e, flush=True)
                continue
            soft, labels, src = {}, {}, {}
            for q, _ in req.questions:
                t = _raw_to_dist(q.id, raw)
                if t is None:
                    continue
                g = c["gold"].get(q.id)
                if q.id == "settled" and g is not None and (_raw_to_dist("loading", raw) or [1, 0])[1] >= 0.5:
                    g = None  # a quiet screen that the teacher sees loading: pixels do not decide "settled"
                d, src[q.id] = _mix(t, g, a.alpha, (c["cp"], q.id) in OVERRIDE)
                soft[q.id], labels[q.id] = [round(x, 4) for x in d], _label_value(q.id, d)
                with lock:
                    stats[f"label_{src[q.id]}"] += 1
                    if g is not None:
                        stats["gold_agrees" if int(np.argmax(t)) == int(np.argmax(g)) else "gold_disagrees"] += 1
            if not soft:
                continue
            key = f"{task_id[:8]}_{c['cp']}_{c['meta']['step']}_{k}"
            paths = [save(img, f"{key}_{role}") for role, img in zip(c["images"], req.images)]
            row = {"id": key, "split": split, "task": task_id, "domain": ep_dir.parent.name, "checkpoint": c["cp"],
                   "state": req.state, "questions": [[q.id, text] for q, text in req.questions], "images": paths,
                   "labels": labels, "soft": soft, "label_source": src, "meta": c["meta"]}
            with lock:
                fh.write(json.dumps(row) + "\n")
                fh.flush()
                stats[f"{split}_{c['cp']}"] += 1
            kept += 1
        with lock:
            stats["episodes"] += 1
            if stats["episodes"] % 10 == 0:
                print(f"[cua_build] {stats['episodes']}/{len(eps)} episodes, {sum(v for k, v in stats.items() if k.startswith(('train_', 'val_')))} examples", flush=True)

    with ThreadPoolExecutor(a.workers) as ex:
        list(ex.map(one, eps))
    fh.close()
    (out / "stats.json").write_text(json.dumps(dict(stats), indent=2))
    print(json.dumps(dict(stats), indent=2))


if __name__ == "__main__":
    main()
