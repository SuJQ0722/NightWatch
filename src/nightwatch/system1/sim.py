"""Simulated System-1 for the simulated desktop: answers from ground truth plus noise.

Pipeline checks only. Numbers produced with it are never results.
"""

from __future__ import annotations

import random

from nightwatch.config import env
from nightwatch.questions import Request
from nightwatch.system1.base import System1


def _el_at(gt: dict | None, x: float, y: float):
    if not gt or x is None:
        return None
    for el in gt.get("elements", []):
        eid, x0, y0, x1, y1 = el[:5]
        if x0 <= x <= x1 and y0 <= y <= y1:
            return el
    return None


def _in(box, x, y) -> bool:
    return bool(box) and box[0] <= x <= box[2] and box[1] <= y <= box[3]


def truth(qid: str, gts: dict, ctx: dict):
    seen, now = gts.get("seen") or gts.get("before") or {}, gts.get("now") or gts.get("now_marked") or {}
    before = gts.get("before") or {}
    x, y = (ctx.get("point") or (None, None))
    if qid == "still_changing":
        return bool(now.get("animating"))
    if qid == "loading":
        return bool(now.get("loading"))
    if qid == "dialog":
        return bool(now.get("popup")) and not seen.get("popup")
    if qid == "error":
        return bool(now.get("error")) and not seen.get("error")
    if qid == "page_changed":
        return now.get("page") != seen.get("page")
    if qid == "toast":
        return bool(now.get("toast")) and not seen.get("toast")
    if qid == "focus_lost":
        return bool(now.get("popup")) and not seen.get("popup")
    if qid in ("target_present", "target_changed"):
        a, b = _el_at(seen, x, y), _el_at(now, x, y)
        same = a is not None and b is not None and a[0] == b[0] and not _in(now.get("popup_box"), x, y)
        return same if qid == "target_present" else not same
    if qid == "target_occluded":
        return _in(now.get("popup_box"), x, y)
    if qid == "target_disabled":
        el = _el_at(now, x, y)
        return bool(el) and len(el) > 5 and not el[5]
    if qid == "effect":
        return now.get("version") != before.get("version")
    if qid == "settled":
        return not now.get("loading") and not now.get("animating")
    if qid == "unexpected":
        return bool(now.get("error"))
    if qid == "long_operation":
        return bool(now.get("long_op"))
    return False


def change_kind(gts: dict, ctx: dict) -> str:
    seen, now = gts.get("seen") or gts.get("before") or {}, gts.get("now") or {}
    box = ctx.get("crop_box")
    amb = now.get("ambient_box")  # the question is about the crop: judge the crop first
    if box and amb and box[0] >= amb[0] - 4 and box[1] >= amb[1] - 4 and box[2] <= amb[2] + 4 and box[3] <= amb[3] + 4:
        return "ambient"
    if now.get("popup") and not seen.get("popup"):
        return "dialog"
    if now.get("error") and not seen.get("error"):
        return "error"
    if now.get("toast") and not seen.get("toast"):
        return "toast"
    if now.get("page") != seen.get("page"):
        return "navigation"
    return "content"


class SimS1(System1):
    name = "sim"
    needs_gt = True

    def __init__(self, noise: float | None = None, flip: float | None = None, seed: int = 0):
        self.noise = float(env("SIM_S1_NOISE", "0.08") if noise is None else noise)
        self.flip = float(env("SIM_S1_FLIP", "0.005") if flip is None else flip)
        self.rng = random.Random(seed)

    def _p(self, val: bool) -> float:
        if self.rng.random() < self.flip:
            val = not val
        p = (0.9 if val else 0.08) + self.rng.gauss(0, self.noise)
        return min(max(p, 0.001), 0.999)

    def _raw(self, req: Request) -> dict:
        from nightwatch.questions import ROLES

        gts = {r: g for (r, _), g in zip(ROLES[req.checkpoint], req.gts)}
        out = {}
        for q, _ in req.questions:
            if q.kind == "choice":
                k = change_kind(gts, req.context)
                if self.rng.random() < self.flip:
                    k = self.rng.choice(list(q.options))
                out[q.id] = {"probs": {o: (0.8 if o == k else 0.2 / (len(q.options) - 1)) for o in q.options}}
            else:
                out[q.id] = {"yes": self._p(truth(q.id, gts, req.context))}  # raw P(yes); fire_on applied later
        return out
