"""Stage 3: deterministic rules for the three checkpoints. System-1 only reads state; code decides.

Thresholds apply to fire probabilities (P(fire_on) of each question). They come from the config
(`rules.thresholds`, default 0.5) and, when `rules.calibration` points to a thresholds.json written by
`python -m nightwatch calibrate`, from the Learn-then-Test calibration (per question).
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from nightwatch.config import ROOT
from nightwatch.questions import Answers


@dataclass
class Decision:
    action: str  # interrupt | invalidate | ignore | none
    reason: str = ""
    p: float = 0.0


@dataclass
class PostView:
    event: str = ""  # a B-group event worth waking for right away
    event_p: float = 0.0
    effect: bool | None = None
    settled: bool | None = None
    changing: bool | None = None
    loading: bool = False
    long_op: bool = False
    benign: bool = False
    detail: dict = field(default_factory=dict)


def load_thresholds(rules_cfg) -> dict:
    th = {k: float(v) for k, v in dict(rules_cfg.get("thresholds") or {}).items()}
    cal = rules_cfg.get("calibration")
    if cal:
        p = Path(cal)
        p = p if p.is_absolute() else ROOT / p
        if p.exists():
            data = json.loads(p.read_text())
            for qid, row in data.get("thresholds", {}).items():
                if row.get("tau") is not None:
                    th[qid] = float(row["tau"])
        elif rules_cfg.get("require_calibration"):
            raise FileNotFoundError(f"calibration file {p} not found (run `python -m nightwatch calibrate`)")
    return th


class Rules:
    def __init__(self, rules_cfg):
        self.cfg = rules_cfg
        self.th = load_thresholds(rules_cfg)
        self.default_tau = float(rules_cfg.get("default_tau", 0.5))
        self.benign_tau = float(rules_cfg.get("benign_tau", 0.6))
        self.interrupt_on = list(rules_cfg.get("interrupt_on", ["dialog", "error", "page_changed"]))
        self.invalidate_on = list(rules_cfg.get("invalidate_on", ["target_present", "target_occluded", "target_changed", "dialog"]))
        self.wake_on = list(rules_cfg.get("wake_on", ["dialog", "error", "toast"]))
        # withholding an action the agent chose costs a whole regeneration when wrong: a stricter bar than waking
        self.invalidate_tau = float(rules_cfg.get("invalidate_tau", 0.0))

    def tau(self, qid: str) -> float:
        return self.th.get(qid, self.default_tau)

    def fires(self, ans: Answers, qid: str, floor: float = 0.0) -> tuple[bool, float]:
        p = ans.p(qid)
        return (p is not None and p >= max(self.tau(qid), floor)), (p or 0.0)

    def _first(self, ans: Answers, qids: list[str], floor: float = 0.0) -> tuple[str, float]:
        """The firing question with the largest margin over its threshold ("", 0) if none fires; `floor` raises
        every threshold to at least that value."""
        best, bp, margin = "", 0.0, -1.0
        for q in qids:
            ok, p = self.fires(ans, q, floor)
            tau = max(self.tau(q), floor)
            if ok and p - tau > margin:
                best, bp, margin = q, p, p - tau
        return best, bp

    # ------------------------------------------------------------------ checkpoints
    def during_reasoning(self, ans: Answers) -> Decision:
        q, p = self._first(ans, self.interrupt_on)
        if q:
            return Decision("interrupt", q, p)
        if ans.benign_change() >= self.benign_tau:
            return Decision("ignore", "benign_change", ans.benign_change())
        return Decision("none")

    def before_execution(self, ans: Answers, keyboard_only: bool = False) -> Decision:
        qids = ["dialog"] if keyboard_only else self.invalidate_on
        q, p = self._first(ans, qids, self.invalidate_tau)
        return Decision("invalidate", q, p) if q else Decision("none")

    def after_execution(self, ans: Answers) -> PostView:
        v = PostView(detail=ans.to_dict()["fire"])
        q, p = self._first(ans, self.wake_on)
        if q:
            v.event, v.event_p = q, p
        if ans.p("effect") is not None:
            v.effect = self.fires(ans, "effect")[0]
        if ans.p("settled") is not None:
            v.settled = self.fires(ans, "settled")[0]
        if ans.p("still_changing") is not None:
            v.changing = self.fires(ans, "still_changing")[0]
        v.loading = self.fires(ans, "loading")[0]
        v.long_op = self.fires(ans, "long_operation")[0]
        v.benign = ans.benign_change() >= self.benign_tau
        # "unexpected" is too vague to wake on zero-shot (Jev puts a median 0.60 on it);
        # it is logged for calibration and only wakes when listed in rules.wake_on
        return v
