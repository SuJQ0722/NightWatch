"""The fixed event-question bank (groups A-D of the proposal, 16 questions) and request building.

Questions are fixed templates; the harness fills the slots (image roles, the parsed action), never an
LLM. Each checkpoint sends one System-1 request with its own ordered images and the subset of the
bank that applies to it (always <= 16 questions, the Intern-Decision limit):

  reasoning  (during VLM generation)   images: seen, now, crop                      -> A3, B1-B5
  pre_exec   (before dispatch)         images: seen, now_marked, target_seen, target_now -> B1, C1-C4
  post_exec  (after the action)        images: before, recent, now, crop            -> A1-A3, B1, B2, B4, D1-D4

Every yes/no question has a `fire_on` answer; `fire` = P(fire_on) is what thresholds apply to. The
`label` names the event class used to calibrate the question (see analysis/labels.py).
`priority` orders the bank for the 4 / 8 / 12 / 16-question ablation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass(frozen=True)
class Q:
    id: str
    group: str
    kind: str  # noul | choice
    text: str
    label: str
    priority: int
    fire_on: str = "yes"
    options: dict | None = None
    sets: tuple = ()


CHANGE_KINDS = {
    "dialog": "a pop-up window, dialog box, permission request or menu",
    "toast": "a short-lived notification, toast, tooltip or status message",
    "error": "an error or warning message",
    "content": "an ordinary content update inside the same page or application",
    "navigation": "a different page, window or application",
    "keyboard": "an on-screen keyboard or input-method popup",
    "ambient": "an ad, carousel, video or animation unrelated to the task",
    "trivial": "the mouse cursor, a text caret, a clock or another trivial change",
}
BENIGN_KINDS = ("ambient", "trivial")

BANK: list[Q] = [
    # ---- A: change and stability
    Q("still_changing", "A", "noul", "Compare {recent} (taken about half a second earlier) with {now}. Is the screen still in the "
      "middle of changing: an animation, a transition, scrolling, or content that is still being drawn?", "changing", 6,
      sets=("post_exec",)),
    Q("loading", "A", "noul", "Does {now} show that something is still loading: a spinner, a progress bar, a busy cursor, "
      "placeholder blocks, or 'Loading' text?", "loading", 7, sets=("post_exec",)),
    Q("change_kind", "A", "choice", "{crop} is a close-up of the part of {now} that just changed. What kind of change is it?",
      "change_kind", 9, options=CHANGE_KINDS, sets=("reasoning", "post_exec")),
    # ---- B: events that make any plan stale
    Q("dialog", "B", "noul", "Compared with {seen}, does {now} show a new pop-up window, dialog box, permission request or "
      "system prompt covering part of the interface?", "dialog", 1, sets=("reasoning", "pre_exec", "post_exec")),
    Q("error", "B", "noul", "Compared with {seen}, does {now} show a new error, warning or failure message?", "error", 5,
      sets=("reasoning", "post_exec")),
    Q("page_changed", "B", "noul", "Is {now} showing a different application, window or page than {seen}, rather than the "
      "same page with small updates?", "navigation", 10, sets=("reasoning",)),
    Q("toast", "B", "noul", "Does {now} show a short-lived notification, toast or status message (for example 'Saved' or "
      "'No network found') that is not in {seen}?", "toast", 11, sets=("reasoning", "post_exec")),
    Q("focus_lost", "B", "noul", "Is the window that is active in {seen} no longer active or no longer visible in {now}?",
      "focus", 13, sets=("reasoning",)),
    # ---- C: is the pending action still valid
    Q("target_present", "C", "noul", "{target_seen} shows the element the agent is about to act on, cut from {seen} around the "
      "action point. {target_now} is the same area of the current screen. Is the same element still there, at the same "
      "place, in {target_now}?", "target_gone", 2, fire_on="no", sets=("pre_exec",)),
    Q("target_occluded", "C", "noul", "In {now_marked} the red circle marks where the agent is about to act; the agent "
      "chose that point while looking at {seen}. Is the point now covered by a pop-up, menu, tooltip, dialog or other "
      "overlay that is NOT in {seen}? (Clicking a button inside a dialog the agent already saw is not covered.)",
      "target_occluded", 8, sets=("pre_exec",)),
    Q("target_changed", "C", "noul", "Comparing {target_seen} with {target_now}, did this area change in a way that matters for "
      "acting on it (different text, a different element, or a moved layout)?", "target_changed", 12, sets=("pre_exec",)),
    Q("target_disabled", "C", "noul", "In {target_now}, does the element in the centre look disabled, greyed out, or not yet "
      "ready to use?", "target_disabled", 14, sets=("pre_exec",)),
    # ---- D: effect of the action and readiness for the next observation
    Q("effect", "D", "noul", "The agent just did this: {action}. Comparing {before} (what the agent saw before acting) with "
      "{now}, did the action produce a visible effect on the screen?", "effect", 3, sets=("post_exec",)),
    Q("settled", "D", "noul", "Is {now} a settled screen that is ready for the next step, with nothing still loading, "
      "animating or half-drawn?", "settled", 4, sets=("post_exec",)),
    Q("unexpected", "D", "noul", "The agent just did this: {action}. Does {now} show something unexpected as a result, such as "
      "the wrong window opening, an error, or the action hitting a different element?", "unexpected", 15, sets=("post_exec",)),
    Q("long_operation", "D", "noul", "Does {now} show a long operation in progress (saving, installing, downloading, "
      "computing) that will take more than a few more seconds?", "long_operation", 16, sets=("post_exec",)),
]
BY_ID = {q.id: q for q in BANK}
assert len(BANK) == 16 and len({q.priority for q in BANK}) == 16

ROLES = {
    "reasoning": [("seen", "the screenshot the agent is currently reasoning about"),
                  ("now", "the current screen"),
                  ("crop", "a close-up of the region of the current screen that changed")],
    "pre_exec": [("seen", "the screenshot the agent based its next action on"),
                 ("now_marked", "the current screen, with a red circle at the point the agent is about to act on"),
                 ("target_seen", "a close-up of the action point in the screenshot the agent saw"),
                 ("target_now", "a close-up of the same area of the current screen")],
    "post_exec": [("before", "the screenshot the agent last looked at (its last action was based on it)"),
                  ("recent", "the screen about half a second ago"),
                  ("now", "the current screen"),
                  ("crop", "a close-up of the region that changed since the action")],
}


def bank_subset(size: int | None = None, ids: list[str] | None = None) -> list[Q]:
    if ids:
        return [BY_ID[i] for i in ids]
    qs = sorted(BANK, key=lambda q: q.priority)
    return qs[: size or len(qs)]


@dataclass
class Request:
    checkpoint: str
    state: str
    images: list  # np.ndarray, in role order
    questions: list[tuple[Q, str]]  # (question, rendered text)
    gts: list = field(default_factory=list)  # per image ground truth (simulated System-1 only)
    context: dict = field(default_factory=dict)  # extra info for the simulated backend

    def qids(self) -> list[str]:
        return [q.id for q, _ in self.questions]


def build_request(checkpoint: str, images: dict[str, np.ndarray], bank: list[Q], task: str = "",
                  action: str = "", gts: dict | None = None, context: dict | None = None,
                  only: list[str] | None = None) -> Request | None:
    roles = ROLES[checkpoint]
    names = {r: f"image {i + 1}" for i, (r, _) in enumerate(roles)}
    fill = dict(names, action=action or "an action")
    fill.setdefault("seen", names.get("before"))  # post_exec: "what the agent last saw" = the pre-action screen
    fill.setdefault("now", names.get("now_marked"))  # pre_exec: the current screen carries the marker
    qs = [(q, q.text.format(**fill)) for q in bank if checkpoint in q.sets and (only is None or q.id in only)]
    if not qs:
        return None
    task = (task or "").strip().replace("\n", " ")
    if len(task) > 240:
        task = task[:237] + "..."
    lines = ["A computer-use agent is operating a desktop. " + (f"Its task: {task}" if task else ""),
             "Images: " + "; ".join(f"{names[r]} is {desc}" for r, desc in roles) + "."]
    if action:
        lines.append(f"The agent's action: {action}.")
    return Request(checkpoint, "\n".join(lines), [images[r] for r, _ in roles], qs,
                   [(gts or {}).get(r) for r, _ in roles], context or {})


@dataclass
class Answers:
    """System-1 output for one request: fire probabilities (and choice distributions)."""
    fire: dict = field(default_factory=dict)  # qid -> P(fire_on)
    probs: dict = field(default_factory=dict)  # qid -> {option: p} for choice questions
    ms: float = 0.0
    backend: str = ""
    server_ms: float | None = None  # http backends: time inside the server (queue + model), and client-side encoding
    encode_ms: float | None = None

    def p(self, qid: str, default: float | None = None) -> float | None:
        return self.fire.get(qid, default)

    def benign_change(self) -> float:
        d = self.probs.get("change_kind")
        return float(sum(d.get(k, 0.0) for k in BENIGN_KINDS)) if d else 0.0

    def to_dict(self) -> dict:
        return {"fire": {k: round(v, 4) for k, v in self.fire.items()},
                "probs": {k: {o: round(p, 4) for o, p in d.items()} for k, d in self.probs.items()},
                "ms": round(self.ms, 2), "backend": self.backend,
                **({"server_ms": round(self.server_ms, 2)} if self.server_ms is not None else {}),
                **({"encode_ms": round(self.encode_ms, 2)} if self.encode_ms is not None else {})}
