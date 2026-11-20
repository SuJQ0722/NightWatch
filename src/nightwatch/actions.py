"""Structured view of an agent action (harness-side parsing; the agent is not asked for anything new).

OSWorld pyautogui actions are Python snippets. We extract the first pointer target (click / moveTo /
drag / scroll at x, y), the kind of action and a one-line description for System-1 prompts. The
pre-execution check only runs for actions with a target; keyboard-only actions get the
"blocking dialog appeared" check instead.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

SPECIAL = ("WAIT", "DONE", "FAIL")
_NUM = r"(-?\d+(?:\.\d+)?)"
_POINTER = re.compile(
    r"pyautogui\.(click|doubleClick|tripleClick|rightClick|middleClick|moveTo|dragTo|mouseDown|mouseUp|scroll|hscroll|vscroll)"
    r"\(\s*(?:x\s*=\s*)?" + _NUM + r"\s*,\s*(?:y\s*=\s*)?" + _NUM)
_SCROLL_ONLY = re.compile(r"pyautogui\.(?:scroll|hscroll|vscroll)\(\s*(-?\d+)\s*(?:,\s*x\s*=\s*" + _NUM + r"\s*,\s*y\s*=\s*" + _NUM + r")?")
_TYPE = re.compile(r"pyautogui\.(?:typewrite|write)\(\s*(['\"])(.*?)\1", re.S)
_KEYS = re.compile(r"pyautogui\.(?:hotkey|press|keyDown)\(([^)]*)\)")


@dataclass
class Act:
    raw: object
    kind: str  # click | double_click | right_click | move | drag | scroll | type | key | wait | done | fail | other
    x: float | None = None
    y: float | None = None
    text: str = ""

    @property
    def has_target(self) -> bool:
        return self.x is not None and self.kind in ("click", "double_click", "right_click", "drag", "scroll", "move")

    @property
    def special(self) -> bool:
        return self.kind in ("wait", "done", "fail")

    def describe(self) -> str:
        if self.kind in ("click", "double_click", "right_click"):
            return f"{self.kind.replace('_', ' ')} at ({self.x:.0f}, {self.y:.0f})"
        if self.kind == "drag":
            return f"drag to ({self.x:.0f}, {self.y:.0f})"
        if self.kind == "scroll":
            return "scroll" + (f" at ({self.x:.0f}, {self.y:.0f})" if self.x is not None else "")
        if self.kind == "type":
            t = self.text if len(self.text) <= 40 else self.text[:37] + "..."
            return f"type the text '{t}'"
        if self.kind == "key":
            return f"press {self.text}"
        if self.kind == "move":
            return f"move the mouse to ({self.x:.0f}, {self.y:.0f})"
        return {"wait": "wait", "done": "declare the task done", "fail": "declare the task infeasible"}.get(self.kind, "run a script")

    def same_as(self, other: "Act", tol: float = 8.0) -> bool:
        if other is None or self.kind != other.kind:
            return False
        if self.x is not None and other.x is not None:
            return abs(self.x - other.x) <= tol and abs(self.y - other.y) <= tol
        return str(self.raw).strip() == str(other.raw).strip()


_AOI_CLICK = re.compile(r"^(triple_click|double_click|click)\s*\(?\s*x?(\d+(?:\.\d+)?)[,\s]+y?(\d+(?:\.\d+)?)", re.I)
_AOI_TYPE_AT = re.compile(r"^type_text_at\s*\(\s*(\d+)[,\s]+(\d+)[,\s]+[\"'](.*?)[\"']", re.I | re.S)
_AOI_TYPE = re.compile(r"^(?:type|enter|input|fill|select)\s*\(\s*(?:#?[\w-]+\s*,\s*)?[\"'](.*?)[\"']", re.I | re.S)
_AOI_KEY = re.compile(r"^key\s*\(\s*([^)]+)\)", re.I)
_AOI_SCROLL = re.compile(r"^scroll\s*\(", re.I)
_AOI_WAIT = re.compile(r"^(?:wait\b|wait\(|i.ll wait|observe|pause|no$)", re.I)
_AOI_DONE = {"done", "complete", "finish", "done.", "task complete", "task done", "done(success)", "done(failure)",
             "stop", "stop()", "stop()."}


def parse_aoi_action(action: str) -> Act:
    """DynaCU-Bench / AOI text actions: click(x, y), type("t"), fill(#id, "t"), key(Enter), scroll(down, 300),
    wait(), done, speak("t")."""
    a = str(action or "").strip().strip("`").strip()
    low = a.lower()
    if low in _AOI_DONE:
        return Act(a, "done")
    m = _AOI_TYPE_AT.match(a)
    if m:
        return Act(a, "click", float(m.group(1)), float(m.group(2)), m.group(3))
    m = _AOI_CLICK.match(low)
    if m:
        kind = {"click": "click", "double_click": "double_click", "triple_click": "double_click"}[m.group(1)]
        return Act(a, kind, float(m.group(2)), float(m.group(3)))
    m = _AOI_TYPE.match(a)
    if m:
        return Act(a, "type", text=m.group(1))
    m = _AOI_KEY.match(a)
    if m:
        return Act(a, "key", text=m.group(1).strip().strip("'\""))
    if _AOI_SCROLL.match(a):
        return Act(a, "scroll")
    if low.startswith("speak"):
        return Act(a, "other", text=a)
    if _AOI_WAIT.match(low) or not low:
        return Act(a, "wait")
    return Act(a, "other")


def parse_action(action, space: str = "pyautogui") -> Act:
    if space == "aoi":
        return parse_aoi_action(action)
    if isinstance(action, dict):  # computer_13 style
        t = str(action.get("action_type", "")).upper()
        if t in SPECIAL:
            return Act(action, t.lower())
        x, y = action.get("parameters", {}).get("x"), action.get("parameters", {}).get("y")
        kind = {"CLICK": "click", "DOUBLE_CLICK": "double_click", "RIGHT_CLICK": "right_click", "MOVE_TO": "move",
                "DRAG_TO": "drag", "SCROLL": "scroll", "TYPING": "type", "PRESS": "key", "HOTKEY": "key"}.get(t, "other")
        return Act(action, kind, x, y, str(action.get("parameters", {}).get("text", "")))
    code = str(action or "").strip()
    if code in SPECIAL:
        return Act(code, code.lower())
    m = _POINTER.search(code)
    if m:
        fn, x, y = m.group(1), float(m.group(2)), float(m.group(3))
        kind = {"click": "click", "doubleClick": "double_click", "tripleClick": "double_click", "rightClick": "right_click",
                "middleClick": "click", "moveTo": "move", "dragTo": "drag", "mouseDown": "click", "mouseUp": "click"}.get(fn, "scroll")
        # moveTo followed by a click without coordinates is still a click at that point
        if kind == "move" and re.search(r"pyautogui\.(click|doubleClick|rightClick|mouseDown)\(\s*\)", code):
            kind = "click"
        if kind == "move" and "dragTo" in code:
            kind = "drag"
        txt = _TYPE.search(code)
        return Act(code, kind, x, y, txt.group(2) if txt else "")
    m = _SCROLL_ONLY.search(code)
    if m:
        return Act(code, "scroll", float(m.group(2)) if m.group(2) else None, float(m.group(3)) if m.group(3) else None)
    m = _TYPE.search(code)
    if m:
        return Act(code, "type", text=m.group(2))
    m = _KEYS.search(code)
    if m:
        keys = "+".join(k.strip().strip("'\"") for k in m.group(1).split(",") if k.strip())
        return Act(code, "key", text=keys)
    return Act(code, "other")


_TRAILING_SLEEP = re.compile(r"^(?P<head>.*?)\s*;?\s*time\.sleep\(\s*(?P<s>\d+(?:\.\d+)?)\s*\)\s*;?\s*(?:#.*)?$")


def strip_trailing_sleep(code: str) -> tuple[str, float]:
    """Remove the `time.sleep(...)` an agent puts after its LAST action (OSWorld's prompt asks for small sleeps
    between lines; models also end the snippet with a long one). That pause is a fixed wait after the action,
    which is exactly what the after-execution waiter replaces. Sleeps between actions stay. Only top-level lines
    are touched, and the original is kept when nothing else would remain or the result does not compile.
    Returns (code, seconds removed)."""
    if not isinstance(code, str) or code.strip() in SPECIAL:
        return code, 0.0
    lines, removed = code.rstrip().split("\n"), 0.0
    while lines:
        ln = lines[-1]
        if not ln.strip() or (ln.lstrip().startswith("#") and ln == ln.lstrip()):
            lines.pop()
            continue
        if ln != ln.lstrip():  # indented: inside a block, leave it alone
            break
        m = _TRAILING_SLEEP.match(ln)
        if not m:
            break
        removed += float(m.group("s"))
        head = m.group("head").strip()
        if head:  # `pyautogui.click(1, 2); time.sleep(1)`
            lines[-1] = head
            break
        lines.pop()
    out = "\n".join(lines)
    if removed == 0 or not re.search(r"pyautogui\.\w+\(", out):
        return code, 0.0
    try:
        compile(out, "<action>", "exec")
    except SyntaxError:
        return code, 0.0
    return out, removed


_ANY_SLEEP = re.compile(r"time\.sleep\(\s*(\d+(?:\.\d+)?)\s*\)")
_LEADING_SLEEP = re.compile(r"^\s*time\.sleep\(\s*(\d+(?:\.\d+)?)\s*\)\s*;?\s*(?:#.*)?$")


def strip_agent_sleeps(code: str, cap_s: float = 1.0) -> tuple[str, float]:
    """Drop the fixed waits an agent writes into its action code; every one is a guess (in 3838 logged OSWorld steps
    no agent reasoning cited a wait time shown on screen; the longest were 30 -> 300 s back-offs on a 403 page).

    - before the first action: removed (waiting for the screen is the harness's event-driven job)
    - after the last action: removed (`strip_trailing_sleep`, the after-execution waiter replaces it)
    - between actions: capped at `cap_s` (pacing inside one snippet, e.g. open a menu then pick an item; the snippet
      runs in one go in the guest, so nothing can observe the screen in between)
    - a snippet that only sleeps becomes WAIT (the waiter waits for the screen instead of the clock)
    Returns (code, seconds removed); the original is kept when the result does not compile."""
    if not isinstance(code, str) or code.strip() in SPECIAL:
        return code, 0.0
    if not re.search(r"pyautogui\.\w+\(", code):
        vals = [float(v) for v in _ANY_SLEEP.findall(code)]
        rest = _ANY_SLEEP.sub("", code)
        rest = re.sub(r"^\s*(?:import\s+[\w ,]+|from\s+\S+\s+import\s+[\w ,]+|#.*|;)?\s*$", "", rest, flags=re.M)
        return ("WAIT", sum(vals)) if vals and not rest.strip() else (code, 0.0)
    out, removed = strip_trailing_sleep(code)
    lines = out.split("\n")
    first = next(i for i, ln in enumerate(lines) if re.search(r"pyautogui\.\w+\(", ln))
    keep = []
    for i, ln in enumerate(lines):
        m = _LEADING_SLEEP.match(ln) if i < first and ln == ln.lstrip() else None
        if m:
            removed += float(m.group(1))
        else:
            keep.append(ln)
    out = "\n".join(keep)

    def cap(m):
        nonlocal removed
        v = float(m.group(1))
        if v <= cap_s:
            return m.group(0)
        removed += v - cap_s
        return f"time.sleep({cap_s:g})"

    out = _ANY_SLEEP.sub(cap, out)
    if removed == 0:
        return code, 0.0
    try:
        compile(out, "<action>", "exec")
    except SyntaxError:
        return code, 0.0
    return out, round(removed, 3)
