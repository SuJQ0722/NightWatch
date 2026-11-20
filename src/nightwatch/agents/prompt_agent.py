"""OSWorld's own PromptAgent (screenshot in, pyautogui code out), unchanged except for its transport.

We subclass `mm_agents.agent.PromptAgent` from the pinned OSWorld checkout and override only
`call_llm`: same system prompt, same history window, same parsing, but the request is streamed and can
be cancelled. With MiMo-v2.6-flash (default `xiaomi/mimo-v2.6-flash` on OpenRouter, or Xiaomi's own
OpenAI-compatible endpoint via AGENT_BASE_URL / AGENT_MODEL) this is the agent all methods share.
"""

from __future__ import annotations

import ast
import io
import os
import threading
import time

from PIL import Image

from nightwatch.agents.base import AgentAdapter, Prediction
from nightwatch.agents.llm import CallStats, Cancelled, Endpoint, stream_chat
from nightwatch.utils import b64

# AOI's narration request (aoi/cu_model.py NARRATION_INSTRUCTION), with OSWorld's code output instead of AOI's ACTION line
AOI_NARRATION = ("Also provide a one-sentence visual narration of what is NEW on screen, in the keyframes and the "
                 "current screenshot. Start your reply with a line\nNARRATION: <one sentence about new visual info, or "
                 "\"No visual change.\">\nthen give your code as usual.")


def fit_image_budget(messages: list, keyframes: list, max_images: int | None) -> tuple[list, list]:
    """Keep a request within an endpoint's image limit (some servers accept at most 4 images per prompt).

    Priority: the current screenshot, then the most recent keyframes (at most max_images - 1), then the newest
    history screenshots. Older history turns keep their text; their image becomes a short placeholder. With no
    keyframes and the default 3-step history the request is unchanged (4 images). No-op when max_images is None."""
    if not max_images:
        return messages, keyframes
    keyframes = list(keyframes)[-(max_images - 1):] if max_images > 1 else []
    budget = max_images - 1 - len(keyframes)  # history images allowed besides the current screenshot
    msgs = list(messages)
    for i in range(len(msgs) - 2, -1, -1):  # newest history first; the last message is the current turn
        content = msgs[i].get("content")
        if msgs[i].get("role") != "user" or not isinstance(content, list):
            continue
        new = []
        for part in content:
            if part.get("type") == "image_url":
                if budget > 0:
                    budget -= 1
                else:
                    part = {"type": "text", "text": "(earlier screenshot omitted)"}
            new.append(part)
        msgs[i] = {**msgs[i], "content": new}
    return msgs, keyframes


def cap_image_bytes(messages: list, max_chars: int | None) -> tuple[list, int]:
    """Keep a request's inline images under a size limit (OpenRouter: "Downloaded image content cannot exceed 30MB",
    HTTP 413; long histories of 20 detailed 1080p PNGs reach it). The oldest history images
    become the same placeholder fit_image_budget uses until the base64 total is <= max_chars; the last message (current
    screenshot, AOI keyframes) is never touched. No-op below the limit. Returns (messages, images dropped)."""
    def size(part):
        return len(((part.get("image_url") or {}).get("url")) or "") if part.get("type") == "image_url" else 0

    if not max_chars:
        return messages, 0
    msgs = list(messages)
    total = sum(size(p) for m in msgs if isinstance(m.get("content"), list) for p in m["content"])
    dropped = 0
    for i in range(len(msgs) - 1):  # oldest first
        if total <= max_chars:
            break
        content = msgs[i].get("content")
        if not isinstance(content, list) or not any(size(p) for p in content):
            continue
        new = []
        for part in content:
            if total > max_chars and size(part):
                total -= size(part)
                dropped += 1
                part = {"type": "text", "text": "(earlier screenshot omitted)"}
            new.append(part)
        msgs[i] = {**msgs[i], "content": new}
    return msgs, dropped


def with_aoi_observation(messages: list, keyframes: list, max_images: int | None = None,
                         narration: str = AOI_NARRATION) -> list:
    """AOI's observation record on OSWorld's PromptAgent: the keyframes captured since the previous action go before
    the current screenshot (AOI ObservationRecord order and labels), plus AOI's narration request. Only the request
    carries them: the agent's own history keeps screenshots and replies (so the narration persists, images do not),
    as in AOI's TrajectoryStore. With max_images, see fit_image_budget."""
    messages, keyframes = fit_image_budget(messages, keyframes, max_images)
    msgs = list(messages)
    last = dict(msgs[-1])
    head = []
    if keyframes:
        lines = [f"[KEYFRAMES — {len(keyframes)} visual change(s) detected since your previous action]"]
        lines += [f"Image {i + 1}: captured {ago:.1f}s before the current screenshot (shows a visual change)"
                  for i, (ago, _) in enumerate(keyframes)]
        head.append({"type": "text", "text": "\n".join(lines)})
        head += [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64(jpg), "detail": "high"}}
                 for _, jpg in keyframes]
        head.append({"type": "text", "text": "[CURRENT SCREENSHOT — what the screen looks like right now]"})
    last["content"] = head + list(last["content"]) + [{"type": "text", "text": narration}]
    msgs[-1] = last
    return msgs


def downscale_obs(obs: dict, size) -> tuple[dict, float, float]:
    """The screenshot the model sees at `size` (None = as is), and the factors that map its coordinates to the screen."""
    if not size:
        return obs, 1.0, 1.0
    size = tuple(size)
    img = Image.open(io.BytesIO(obs["screenshot"]))
    if img.size == size:
        return obs, 1.0, 1.0
    buf = io.BytesIO()
    img.convert("RGB").resize(size, Image.BICUBIC).save(buf, "PNG")
    return {**obs, "screenshot": buf.getvalue()}, img.width / size[0], img.height / size[1]


_POINTER_CALLS = {"click", "leftClick", "rightClick", "middleClick", "doubleClick", "tripleClick", "moveTo", "dragTo",
                  "mouseDown", "mouseUp", "moveRel", "dragRel", "move", "drag"}
_SCROLL_CALLS = {"scroll", "hscroll", "vscroll"}  # scroll(clicks, x, y): the first argument is not a coordinate


_SELF_SCALED = {"sx", "sy", "scale", "scale_x", "scale_y", "x_scale", "y_scale", "ratio", "factor"}


def is_noop_code(code) -> bool:
    """True for a code block with nothing to run: only blank lines and comments.

    Qwen3.8-27B sometimes puts its explanation in a separate ```python block of comments, and when it degenerates it
    repeats "# done" blocks by the hundred (one step had 174). OSWorld's parser turns every block into an action, and a
    loop that waits for the screen after each action spent minutes per step on them."""
    if not isinstance(code, str):
        return False
    lines = [ln.strip() for ln in code.splitlines()]
    return bool(code.strip()) and all(not ln or ln.startswith("#") for ln in lines)


def rescale_coordinates(code: str, sx: float, sy: float) -> str:
    """Map the screen coordinates of the pyautogui calls in `code` from the model's image to the screen, (x*sx, y*sy).

    Numeric literals are rewritten in place (so the loop's target parsing sees screen pixels). A coordinate given by a
    variable or an expression (`click(x, y)` in a loop, `774 - i*18`, `click(*cells[0])`, `click(center(p))`) is
    mapped at run time (None passes through), Values in screen pixels
    (from pyautogui.locate* / position() / size()) and expressions the model scaled itself (`821 * sx`) are left alone;
    code that does not parse (WAIT, DONE, broken snippets) comes back unchanged."""
    try:
        tree = ast.parse(code)
    except (SyntaxError, ValueError):
        return code
    # values already in screen pixels: names assigned (directly or through other such names) from the screen size, the
    # pointer position or image search on the real screen. Only those are left alone; every other coordinate is in the
    # model's image and is mapped, so click(x, y) with x, y = 100, 200 and click(100, 200) land on the same pixel.
    def screen_call(n):
        return (isinstance(n, ast.Call) and isinstance(n.func, ast.Attribute) and isinstance(n.func.value, ast.Name)
                and n.func.value.id == "pyautogui" and (n.func.attr.startswith("locate") or n.func.attr in ("position", "size")))

    # Source order matters: a name counts as screen pixels at a use only if its latest assignment before that use came
    # from such a value (x, y = position() and then x, y = 100, 200 leaves 100, 200 in image pixels).
    def pos(n):
        return (getattr(n, "lineno", 0), getattr(n, "col_offset", 0))

    bindings: list = []  # (position, name, from the screen?)

    def screen_at(name, at):
        prior = [b for b in bindings if b[1] == name and b[0] < at]
        return max(prior)[2] if prior else False

    def from_screen(node, at=None):
        at = pos(node) if at is None else at
        return any(screen_call(n) or (isinstance(n, ast.Name) and screen_at(n.id, at)) for n in ast.walk(node))

    assigns = []
    for n in ast.walk(tree):
        if isinstance(n, ast.Assign):
            assigns.append((pos(n), n.value, n.targets, False))
        elif isinstance(n, ast.AugAssign):
            assigns.append((pos(n), n.value, [n.target], True))
        elif isinstance(n, (ast.AnnAssign, ast.NamedExpr)) and n.value is not None:
            assigns.append((pos(n), n.value, [n.target], False))
        elif isinstance(n, ast.For):
            assigns.append((pos(n), n.iter, [n.target], False))
        elif isinstance(n, ast.comprehension):
            assigns.append((pos(n.target), n.iter, [n.target], False))
    for at, value, targets, aug in sorted(assigns, key=lambda a: a[0]):
        for t in targets:
            for m in ast.walk(t):
                if isinstance(m, ast.Name):
                    bindings.append((at, m.id, from_screen(value, at) or (aug and screen_at(m.id, at))))
    edits = []

    def self_scaled(node):
        for n in ast.walk(node):
            if isinstance(n, ast.Name) and n.id.lower() in _SELF_SCALED:
                return True
        return from_screen(node)

    def scale(node, s):
        if isinstance(node, ast.UnaryOp) and isinstance(node.op, ast.USub) and isinstance(node.operand, ast.Constant):
            node = node.operand
        if isinstance(node, ast.Constant):
            if type(node.value) in (int, float):
                edits.append((node, str(int(round(node.value * s)))))
        elif not isinstance(node, ast.Starred) and not self_scaled(node):  # None (pyautogui: current position) passes
            edits.append((node, f"(lambda _v: _v if _v is None else int(round(_v * {s!r})))({ast.get_source_segment(code, node)})"))

    def scale_point(node):  # one argument that holds both coordinates: click(p), click(*p)
        if self_scaled(node):
            return
        inner = node.value if isinstance(node, ast.Starred) else node
        src = ast.get_source_segment(code, inner)
        if isinstance(node, ast.Starred):
            edits.append((node, f"*[_v if _v is None else int(round(_v * _s)) for _v, _s in zip(tuple({src})[:2], "
                                f"({sx!r}, {sy!r}))]"))
        else:  # click(p) with a point, or click(x) with only x (pyautogui keeps the current y)
            edits.append((node, f"(lambda _p: _p if _p is None else int(round(_p * {sx!r})) if isinstance(_p, (int, float)) "
                                f"else [_v if _v is None else int(round(_v * _s)) for _v, _s in zip(tuple(_p)[:2], ({sx!r}, {sy!r}))])({src})"))

    for call in ast.walk(tree):
        if not (isinstance(call, ast.Call) and isinstance(call.func, ast.Attribute)
                and isinstance(call.func.value, ast.Name) and call.func.value.id == "pyautogui"):
            continue
        has_y = any(kw.arg == "y" for kw in call.keywords)
        if call.func.attr in _POINTER_CALLS:
            xy = call.args[:2]
            if xy and isinstance(xy[0], (ast.Tuple, ast.List)):  # click((x, y)), click([x, y])
                xy = xy[0].elts[:2]
            elif xy and (isinstance(xy[0], ast.Starred) or (len(call.args) == 1 and not has_y
                                                            and not isinstance(xy[0], (ast.Constant, ast.UnaryOp)))):
                scale_point(xy[0])
                xy = []
        elif call.func.attr in _SCROLL_CALLS:
            xy = call.args[1:3]
        else:
            continue
        for node, s in zip(xy, (sx, sy)):
            scale(node, s)
        for kw in call.keywords:
            if kw.arg in ("x", "y"):
                scale(kw.value, sx if kw.arg == "x" else sy)
    # nested edits (a call inside another call's argument) cannot be spliced twice: keep the outermost
    spans = sorted(((n.lineno, n.col_offset, n.end_lineno, n.end_col_offset), new) for n, new in edits)
    kept = []
    for span, new in spans:
        if kept and (span[0], span[1]) < (kept[-1][0][2], kept[-1][0][3]):
            continue
        kept.append((span, new))
    edits = [(type("N", (), {"lineno": s[0], "col_offset": s[1], "end_lineno": s[2], "end_col_offset": s[3]}), new)
             for s, new in kept]
    # ast offsets are UTF-8 byte columns
    raw = code.encode("utf-8")
    starts, pos = [], 0
    for line in raw.splitlines(keepends=True):
        starts.append(pos)
        pos += len(line)
    spans = [(starts[n.lineno - 1] + n.col_offset, starts[n.end_lineno - 1] + n.end_col_offset, new) for n, new in edits]
    for a, b, new in sorted(spans, reverse=True):
        raw = raw[:a] + new.encode("utf-8") + raw[b:]
    return raw.decode("utf-8")


def _make_class():
    from mm_agents.agent import PromptAgent  # from third_party/OSWorld (on sys.path via osworld.pth)

    class StreamingPromptAgent(PromptAgent):
        endpoint: Endpoint
        cancel: threading.Event | None = None
        stats: CallStats | None = None
        cancelled = False
        aoi_keyframes: list | None = None  # set per call by the aoi_visual baseline (None = plain PromptAgent)

        def call_llm(self, payload):  # noqa: D401 - same contract as PromptAgent.call_llm
            messages = payload["messages"]
            max_images = int(os.environ.get("AGENT_MAX_IMAGES") or 0) or None
            if self.aoi_keyframes is not None:
                messages = with_aoi_observation(messages, self.aoi_keyframes, max_images)
            else:
                messages, _ = fit_image_budget(messages, [], max_images)
            try:
                return stream_chat(self.endpoint, messages, self.cancel, self.stats,
                                   max_tokens=payload.get("max_tokens"), temperature=payload.get("temperature"),
                                   top_p=payload.get("top_p"))
            except Cancelled:
                # Not raised: PromptAgent.predict treats an exception as an empty response and sleeps 2 s
                # before retrying. A non-empty placeholder returns at once; the loop discards this
                # prediction and restores the agent's history (snapshot) on every interrupt.
                self.cancelled = True
                return "WAIT"

    return StreamingPromptAgent


class PromptAgentAdapter(AgentAdapter):
    def __init__(self, agent_cfg, client_password: str = "password"):
        cls = _make_class()
        ep = Endpoint.from_env("AGENT")
        for k in ("model", "max_tokens", "temperature", "top_p"):
            if agent_cfg.get(k) is not None:
                setattr(ep, k, agent_cfg.get(k))
        self.agent = cls(platform="ubuntu", model=ep.model, max_tokens=ep.max_tokens, top_p=ep.top_p,
                         temperature=ep.temperature, action_space="pyautogui",
                         observation_type=agent_cfg.get("observation_type", "screenshot"),
                         max_trajectory_length=int(agent_cfg.get("max_trajectory_length", 3)),
                         client_password=client_password)
        self.agent.endpoint = ep
        # e.g. [1280, 720]: the model sees this size and its clicks are mapped back to the screen (null = screenshot as is).
        # MiMo-v2.6-flash answers 1920x1080 screenshots in 1280x720 coordinates.
        self.obs_size = tuple(agent_cfg.get("obs_size") or ()) or None
        # Appended to the task line of the system prompt; {w}/{h} = the size the model sees. Qwen3.8-27B needs it to answer
        # in pixels of that image.
        note = agent_cfg.get("screen_note")
        w, h = self.obs_size or (1920, 1080)
        self.screen_note = note.format(w=w, h=h) if note else ""
        self.supports_cancel = ep.stream and bool(agent_cfg.get("cancellable", True))
        self.name = f"prompt_agent:{ep.model}"

    def reset(self, vm_ip=None) -> None:
        import logging

        self.agent.reset(logging.getLogger("nightwatch.agent"), vm_ip=vm_ip)

    def predict(self, instruction, obs, cancel) -> Prediction:
        a = self.agent
        a.cancel = cancel if self.supports_cancel else None
        a.stats = CallStats()
        a.cancelled = False
        a.aoi_keyframes = obs.get("aoi_keyframes")
        obs, sx, sy = downscale_obs(obs, self.obs_size)
        t = time.perf_counter()
        response, actions = a.predict(instruction + ("\n" + self.screen_note if self.screen_note else ""), obs)
        if (sx, sy) != (1.0, 1.0) and actions:  # a new list: the agent's history keeps the coordinates of the image it saw
            actions = [rescale_coordinates(c, sx, sy) if isinstance(c, str) else c for c in actions]
        if actions:  # comment-only blocks run nothing; drop them so that no loop waits or observes after them
            actions = [c for c in actions if not is_noop_code(c)]
        s = a.stats
        return Prediction(response=response or "", actions=actions, cancelled=a.cancelled,
                          seconds=time.perf_counter() - t, t_first_token=s.t_first_token,
                          input_tokens=s.input_tokens, output_tokens=s.output_tokens,
                          tokens_estimated=s.estimated, retries=s.retries, extra={"http": s.http})

    def snapshot(self):
        a = self.agent
        return list(a.observations), list(a.actions), list(a.thoughts)

    def restore(self, snap) -> None:
        a = self.agent
        a.observations, a.actions, a.thoughts = list(snap[0]), list(snap[1]), list(snap[2])
