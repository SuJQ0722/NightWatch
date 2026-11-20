"""System-1 interface: one typed multi-question request on a few images -> calibrated probabilities.

Backends implement `_raw(req) -> {qid: {"yes": p} | {"probs": {option: p}}}`; `ask()` converts to
`Answers` (fire probabilities per question) and times the call. `Meter` wraps any backend and counts
calls / milliseconds per checkpoint, which the step log reports as `jev_calls` / `jev_ms_total`.
"""

from __future__ import annotations

import threading
import time
from collections import defaultdict

from nightwatch.questions import Answers, Request


class System1:
    name = "base"
    supports_choice = True
    needs_gt = False

    def _raw(self, req: Request) -> dict:
        raise NotImplementedError

    def warmup(self) -> None:
        pass

    def ask(self, req: Request) -> Answers:
        t = time.perf_counter()
        raw = self._raw(req)
        ms = (time.perf_counter() - t) * 1000
        timing = raw.pop("_timing", None) or {}  # http backends: {"server_ms", "encode_ms"}
        ans = to_answers(req, raw, ms, self.name)
        ans.server_ms, ans.encode_ms = timing.get("server_ms"), timing.get("encode_ms")
        return ans


def to_answers(req: Request, raw: dict, ms: float, backend: str) -> Answers:
    out = Answers(ms=ms, backend=backend)
    for q, _ in req.questions:
        r = raw.get(q.id) or {}
        if q.kind == "choice":
            probs = r.get("probs") or {}
            tot = sum(max(0.0, float(v)) for v in probs.values()) or 1.0
            out.probs[q.id] = {o: max(0.0, float(probs.get(o, 0.0))) / tot for o in q.options}
        else:
            if "yes" not in r:
                continue  # unanswered: no evidence either way (never fires)
            p = float(r["yes"])
            p = min(max(p, 0.0), 1.0)
            out.fire[q.id] = p if q.fire_on == "yes" else 1.0 - p
    return out


def choice_as_nouls(req: Request) -> list[tuple[str, str, str]]:
    """For yes/no-only backends: (qid, option, text) asking 'is it <option>?' per option."""
    out = []
    for q, text in req.questions:
        if q.kind == "choice":
            for o, desc in q.options.items():
                out.append((q.id, o, f"{text} Is it {desc}?"))
    return out


FAST_IDS = {"settled", "loading", "still_changing"}


class Split(System1):
    """The waiter's fast stability tier (only FAST_IDS questions) on its own backend, so it never queues behind the
    full checks it races against (method key `system1_fast`); everything else goes to `full`."""

    def __init__(self, full: System1, fast: System1):
        self.full, self.fast = full, fast
        self.name = f"{full.name} + fast {fast.name}"
        self.needs_gt = full.needs_gt or fast.needs_gt

    def warmup(self) -> None:
        self.full.warmup()
        self.fast.warmup()
        self.name = f"{self.full.name} + fast {self.fast.name}"

    def ask(self, req: Request) -> Answers:
        tier = self.fast if all(q.id in FAST_IDS for q, _ in req.questions) else self.full
        return tier.ask(req)


class Meter(System1):
    """Counts calls and latency per checkpoint; thread-safe (monitor thread + main thread)."""

    def __init__(self, inner: System1):
        self.inner = inner
        self.name = inner.name
        self.needs_gt = inner.needs_gt
        self.lock = threading.Lock()
        self.calls: dict[str, int] = defaultdict(int)
        self.ms: dict[str, float] = defaultdict(float)
        self.errors = 0
        self.answer_errors = 0  # the watcher answered, but unusably (rejected request, invalid answer): not infrastructure
        self.skipped = 0
        # after a failure, calls return None at once for this long: a down backend must not cost a full
        # request budget on every probe of the waiter
        from nightwatch.config import env

        self.cooldown_s = float(env("S1_COOLDOWN_S", "3"))
        self._down_until = 0.0
        self._pending = 0
        self._idle = threading.Condition()

    def wait_idle(self) -> None:
        """Wait for outstanding calls so late failures enter the episode error count."""
        with self._idle:
            while self._pending:
                self._idle.wait(timeout=1)

    def warmup(self) -> None:
        self.inner.warmup()

    def ask(self, req: Request) -> Answers | None:
        if not _tracked(self.name):
            return self._ask(req)
        with self._idle:
            self._pending += 1
        try:
            return self._ask(req)
        finally:
            with self._idle:
                self._pending -= 1
                self._idle.notify_all()

    def _ask(self, req: Request) -> Answers | None:
        if time.monotonic() < self._down_until:
            with self.lock:
                self.skipped += 1
            return None
        try:
            ans = self.inner.ask(req)
        except Exception as e:  # System-1 down: callers fall back to pixel-only behaviour
            with self.lock:
                self.errors += 1
                self.answer_errors += "rejected" in str(e)
                self._down_until = time.monotonic() + self.cooldown_s
                if self.errors <= 3:
                    import logging

                    logging.getLogger("nightwatch.system1").warning("System-1 call failed: %s", e)
            return None
        with self.lock:
            self.calls[req.checkpoint] += 1
            self.ms[req.checkpoint] += ans.ms
        return ans

    def snapshot(self) -> tuple[dict, dict]:
        with self.lock:
            return dict(self.calls), dict(self.ms)


def _tracked(name: str) -> bool:
    """Watchers whose outstanding calls are awaited at the end of an episode (slow comparator services)."""
    return name.startswith("d3_vllm:")
