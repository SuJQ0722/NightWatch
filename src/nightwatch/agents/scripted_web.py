"""A scripted agent for DynaCU pipeline checks (no API key): it waits, then clicks the interactive
elements AOI lists for the page, one per step, with a configurable "thinking" latency. Pipeline checks only."""

from __future__ import annotations

import random
import re
import time

from nightwatch.agents.base import AgentAdapter, Prediction


class ScriptedWebAgent(AgentAdapter):
    name = "scripted_web"

    def __init__(self, latency_s: float = 1.5, seed: int = 0):
        self.latency_s = latency_s
        self.rng = random.Random(seed)
        self.env = None
        self.i = 0

    def bind_env(self, env) -> None:
        self.env = env

    def reset(self, vm_ip=None) -> None:
        self.i = 0

    def predict(self, instruction, obs, cancel) -> Prediction:
        t = time.perf_counter()
        end = time.monotonic() + self.latency_s * self.rng.uniform(0.6, 1.4)
        while time.monotonic() < end:
            if cancel.is_set():
                return Prediction(cancelled=True, seconds=time.perf_counter() - t, tokens_estimated=True)
            time.sleep(0.01)
        pts = re.findall(r"at \((\d+),(\d+)\)", self.env.page_elements() if self.env else "")
        self.i += 1
        action = "wait()" if not pts or self.i % 3 == 1 else "click(%s, %s)" % pts[(self.i // 3) % len(pts)]
        return Prediction(response=f"NARRATION: scripted.\nACTION: {action}", actions=[action],
                          seconds=time.perf_counter() - t)

    def snapshot(self):
        return self.i

    def restore(self, snap) -> None:
        self.i = snap
