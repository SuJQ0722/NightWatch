"""A scripted OSWorld agent for infrastructure runs without an LLM key: WAIT, a click at the screen centre, DONE,
with a fixed "thinking" latency. It exercises the real sandbox, task setup, frame stream, System-1, waiters and
OSWorld's evaluator; its success rate means nothing."""

from __future__ import annotations

import time

from nightwatch.agents.base import AgentAdapter, Prediction


class ScriptedOSWorldAgent(AgentAdapter):
    name = "scripted_osworld"

    def __init__(self, latency_s: float = 2.0, screen=(1920, 1080)):
        self.latency_s, self.screen, self.i = latency_s, screen, 0

    def reset(self, vm_ip=None) -> None:
        self.i = 0

    def predict(self, instruction, obs, cancel) -> Prediction:
        t = time.perf_counter()
        end = time.monotonic() + self.latency_s
        while time.monotonic() < end:
            if cancel.is_set():
                return Prediction(cancelled=True, seconds=time.perf_counter() - t, tokens_estimated=True)
            time.sleep(0.01)
        plan = ["WAIT", f"pyautogui.click({self.screen[0] // 2}, {self.screen[1] // 2})", "DONE"]
        a = plan[min(self.i, len(plan) - 1)]
        self.i += 1
        return Prediction(response=f"```{a}```", actions=[a], seconds=time.perf_counter() - t)

    def snapshot(self):
        return self.i

    def restore(self, snap) -> None:
        self.i = snap
