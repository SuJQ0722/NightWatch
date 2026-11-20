"""A slow, perfectly perceiving scripted agent for the simulated desktop (pipeline checks only).

It reads the ground truth attached to the observation (the state at screenshot time), "thinks" for a
log-normal latency during which it can be cancelled, and acts on what it saw: so it fails exactly when
the screen it saw was stale or captured mid-transition.
"""

from __future__ import annotations

import random
import time

from nightwatch.agents.base import AgentAdapter, Prediction


class SimAgent(AgentAdapter):
    name = "sim_agent"

    def __init__(self, latency_s: float = 1.2, jitter: float = 0.3, seed: int = 0, cancellable: bool = True):
        self.latency_s, self.jitter = latency_s, jitter
        self.rng = random.Random(seed)
        self.supports_cancel = cancellable
        self.history: list = []

    def reset(self, vm_ip=None) -> None:
        self.history = []

    def _decide(self, gt: dict) -> str:
        if not gt:
            return "WAIT"
        els = gt.get("elements", [])
        if gt.get("popup"):
            x0, y0, x1, y1 = els[0][1:5]
            return f"pyautogui.click({(x0 + x1) // 2}, {(y0 + y1) // 2})"
        if gt.get("success_visible"):
            return "DONE"
        if gt.get("loading"):
            return "WAIT"
        for el in els:
            if el[0].startswith("btn"):
                x0, y0, x1, y1 = el[1:5]
                return f"pyautogui.click({(x0 + x1) // 2}, {(y0 + y1) // 2})"
        return "WAIT"

    def predict(self, instruction, obs, cancel) -> Prediction:
        t = time.perf_counter()
        lat = max(0.05, self.rng.lognormvariate(0, self.jitter) * self.latency_s)
        end = time.monotonic() + lat
        while time.monotonic() < end:
            if self.supports_cancel and cancel.is_set():
                self.history.append(("cancelled", None))  # undone by restore()
                return Prediction(cancelled=True, seconds=time.perf_counter() - t, output_tokens=40 * (lat - (end - time.monotonic())),
                                  tokens_estimated=True)
            time.sleep(0.01)
        action = self._decide(obs.get("gt"))
        self.history.append((obs.get("gt", {}).get("t"), action))
        resp = f"Looking at the screen.\n```python\n{action}\n```" if action not in ("WAIT", "DONE", "FAIL") else f"```{action}```"
        return Prediction(response=resp, actions=[action], seconds=time.perf_counter() - t,
                          t_first_token=0.2 * lat, input_tokens=1500, output_tokens=40 * lat)

    def snapshot(self):
        return list(self.history)

    def restore(self, snap) -> None:
        self.history = list(snap)
