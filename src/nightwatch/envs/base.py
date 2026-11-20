"""Environment interface the episode loop needs (OSWorld adapter and the simulated desktop)."""

from __future__ import annotations


class Env:
    screen_size: tuple[int, int] = (1920, 1080)
    vm_ip: str | None = None
    name = "env"

    def reset(self, task: dict) -> None:
        raise NotImplementedError

    def instruction(self, task: dict) -> str:
        return task.get("instruction", "")

    def grab(self) -> tuple[bytes | None, dict | None]:
        """Screenshot bytes (+ ground truth for the simulator). Called from the frame-stream thread."""
        raise NotImplementedError

    def obs(self) -> dict:
        data, gt = self.grab()
        return {"screenshot": data, "accessibility_tree": None, "terminal": None, "instruction": None, "gt": gt}

    def execute(self, action) -> dict:
        """Run one action; no sleep and no observation (the waiter decides when to look). -> info."""
        raise NotImplementedError

    def a11y(self) -> str | None:
        return None

    def evaluate(self) -> float:
        raise NotImplementedError

    def close(self) -> None:
        pass
