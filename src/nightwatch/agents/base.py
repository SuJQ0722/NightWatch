"""Agent adapter interface used by the episode loop.

The loop runs `predict` in a worker thread so it can watch the screen meanwhile. `snapshot()` /
`restore()` undo an interrupted or invalidated generation in the agent's own history, so the agent
never "remembers" an action that was not executed. Adapters that cannot stop a generation set
`supports_cancel = False`; the loop then degrades an interrupt into "let it finish, then invalidate".
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field


@dataclass
class Prediction:
    response: str = ""
    actions: list | None = None
    cancelled: bool = False
    error: str = ""
    seconds: float = 0.0
    t_first_token: float | None = None
    input_tokens: float = 0.0
    output_tokens: float = 0.0
    tokens_estimated: bool = False
    retries: int = 0
    extra: dict = field(default_factory=dict)


class AgentAdapter:
    supports_cancel = True
    name = "agent"

    def reset(self, vm_ip: str | None = None) -> None:
        raise NotImplementedError

    def predict(self, instruction: str, obs: dict, cancel: threading.Event) -> Prediction:
        raise NotImplementedError

    def snapshot(self):
        raise NotImplementedError

    def restore(self, snap) -> None:
        raise NotImplementedError
