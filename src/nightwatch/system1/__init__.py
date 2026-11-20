"""System-1 backends.

  http      perception server client (`python -m nightwatch serve`, holds Intern-Decision on a GPU)  [default]
  d3        vLLM-SR Decision 3.0 in-process (D3_MODEL: d3-mini 4B, d3-lite 0.8B, or a fine-tuned package)
  intern    Intern-Decision in-process (only inside the System-1 env; the server uses this)
  jev       hosted Jev 1.14 (OpenRouter / TypeSafe)
  frontier  a large VLM answering every question in JSON (equal-budget observer comparator)
  logprob   one forward per question on an OpenAI-compatible server, P(yes) from logprobs
  sim       ground truth + noise, simulated desktop only (pipeline checks, never results)
  none      no System-1 (pixel-only methods)
"""

from __future__ import annotations

from nightwatch.config import env
from nightwatch.system1.base import Meter, System1


def make_system1(name: str | None = None) -> System1 | None:
    name = (name or env("SYSTEM1_BACKEND", "http")).strip()
    name = name if name.startswith("http") else name.lower()
    if name in ("none", ""):
        return None
    if name == "http" or name.startswith("http://") or name.startswith("https://"):
        from nightwatch.system1.remote import HttpS1

        return HttpS1(None if name == "http" else name)
    if name == "d3":
        from nightwatch.system1.d3 import D3S1

        return D3S1()
    if name == "d3_vllm":
        from nightwatch.system1.d3_vllm import D3VllmS1

        return D3VllmS1()
    if name == "intern":
        from nightwatch.system1.intern import InternS1

        return InternS1()
    if name == "jev":
        from nightwatch.system1.remote import JevS1

        return JevS1()
    if name == "frontier":
        from nightwatch.system1.remote import FrontierS1

        return FrontierS1()
    if name == "logprob":
        from nightwatch.system1.remote import LogprobS1

        return LogprobS1()
    if name == "sim":
        from nightwatch.system1.sim import SimS1

        return SimS1()
    raise ValueError(f"unknown System-1 backend {name!r}")


__all__ = ["make_system1", "Meter", "System1"]
