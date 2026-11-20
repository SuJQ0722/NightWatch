"""Official D3 decision endpoint: native probabilities, no text generation."""
from __future__ import annotations

import fcntl
import math
import os
import time
from contextlib import contextmanager
from pathlib import Path

import httpx

from nightwatch.config import env
from nightwatch.questions import Request
from nightwatch.system1.base import System1
from nightwatch.system1.remote import _post
from nightwatch.utils import data_url_jpeg



@contextmanager
def request_slot(env_var: str, slots: int):
    """Bound in-flight requests across threads / processes (lock files in the directory named by `env_var`).

    Waiting for a slot happens on the caller's clock, so it stays in wall time; only the server queue is bounded."""
    directory = os.environ.get(env_var)
    if not directory:
        yield
        return
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    held = None
    while held is None:
        for index in range(slots):
            candidate = (root / str(index)).open("a")
            try:
                fcntl.flock(candidate, fcntl.LOCK_EX | fcntl.LOCK_NB)
                held = candidate
                break
            except BlockingIOError:
                candidate.close()
        if held is None:
            time.sleep(0.02)
    try:
        yield
    finally:
        held.close()


class D3VllmS1(System1):
    def __init__(self):
        self.url = env("D3_VLLM_URL", "http://127.0.0.1:8840").rstrip("/")
        self.model = env("D3_VLLM_MODEL", "d3-fp8")
        self.max_side = int(env("S1_MAX_SIDE", "1280"))
        # Permit a slow comparator to answer instead of silently scoring pixel fallback.
        # This is a transport timeout; elapsed inference still enters loop wall time.
        self.budget_s = float(env("D3_VLLM_BUDGET_S", "60"))
        self.name = f"d3_vllm:{self.model}"
        self.client = httpx.Client(trust_env=False, timeout=8)

    def warmup(self):
        self.client.get(self.url + "/health").raise_for_status()
        response = self.client.get(self.url + "/v1/models")
        response.raise_for_status()
        if self.model not in {item["id"] for item in response.json()["data"]}:
            raise RuntimeError(f"D3 service does not serve {self.model}")

    def _raw(self, req: Request) -> dict:
        start = time.perf_counter()
        questions = {
            q.id: {"type": q.kind, "instructions": text,
                   **({"criteria": dict(q.options)} if q.kind == "choice" else {})}
            for q, text in req.questions
        }
        body = {"model": self.model, "state": req.state,
                "images": [data_url_jpeg(im, max_side=self.max_side) for im in req.images],
                "questions": questions}
        encode_ms = (time.perf_counter() - start) * 1000
        # D3_REQUEST_SLOTS bounds in-flight requests to the 27B server (default 2 via D3_SLOTS): on flashing
        # screens the unbounded stream queued 40+ requests and timed out.
        with request_slot("D3_REQUEST_SLOTS", int(env("D3_SLOTS", "2"))):
            response = _post(self.client, self.url + "/v1/systemone", body, {}, budget_s=self.budget_s)
        answers = response.get("answers", {})
        if set(answers) != set(questions):
            raise RuntimeError("D3 response question IDs differ from request")
        raw = {}
        for q, _ in req.questions:
            answer = answers[q.id]
            probabilities = answer.get("probabilities", {})
            expected = set(q.options) if q.kind == "choice" else {"yes", "no"}
            if answer.get("type") != q.kind or set(probabilities) != expected:
                raise RuntimeError(f"D3 response schema mismatch for {q.id}")
            if not all(isinstance(p, (int, float)) and math.isfinite(p) and 0 <= p <= 1
                       for p in probabilities.values()) or abs(sum(probabilities.values()) - 1) > 0.002:
                raise RuntimeError(f"D3 invalid probabilities for {q.id}")
            raw[q.id] = {"probs": probabilities} if q.kind == "choice" else {"yes": probabilities["yes"]}
        raw["_timing"] = {"encode_ms": encode_ms}
        return raw
