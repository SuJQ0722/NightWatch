"""Remote System-1 backends: hosted Jev, the perception server, and the two "big model as observer"
comparators (a frontier VLM answering in text; an OpenAI-compatible server read through logprobs)."""

from __future__ import annotations

import json
import math
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor

import httpx

from nightwatch.config import env
from nightwatch.questions import Request
from nightwatch.system1.base import System1, choice_as_nouls
from nightwatch.utils import b64, data_url_jpeg, encode_jpeg


def _post(client: httpx.Client, url: str, body: dict, headers: dict, budget_s: float | None = None) -> dict:
    """POST with retries inside one time budget (S1_BUDGET_S, default 8 s). Every System-1 call sits on the
    loop's critical path: a late answer is worth nothing, so a failing backend must fail fast and let the
    caller fall back to pixel-only behaviour (offline analyses can raise the budget)."""
    budget = float(budget_s if budget_s is not None else env("S1_BUDGET_S", "8"))
    deadline = time.monotonic() + budget
    last, attempt = None, 0
    while True:
        left = deadline - time.monotonic()
        if left <= 0.05:
            break
        try:
            r = client.post(url, json=body, headers=headers, timeout=left)
        except httpx.TransportError as e:
            last = e
        else:
            if r.status_code < 400:
                return r.json()
            if r.status_code != 429 and r.status_code < 500:
                raise RuntimeError(f"HTTP {r.status_code} from {url}: {r.text[:500]}")
            last = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
        pause = min(0.5 * 2**attempt, deadline - time.monotonic())
        if pause > 0:
            time.sleep(pause)
        attempt += 1
    raise RuntimeError(f"request to {url} failed within {budget:.0f} s: {last}")


class HttpS1(System1):
    """Client of `python -m nightwatch serve` (the server holds the model; many harness processes share it).

    `url` may list replicas of the same model ("http://a:8711,http://b:8751", one per GPU): each request goes to the
    replica with the fewest requests in flight; a replica that fails is skipped for a few seconds and the request
    moves on to the next one within the same time budget (S1_BUDGET_S)."""

    def __init__(self, url: str | None = None):
        spec = url or env("S1_URL", "http://127.0.0.1:8710")
        self.urls = [u.strip().rstrip("/") for u in spec.split(",") if u.strip()]
        self.url = self.urls[0]
        self.max_side = int(env("S1_MAX_SIDE", "1280"))
        self.client = httpx.Client(timeout=float(env("S1_TIMEOUT", "60")),
                                   limits=httpx.Limits(max_connections=64, max_keepalive_connections=32))
        self.name = "http"
        self._lock = threading.Lock()
        self._busy = {u: 0 for u in self.urls}
        self._down_until = {u: 0.0 for u in self.urls}

    def warmup(self) -> None:
        names, errs = [], []
        for u in self.urls:
            try:
                names.append(self.client.get(u + "/health", timeout=10).json().get("backend", "http"))
            except Exception as e:
                errs.append(f"{u}: {e}")
                self._down_until[u] = time.monotonic() + 30
        if not names:
            raise RuntimeError("no perception server reachable: " + "; ".join(errs))
        self.name = names[0] + (f" x{len(names)}" if len(names) > 1 else "")

    def _pick(self, tried: set) -> str | None:
        now = time.monotonic()
        with self._lock:
            cand = [u for u in self.urls if u not in tried]
            if not cand:
                return None
            u = min(cand, key=lambda u: (self._down_until[u] > now, self._busy[u]))
            self._busy[u] += 1
            return u

    def _raw(self, req: Request) -> dict:
        t = time.perf_counter()
        body = {"checkpoint": req.checkpoint, "state": req.state,
                "images": [b64(encode_jpeg(im, 92, self.max_side)) for im in req.images],
                "questions": [{"id": q.id, "kind": q.kind, "text": t, "options": q.options} for q, t in req.questions]}
        enc = (time.perf_counter() - t) * 1000

        def done(js: dict) -> dict:
            return {**js["raw"], "_timing": {"server_ms": js.get("ms"), "encode_ms": enc}}

        if len(self.urls) == 1:
            return done(_post(self.client, self.url + "/v1/ask", body, {}))
        deadline = time.monotonic() + float(env("S1_BUDGET_S", "8"))
        tried: set = set()
        last = None
        while time.monotonic() < deadline - 0.05:
            u = self._pick(tried)
            if u is None:  # every replica failed once: start over while time is left
                tried.clear()
                time.sleep(min(0.2, max(0.0, deadline - time.monotonic())))
                continue
            tried.add(u)
            try:
                r = self.client.post(u + "/v1/ask", json=body, timeout=deadline - time.monotonic())
                if r.status_code < 400:
                    return done(r.json())
                last = RuntimeError(f"HTTP {r.status_code} from {u}: {r.text[:200]}")
            except httpx.TransportError as e:
                last = e
            finally:
                with self._lock:
                    self._busy[u] -= 1
            with self._lock:
                self._down_until[u] = time.monotonic() + float(env("S1_COOLDOWN_S", "3"))
        raise RuntimeError(f"no replica answered within the budget: {last}")


def fit_jev(img, lo: int = 288, hi: int = 1568):
    """Jev rejects a side below 280 px (tested: 600x280 ok, 512x256 / 300x256 not) or above
    1568 px. Scale every side into [288, 1568], pad if the aspect
    ratio makes that impossible (thin crops), keeping the content unchanged otherwise."""
    import numpy as np
    from PIL import Image

    im = Image.fromarray(img) if isinstance(img, np.ndarray) else img
    w, h = im.size
    s = max(1.0, lo / min(w, h))
    if max(w, h) * s > hi:
        s = hi / max(w, h)
    if s != 1.0:
        im = im.resize((max(1, round(w * s)), max(1, round(h * s))), Image.BILINEAR)
    w, h = im.size
    if w < lo or h < lo:
        canvas = Image.new("RGB", (max(w, lo), max(h, lo)), (128, 128, 128))
        canvas.paste(im, ((canvas.width - w) // 2, (canvas.height - h) // 2))
        im = canvas
    return np.asarray(im)


class JevS1(System1):
    """Hosted Jev 1.14 (images are an experimental feature): one request, all questions, all images."""

    def __init__(self):
        self.provider = env("JEV_PROVIDER", "openrouter")
        if self.provider == "openrouter":
            self.url = env("JEV_BASE_URL", "https://openrouter.ai/api").rstrip("/") + "/alpha/decisions"
            self.model = env("JEV_MODEL", "typesafe/jev-1.14")
            self.key = env("OPENROUTER_API_KEY")
        else:
            self.url = env("JEV_BASE_URL", "https://api.typesafe.ai").rstrip("/") + "/v1/systemone"
            self.model = env("JEV_MODEL", "jev-1.14.0")
            self.key = env("TYPESAFE_API_KEY")
        self.name = f"jev:{self.model}"
        self.client = httpx.Client(timeout=float(env("JEV_TIMEOUT", "60")))

    def _raw(self, req: Request) -> dict:
        if not self.key:
            raise RuntimeError("hosted Jev needs OPENROUTER_API_KEY (or TYPESAFE_API_KEY with JEV_PROVIDER=typesafe)")
        qs = {}
        for q, text in req.questions:
            qs[q.id] = ({"type": "choice", "instructions": text, "criteria": dict(q.options)} if q.kind == "choice"
                        else {"type": "noul", "instructions": text})
        headers = {"Authorization": f"Bearer {self.key}"}
        if self.provider == "openrouter":
            state = [req.state] + [{"type": "image_url", "image_url": {"url": data_url_jpeg(fit_jev(im), max_side=None)}}
                                   for im in req.images]
            body = {"model": self.model, "state": state, "questions": qs}
        else:
            headers["TypeSafe-Danger-Zone-Images"] = "true"
            assets = {f"image{i + 1}": {"type": "image", "format": "jpeg", "data_base64": b64(encode_jpeg(fit_jev(im), 90))}
                      for i, im in enumerate(req.images)}
            body = {"model": self.model, "state": {"description": req.state, **{k: k for k in assets}},
                    "questions": qs, "assets": assets}
        data = _post(self.client, self.url, body, headers)
        out = {}
        for qid, a in (data.get("answers") or {}).items():
            if a.get("type") == "choice" or "choice" in a:
                out[qid] = {"probs": a.get("probabilities", {})}
            else:
                p = a.get("noul")
                if p is None:
                    pr = a.get("probabilities", {})
                    p = pr.get("yes", pr.get("true", 0.5))
                out[qid] = {"yes": float(p)}
        return out


def _chat_endpoint(prefix: str) -> tuple[str, str, str]:
    base = env(f"{prefix}_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
    key = env(f"{prefix}_API_KEY") or (env("OPENROUTER_API_KEY") if "openrouter.ai" in base else env("OPENAI_API_KEY"))
    return base, env(f"{prefix}_MODEL"), key


class FrontierS1(System1):
    """Equal-budget comparator: a large VLM answers every question with a probability in JSON."""

    def __init__(self):
        self.base, self.model, self.key = _chat_endpoint("OBSERVER")
        self.model = self.model or env("AGENT_MODEL", "xiaomi/mimo-v2.6-flash")
        self.name = f"frontier:{self.model}"
        self.client = httpx.Client(timeout=float(env("OBSERVER_TIMEOUT", "120")), proxy=env("OBSERVER_PROXY") or None)

    def _raw(self, req: Request) -> dict:
        spec = []
        for q, text in req.questions:
            if q.kind == "choice":
                spec.append(f'"{q.id}": {{option: probability}} over options {list(q.options)}. Question: {text} '
                            + "; ".join(f"{k} = {v}" for k, v in q.options.items()))
            else:
                spec.append(f'"{q.id}": probability (0-1) that the answer is yes. Question: {text}')
        prompt = (req.state + "\n\nAnswer every question. Reply with one JSON object only, with these keys:\n- "
                  + "\n- ".join(spec))
        content = [{"type": "text", "text": prompt}] + [
            {"type": "image_url", "image_url": {"url": data_url_jpeg(im)}} for im in req.images]
        body = {"model": self.model, "messages": [{"role": "user", "content": content}], "temperature": 0,
                "max_tokens": int(env("OBSERVER_MAX_TOKENS", "1024"))}
        data = _post(self.client, self.base + "/chat/completions", body, {"Authorization": f"Bearer {self.key}"})
        text = data["choices"][0]["message"].get("content") or ""
        m = re.search(r"\{.*\}", text, re.S)
        try:
            obj = json.loads(m.group(0)) if m else {}
        except json.JSONDecodeError:
            obj = {}
        out = {}
        for q, _ in req.questions:
            v = obj.get(q.id)
            if q.kind == "choice":
                out[q.id] = {"probs": v if isinstance(v, dict) else {}}
            else:
                try:
                    out[q.id] = {"yes": float(v)}
                except (TypeError, ValueError):
                    out[q.id] = {"yes": 0.5}
        return out


class LogprobS1(System1):
    """Same-budget comparator: one forward per question on an OpenAI-compatible server (e.g. vLLM with
    prefix caching), P(yes) read from the first token's logprobs."""

    def __init__(self):
        self.base, self.model, self.key = _chat_endpoint("LOGPROB")
        self.base = env("LOGPROB_BASE_URL", "http://127.0.0.1:8901/v1").rstrip("/")
        self.model = self.model or "Qwen/Qwen3.5-4B"
        self.key = env("LOGPROB_API_KEY", "EMPTY")
        self.name = f"logprob:{self.model}"
        self.client = httpx.Client(timeout=60)
        self.pool = ThreadPoolExecutor(int(env("LOGPROB_CONCURRENCY", "16")))

    def _one(self, req: Request, text: str) -> float:
        content = [{"type": "image_url", "image_url": {"url": data_url_jpeg(im)}} for im in req.images]
        content.append({"type": "text", "text": f"{req.state}\n\n{text}\nAnswer with yes or no."})
        body = {"model": self.model, "messages": [{"role": "user", "content": content}], "max_tokens": 1,
                "temperature": 0, "logprobs": True, "top_logprobs": 20}
        data = _post(self.client, self.base + "/chat/completions", body, {"Authorization": f"Bearer {self.key}"})
        top = data["choices"][0]["logprobs"]["content"][0]["top_logprobs"]
        py = sum(math.exp(t["logprob"]) for t in top if t["token"].strip().lower() == "yes")
        pn = sum(math.exp(t["logprob"]) for t in top if t["token"].strip().lower() == "no")
        return py / (py + pn) if py + pn > 0 else 0.5

    def _raw(self, req: Request) -> dict:
        jobs = [(q.id, None, t) for q, t in req.questions if q.kind != "choice"] + choice_as_nouls(req)
        ps = list(self.pool.map(lambda j: self._one(req, j[2]), jobs))
        out: dict = {}
        for (qid, opt, _), p in zip(jobs, ps):
            if opt is None:
                out[qid] = {"yes": p}
            else:
                out.setdefault(qid, {"probs": {}})["probs"][opt] = p
        return out
