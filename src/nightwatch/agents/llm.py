"""Streaming, cancellable OpenAI-compatible chat call (the only thing NightWatch changes about the agent:
its transport, so that an interrupt can stop the generation mid-stream).

`stream_chat` streams with `stream_options.include_usage`; a watcher thread closes the HTTP response
as soon as `cancel` is set, which unblocks the reader even while the server is still prefilling.
On cancel it raises `Cancelled` carrying the partial text and a token estimate.
"""

from __future__ import annotations

import json
import threading
import time
from dataclasses import dataclass, field

import httpx

from nightwatch.config import env


class Cancelled(Exception):
    def __init__(self, partial: str, reasoning: str):
        super().__init__("generation cancelled")
        self.partial, self.reasoning = partial, reasoning


@dataclass
class Endpoint:
    base_url: str
    model: str
    api_key: str
    max_tokens: int = 1500
    temperature: float = 0.5
    top_p: float = 0.9
    extra_body: dict = field(default_factory=dict)
    timeout: float = 300.0
    retries: int = 6
    stream: bool = True
    price_in: float = 0.0  # USD / 1M tokens
    price_out: float = 0.0
    proxy: str = ""  # e.g. socks5h://127.0.0.1:1080 for an endpoint reachable only through a proxy

    @classmethod
    def from_env(cls, prefix: str = "AGENT") -> "Endpoint":
        base = env(f"{prefix}_BASE_URL", "https://openrouter.ai/api/v1").rstrip("/")
        key = env(f"{prefix}_API_KEY") or (env("OPENROUTER_API_KEY") if "openrouter.ai" in base else "") or env("OPENAI_API_KEY")
        extra = env(f"{prefix}_EXTRA_BODY")
        return cls(base_url=base, model=env(f"{prefix}_MODEL", "xiaomi/mimo-v2.6-flash"), api_key=key,
                   max_tokens=int(env(f"{prefix}_MAX_TOKENS", "1500")),
                   temperature=float(env(f"{prefix}_TEMPERATURE", "0.5")), top_p=float(env(f"{prefix}_TOP_P", "0.9")),
                   extra_body=json.loads(extra) if extra else {}, timeout=float(env(f"{prefix}_TIMEOUT", "300")),
                   stream=env(f"{prefix}_STREAM", "1") != "0", retries=int(env(f"{prefix}_RETRIES", "6")),
                   proxy=env(f"{prefix}_PROXY", ""),
                   price_in=float(env(f"{prefix}_PRICE_IN_PER_MTOK", "0") or 0),
                   price_out=float(env(f"{prefix}_PRICE_OUT_PER_MTOK", "0") or 0))


@dataclass
class CallStats:
    t_first_token: float | None = None  # seconds after the call started
    input_tokens: float = 0.0
    output_tokens: float = 0.0
    reasoning_text_chars: int = 0
    estimated: bool = False
    retries: int = 0
    seconds: float = 0.0
    images_capped: int = 0  # history images dropped to stay under the request size limit (qwen35 agent)
    http: list = field(default_factory=list)  # one per HTTP attempt: unix send / first_token / return, outcome


def _estimate_tokens(text: str) -> float:
    return len(text) / 3.5


def flatten_text_system(messages: list) -> list:
    """Send a text-only system message as a plain string (same text, joined by newlines).

    OSWorld's PromptAgent sends `system` as a list of text parts. Some OpenAI-compatible servers silently drop a
    list-form system message, so the agent would never see the task or the output format."""
    out = []
    for m in messages:
        c = m.get("content")
        if m.get("role") == "system" and isinstance(c, list) and all(p.get("type") == "text" for p in c):
            m = {**m, "content": "\n".join(p["text"] for p in c)}
        out.append(m)
    return out


def tool_calls_as_xml(calls: list | None) -> str:
    """Render OpenAI-style tool calls as the `<tool_call><function=...>` text that Qwen agents parse.

    Some servers strip a generated tool call from `content` and return it only when the request has `tools`."""
    out = []
    for call in calls or []:
        fn = call.get("function") or {}
        try:
            args = json.loads(fn.get("arguments") or "{}")
        except json.JSONDecodeError:
            continue
        if not isinstance(args, dict):
            continue
        params = "".join(f"<parameter={k}>\n{json.dumps(v, ensure_ascii=False) if isinstance(v, (list, dict)) else v}\n</parameter>\n"
                         for k, v in args.items())
        out.append(f"\n<tool_call>\n<function={fn.get('name', '')}>\n{params}</function>\n</tool_call>")
    return "".join(out)


def stream_chat(ep: Endpoint, messages: list, cancel: threading.Event | None = None,
                stats: CallStats | None = None, **overrides) -> str:
    stats = stats if stats is not None else CallStats()
    ov = {k: v for k, v in overrides.items() if v is not None}
    messages = flatten_text_system(messages)
    body = {"model": ep.model, "messages": messages, "max_tokens": ov.get("max_tokens", ep.max_tokens),
            "temperature": ov.get("temperature", ep.temperature), "top_p": ov.get("top_p", ep.top_p)}
    body.update(ep.extra_body)
    if ov.get("tools"):
        body["tools"] = ov["tools"]
    if ep.stream:
        body.update(stream=True, stream_options={"include_usage": True})
    headers = {"Authorization": f"Bearer {ep.api_key}", "Content-Type": "application/json"}
    url = ep.base_url + "/chat/completions"
    t0 = time.perf_counter()
    last_err: Exception | None = None
    for attempt in range(ep.retries):
        if cancel is not None and cancel.is_set():
            raise Cancelled("", "")
        text, reasoning = [], []
        tool_parts: dict = {}
        call = {"send": time.time(), "first_token": None, "return": None, "outcome": "error"}
        stats.http.append(call)
        try:
            with httpx.Client(timeout=httpx.Timeout(ep.timeout, connect=30), proxy=ep.proxy or None) as client:
                if not ep.stream:
                    r = client.post(url, json=body, headers=headers)
                    if r.status_code == 429 or r.status_code >= 500:
                        raise httpx.HTTPStatusError(r.text[:300], request=r.request, response=r)
                    r.raise_for_status()
                    data = r.json()
                    if data.get("provider"):
                        call["provider"] = data["provider"]
                    u = data.get("usage") or {}
                    stats.input_tokens += float(u.get("prompt_tokens", 0) or 0)
                    stats.output_tokens += float(u.get("completion_tokens", 0) or 0)
                    stats.seconds = time.perf_counter() - t0
                    call.update(outcome="ok", **{"return": time.time()})
                    msg = data["choices"][0]["message"]
                    return (msg.get("content") or "") + tool_calls_as_xml(msg.get("tool_calls"))
                with client.stream("POST", url, json=body, headers=headers) as resp:
                    if resp.status_code == 429 or resp.status_code >= 500:
                        resp.read()
                        raise httpx.HTTPStatusError(resp.text[:300], request=resp.request, response=resp)
                    if resp.status_code >= 400:
                        resp.read()
                        raise RuntimeError(f"HTTP {resp.status_code}: {resp.text[:500]}")
                    done = threading.Event()

                    def watch():
                        while not done.is_set():
                            if cancel is not None and cancel.wait(0.02):
                                try:
                                    resp.close()
                                except Exception:
                                    pass
                                return

                    if cancel is not None:
                        threading.Thread(target=watch, daemon=True).start()
                    usage = None
                    try:
                        for line in resp.iter_lines():
                            if cancel is not None and cancel.is_set():
                                break
                            if not line.startswith("data:"):
                                continue
                            payload = line[5:].strip()
                            if payload == "[DONE]":
                                break
                            try:
                                chunk = json.loads(payload)
                            except json.JSONDecodeError:
                                continue
                            if chunk.get("usage"):
                                usage = chunk["usage"]
                            if chunk.get("provider") and not call.get("provider"):  # OpenRouter: who served it
                                call["provider"] = chunk["provider"]
                            for ch in chunk.get("choices") or []:
                                d = ch.get("delta") or {}
                                piece = d.get("content") or ""
                                rpiece = d.get("reasoning_content") or d.get("reasoning") or ""
                                if (piece or rpiece) and call["first_token"] is None:
                                    call["first_token"] = time.time()
                                if (piece or rpiece) and stats.t_first_token is None:
                                    stats.t_first_token = time.perf_counter() - t0
                                text.append(piece)
                                reasoning.append(rpiece)
                                for tc in d.get("tool_calls") or []:  # arguments arrive in fragments
                                    slot = tool_parts.setdefault(tc.get("index", 0), {"name": "", "arguments": ""})
                                    fn = tc.get("function") or {}
                                    slot["name"] += fn.get("name") or ""
                                    slot["arguments"] += fn.get("arguments") or ""
                    except (httpx.StreamError, httpx.ReadError, httpx.RemoteProtocolError):
                        if not (cancel is not None and cancel.is_set()):
                            raise
                    finally:
                        done.set()
                    full, rfull = "".join(text), "".join(reasoning)
                    full += tool_calls_as_xml([{"function": tool_parts[i]} for i in sorted(tool_parts)])
                    stats.reasoning_text_chars += len(rfull)
                    if cancel is not None and cancel.is_set():
                        stats.estimated = True
                        stats.output_tokens += _estimate_tokens(full + rfull)
                        stats.seconds = time.perf_counter() - t0
                        call.update(outcome="cancelled", **{"return": time.time()})
                        raise Cancelled(full, rfull)
                    if usage:
                        stats.input_tokens += float(usage.get("prompt_tokens", 0) or 0)
                        stats.output_tokens += float(usage.get("completion_tokens", 0) or 0)
                    else:
                        stats.estimated = True
                        stats.output_tokens += _estimate_tokens(full + rfull)
                    stats.seconds = time.perf_counter() - t0
                    call.update(outcome="ok", **{"return": time.time()})
                    return full
        except Cancelled:
            call.update(outcome="cancelled", **{"return": time.time()})
            raise
        except (httpx.TransportError, httpx.HTTPStatusError) as e:
            if cancel is not None and cancel.is_set():
                stats.seconds = time.perf_counter() - t0
                call.update(outcome="cancelled", **{"return": time.time()})
                raise Cancelled("".join(text), "".join(reasoning))
            call.update(outcome=f"error: {str(e)[:120]}", **{"return": time.time()})
            last_err = e
            stats.retries += 1
            wait = min(2 ** attempt, 30)
            if cancel is not None:
                if cancel.wait(wait):
                    raise Cancelled("", "")
            else:
                time.sleep(wait)
        finally:
            if call["return"] is None:  # an exception that is not retried (e.g. HTTP 4xx)
                call["return"] = time.time()
    raise RuntimeError(f"LLM call failed after {ep.retries} tries: {type(last_err).__name__}")
