"""Perception server: one System-1 model shared by every harness process (stdlib HTTP, no extra deps).

  GET  /health          -> {"backend": ..., "device": ...}
  POST /v1/ask          {"checkpoint", "state", "images": [b64 jpeg], "questions": [{id, kind, text, options}]}
                        -> {"raw": {qid: {"yes": p} | {"probs": {...}}}, "ms": float}
  POST /v1/embed        {"images": [b64 jpeg]} -> {"embeddings": [[...]]}   (CLIP, for the clip-stable baseline)

Runs in the System-1 env:  .conda/envs/nightwatch-s1/bin/python -m nightwatch serve   (scripts/serve_system1.sh)
"""

from __future__ import annotations

import base64
import contextlib
import io
import json
import logging
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from PIL import Image

from nightwatch.config import env, model_path
from nightwatch.questions import BY_ID, Q, Request
from nightwatch.system1.base import FAST_IDS

log = logging.getLogger("nightwatch.server")


def _img(b: str) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(b))).convert("RGB"))


class _Clip:
    def __init__(self):
        import torch
        from transformers import CLIPModel, CLIPProcessor

        path = model_path("CLIP_PATH", "clip-vit-base-patch32", "openai/clip-vit-base-patch32")
        self.dev = "cuda" if torch.cuda.is_available() and env("CLIP_DEVICE", "auto") != "cpu" else "cpu"
        self.model = CLIPModel.from_pretrained(path).to(self.dev).eval()
        self.proc = CLIPProcessor.from_pretrained(path)
        self.torch = torch
        self.lock = threading.Lock()

    def embed(self, imgs: list[np.ndarray]) -> list[list[float]]:
        with self.lock, self.torch.no_grad():
            x = self.proc(images=[Image.fromarray(i) for i in imgs], return_tensors="pt").to(self.dev)
            e = self.model.get_image_features(**x)
            e = e / e.norm(dim=-1, keepdim=True)
            return e.float().cpu().numpy().tolist()


class _PriorityGate:
    """One request on the model at a time; a waiting high-priority request is served before any waiting normal one."""

    def __init__(self):
        self.cv = threading.Condition()
        self.busy, self.high_waiting = False, 0

    @contextlib.contextmanager
    def hold(self, high: bool):
        with self.cv:
            self.high_waiting += high
            while self.busy or (not high and self.high_waiting):
                self.cv.wait()
            self.high_waiting -= high
            self.busy = True
        try:
            yield
        finally:
            with self.cv:
                self.busy = False
                self.cv.notify_all()


def serve(backend: str | None = None, host: str = "127.0.0.1", port: int | None = None) -> None:
    from nightwatch.system1 import make_system1

    backend = backend or env("S1_SERVER_BACKEND", "intern")
    frac = env("S1_GPU_MEM_FRACTION")
    if frac:  # cap this process on its (shared) GPU, e.g. 0.85
        import torch

        if torch.cuda.is_available():
            torch.cuda.set_per_process_memory_fraction(float(frac), 0)
    s1 = make_system1(backend)
    s1.warmup()
    if env("S1_WARMUP", "1") == "1":  # compile / autotune kernels (fla, SDPA) on the request shapes we will send
        from nightwatch.questions import BANK, ROLES, build_request

        rng = np.random.default_rng(0)
        for cp, roles in ROLES.items():
            imgs = {r: (rng.random((720, 1280, 3)) * 255).astype(np.uint8) if i < 3 else
                    (rng.random((200, 320, 3)) * 255).astype(np.uint8) for i, (r, _) in enumerate(roles)}
            t = time.perf_counter()
            try:
                s1.ask(build_request(cp, imgs, BANK, "warm up", "click at (10, 10)"))
                print(f"[nightwatch serve] warm-up {cp}: {1000 * (time.perf_counter() - t):.0f} ms", flush=True)
            except Exception as e:
                print(f"[nightwatch serve] warm-up {cp} failed: {e}", flush=True)
    reserve = env("S1_GPU_RESERVE")
    if reserve:  # hold this share of the GPU in torch's cache (requests reuse it), e.g. 0.8 on a shared machine
        import torch

        if torch.cuda.is_available():
            need = int(float(reserve) * torch.cuda.get_device_properties(0).total_memory) - torch.cuda.memory_reserved(0)
            if need > 0:
                torch.empty(need, dtype=torch.uint8, device="cuda:0")  # freed at once, stays cached
            print(f"[nightwatch serve] reserved {torch.cuda.memory_reserved(0) / 2**30:.1f} GiB", flush=True)
    port = port or int(env("S1_PORT", "8710"))
    gate = _PriorityGate()
    clip: dict = {}
    info = {"backend": s1.name, "device": getattr(s1, "device", "")}

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code: int, obj) -> None:
            data = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_GET(self):
            self._send(200, info) if self.path == "/health" else self._send(404, {"error": "not found"})

        def do_POST(self):
            try:
                body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
                if self.path == "/v1/ask":
                    qs = []
                    for q in body["questions"]:
                        base = BY_ID.get(q["id"])
                        qq = base if base else Q(q["id"], "?", q.get("kind", "noul"), q["text"], "", 99,
                                                 options=q.get("options"))
                        qs.append((qq, q["text"]))
                    req = Request(body.get("checkpoint", "?"), body.get("state", ""),
                                  [_img(b) for b in body["images"]], qs)
                    t = time.perf_counter()
                    # on the critical path (the waiter's fast stability tier, the pre-execution check): ahead of
                    # queued checks, otherwise the answer comes after the pixel rules / the action already went
                    with gate.hold(req.checkpoint == "pre_exec" or all(q.id in FAST_IDS for q, _ in qs)):
                        raw = s1._raw(req)
                    self._send(200, {"raw": raw, "ms": (time.perf_counter() - t) * 1000})
                elif self.path == "/v1/embed":
                    if "m" not in clip:
                        clip["m"] = _Clip()
                    self._send(200, {"embeddings": clip["m"].embed([_img(b) for b in body["images"]])})
                else:
                    self._send(404, {"error": "not found"})
            except Exception as e:  # report, keep serving
                log.exception("request failed")
                self._send(500, {"error": repr(e)})

    srv = ThreadingHTTPServer((host, port), H)
    print(f"[nightwatch serve] {info} on http://{host}:{port}", flush=True)
    srv.serve_forever()
