"""Local Intern-Decision (0.8B / 2B / 4B) as System-1: one causal forward pass scores every question.

Runs inside the System-1 env (torch + transformers 5.x), normally behind `python -m nightwatch serve`.
The released `inference.py` takes 1-16 questions and 1-8 images (paths, resolved against
media_root); calibration (the checkpoint's temperature) is applied by the engine.
"""

from __future__ import annotations

import importlib.util
import sys
import tempfile
import threading
import uuid
from pathlib import Path

from PIL import Image

from nightwatch.config import env, model_path
from nightwatch.questions import Request
from nightwatch.system1.base import System1


def pick_device(spec: str, min_free_gb: float = 4.0) -> str:
    if spec != "auto":
        return spec
    try:
        import torch

        if not torch.cuda.is_available():
            return "cpu"
        best, free_best = None, 0.0
        for i in range(torch.cuda.device_count()):
            free, _ = torch.cuda.mem_get_info(i)
            if free / 2**30 > free_best:
                best, free_best = i, free / 2**30
        return f"cuda:{best}" if best is not None and free_best >= min_free_gb else "cpu"
    except Exception:
        return "cpu"


class InternS1(System1):
    name = "intern"

    def __init__(self, path: str | None = None, device: str | None = None):
        size = env("INTERN_DECISION_SIZE", "0.8B")
        self.path = path or model_path("INTERN_DECISION_PATH", f"Intern-Decision-{size}", f"internlm/Intern-Decision-{size}")
        self.device_spec = device or env("INTERN_DEVICE", "auto")
        self.dtype = env("INTERN_DTYPE", "bfloat16")
        self.max_side = int(env("S1_MAX_SIDE", "1280"))
        self.name = f"intern:{Path(self.path).name}"
        self._engine = None
        self._lock = threading.Lock()
        self.media = Path(tempfile.mkdtemp(prefix="nightwatch-s1-"))

    def warmup(self) -> None:
        with self._lock:
            if self._engine is not None:
                return
            d = Path(self.path)
            if not (d / "inference.py").exists():
                raise FileNotFoundError(f"{d}/inference.py not found: run `bash scripts/setup_env.sh models` "
                                        "or set INTERN_DECISION_PATH")
            spec = importlib.util.spec_from_file_location("intern_decision_inference", d / "inference.py")
            mod = importlib.util.module_from_spec(spec)
            sys.modules[spec.name] = mod
            spec.loader.exec_module(mod)
            self.device = pick_device(self.device_spec)
            kw = {}
            tf = d / "nightwatch_temperature.json"  # written by training/ (temperature refit after fine-tuning)
            if tf.exists():
                import json

                kw["temperature"] = float(json.loads(tf.read_text())["temperature"])
            self._engine = mod.DecisionEngine(checkpoint=str(d), device=self.device, **kw,
                                              dtype=self.dtype if self.device != "cpu" else "float32",
                                              media_root=str(self.media))

    def _raw(self, req: Request) -> dict:
        self.warmup()
        names = []
        for img in req.images:
            im = Image.fromarray(img)
            if max(im.size) > self.max_side:
                s = self.max_side / max(im.size)
                im = im.resize((round(im.width * s), round(im.height * s)), Image.BILINEAR)
            name = f"{uuid.uuid4().hex}.png"
            im.save(self.media / name, compress_level=1)
            names.append(name)
        qs = {}
        for q, text in req.questions:
            if q.kind == "choice":
                qs[q.id] = {"type": "choice", "instructions": text, "criteria": dict(q.options)}
            else:
                qs[q.id] = {"type": "noul", "instructions": text}
        try:
            with self._lock:
                res = self._engine.predict({"state": req.state, "questions": qs, "images": names})
        finally:
            for n in names:
                (self.media / n).unlink(missing_ok=True)
        out = {}
        for qid, a in res["answers"].items():
            if a.get("type") == "choice":
                out[qid] = {"probs": a.get("probabilities", {})}
            else:
                out[qid] = {"yes": float(a["noul"])}
        return out
