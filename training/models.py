"""Trainable wrappers around the two decision-model families, scoring exactly as their own inference code.

Both expose:
  option_logits(example_images, state, question_dict) -> 1-D tensor of option logits (option order of the
      question: noul = [no, yes]; choice = criteria order), with gradients;
  save(out_dir)  -> a package the normal loaders read (D3S1 / InternS1), with LoRA merged;
  temperature    -> the softmax temperature the deployed package applies (refit after training).

D3 (vllm-sr d3-mini / d3-lite):  prompt = runtime.image_text(state, question, n_images); processor encodes
    text + images; logits = backbone(...).last_hidden_state[:, -1] @ readout.T over the question's codes
    (exactly `D3.image_logits`).  The package is re-written with merged weights, the trained readout and the
    fitted temperature; MODEL_MANIFEST.json is dropped (an unverified local export, which the runtime accepts).
Intern-Decision:  the request is compiled by the checkpoint's own `compile_row` + `HFBackend.encode`; logits
    at each decision placeholder over that field's candidate symbols (exactly `DecisionEngine.predict`).
    The package is the checkpoint folder with merged weights; the temperature goes to nightwatch_temperature.json
    (InternS1 reads it).
"""

from __future__ import annotations

import importlib.util
import os
import json
import shutil
import sys
import tempfile
import uuid
from pathlib import Path

import torch
from PIL import Image

LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "in_proj_qkvz",
                "in_proj_ba", "out_proj"]


def _resize(img: Image.Image, max_side: int) -> Image.Image:
    if max(img.size) > max_side:
        s = max_side / max(img.size)
        img = img.resize((max(28, round(img.width * s)), max(28, round(img.height * s))), Image.BILINEAR)
    return img


def _lora(module, r: int, alpha: int, dropout: float, freeze_vision: bool):
    from peft import LoraConfig, get_peft_model

    names = {n.split(".")[-1] for n, m in module.named_modules() if isinstance(m, torch.nn.Linear)}
    targets = [t for t in LORA_TARGETS if t in names]
    excl = r".*visual.*" if freeze_vision else None
    cfg = LoraConfig(r=r, lora_alpha=alpha, lora_dropout=dropout, target_modules=targets, exclude_modules=excl)
    return get_peft_model(module, cfg)


class D3Trainable(torch.nn.Module):
    family = "d3"

    def __init__(self, path: str, device: str = "cpu", max_side: int = 1280, lora_r: int = 16, lora_alpha: int = 32,
                 lora_dropout: float = 0.05, full: bool = False, freeze_vision: bool = True, grad_ckpt: bool = True):
        super().__init__()
        from transformers import AutoModel

        self.path = Path(path)
        m = AutoModel.from_pretrained(str(self.path), trust_remote_code=True, device=device)
        self.rt = m.runtime
        self.mod = sys.modules[type(self.rt).__module__]
        self.device = torch.device(device)
        bb = self.rt.backbone
        if self.device.type == "cpu" and os.environ.get("NIGHTWATCH_TRAIN_FP32", "1") == "1":
            bb.float()  # CPU training in fp32 (bf16 matmuls on CPU are slow and lossy); GPU keeps the checkpoint bf16
        if grad_ckpt and hasattr(bb, "gradient_checkpointing_enable"):
            bb.gradient_checkpointing_enable()
        if full:
            for n, p in bb.named_parameters():
                p.requires_grad_(not (freeze_vision and n.startswith("visual")))
            self.backbone = bb
        else:
            for p in bb.parameters():
                p.requires_grad_(False)
            self.backbone = _lora(bb, lora_r, lora_alpha, lora_dropout, freeze_vision)
        self.readout = torch.nn.Parameter(self.rt.readout.detach().float().clone())
        self.temperature = float(self.rt.temperature)
        self.max_side = max_side
        self.full = full

    def option_logits(self, images: list[Image.Image], state: str, question: dict, name: str = "q") -> torch.Tensor:
        q = self.mod.normalize_question(question)
        imgs = [_resize(i, self.max_side) for i in images]
        text = self.rt.image_text(state, q.rendered, len(imgs))
        enc = self.rt.processor(text=[text], images=imgs, return_tensors="pt", padding=True)
        enc = {k: v.to(self.device) for k, v in enc.items()}
        # exactly D3.image_logits: the runtime's image embeddings + multimodal RoPE, then its PromptEncoder
        # (bidirectional softmax attention over the prompt; causal Gated-DeltaNet scan). The stock causal HF
        # forward is NOT the d3 model and disagrees by up to 0.3 in probability on multi-image requests.
        embeds, mrope = self.rt.image_embeddings(enc)
        hidden = self.rt.encoder(embeds, enc["attention_mask"], mrope)[:, -1]
        logits = hidden.float() @ self.readout.T
        return logits[0, : len(q.keys)]

    def save(self, out: Path, temperature: float | None = None) -> Path:
        from safetensors.torch import save_file

        out = Path(out)
        if out.exists():
            shutil.rmtree(out)
        shutil.copytree(self.path, out, ignore=shutil.ignore_patterns("*.safetensors", "MODEL_MANIFEST.json", "*.index.json"),
                        symlinks=False)
        bb = self.backbone.merge_and_unload() if not self.full else self.backbone
        sd = {k: v.detach().to(torch.bfloat16).contiguous().cpu() for k, v in bb.state_dict().items()}
        save_file(sd, str(out / "model-00001-of-00001.safetensors"), metadata={"format": "pt"})
        (out / "model.safetensors.index.json").write_text(json.dumps(
            {"metadata": {}, "weight_map": {k: "model-00001-of-00001.safetensors" for k in sd}}))
        save_file({"weight": self.readout.detach().float().cpu().contiguous()}, str(out / "readout.safetensors"))
        cfg = json.loads((out / "decision_config.json").read_text())
        cfg["temperature"] = float(temperature or self.temperature)
        cfg["nightwatch"] = {"base": str(self.path), "note": "fine-tuned by training/"}
        (out / "decision_config.json").write_text(json.dumps(cfg, indent=2))
        return out


class InternTrainable(torch.nn.Module):
    family = "intern"

    def __init__(self, path: str, device: str = "cpu", max_side: int = 1280, lora_r: int = 16, lora_alpha: int = 32,
                 lora_dropout: float = 0.05, full: bool = False, freeze_vision: bool = True, grad_ckpt: bool = True):
        super().__init__()
        self.path = Path(path)
        spec = importlib.util.spec_from_file_location("intern_decision_inference_train", self.path / "inference.py")
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod
        spec.loader.exec_module(mod)
        self.mod = mod
        self.media = Path(tempfile.mkdtemp(prefix="nightwatch-train-"))
        self.be = mod.HFBackend(str(self.path), media_root=str(self.media), device=device,
                                dtype="float32" if device == "cpu" else "bfloat16")
        self.device = torch.device(device)
        model = self.be.model
        if grad_ckpt and hasattr(model, "gradient_checkpointing_enable"):
            model.gradient_checkpointing_enable()
        if full:
            for n, p in model.named_parameters():
                p.requires_grad_(not (freeze_vision and "visual" in n))
            self.model = model
        else:
            for p in model.parameters():
                p.requires_grad_(False)
            self.model = _lora(model, lora_r, lora_alpha, lora_dropout, freeze_vision)
        tf = self.path / "nightwatch_temperature.json"
        self.temperature = float(json.loads(tf.read_text())["temperature"]) if tf.exists() else float(mod.DEFAULT_TEMPERATURE)
        self.max_side = max_side
        self.full = full

    def multi_logits(self, images: list[Image.Image], state: str, questions: dict) -> dict[str, torch.Tensor]:
        """All fields in one forward (the family's native mode): {qid: option logits}."""
        names = []
        for img in images:
            n = f"{uuid.uuid4().hex}.png"
            _resize(img, self.max_side).save(self.media / n)
            names.append(n)
        row = self.mod.validate_request({"state": state, "questions": questions, "images": names})
        try:
            compiled, batch, positions = self.be.encode(row)
        finally:
            for n in names:
                (self.media / n).unlink(missing_ok=True)
        batch = batch.to(self.device)
        logits = self.model(**batch, use_cache=False, logits_to_keep=positions.to(self.device)).logits[0]
        out = {}
        for i, f in enumerate(compiled.fields):
            ids = [self.be.tokenizer.encode(s, add_special_tokens=False)[0] for s in compiled.symbols[f]]
            out[f] = logits[i, ids].float()
        return out

    def option_logits(self, images, state, question, name: str = "q") -> torch.Tensor:
        # the field name is part of Intern-Decision's prompt (JSON skeleton): use the question id, as InternS1 does
        return self.multi_logits(images, state, {name: question})[name]

    def save(self, out: Path, temperature: float | None = None) -> Path:
        out = Path(out)
        if out.exists():
            shutil.rmtree(out)
        shutil.copytree(self.path, out, ignore=shutil.ignore_patterns("*.safetensors", "*.index.json", "__pycache__"))
        model = self.model.merge_and_unload() if not self.full else self.model
        model.save_pretrained(str(out), safe_serialization=True)
        (out / "nightwatch_temperature.json").write_text(json.dumps({"temperature": float(temperature or self.temperature)}))
        return out


def load_trainable(family: str, path: str, **kw):
    return (D3Trainable if family == "d3" else InternTrainable)(path, **kw)
