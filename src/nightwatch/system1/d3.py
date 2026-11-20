"""vLLM-SR Decision 3.0 (d3-mini 4B, d3-lite 0.8B, ...) as System-1, in-process.

Loaded with `AutoModel.from_pretrained(path, trust_remote_code=True)` (the d3 runtime shipped in the model
repo; transformers 5.17.0). A request's images are shared by all of its questions; each question is one
forward with a 255-way answer-code readout at the last position (batched by the runtime). Fine-tuned
checkpoints written by `training.sft.torch_sft` keep the same package layout, so the same loader serves them.
"""

from __future__ import annotations

import threading
from pathlib import Path

from PIL import Image

from nightwatch.config import MODELS_DIR, env
from nightwatch.questions import Request
from nightwatch.system1.base import System1
from nightwatch.system1.intern import pick_device


def _repeat_cache(cache, n: int) -> None:
    """Repeat a (hybrid) HF cache along the batch dimension: attention layers have batch_repeat_interleave; the
    Gated-DeltaNet layers keep conv / recurrent states in per-state containers."""
    import torch

    for layer in getattr(cache, "layers", []):
        if hasattr(layer, "batch_repeat_interleave"):
            layer.batch_repeat_interleave(n)
            continue
        for attr in ("conv_states", "recurrent_states"):
            box = getattr(layer, attr, None)
            if isinstance(box, dict):
                for k, v in box.items():
                    if torch.is_tensor(v) and v.numel():
                        box[k] = v.repeat_interleave(n, dim=0)
            elif isinstance(box, list):
                for k, v in enumerate(box):
                    if torch.is_tensor(v) and v.numel():
                        box[k] = v.repeat_interleave(n, dim=0)
            elif torch.is_tensor(box) and box.numel():
                setattr(layer, attr, box.repeat_interleave(n, dim=0))


class D3S1(System1):
    def __init__(self, model: str | None = None, device: str | None = None):
        spec = model or env("D3_MODEL", "d3-mini")
        p = Path(spec)
        self.path = str(p if p.exists() else MODELS_DIR / spec.split("/")[-1])
        self.name = f"d3:{Path(self.path).name}"
        self.device_spec = device or env("D3_DEVICE", "auto")
        self.max_side = int(env("S1_MAX_SIDE", "1280"))
        self._m = None
        self._lock = threading.Lock()

    def warmup(self) -> None:
        with self._lock:
            if self._m is None:
                from transformers import AutoModel

                if not Path(self.path).exists():
                    raise FileNotFoundError(f"{self.path} missing: MODELS=\"d3-mini d3-lite\" bash scripts/setup_env.sh models")
                self.device = pick_device(self.device_spec, min_free_gb=6)
                # batch_size = questions per forward pass; every question re-encodes all images, so on a shared
                # GPU keep it small (D3_BATCH)
                self._m = AutoModel.from_pretrained(self.path, trust_remote_code=True, device=self.device,
                                                    batch_size=int(env("D3_BATCH", "8"))).eval()

    # ------------------------------------------------------------------ shared-vision fast path
    def _prefix_scores(self, state: str, qs: dict, imgs: list) -> dict:
        """The runtime re-encodes every image for every question (one forward per question). Here the images go
        through the vision tower ONCE per request; each question's full prompt (shared prefix with the image
        tokens + its own question text, plus the reversed-option prompt the runtime averages choice questions with)
        then runs through the language model as one left-padded batch with those image embeddings scattered in.
        It is the runtime's computation without the repeated vision work (KV-cache continuation is not an option:
        the Gated-DeltaNet layers only resume their recurrent state for single-token steps)."""
        import os

        import torch

        rt = self._m.runtime
        mod = __import__("sys").modules[type(rt).__module__]
        items = []  # (qid, reversed?, Question, text)
        for qid, qd in qs.items():
            q = mod.normalize_question(qd)
            items.append((qid, False, q, rt.image_text(state, q.rendered, len(imgs))))
            rev = mod.reversed_question(q)
            if rev is not None:
                items.append((qid, True, q, rt.image_text(state, rev, len(imgs))))
        texts = [t for *_, t in items]
        cut = os.path.commonprefix(texts).rfind("\n") + 1
        if cut <= 0:
            raise ValueError("no shared prefix")
        tok = rt.processor.tokenizer
        dev = torch.device(self.device)
        bb = rt.backbone
        enc = rt.processor(text=[texts[0][:cut]], images=imgs, return_tensors="pt")
        prefix = enc["input_ids"][0].tolist()
        rows = [prefix + tok(t[cut:], add_special_tokens=False)["input_ids"] for t in texts]
        img_id = bb.config.image_token_id
        chunk = int(env("D3_LM_BATCH", "16"))
        with torch.inference_mode():
            feats = bb.get_image_features(enc["pixel_values"].to(dev), enc["image_grid_thw"].to(dev), return_dict=True)
            img_emb = torch.cat(feats.pooler_output, dim=0)
            emb = bb.get_input_embeddings()
            hs = []
            for s0 in range(0, len(rows), chunk):
                part = rows[s0 : s0 + chunk]
                n, width = len(part), max(len(r) for r in part)
                ids = torch.full((n, width), tok.pad_token_id or 0, dtype=torch.long)
                mask = torch.zeros((n, width), dtype=torch.long)
                for i, r in enumerate(part):  # left padding, as the runtime's processor does
                    ids[i, width - len(r):] = torch.tensor(r)
                    mask[i, width - len(r):] = 1
                ids, mask = ids.to(dev), mask.to(dev)
                e = emb(ids)
                im = (ids == img_id).unsqueeze(-1).expand_as(e)
                e = e.masked_scatter(im, img_emb.to(e.dtype).repeat(n, 1))
                grid = enc["image_grid_thw"].to(dev).repeat(n, 1)
                types = torch.zeros_like(ids)
                if "mm_token_type_ids" in enc:
                    tt = enc["mm_token_type_ids"][0].to(dev)
                    for i, r in enumerate(part):
                        types[i, width - len(r): width - len(r) + len(prefix)] = tt
                pos = bb.compute_3d_position_ids(input_ids=ids, inputs_embeds=e, image_grid_thw=grid, video_grid_thw=None,
                                                 attention_mask=mask, past_key_values=None, mm_token_type_ids=types)
                # d3's own prompt encoder: bidirectional softmax attention over the prompt, causal DeltaNet scan
                hs.append(rt.encoder(e, mask, pos)[:, -1])
            h = torch.cat(hs)
            logits = (h.float() @ rt.readout.float().T) / rt.temperature
        return self._finish(items, logits, mod)

    @staticmethod
    def _finish(items, logits, mod) -> dict:
        import torch

        fwd: dict = {}
        bwd: dict = {}
        for (qid, rev, q, _), lg in zip(items, logits):
            c = len(q.keys)
            p = torch.softmax(lg[:c], -1).tolist()
            (bwd if rev else fwd)[qid] = (q, p)
        out_ans = {}
        for qid, (q, p) in fwd.items():
            if qid in bwd:
                p = mod.average_orders(p, bwd[qid][1])
            out_ans[qid] = mod.product_answer(q, p) if hasattr(mod, "product_answer") else {"probabilities": dict(zip(q.keys, p))}
        return out_ans

    def _raw(self, req: Request) -> dict:
        self.warmup()
        imgs = []
        for a in req.images:
            im = Image.fromarray(a)
            if max(im.size) > self.max_side:
                s = self.max_side / max(im.size)
                im = im.resize((round(im.width * s), round(im.height * s)), Image.BILINEAR)
            imgs.append(im)
        qs = {}
        for q, text in req.questions:
            qs[q.id] = ({"type": "choice", "instructions": text, "criteria": dict(q.options)} if q.kind == "choice"
                        else {"type": "noul", "instructions": text})
        answers = None
        if imgs and env("D3_PREFIX", "1") == "1" and str(self.device).startswith("cuda"):
            try:
                with self._lock:
                    answers = self._prefix_scores(req.state, qs, imgs)
            except Exception as e:  # fall back to the runtime's own path
                import logging

                logging.getLogger("nightwatch.d3").warning("prefix path failed (%s); using system_one", e)
        if answers is None:
            with self._lock:
                answers = self._m.system_one(state=req.state, questions=qs, images=imgs or None).get("answers") or {}
        out = {}
        for qid, a in answers.items():
            if not isinstance(a, dict):
                continue
            if a.get("type") == "choice":
                out[qid] = {"probs": a.get("probabilities", {})}
            elif a.get("noul") is not None:
                out[qid] = {"yes": float(a["noul"])}
        return out
