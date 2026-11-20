"""Stage 1: supervised fine-tuning of the System-1 decision model on the event-question bank.

  python -m training.sft.torch_sft --family d3 --model data/models/d3-mini --data data/synthweb/ds --out data/ckpt/d3-mini-sft
  torchrun --nproc_per_node 4 -m training.sft.torch_sft ...      (data parallel; gradients all-reduced by hand)

Loss: cross-entropy on each labelled question's option logits, scored exactly as at inference (models.py),
with positives of rare questions up-weighted (sqrt of the class ratio, capped). d3 scores one question per
forward, so each example contributes `--questions-per-example` sampled questions; Intern-Decision scores all
of an example's questions in one forward. After training, one temperature is refitted on the validation split
(NLL), the package is exported with merged weights, and val/test metrics (AUROC, accuracy, Brier, ECE per
question) are written next to it.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from nightwatch.analysis.pilot import auroc, ece
from nightwatch.questions import BY_ID, CHANGE_KINDS


def qdict(qid: str, text: str) -> dict:
    q = BY_ID[qid]
    if q.kind == "choice":
        return {"type": "choice", "instructions": text, "criteria": dict(CHANGE_KINDS)}
    return {"type": "noul", "instructions": text}


def target_index(qid: str, value) -> int:
    if BY_ID[qid].kind == "choice":
        return list(CHANGE_KINDS).index(value)
    return 1 if value else 0


def load_examples(data: Path | str, split: str, limit: int | None = None) -> list[dict]:
    """examples.jsonl rows of one split; `data` may list several dataset folders ("a,b"), each row remembers its
    own folder (`_root`) for its images. "folder:N" keeps at most N rows of that folder (a fixed random sample),
    which sets the mixture. With a limit, rows are taken evenly from the folders."""
    per = []
    for spec in str(data).split(","):
        d, _, cap = spec.strip().partition(":")
        d = Path(d)
        rows = [json.loads(l) for l in open(d / "examples.jsonl") if f'"split": "{split}"' in l]
        rows = [dict(r, _root=str(d)) for r in rows if r["split"] == split]
        if cap and len(rows) > int(cap):
            rows = random.Random(0).sample(rows, int(cap))
        per.append(rows)
    if limit:
        per = [rows[: max(1, limit // len(per))] for rows in per]
    return [r for rows in per for r in rows]


def open_images(data: Path, ex: dict) -> list[Image.Image]:
    root = Path(ex.get("_root") or data)
    out = []
    for p in ex["images"]:
        q = Path(p)
        out.append(Image.open(q if q.is_absolute() else root / q).convert("RGB"))
    return out


def soft_target(qid: str, ex: dict) -> list[float] | None:
    """The target distribution over the question's options in canonical order (noul: [no, yes]; choice:
    CHANGE_KINDS order) when the example carries one (distillation: teacher mixed with gold), else None."""
    q = (ex.get("soft") or {}).get(qid)
    if q is None:
        return None
    q = [max(0.0, float(v)) for v in q]
    s = sum(q) or 1.0
    return [v / s for v in q]


def class_weights(rows: list[dict], cap: float = 4.0) -> dict:
    pos, n = defaultdict(int), defaultdict(int)
    for r in rows:
        for q, v in r["labels"].items():
            if BY_ID[q].kind != "choice":
                n[q] += 1
                pos[q] += bool(v)
    return {q: min(cap, math.sqrt(max(1, n[q] - pos[q]) / max(1, pos[q]))) for q in n}


def example_logits(model, data: Path, ex: dict, k: int | None, rng: random.Random,
                   permute: bool = False) -> list[tuple[str, torch.Tensor, int]]:
    """Option logits per labelled question. With `permute` (training), choice questions show their options in a
    random order (JevAgentHarness recipe; the d3 runtime averages both orders at inference anyway) and the
    target index follows the permutation."""
    imgs = open_images(data, ex)
    qs = [(qid, text) for qid, text in ex["questions"] if qid in ex["labels"]]
    if model.family == "d3" and k and len(qs) > k:
        qs = rng.sample(qs, k)
    dicts, targets = {}, {}
    for qid, t in qs:
        d, y = qdict(qid, t), target_index(qid, ex["labels"][qid])
        soft = soft_target(qid, ex)
        if permute and d["type"] == "choice":
            keys = list(d["criteria"])
            order = keys[:]
            rng.shuffle(order)
            d = dict(d, criteria={o: d["criteria"][o] for o in order})
            y = order.index(keys[y])
            if soft is not None:
                soft = [soft[keys.index(o)] for o in order]
        dicts[qid], targets[qid] = d, (soft if soft is not None else y)
    if model.family == "intern":
        lg = model.multi_logits(imgs, ex["state"], dicts)
        return [(qid, lg[qid], targets[qid]) for qid, _ in qs]
    return [(qid, model.option_logits(imgs, ex["state"], dicts[qid], name=qid), targets[qid]) for qid, _ in qs]


def loss_fn(lg: torch.Tensor, y, kind: str) -> torch.Tensor:
    """ce, or soft_ce_brier (JevAgentHarness SFT): CE + Brier score of the option distribution (keeps it calibrated).
    `y` is the target option index, or a target distribution over the options (soft labels): CE against it."""
    logp = torch.log_softmax(lg.float(), -1)
    if isinstance(y, int):
        q = torch.zeros_like(logp)
        q[y] = 1.0
    else:
        q = torch.tensor(y, dtype=logp.dtype, device=logp.device)
    ce = -(q * logp).sum()
    if kind == "ce":
        return ce
    return ce + ((logp.exp() - q) ** 2).sum()


@torch.no_grad()
def evaluate(model, data: Path, rows: list[dict], temperature: float = 1.0) -> tuple[dict, list]:
    model.eval()
    raw = []  # (qid, logits np, target)
    for ex in rows:
        for qid, lg, _ in example_logits(model, data, ex, None, random.Random(0)):
            # metrics on the hard label (gold, or the teacher's argmax), options in canonical order here
            raw.append((qid, lg.detach().float().cpu().numpy(), target_index(qid, ex["labels"][qid])))
    model.train()
    return metrics(raw, temperature), raw


def metrics(raw, temperature: float) -> dict:
    by = defaultdict(list)
    for qid, lg, y in raw:
        z = lg / temperature
        p = np.exp(z - z.max())
        p /= p.sum()
        by[qid].append((p, y))
    out = {}
    for qid, rows in by.items():
        if BY_ID[qid].kind == "choice":
            out[qid] = {"n": len(rows), "acc": float(np.mean([p.argmax() == y for p, y in rows]))}
        else:
            ps, ys = np.array([p[1] for p, _ in rows]), np.array([y for _, y in rows], bool)
            out[qid] = {"n": len(rows), "pos": int(ys.sum()), "auroc": auroc(ps, ys), "acc": float(((ps >= .5) == ys).mean()),
                        "brier": float(((ps - ys) ** 2).mean()), "ece": ece(ps, ys)}
    return out


def fit_temperature(raw) -> float:
    best, bt = 1e18, 1.0
    for t in np.exp(np.linspace(np.log(0.25), np.log(8), 60)):
        nll = 0.0
        for _, lg, y in raw:
            z = lg / t
            nll -= z[y] - (z.max() + np.log(np.exp(z - z.max()).sum()))
        if nll < best:
            best, bt = nll, float(t)
    return bt


def _dist():
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        import torch.distributed as dist

        dist.init_process_group("nccl" if torch.cuda.is_available() else "gloo")
        return dist.get_rank(), dist.get_world_size(), dist
    return 0, 1, None


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["d3", "intern"], required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--data", default="data/synthweb/ds", help="dataset folder(s), comma separated")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--epochs", type=float, default=2.0)
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--lora-r", type=int, default=16)
    ap.add_argument("--full", action="store_true", help="full fine-tune instead of LoRA (vision tower stays frozen)")
    ap.add_argument("--max-side", type=int, default=1280)
    ap.add_argument("--questions-per-example", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--val-limit", type=int, default=400)
    ap.add_argument("--eval-every", type=int, default=500)
    ap.add_argument("--save-every", type=int, default=50, help="optimizer steps between resume checkpoints (<out>/resume.pt)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--loss", choices=["ce", "soft_ce_brier"], default="soft_ce_brier")
    ap.add_argument("--no-permute", action="store_true", help="keep choice options in canonical order")
    a = ap.parse_args(argv)
    rank, world, dist = _dist()
    device = f"cuda:{int(os.environ.get('LOCAL_RANK', 0))}" if a.device == "cuda" and world > 1 else a.device
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed + rank)
    data, out = Path(a.data), Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    from training.models import load_trainable

    model = load_trainable(a.family, a.model, device=device, max_side=a.max_side, lora_r=a.lora_r, full=a.full)
    reserve = os.environ.get("TRAIN_GPU_RESERVE")
    if reserve and device.startswith("cuda"):  # hold this share of a shared GPU in torch's cache (training reuses it)
        need = int(float(reserve) * torch.cuda.get_device_properties(device).total_memory) - torch.cuda.memory_reserved(device)
        if need > 0:
            torch.empty(need, dtype=torch.uint8, device=device)
    train = load_examples(data, "train", a.limit)
    val = load_examples(data, "val", a.val_limit)
    w = class_weights(train)
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr, weight_decay=0.0)
    total = a.max_steps if a.max_steps is not None else int(a.epochs * len(train) / (a.accum * world))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / max(20, total // 20)) * max(0.05, 1 - s / max(1, total)))
    log = open(out / "train_log.jsonl", "a")
    if rank == 0:
        base_m, _ = evaluate(model, data, val[:100])
        print("[sft] zero-shot val (100 ex):", json.dumps({k: {kk: (round(vv, 3) if isinstance(vv, float) else vv) for kk, vv in v.items()} for k, v in base_m.items()}), flush=True)
    model.train()  # gradient checkpointing only acts in train mode
    order = [i for i in range(len(train)) if i % world == rank]
    step, t0, run = 0, time.time(), 0.0
    named = {n: p for n, p in model.named_parameters() if p.requires_grad}
    ck = out / "resume.pt"
    skip = 0
    if ck.exists():  # a killed run continues from its last checkpoint (data order replayed by the same seed)
        st = torch.load(ck, map_location=device, weights_only=False)
        with torch.no_grad():
            for n, v in st["params"].items():
                named[n].copy_(v)
        opt.load_state_dict(st["opt"])
        sched.load_state_dict(st["sched"])
        step, skip = st["step"], st["step"] * a.accum
        if rank == 0:
            print(f"[sft] resumed at step {step}/{total}", flush=True)
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if "cuda" in str(device) else torch.autocast("cpu", enabled=False)
    while step < total:
        rng.shuffle(order)
        for k, i in enumerate(order):
            if skip:
                skip -= 1
                continue
            ex = train[i]
            with autocast:
                outs = example_logits(model, data, ex, a.questions_per_example, rng, permute=not a.no_permute)
            loss = 0.0
            for qid, lg, y in outs:
                pos = (y == 1) if isinstance(y, int) else (len(y) == 2 and y[1] >= 0.5)
                wt = w.get(qid, 1.0) if (BY_ID[qid].kind != "choice" and pos) else 1.0
                loss = loss + wt * loss_fn(lg, y, a.loss)
            if not outs:
                continue
            (loss / (len(outs) * a.accum)).backward()
            run += float(loss.detach()) / len(outs)
            if (k + 1) % a.accum:
                continue
            if dist is not None:
                for p in params:
                    if p.grad is not None:
                        dist.all_reduce(p.grad)
                        p.grad /= world
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if rank == 0 and step % 10 == 0:
                rec = {"step": step, "loss": run / (10 * a.accum), "lr": sched.get_last_lr()[0], "s": round(time.time() - t0)}
                log.write(json.dumps(rec) + "\n")
                log.flush()
                print("[sft]", rec, flush=True)
                run = 0.0
            if rank == 0 and step % a.save_every == 0:
                torch.save({"params": {n: p.detach().cpu() for n, p in named.items()}, "opt": opt.state_dict(),
                            "sched": sched.state_dict(), "step": step}, out / "resume.tmp")
                (out / "resume.tmp").replace(ck)
            if rank == 0 and step % a.eval_every == 0:
                m, _ = evaluate(model, data, val[:200])
                log.write(json.dumps({"step": step, "val": m}) + "\n")
            if step >= total:
                break
    if rank == 0:
        _, raw = evaluate(model, data, val)
        temp = fit_temperature(raw) if len(raw) >= 50 else model.temperature  # too little validation data: keep it
        pkg = model.save(out / "package", temperature=temp)
        report = {"temperature": temp, "val": metrics(raw, temp)}
        test = load_examples(data, "test", 600)
        if test:
            report["test"] = evaluate(model, data, test, temp)[0]
        hold = load_examples(data, "holdout", 600)
        if hold:
            report["holdout"] = evaluate(model, data, hold, temp)[0]
        (out / "metrics.json").write_text(json.dumps(report, indent=2))
        print(f"[sft] done: package {pkg}, temperature {temp:.3f}", flush=True)
    if dist is not None:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
