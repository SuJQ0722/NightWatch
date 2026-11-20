"""Stage 2: reinforcement learning of the decisions themselves (wake / interrupt / invalidate).

The System-1's per-question probabilities define a stochastic policy, composed exactly like rules.py:

  wake (after an action)    p_w(f) = p_event + (1 - p_event) * p_ready
                            p_event = 1 - (1-p_dialog)(1-p_error)(1-p_toast)(1-p_unexpected)
                            p_ready = p_settled * (1 - p_still_changing) * (1 - p_loading)
  interrupt (during VLM)    p_i(f) = 1 - (1-p_dialog)(1-p_error)(1-p_page_changed)
  invalidate (pre-exec)     p_v    = 1 - (1-p_target_gone)(1-p_target_occluded)(1-p_target_changed)(1-p_dialog)

Episodes come from SynthWeb recordings, so the environment's future is known: the policy acts only on the
frames stage 1 would send to System-1 (pixel change, or a probe every `probe_s`), and the first frame where
it fires ends the episode. The expected return is therefore exact, no sampling needed:
  E[R] = sum_k  p_k prod_{j<k}(1 - p_j) R(t_k)  +  prod_j (1 - p_j) R(timeout)
and its gradient flows into the decision model (LoRA on top of the SFT package).

Rewards are in seconds of wall clock, the quantity the watcher saves:
  wake        R = -(t_w - t_action) - c_early [screen not ready yet: the next step is invalid]
                  - c_miss [a transient toast/error came and went unseen]
  interrupt   event at t_e:  interrupt at t >= t_e: -(t - t_e);  no interrupt: -(t0+L - t_e) - c_stale
              no event / too early:  -(t - t0) - c_regen        (the generation thrown away)
  invalidate  stale target:  invalidate: -L;  execute: -c_invalid       valid target: invalidate: -L; execute: 0
Loss = -E[R]/scale + beta * KL(policy || SFT reference, per question) + gamma * CE(labels)  (keeps calibration).

  python -m training.rl.offline_rl --family d3 --model data/ckpt/d3-mini-sft/package --rec data/synthweb/rec --out data/ckpt/d3-mini-rl
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from nightwatch.diff import ChangeDetector
from nightwatch.questions import BANK, BY_ID, build_request
from training.data.build import TASKS, label
from training.sft.torch_sft import qdict, target_index
from nightwatch.utils import box_around, crop, draw_marker, small_gray

WAKE_Q = ["dialog", "error", "toast", "unexpected", "settled", "still_changing", "loading"]
INT_Q = ["dialog", "error", "page_changed"]
INV_Q = ["target_present", "target_occluded", "target_changed", "dialog"]


def _key(g):
    return (g["version"], g["dialog"], g["toast"], g["error"], g["loading"], g["animating"], g["page"])


class Recording:
    def __init__(self, d: Path, det_cache: dict):
        self.d = d
        self.spec = json.loads((d / "spec.json").read_text())
        self.frames = [json.loads(l) for l in open(d / "frames.jsonl")]
        self.actions = [json.loads(l) for l in open(d / "actions.jsonl")]
        self.ts = np.array([f["t"] for f in self.frames])
        self._img: dict = {}
        self.det = ChangeDetector((self.spec["width"], self.spec["height"]))
        self._sig = None

    def img(self, i) -> np.ndarray:
        if i not in self._img:
            self._img[i] = np.asarray(Image.open(self.d / "frames" / self.frames[i]["file"]).convert("RGB"))
            if len(self._img) > 64:
                self._img.pop(next(iter(self._img)))
        return self._img[i]

    def sig(self) -> list[bool]:
        if self._sig is None:
            prev, out = None, []
            for i in range(len(self.frames)):
                s = small_gray(self.img(i))
                out.append(prev is not None and self.det.compare(prev, s).significant)
                prev = s
            self._sig = out
        return self._sig

    def at(self, t) -> int:
        return int(np.clip(np.searchsorted(self.ts, t, side="right") - 1, 0, len(self.frames) - 1))

    def gated(self, i0: int, i1: int, probe_s: float) -> list[int]:
        """Frames stage 1 would pass in (i0, i1]: pixel change or a probe every probe_s."""
        sig, out, last = self.sig(), [], self.ts[i0]
        for i in range(i0 + 1, i1 + 1):
            if sig[i] or self.ts[i] - last >= probe_s:
                out.append(i)
                last = self.ts[i]
        return out


def wake_episode(rec: Recording, a: dict, cfg: dict) -> dict | None:
    i_b = a["idx_before"]
    t_a = a["t"]
    i_end = rec.at(t_a + cfg["horizon"])
    if i_end - i_b < 10:
        return None
    gts = [rec.frames[i]["gt"] for i in range(i_b, i_end + 1)]
    ts = rec.ts[i_b:i_end + 1]
    changes = [ts[k] for k in range(1, len(gts)) if _key(gts[k]) != _key(gts[k - 1]) and ts[k] > t_a]
    first = changes[0] if changes and changes[0] - t_a < 2.0 else None
    t_ready = None
    for k in range(len(gts)):
        t = ts[k]
        if t < t_a or (first is not None and t < first) or gts[k]["loading"] or gts[k]["animating"]:
            continue
        if not any(t < c <= t + 0.4 for c in changes):
            t_ready = t
            break
    t_ready = t_ready if t_ready is not None else ts[-1]
    windows = []
    for kind in ("toast", "error"):
        on = None
        for k in range(1, len(gts)):
            if gts[k][kind] and not gts[k - 1][kind] and ts[k] > t_a:
                on = ts[k]
            if on is not None and not gts[k][kind]:
                windows.append((on, ts[k]))
                on = None
    i_seen = rec.at(t_a - random.uniform(0.3, 5.0))  # the agent's screenshot (loop: post_exec before = obs_frame)
    return {"kind": "wake", "rec": rec, "i_b": i_b, "i_seen": i_seen, "t_a": t_a, "frames": rec.gated(i_b, i_end, cfg["probe_s"]),
            "t_ready": float(t_ready), "windows": windows, "horizon_t": float(ts[-1]),
            "action": f"click at ({a['x']:.0f}, {a['y']:.0f})" if a["kind"] != "type" else f"type the text '{a['text']}'"}


def wake_reward(ep: dict, t_w: float, c: dict) -> float:
    r = -(t_w - ep["t_a"])
    for on, off in ep["windows"]:
        if on < ep["t_ready"] + 1.0:
            if on + 0.15 <= t_w <= off:
                return r
            if t_w > off:
                return r - c["c_miss"]
    if t_w < ep["t_ready"]:
        return r - c["c_early"]
    return r


def interrupt_episode(rec: Recording, rng: random.Random, cfg: dict) -> dict | None:
    if len(rec.frames) < 60:
        return None
    i0 = rng.randrange(0, len(rec.frames) - 40)
    L = min(cfg["lat_max"], rng.lognormvariate(math.log(cfg["lat_median"]), 0.45))
    i1 = rec.at(rec.ts[i0] + L)
    seen = rec.frames[i0]["gt"]
    t_e = None
    for i in range(i0 + 1, i1 + 1):
        g = rec.frames[i]["gt"]
        if (g["dialog"] and not seen["dialog"]) or (g["error"] and not seen["error"]) or g["page"] != seen["page"]:
            t_e = float(rec.ts[i])
            break
    return {"kind": "interrupt", "rec": rec, "i0": i0, "t0": float(rec.ts[i0]), "L": L, "t_e": t_e,
            "frames": rec.gated(i0, i1, cfg["probe_s"])}


def interrupt_reward(ep: dict, t: float | None, c: dict) -> float:
    t0, te = ep["t0"], ep["t_e"]
    if t is None:  # never interrupted
        return -(t0 + ep["L"] - te) - c["c_stale"] if te is not None else 0.0
    if te is not None and t >= te:
        return -(t - te)
    return -(t - t0) - c["c_regen"]


def invalidate_episode(rec: Recording, rng: random.Random, cfg: dict) -> dict | None:
    i_s = rng.randrange(0, max(1, len(rec.frames) - 40))
    els = rec.frames[i_s]["gt"]["elements"]
    if not els:
        return None
    e = rng.choice(els)
    x, y = (e[1] + e[3]) / 2, (e[2] + e[4]) / 2
    i_n = rec.at(rec.ts[i_s] + rng.uniform(0.5, 6.0))
    g = {"seen": rec.frames[i_s]["gt"], "now_marked": rec.frames[i_n]["gt"]}
    stale = (not label("target_present", g, {"point": (x, y)})) or label("target_occluded", g, {"point": (x, y)})
    return {"kind": "invalidate", "rec": rec, "i_s": i_s, "i_n": i_n, "point": (x, y), "stale": bool(stale),
            "L": min(cfg["lat_max"], rng.lognormvariate(math.log(cfg["lat_median"]), 0.45))}


def invalidate_reward(ep: dict, invalidated: bool, c: dict) -> float:
    if invalidated:
        return -ep["L"]
    return -c["c_invalid"] if ep["stale"] else 0.0


# ------------------------------------------------------------------------------------------- policy
def fires(model, images, state, qids, req_questions) -> dict[str, torch.Tensor]:
    """P(fire) per question, differentiable."""
    text = dict(req_questions)
    out = {}
    if model.family == "intern":
        lg = model.multi_logits(images, state, {q: qdict(q, text[q]) for q in qids})
    else:
        lg = {q: model.option_logits(images, state, qdict(q, text[q]), name=q) for q in qids}
    for q in qids:
        p = torch.softmax(lg[q] / model.temperature, -1)
        out[q] = p[1] if BY_ID[q].fire_on == "yes" else p[0]
    return out


def _pil(a):
    return Image.fromarray(a) if isinstance(a, np.ndarray) else a


def frame_request(ep: dict, i: int, task: str):
    rec = ep["rec"]
    if ep["kind"] == "wake":
        i_b = ep["i_b"]
        prev = rec.at(rec.ts[i] - 0.12)
        ch_mov = rec.det.compare(small_gray(rec.img(prev)), small_gray(rec.img(i)))
        ch_pre = rec.det.compare(small_gray(rec.img(i_b)), small_gray(rec.img(i)))
        cb = ch_mov.union if ch_mov.significant else ch_pre.union
        imgs = {"before": rec.img(ep.get("i_seen", i_b)), "recent": rec.img(rec.at(rec.ts[i] - 0.5)), "now": rec.img(i),
                "crop": crop(rec.img(i), cb, 24, 160) if cb else rec.img(i)}
        return build_request("post_exec", imgs, BANK, task, ep["action"]), WAKE_Q
    if ep["kind"] == "interrupt":
        i0 = ep["i0"]
        ch = rec.det.compare(small_gray(rec.img(i0)), small_gray(rec.img(i)))
        imgs = {"seen": rec.img(i0), "now": rec.img(i), "crop": crop(rec.img(i), ch.union, 24, 160) if ch.union else rec.img(i)}
        return build_request("reasoning", imgs, BANK, task), INT_Q
    x, y = ep["point"]
    tb = box_around(x, y, 80)
    imgs = {"seen": rec.img(ep["i_s"]), "now_marked": draw_marker(rec.img(i), x, y),
            "target_seen": crop(rec.img(ep["i_s"]), tb, 0, 160), "target_now": crop(rec.img(i), tb, 0, 160)}
    return build_request("pre_exec", imgs, BANK, task, f"click at ({x:.0f}, {y:.0f})"), INV_Q


def compose(kind: str, f: dict) -> torch.Tensor:
    if kind == "wake":
        ev = 1 - (1 - f["dialog"]) * (1 - f["error"]) * (1 - f["toast"]) * (1 - f["unexpected"])
        ready = f["settled"] * (1 - f["still_changing"]) * (1 - f["loading"])
        return ev + (1 - ev) * ready
    if kind == "interrupt":
        return 1 - (1 - f["dialog"]) * (1 - f["error"]) * (1 - f["page_changed"])
    return 1 - (1 - f["target_present"]) * (1 - f["target_occluded"]) * (1 - f["target_changed"]) * (1 - f["dialog"])


def episode_loss(model, ep: dict, c: dict, task: str, beta: float, max_side_frames: int) -> tuple[torch.Tensor, dict]:
    rec = ep["rec"]
    frames = [ep["i_n"]] if ep["kind"] == "invalidate" else ep["frames"][:max_side_frames]
    ps, kls, t_list = [], [], []
    for i in frames:
        req, qids = frame_request(ep, i, task)
        imgs = [_pil(a) for a in req.images]
        f = fires(model, imgs, req.state, qids, [(q.id, t) for q, t in req.questions])
        with torch.no_grad(), _reference(model):
            fr = fires(model, imgs, req.state, qids, [(q.id, t) for q, t in req.questions])
        for q in qids:
            p, r = f[q].clamp(1e-5, 1 - 1e-5), fr[q].clamp(1e-5, 1 - 1e-5)
            kls.append(p * torch.log(p / r) + (1 - p) * torch.log((1 - p) / (1 - r)))
        ps.append(compose(ep["kind"], f).clamp(1e-5, 1 - 1e-5))
        t_list.append(float(rec.ts[i]))
    if ep["kind"] == "invalidate":
        p = ps[0]
        er = p * invalidate_reward(ep, True, c) + (1 - p) * invalidate_reward(ep, False, c)
    else:
        surv = torch.ones((), device=ps[0].device) if ps else torch.ones(())
        er = 0.0
        for p, t in zip(ps, t_list):
            r = wake_reward(ep, t, c) if ep["kind"] == "wake" else interrupt_reward(ep, t, c)
            er = er + surv * p * r
            surv = surv * (1 - p)
        r_none = wake_reward(ep, ep["horizon_t"], c) if ep["kind"] == "wake" else interrupt_reward(ep, None, c)
        er = er + surv * r_none
    kl = torch.stack(kls).mean() if kls else torch.zeros(())
    loss = -er / c["scale"] + beta * kl
    return loss, {"er": float(er.detach() if torch.is_tensor(er) else er), "kl": float(kl.detach()), "n_frames": len(frames)}


def sample_episode(rng, recs, cfg, mix):
    while True:
        rec = Recording(rng.choice(recs), {})
        kind = rng.choices([k for k, _ in mix], [w for _, w in mix])[0]
        if kind == "wake":
            acts = [x for x in rec.actions if x["idx_before"] + 10 < len(rec.frames)]
            ep = wake_episode(rec, rng.choice(acts), cfg) if acts else None
        elif kind == "interrupt":
            ep = interrupt_episode(rec, rng, cfg)
        else:
            ep = invalidate_episode(rec, rng, cfg)
        if ep is not None and (ep["kind"] == "invalidate" or ep["frames"]):
            return rec, ep


@torch.no_grad()
def evaluate_decisions(model, recs, cfg, c, mix, a, out: Path) -> dict:
    """Mean E[R] of the stochastic policy and metrics of the deployed threshold policy (p >= tau fires)."""
    model.eval()
    rng = random.Random(a.seed + 777)
    acc: dict = {k: [] for k, _ in mix}
    for n in range(a.episodes):
        rec, ep = sample_episode(rng, recs, cfg, mix)
        task = rng.choice(TASKS).format(x=rng.choice(rec.spec["nav"]))
        _, info = episode_loss(model, ep, c, task, 0.0, a.max_frames)
        frames = [ep["i_n"]] if ep["kind"] == "invalidate" else ep["frames"][: a.max_frames]
        fired_t = None
        for i in frames:
            req, qids = frame_request(ep, i, task)
            f = fires(model, [_pil(x) for x in req.images], req.state, qids, [(q.id, t) for q, t in req.questions])
            if float(compose(ep["kind"], f)) >= a.tau:
                fired_t = float(rec.ts[i])
                break
        row = {"er": info["er"]}
        if ep["kind"] == "wake":
            t_w = fired_t if fired_t is not None else ep["horizon_t"]
            row.update(R=wake_reward(ep, t_w, c), delay=t_w - ep["t_a"], early=t_w < ep["t_ready"],
                       late=max(0.0, t_w - ep["t_ready"]),
                       missed=any(t_w > off for on, off in ep["windows"] if on < ep["t_ready"] + 1.0))
        elif ep["kind"] == "interrupt":
            row.update(R=interrupt_reward(ep, fired_t, c), event=ep["t_e"] is not None, fired=fired_t is not None,
                       reaction=(fired_t - ep["t_e"]) if (fired_t is not None and ep["t_e"] is not None and fired_t >= ep["t_e"]) else None)
        else:
            row.update(R=invalidate_reward(ep, fired_t is not None, c), stale=ep["stale"], fired=fired_t is not None)
        acc[ep["kind"]].append(row)
    rep = {}
    for k, rows in acc.items():
        if not rows:
            continue
        r = {"n": len(rows), "E[R]": float(np.mean([x["er"] for x in rows])), "R_tau": float(np.mean([x["R"] for x in rows]))}
        if k == "wake":
            r.update(delay_mean=float(np.mean([x["delay"] for x in rows])), early_rate=float(np.mean([x["early"] for x in rows])),
                     late_mean=float(np.mean([x["late"] for x in rows])), miss_rate=float(np.mean([x["missed"] for x in rows])))
        else:
            pos = [x for x in rows if x.get("event", x.get("stale"))]
            neg = [x for x in rows if not x.get("event", x.get("stale"))]
            tp = sum(x["fired"] for x in pos)
            fp = sum(x["fired"] for x in neg)
            r.update(recall=tp / max(1, len(pos)), precision=tp / max(1, tp + fp), false_alarm_rate=fp / max(1, len(neg)))
            if k == "interrupt":
                re_ = [x["reaction"] for x in rows if x["reaction"] is not None]
                r["reaction_mean"] = float(np.mean(re_)) if re_ else None
        rep[k] = r
    (out / f"decisions_{a.split}.json").write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    return rep


class _reference:
    """Disable the RL adapter so the model is the SFT reference."""

    def __init__(self, model):
        self.m = model.backbone if model.family == "d3" else model.model

    def __enter__(self):
        self.ctx = self.m.disable_adapter() if hasattr(self.m, "disable_adapter") else None
        return self.ctx.__enter__() if self.ctx else None

    def __exit__(self, *a):
        return self.ctx.__exit__(*a) if self.ctx else False


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--family", choices=["d3", "intern"], required=True)
    ap.add_argument("--model", required=True, help="the SFT package (RL adds a new LoRA on top)")
    ap.add_argument("--rec", default="data/synthweb/rec")
    ap.add_argument("--out", required=True)
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    ap.add_argument("--episodes", type=int, default=4000)
    ap.add_argument("--mix", default="wake:0.5,interrupt:0.25,invalidate:0.25")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--accum", type=int, default=8)
    ap.add_argument("--beta", type=float, default=0.05)
    ap.add_argument("--max-side", type=int, default=1280)
    ap.add_argument("--max-frames", type=int, default=24, help="decision frames per episode (cost bound)")
    ap.add_argument("--horizon", type=float, default=8.0)
    ap.add_argument("--probe-s", type=float, default=1.0)
    ap.add_argument("--lat-median", type=float, default=3.5, help="VLM latency median (s); measure MiMo with `nightwatch ping`")
    ap.add_argument("--lat-max", type=float, default=12.0)
    ap.add_argument("--c-early", type=float, default=4.0, help="cost of an observation taken too early (one invalid step)")
    ap.add_argument("--c-miss", type=float, default=6.0)
    ap.add_argument("--c-stale", type=float, default=4.0)
    ap.add_argument("--c-regen", type=float, default=1.0)
    ap.add_argument("--c-invalid", type=float, default=7.0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", default="train")
    ap.add_argument("--eval-only", action="store_true", help="no training: E[R] and threshold-policy metrics on --split")
    ap.add_argument("--tau", type=float, default=0.5, help="threshold of the deterministic (deployed) policy in --eval-only")
    a = ap.parse_args(argv)
    from training.models import load_trainable

    rng = random.Random(a.seed)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    model = load_trainable(a.family, a.model, device=a.device, max_side=a.max_side)
    recs = []
    for d in sorted(Path(a.rec).iterdir()):
        if not (d / "done").exists():
            continue
        seed = int(d.name)
        split = "train" if seed % 10 < 8 else "val" if seed % 10 == 8 else "test"
        if split == a.split:
            recs.append(d)
    if not recs:
        raise SystemExit(f"no finished recordings of split {a.split!r} under {a.rec}")
    cfg = {"horizon": a.horizon, "probe_s": a.probe_s, "lat_median": a.lat_median, "lat_max": a.lat_max}
    c = {"c_early": a.c_early, "c_miss": a.c_miss, "c_stale": a.c_stale, "c_regen": a.c_regen, "c_invalid": a.c_invalid,
         "scale": 5.0}
    mix = [(k, float(v)) for k, v in (x.split(":") for x in a.mix.split(","))]
    if a.eval_only:
        return evaluate_decisions(model, recs, cfg, c, mix, a, out)
    model.train()  # gradient checkpointing (and LoRA dropout) only act in train mode; d3's runtime leaves eval on
    params = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=a.lr)
    log = open(out / "rl_log.jsonl", "a")
    t0, n, run = time.time(), 0, {"er": 0.0, "kl": 0.0}
    autocast = torch.autocast("cuda", dtype=torch.bfloat16) if "cuda" in a.device else torch.autocast("cpu", enabled=False)
    while n < a.episodes:
        rec, ep = sample_episode(rng, recs, cfg, mix)
        task = rng.choice(TASKS).format(x=rng.choice(rec.spec["nav"]))
        with autocast:
            loss, info = episode_loss(model, ep, c, task, a.beta, a.max_frames)
        (loss / a.accum).backward()
        n += 1
        run["er"] += info["er"]
        run["kl"] += info["kl"]
        if n % a.accum == 0:
            torch.nn.utils.clip_grad_norm_(params, 1.0)
            opt.step()
            opt.zero_grad(set_to_none=True)
        if n % 20 == 0:
            rec_ = {"episodes": n, "E[R]": run["er"] / 20, "kl": run["kl"] / 20, "s": round(time.time() - t0)}
            log.write(json.dumps(rec_) + "\n")
            log.flush()
            print("[rl]", rec_, flush=True)
            run = {"er": 0.0, "kl": 0.0}
    pkg = model.save(out / "package")
    print(f"[rl] done -> {pkg}")


if __name__ == "__main__":
    main()
