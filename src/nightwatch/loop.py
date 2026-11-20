"""The event-driven episode loop (all methods, baselines included, run through this one loop).

Per step:
  observe ─▶ agent.predict in a worker thread ──(checkpoint 1: during reasoning)── monitor frames:
             stage-1 change vs the screenshot the agent saw ─▶ System-1 B-group ─▶ rules: interrupt?
          ─▶ for each action (checkpoint 2: before execution): stage-1 near the target ─▶ System-1 C-group
             ─▶ rules: invalidate? (undo the generation in the agent's history, re-observe, re-predict)
          ─▶ execute (no sleep) ─▶ (checkpoint 3: after execution) the method's waiter decides when to look:
             fixed sleep | pixel-stable | a11y-stable | clip-stable | nightwatch (A/B/D groups + rules)
          ─▶ observe
Every step is logged with its timestamps plus `segments` (obs / vlm / vlm_wasted /
check / exec / wait), from which the wall-clock decomposition is computed.
"""

from __future__ import annotations

import io
import logging
import threading
import time
import traceback
from collections import Counter
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
from PIL import Image

from nightwatch.actions import Act, parse_action, strip_agent_sleeps
from nightwatch.agents.base import AgentAdapter, Prediction
from nightwatch.diff import ChangeDetector
from nightwatch.envs.base import Env
from nightwatch.frames import Frame, FrameStream, frame_from_bytes
from nightwatch.questions import bank_subset, build_request
from nightwatch.rules import Decision, Rules
from nightwatch.system1.base import Meter
from nightwatch.utils import Clock, box_around, crop, draw_marker, encode_jpeg, jdump, jline
from nightwatch.system1.base import _tracked

log = logging.getLogger("nightwatch.loop")



PUSV_KEYWORDS = ("security", "alert", "warning", "zenity", "systemoverlay")


def pusv_verdict(g0: np.ndarray, g1: np.ndarray, x: float, y: float, w_obs: list | None, w_now: list | None,
                 tau1: float = 0.92, noise: int = 20, tau2a: float = 0.002) -> tuple[str, float]:
    """PUSV's three layers on grayscale screenshots g0 (T_obs) and g1 (just before the click), in the PUSV paper's order:
    L1  SSIM of the 160x160 patch centred at the click < tau1;
    L2a share of pixels outside that patch whose grayscale differs by > noise exceeds tau2a;
    L2b a window listed now and not at T_obs (`wmctrl -l` lines) has a title containing a PUSV keyword
        (case-insensitive substring).
    Returns (layer, value) for the first layer that fires, ("", 0.0) when the click may proceed."""
    from skimage.metrics import structural_similarity

    if g0.shape != g1.shape:
        return "", 0.0
    H, W = g0.shape
    xi, yi = min(max(int(x), 0), W - 1), min(max(int(y), 0), H - 1)
    x0, y0 = min(max(xi - 80, 0), W - 160), min(max(yi - 80, 0), H - 160)
    ssim = float(structural_similarity(g0[y0:y0 + 160, x0:x0 + 160], g1[y0:y0 + 160, x0:x0 + 160], data_range=255))
    if ssim < tau1:
        return "pusv_l1_ssim", ssim
    changed = np.abs(g0.astype(np.int16) - g1.astype(np.int16)) > noise
    changed[y0:y0 + 160, x0:x0 + 160] = False
    ratio = float(changed.sum()) / float(H * W - 160 * 160)
    if ratio > tau2a:
        return "pusv_l2a_global", ratio
    if w_obs is not None and w_now is not None:
        new = [w for w in w_now if w not in w_obs]
        if any(k in w.lower() for w in new for k in PUSV_KEYWORDS):
            return "pusv_l2b_window", 1.0
    return "", 0.0

@dataclass
class StepRecord:
    step: int
    t_obs: float
    t_obs_final: float = 0.0
    obs_count: int = 1
    t_vlm_start: float = 0.0
    t_vlm_end: float = 0.0
    vlm_calls: int = 0
    interrupted: int = 0
    t_interrupt: float | None = None
    interrupt_reason: str = ""
    interrupts: list = field(default_factory=list)
    t_dispatch: float | None = None
    invalidated: int = 0
    invalidate_reason: str = ""
    invalidations: list = field(default_factory=list)
    vetoes: int = 0
    t_action_done: float | None = None
    t_wake: float | None = None
    wake_reason: str = ""
    wake_diag: str = ""
    wait_s: float = 0.0
    jev_calls: int = 0
    jev_ms_total: float = 0.0
    jev_ms_critical: float = 0.0  # System-1 time spent on the critical path (pre-exec checks, post-exec waiting)
    diff_frames: int = 0
    frames_total: int = 0
    step_type: str = "normal"
    action: str = ""
    act_kind: str = ""
    n_actions: int = 0
    stale_px: float | None = None
    effect_px: float | None = None
    input_tokens: float = 0.0
    output_tokens: float = 0.0
    wasted_tokens: float = 0.0
    stripped_sleep_s: float = 0.0  # agent-written fixed waits removed from the action code (strip_sleep)
    keyframes: int = 0  # AOI keyframes handed over with the observation that closes this step (aoi_visual)
    obs_file: str = ""  # obs/<file> the executed prediction was based on (frames.save_obs)
    llm_calls: list = field(default_factory=list)  # every agent LLM call of the step: send / return timestamps
    tokens_estimated: bool = False
    t_first_token: float | None = None
    response: str = ""
    segments: list = field(default_factory=list)  # [kind, t0, t1]
    error: str = ""


class Episode:
    def __init__(self, env: Env, agent: AgentAdapter, s1: Meter | None, method, cfg, task: dict, out_dir: Path):
        self.env, self.agent, self.s1, self.m, self.cfg, self.task = env, agent, s1, method, cfg, task
        self.out = Path(out_dir)
        self.out.mkdir(parents=True, exist_ok=True)
        self.clock = Clock()
        st1 = dict(cfg.get("stage1", {}))
        self.det = ChangeDetector(env.screen_size, **{k: v for k, v in st1.items() if k in (
            "small_width", "pix_thresh", "min_frac", "min_area", "benign_regions")})
        W, H = env.screen_size
        self.benign_max_area = float(st1.get("benign_learn_max_frac", 0.02)) * W * H
        fr = cfg.get("frames", {})
        rec_dir = self.out if (fr.get("record") or method.get("record")) else None
        self.save_obs = bool(fr.get("save_obs", False))  # every screenshot the agent receives -> obs/NNN_t<s>.jpg
        self._obs_n, self._obs_file = 0, ""
        self.stream = FrameStream(env, self.clock, self.det, hz=float(method.get("frames_hz") or fr.get("hz", 10)),
                                  small_width=int(st1.get("small_width", 480)), record_dir=rec_dir,
                                  record_quality=int(fr.get("record_quality", 85)),
                                  record_max_side=fr.get("record_max_side"))
        self.rules = Rules(method.get("rules", {}))
        self.bank = bank_subset(method.get("bank_size"), method.get("bank_ids"))
        self.pcfg = method.get("post_cfg", {})
        self.post = method.get("post", "fixed")
        self.use_stage1 = bool(method.get("stage1", True))
        self.use_crop = bool(method.get("crop", True))
        self.lim = cfg.get("limits", {})
        self.instruction = env.instruction(task)
        self.space = getattr(env, "action_space", "pyautogui")
        if hasattr(env, "native_wait"):
            env.native_wait = self.post == "fixed"  # fixed rhythm keeps the benchmark's own wait(); others use the waiter
        self.vlm_calls = 0
        self.last_interrupt_t = -1e9
        self.benign_boxes: list = []
        self.prev_act: Act | None = None
        self.sig_log: list = []
        self._s1_fh = open(self.out / "system1.jsonl", "a", encoding="utf-8") if s1 else None
        # one line per System-1-driven decision (interrupt / invalidate / wake on an event); none of them is a step
        self._ev_fh = open(self.out / "events.jsonl", "a", encoding="utf-8")
        self._ev_n = 0
        self.event_counts: Counter = Counter()
        self._intr_frame = self._pre_frame = self._wake_frame = None
        self._clip = None
        # AOI's inter-step keyframes (the aoi_visual baseline): AOI's own extractor, fed by the frame stream
        self.kf = self._aoi_keyframes() if method.get("keyframes") == "aoi" else None
        self._kf_stop = threading.Event()

    # ================================================================== helpers
    def now(self) -> float:
        return self.clock.now()

    def _log_llm_call(self, rec: StepRecord, pred: Prediction, t0: float, t1: float, status: str) -> None:
        """One agent prediction: loop-clock start/end (t, seconds since the episode loop started), the unix time of
        every HTTP attempt inside it (send, first token, return), and what became of it."""
        e0 = self.clock.epoch0
        rel = lambda u: None if u is None else round(u - e0, 4)  # noqa: E731
        http = [{"send_unix": round(h["send"], 4), "first_token_unix": h["first_token"] and round(h["first_token"], 4),
                 "return_unix": h["return"] and round(h["return"], 4), "send_t": rel(h["send"]),
                 "first_token_t": rel(h["first_token"]), "return_t": rel(h["return"]), "outcome": h["outcome"],
                 **({"provider": h["provider"]} if h.get("provider") else {}),
                 **{k: h[k] for k in ('reasoning_effort', 'max_tokens', 'reasoning_chars', 'content_chars', 'finish_reason') if k in h}}
                for h in (pred.extra or {}).get("http", [])]
        rec.llm_calls.append({"call": self.vlm_calls, "status": status, "t_start": round(t0, 4), "t_end": round(t1, 4),
                              "start_unix": round(e0 + t0, 4), "end_unix": round(e0 + t1, 4), "seconds": round(t1 - t0, 4),
                              "ttft_s": pred.t_first_token and round(pred.t_first_token, 4),
                              "input_tokens": pred.input_tokens, "output_tokens": pred.output_tokens,
                              "retries": pred.retries, "http": http,
                              **({"images_capped": pred.extra["images_capped"]} if (pred.extra or {}).get("images_capped") else {})})

    def _aoi_keyframes(self):
        """AOI's KeyframeExtractor from the pinned checkout, with AOI's defaults (pixel gate 1 %, CLIP ViT-B/16
        theta 0.04, capture on >= 3 % pixel change, at most 5 keyframes per step)."""
        import sys

        from nightwatch.config import ROOT
        from nightwatch.config import env as getenv

        aoi_dir = str(ROOT / "third_party" / "aoi")
        if aoi_dir not in sys.path:
            sys.path.insert(0, aoi_dir)
        from aoi.keyframe_extractor import KeyframeExtractor

        dev = getenv("AOI_CLIP_DEVICE")
        if not dev:
            import torch

            dev = "cuda" if torch.cuda.is_available() else "cpu"
        return KeyframeExtractor(device=dev, max_keyframes=int(self.m.get("max_keyframes", 5)))

    def _feed_keyframes(self) -> None:
        """AOI's capture thread: every new frame of the stream goes through the extractor."""
        from PIL import Image

        last = -1
        while not self._kf_stop.is_set():
            f = self.stream.wait_new(last, timeout=0.2)
            if f is None:
                continue
            last = f.idx
            try:
                self.kf.on_sample(Image.fromarray(f.rgb), timestamp=f.t)
            except Exception as e:
                log.warning("keyframe extraction failed: %s", e)

    def _seg(self, rec: StepRecord, kind: str, t0: float, t1: float | None = None) -> None:
        rec.segments.append([kind, round(t0, 4), round(t1 if t1 is not None else self.now(), 4)])

    def _observe(self, rec: StepRecord | None = None) -> None:
        t0 = self.now()
        o = self.env.obs()
        t1 = self.now()
        if not o.get("screenshot"):  # the official loop would crash here; retry once
            time.sleep(1.0)
            o = self.env.obs()
            t1 = self.now()
        self.obs = o
        if self.save_obs:
            self._obs_n += 1
            self._obs_file = f"{self._obs_n:03d}_t{t1:.1f}.jpg"
            (self.out / "obs").mkdir(exist_ok=True)
            (self.out / "obs" / self._obs_file).write_bytes(
                encode_jpeg(np.asarray(Image.open(io.BytesIO(o["screenshot"])).convert("RGB")), 85))
        if self.m.get("validate") == "pusv" and hasattr(self.env, "windows"):
            self._w_obs = self.env.windows()  # PUSV's W_obs: the window list at T_obs
        if self.kf is not None:  # an AOI interval ends at every observation
            kfs = self.kf.get_and_reset()
            o["aoi_keyframes"] = [(round(t1 - k.timestamp, 1), encode_jpeg(np.asarray(k.image.convert("RGB")), 85))
                                  for k in kfs]
            if rec is not None:
                rec.keyframes += len(kfs)
        self.obs_frame = frame_from_bytes(o["screenshot"], t1, gt=o.get("gt"),
                                          small_width=int(self.cfg.get("stage1", {}).get("small_width", 480)))
        self.t_obs = t1
        if rec is not None:
            self._seg(rec, "obs", t0, t1)

    def _crop_or_full(self, img: np.ndarray, box) -> np.ndarray:
        if not self.use_crop or box is None:
            return img
        return crop(img, box, pad=int(self.pcfg.get("crop_pad", 24)), min_side=int(self.pcfg.get("crop_min", 160)))

    def _ask(self, checkpoint: str, images: dict, gts: dict, ctx: dict, action: str = "", only=None,
             critical: bool = False, rec: StepRecord | None = None, frame_idx: int | None = None):
        if self.s1 is None:
            return None
        req = build_request(checkpoint, images, self.bank, self.instruction, action, gts, ctx, only)
        if req is None:
            return None
        ans = self.s1.ask(req)
        if ans is None:
            return None
        if rec is not None:
            rec.jev_calls += 1
            rec.jev_ms_total += ans.ms
            if critical:
                rec.jev_ms_critical += ans.ms
        if self._s1_fh:
            jline(self._s1_fh, {"t": round(self.now(), 4), "step": rec.step if rec else None, "checkpoint": checkpoint,
                                "frame": frame_idx, "obs_t": round(self.t_obs, 4), "ctx": {k: v for k, v in ctx.items()},
                                **ans.to_dict(),
                                # questions at or over their stage-3 threshold (what the rules act on)
                                "over_tau": {q: round(p, 4) for q, p in ans.fire.items() if p >= self.rules.tau(q)}})
        return ans

    def _event(self, kind: str, rec: StepRecord, seen: Frame | None, now: Frame | None, **info) -> None:
        """Log one decision with the two screens it was judged on: what the agent last saw and the frame that
        triggered it (events/NNN_<kind>_{seen,now}.jpg, for case studies)."""
        self._ev_n += 1
        self.event_counts[kind] += 1
        row = {"n": self._ev_n, "kind": kind, "step": rec.step, "t": round(self.now(), 4), **info}
        d = self.out / "events"
        for tag, fr in (("seen", seen), ("now", now)):
            if fr is not None:
                d.mkdir(exist_ok=True)
                name = f"{self._ev_n:03d}_{kind}_{tag}.jpg"
                (d / name).write_bytes(encode_jpeg(fr.rgb, 85, 960))
                row[tag] = {"file": name, "frame": fr.idx, "t": round(fr.t, 3)}
        jline(self._ev_fh, row)

    def _compare(self, a: Frame, b: Frame):
        return self.det.compare(a.small, b.small, ignore=self.benign_boxes)

    # ================================================================== checkpoint 1
    def _monitored_predict(self, rec: StepRecord) -> tuple[Prediction, Decision | None]:
        cancel = threading.Event()
        box: dict = {}
        obs = dict(self.obs)

        def work():
            try:
                box["pred"] = self.agent.predict(self.instruction, obs, cancel)
            except Exception as e:  # never let the agent thread die silently
                box["pred"] = Prediction(error=f"{type(e).__name__}: {e}")
                log.warning("agent.predict failed: %s", traceback.format_exc(limit=3))

        th = threading.Thread(target=work, name="agent-predict", daemon=True)
        th.start()
        interrupt: Decision | None = None
        last = self.stream.latest()
        last_idx = last.idx if last else -1
        last_checked = self.obs_frame
        pixel_rules = self.m.get("pixel_rules", False)
        allow = (self.m.get("interrupt", False) and (self.s1 is not None or pixel_rules)
                 and rec.interrupted < int(self.lim.get("max_interrupts_per_step", 2)))
        min_gap = float(self.m.get("rules", {}).get("min_interval_s", 1.0))
        # System-1 checks during reasoning run on their own thread: they must never delay noticing that the
        # agent has finished (a slow check is abandoned the moment the prediction is in).
        pending: dict = {}

        def check(f, ch):
            try:
                pending["frame"] = f
                pending["ans"] = self._ask("reasoning", {"seen": self.obs_frame.rgb, "now": f.rgb,
                                                         "crop": self._crop_or_full(f.rgb, ch.union)},
                                           {"seen": self.obs_frame.gt, "now": f.gt, "crop": f.gt}, {"crop_box": ch.union},
                                           rec=rec, frame_idx=f.idx)
            except Exception as e:
                log.debug("reasoning check failed: %s", e)
                pending["ans"] = None

        checker: threading.Thread | None = None
        checker_ch = None
        while th.is_alive():
            if checker is not None and not checker.is_alive():  # a finished check: apply the rules
                ans, checker = pending.pop("ans", None), None
                if ans is not None:
                    dec = self.rules.during_reasoning(ans)
                    self._update_benign(ans, checker_ch.boxes)
                    if dec.action == "interrupt" and interrupt is None:
                        self._intr_frame = pending.get("frame")
                        interrupt = dec
                        interrupt.p = round(dec.p, 4)
                        self.last_interrupt_t = self.now()
                        if self.agent.supports_cancel:
                            cancel.set()
                            th.join(timeout=float(self.lim.get("cancel_join_s", 10)))
                            break
            f = self.stream.wait_new(last_idx, timeout=0.05)
            if f is None:
                continue
            last_idx = f.idx
            if f.t - f.grab_s <= self.obs_frame.t:  # captured before the agent's observation (lagging stream): not news
                continue
            if not allow or interrupt is not None or checker is not None:
                continue  # one check in flight at a time; newer frames are judged by the next one
            if self.now() - self.last_interrupt_t < min_gap:
                continue
            ch = self._compare(self.obs_frame, f)
            if pixel_rules:
                # The frozen stage-1 gate is the complete decision rule in this ablation.
                if ch.significant:
                    self._intr_frame = f
                    interrupt = Decision("interrupt", "pixel_screen_changed", 1.0)
                    self.last_interrupt_t = self.now()
                    if self.agent.supports_cancel:
                        cancel.set()
                    break
                continue
            if self.use_stage1:
                if not ch.significant:
                    continue
                if last_checked is not self.obs_frame and not self._compare(last_checked, f).significant:
                    continue  # this screen was already judged
            last_checked, checker_ch = f, ch
            checker = threading.Thread(target=check, args=(f, ch), name="reasoning-check", daemon=True)
            checker.start()
        th.join()
        return box.get("pred") or Prediction(error="no prediction"), interrupt

    # ================================================================== checkpoint 2
    def _pusv(self, act: Act, rec: StepRecord) -> Decision:
        """PUSV (Xu, arXiv 2604.18860, sec. 5), the published pre-execution check, as a baseline: before a click, a
        fresh screenshot through the OSWorld API and the window list are checked against the observation the agent
        reasoned on (`pusv_verdict`). Clicks only, as in PUSV; an aborted click is not executed and the agent
        decides again on the new screen (PUSV only says the action is aborted)."""
        if act.kind not in ("click", "double_click", "right_click") or act.x is None:
            return Decision("none")
        from PIL import Image

        t0 = self.now()
        try:
            # OSWorld: a fresh screenshot through its API, as in PUSV; other environments (DynaCU's browser):
            # their own grab. Without OS windows (a single browser page), layer L2b has nothing to check.
            ctl = getattr(self.env, "controller", None)
            shot = ctl.get_screenshot() if ctl is not None else self.env.grab()[0]
            if not shot:
                return Decision("none")
            now = frame_from_bytes(shot, self.now(), small_width=int(self.cfg.get("stage1", {}).get("small_width", 480)))
            self._pre_frame = now
            g0 = np.asarray(Image.fromarray(self.obs_frame.rgb).convert("L"))
            g1 = np.asarray(Image.fromarray(now.rgb).convert("L"))
            w_now = self.env.windows() if hasattr(self.env, "windows") else None
            reason, value = pusv_verdict(g0, g1, act.x, act.y, getattr(self, "_w_obs", None), w_now)
            return Decision("invalidate", reason, value) if reason else Decision("none")
        finally:
            self._seg(rec, "check", t0)

    def _pixel_pre_exec(self, act: Act, rec: StepRecord) -> Decision:
        """Void targeted actions if >5% of their 160px patch changed by >16 gray levels.

        Uses the existing target-region comparison and stale threshold, with a fresh grab
        as in PUSV. Untargeted actions have no target patch and are left unchanged.
        No System-1 calls, global fallback, or learned benign-region suppression.
        """
        if not act.has_target:
            return Decision("none")
        t0 = self.now()
        try:
            shot = self.env.grab()[0]
            f = frame_from_bytes(shot, self.now(), small_width=int(self.cfg.get("stage1", {}).get("small_width", 480)))
            self._pre_frame = f
            w, h = self.env.screen_size
            x, y = min(max(act.x, 0), w - 1), min(max(act.y, 0), h - 1)
            box = box_around(x, y, int(self.pcfg.get("target_half", 80)))
            change = self._region_change(self.obs_frame, f, box)
            threshold = float(self.cfg.get("step_types", {}).get("stale_px", 0.05))
            return Decision("invalidate", "pixel_target_changed", change) if change >= threshold else Decision("none")
        finally:
            self._seg(rec, "check", t0)

    def _fresh_frame(self) -> Frame | None:
        """A frame captured after the agent's observation. Judged by capture time (`t - grab_s`: the AGS daemon reports
        the capture age), not by when the grab returned. A lagging stream can hold only frames from before the
        observation (a pre-execution check compared a frame 1.74 s older than the observation and voided the
        action on the difference); then wait briefly for a new frame, else take a screenshot through the OSWorld API
        (as PUSV does), not through the same lagging stream."""
        def fresh(f):
            return f is not None and f.t - f.grab_s > self.obs_frame.t

        f = self.stream.latest()
        if fresh(f):
            return f
        f = self.stream.wait_new(f.idx if f is not None else -1, timeout=0.3)
        if fresh(f):
            return f
        self.stale_frame_refreshes = getattr(self, "stale_frame_refreshes", 0) + 1
        try:
            ctl = getattr(self.env, "controller", None)
            shot = ctl.get_screenshot() if ctl is not None else self.env.grab()[0]
            if not shot:
                return None
            return frame_from_bytes(shot, self.now(), small_width=int(self.cfg.get("stage1", {}).get("small_width", 480)))
        except Exception as e:  # no fresh view: no check rather than a check against the past
            log.warning("fresh screenshot for the pre-execution check failed: %s", e)
            return None

    def _pre_exec(self, act: Act, rec: StepRecord) -> Decision:
        f = self._fresh_frame()
        self._pre_frame = f
        if f is None:
            return Decision("none")  # (a WAIT decided on a stale screen is invalidated too: the agent re-looks)
        ch = self._compare(self.obs_frame, f)
        if self.use_stage1 and not ch.significant:
            return Decision("none")
        t0 = self.now()
        W, H = self.env.screen_size
        if act.has_target:
            x, y = min(max(act.x, 0), W - 1), min(max(act.y, 0), H - 1)
            half = int(self.pcfg.get("target_half", 80))
            tb = box_around(x, y, half)
            # the target area looks as the agent saw it: it cannot have vanished, changed or been covered since,
            # so only a new pop-up elsewhere (a modal dialog swallows the click) can make the action stale
            region = self._region_change(self.obs_frame, f, tb)
            only = ["dialog"] if region < float(self.cfg.get("step_types", {}).get("stale_px", 0.05)) else None
            ans = self._ask("pre_exec", {"seen": self.obs_frame.rgb, "now_marked": draw_marker(f.rgb, x, y),
                                         "target_seen": crop(self.obs_frame.rgb, tb, pad=0, min_side=2 * half),
                                         "target_now": crop(f.rgb, tb, pad=0, min_side=2 * half)},
                            {"seen": self.obs_frame.gt, "now_marked": f.gt, "target_seen": self.obs_frame.gt, "target_now": f.gt},
                            {"point": (x, y), "target_region_change": round(region, 4)}, act.describe(), only=only,
                            critical=True, rec=rec, frame_idx=f.idx)
            dec = self.rules.before_execution(ans) if ans else Decision("none")
        else:  # keyboard / DONE / FAIL: did something happen that makes the plan stale?
            ans = self._ask("reasoning", {"seen": self.obs_frame.rgb, "now": f.rgb, "crop": self._crop_or_full(f.rgb, ch.union)},
                            {"seen": self.obs_frame.gt, "now": f.gt, "crop": f.gt}, {"crop_box": ch.union},
                            only=list(self.m.get("rules", {}).get("invalidate_untargeted", ["dialog", "error", "toast", "page_changed"])),
                            critical=True, rec=rec, frame_idx=f.idx)
            dec = Decision("none")
            if ans:
                q, p = self.rules._first(ans, list(self.m.get("rules", {}).get(
                    "invalidate_untargeted", ["dialog", "error", "toast", "page_changed"])), self.rules.invalidate_tau)
                if q:
                    dec = Decision("invalidate", q, p)
        self._seg(rec, "check", t0)
        return dec

    # ================================================================== checkpoint 3: waiters
    def _wait(self, act: Act, pre: Frame, t_done: float, rec: StepRecord) -> tuple[str, str]:
        post = self.post
        p = self.pcfg
        if act.kind in ("done", "fail"):
            return "episode_end", ""
        if post == "fixed":
            pause = float(self.m.get("sleep", 2.0))
            time.sleep(pause * (2 if act.kind == "wait" else 1))  # DesktopEnv.step sleeps twice on WAIT
            return "fixed", ""
        max_wait = float(p.get("max_wait_s", 10.0))
        no_eff = float(p.get("wait_action_s", 3.0) if act.kind == "wait" else p.get("no_effect_timeout_s", 1.0))
        quiet_s = float(p.get("quiet_s", 0.4))
        if post == "a11y":
            return self._wait_a11y(t_done, max_wait)
        if post == "clip":
            return self._wait_clip(pre, t_done, max_wait, quiet_s, no_eff)
        if post == "nightwatch" and p.get("optimistic", True) and self.use_stage1:
            return self._wait_optimistic(act, pre, t_done, rec, no_eff, quiet_s, max_wait)
        last_idx, any_change, quiet_since = pre.idx, False, None
        # "no visible change" needs a frame whose capture STARTED after the action: on a slow stream the
        # timeout alone would mean "no new frame yet", not "the screen did not change"
        seen_after = False
        view, view_t, last_s1_t, last_s1_frame, pending = None, -1.0, -1e9, None, None
        probe = float(p.get("probe_every_s", 1.0))
        stuck_s = float(p.get("stuck_s", 8.0))
        while True:
            f = self.stream.wait_new(last_idx, timeout=0.1)
            el = self.now() - t_done
            if f is not None:
                last_idx = f.idx
                seen_after = seen_after or f.t - f.grab_s >= t_done
                prev = self.stream.at_or_before(f.t - 1e-6)
                ch_mov = self._compare(prev, f) if (prev is not None and prev is not f) else None
                moving = bool(ch_mov and ch_mov.significant)
                ch_pre = self._compare(pre, f)
                any_change = any_change or ch_pre.significant
                quiet_since = None if moving else (quiet_since if quiet_since is not None else f.t)
                quiet = (f.t - quiet_since) if quiet_since is not None else 0.0
                if post == "pixel":
                    if any_change and quiet >= quiet_s:
                        return "pixel_stable", ""
                elif post == "nightwatch":
                    fresh = last_s1_frame is None or self._compare(last_s1_frame, f).significant
                    # the forced probe is for a screen that changed and then stopped on a static "Loading..."; with
                    # no change at all the step ends at the no-effect timeout whatever System-1 says
                    need = any_change and ((fresh and (moving or quiet >= quiet_s)) or (self.now() - last_s1_t >= probe))
                    if any_change and quiet >= quiet_s and view is not None and view_t >= quiet_since:
                        need = False  # the quiet screen was already judged
                    if not self.use_stage1:
                        need = True
                    if need and self.s1 is not None:
                        recent = self.stream.at_or_before(f.t - 0.5) or pre
                        cbox = ch_mov.union if moving else ch_pre.union  # what is changing right now
                        # "before" = the screenshot the agent last saw: an event matters if the AGENT has not seen it,
                        # even when it appeared while the agent was still thinking (pixel effect checks keep `pre`)
                        seen = self.obs_frame
                        ans = self._ask("post_exec", {"before": seen.rgb, "recent": recent.rgb, "now": f.rgb,
                                                      "crop": self._crop_or_full(f.rgb, cbox)},
                                        {"before": seen.gt, "recent": recent.gt, "now": f.gt, "crop": f.gt},
                                        {"crop_box": cbox}, act.describe(), critical=True, rec=rec,
                                        frame_idx=f.idx)
                        last_s1_t, last_s1_frame = self.now(), f
                        if ans is not None:
                            view, view_t = self.rules.after_execution(ans), f.t
                            if view.event and pending is None:
                                pending = (view.event, view.event_p, self.now(), f)
                            self._update_benign(ans, ch_mov.boxes if moving else [])
                    if pending is not None and (quiet >= float(p.get("event_quiet_s", 0.15))
                                                or self.now() - pending[2] >= float(p.get("event_max_delay_s", 1.0))):
                        self._wake_frame = pending[3]
                        return f"event:{pending[0]}", f"p={pending[1]:.2f}"  # wake as soon as the event has drawn
                    if any_change and quiet >= quiet_s:
                        if self.s1 is None or view is None:
                            if self.s1 is None or self.s1.errors:
                                return "pixel_stable_fallback", ""
                        elif view_t >= quiet_since and view.settled is not False and not view.loading and view.changing is not True:
                            if view.effect is not False:
                                return "effect_stable", ""
                            if el >= no_eff:
                                return "settled_no_effect", ""
                    if view is not None and el >= stuck_s and (view.loading or view.changing or view.settled is False):
                        if not view.long_op or el >= float(p.get("long_op_max_s", 30.0)):
                            return "stuck", f"no settled screen {el:.1f}s after the action"
            if not any_change and seen_after and el >= no_eff:
                return "no_effect_timeout", f"no visible change {el:.1f}s after the action"
            long_ok = view is not None and view.long_op and post == "nightwatch"
            if el >= (float(p.get("long_op_max_s", 30.0)) if long_ok else max_wait):
                return "max_wait", ""

    def _wait_optimistic(self, act: Act, pre: Frame, t_done: float, rec: StepRecord, no_eff: float, quiet_s: float,
                         max_wait: float) -> tuple[str, str]:
        """Checkpoint 3 with System-1 off the critical path. Wake as soon as the changed screen has been quiet for
        quiet_s; System-1 runs on a side thread on every new screen and only (a) wakes earlier on an event the agent
        has not seen (dialog / error / toast), or (b) holds the wake while the SAME screen is judged loading / still
        changing (re-checked every probe_every_s, up to stuck_s / long_op_max_s). A page that finishes loading after
        the wake is caught by checkpoint 1: the agent is interrupted while it reasons on the stale screen.
        With no change at all the step ends at the no-effect timeout, exactly as in the synchronous waiter.

        s1_wake (fast stability tier): on every new quiet frame (quiet >= min_quiet_s after a change, or no change
        for noeff_min_s after the action) System-1 is asked only `fast_questions` (~90 ms instead of ~480 ms for the
        full post-exec set); "settled" >= wake_tau with nothing loading (or changing, when still_changing is asked) wakes
        the agent before the pixel rules (quiet_s / no_effect_timeout_s) would. still_changing is not asked by default:
        it describes the half second before the frame, so it is "yes" on the first quiet frame and would block the
        earliest wakes that "settled" (trained on what happens next) allows. The full check above keeps running for events."""
        p = self.pcfg
        probe, stuck_s = float(p.get("probe_every_s", 1.0)), float(p.get("stuck_s", 8.0))
        long_max = float(p.get("long_op_max_s", 30.0))
        last_idx, any_change, quiet_since, seen_after = pre.idx, False, None, False
        box: dict = {}
        th: threading.Thread | None = None
        asked, last_ask_t, view, view_frame, pending = None, -1e9, None, None, None
        s1_wake = bool(p.get("s1_wake", False)) and self.s1 is not None
        wake_tau, min_quiet = float(p.get("wake_tau", 0.99)), float(p.get("min_quiet_s", 0.3))
        noeff_min = float(p.get("noeff_min_s", 0.3))
        full_gap = float(p.get("full_min_interval_s", 0.0))
        fast_qs = list(p.get("fast_questions", ["settled", "loading"]))
        fbox: dict = {}
        fth: threading.Thread | None = None
        fast_asked: Frame | None = None

        def fast(f: Frame, cbox) -> None:
            try:
                recent = self.stream.at_or_before(f.t - 0.5) or pre
                seen = self.obs_frame
                fbox["ans"] = self._ask("post_exec", {"before": seen.rgb, "recent": recent.rgb, "now": f.rgb,
                                                      "crop": self._crop_or_full(f.rgb, cbox)},
                                        {"before": seen.gt, "recent": recent.gt, "now": f.gt, "crop": f.gt},
                                        {"crop_box": cbox, "tier": "fast"}, act.describe(), only=fast_qs, rec=rec,
                                        frame_idx=f.idx)
            except Exception as e:
                log.debug("fast settle check failed: %s", e)
                fbox["ans"] = None
            fbox["frame"] = f

        def check(f: Frame, cbox, boxes) -> None:
            try:
                recent = self.stream.at_or_before(f.t - 0.5) or pre
                seen = self.obs_frame  # an event matters if the AGENT has not seen it
                box["ans"] = self._ask("post_exec", {"before": seen.rgb, "recent": recent.rgb, "now": f.rgb,
                                                     "crop": self._crop_or_full(f.rgb, cbox)},
                                       {"before": seen.gt, "recent": recent.gt, "now": f.gt, "crop": f.gt},
                                       {"crop_box": cbox}, act.describe(), rec=rec, frame_idx=f.idx)
            except Exception as e:
                log.debug("post-exec check failed: %s", e)
                box["ans"] = None
            box["frame"], box["boxes"] = f, boxes

        while True:
            f = self.stream.wait_new(last_idx, timeout=0.05)
            el = self.now() - t_done
            if th is not None and not th.is_alive():  # a finished check: apply the rules
                ans, vf, boxes = box.pop("ans", None), box.pop("frame", None), box.pop("boxes", [])
                th = None
                if ans is not None:
                    view, view_frame = self.rules.after_execution(ans), vf
                    self._update_benign(ans, boxes)
                    if view.event and pending is None:
                        pending = (view.event, view.event_p, self.now(), vf)
            if f is not None:
                last_idx = f.idx
                seen_after = seen_after or f.t - f.grab_s >= t_done
                prev = self.stream.at_or_before(f.t - 1e-6)
                ch_mov = self._compare(prev, f) if (prev is not None and prev is not f) else None
                moving = bool(ch_mov and ch_mov.significant)
                ch_pre = self._compare(pre, f)
                any_change = any_change or ch_pre.significant
                quiet_since = None if moving else (quiet_since if quiet_since is not None else f.t)
                # full checks back to back from every worker saturate the shared GPU and slow the fast tier that
                # decides the wake: keep full_min_interval_s between them, and none while a fast check is out
                if (any_change and th is None and self.s1 is not None and fth is None
                        and self.now() - last_ask_t >= full_gap):
                    fresh = asked is None or self._compare(asked, f).significant
                    if fresh or self.now() - last_ask_t >= probe:
                        asked, last_ask_t = f, self.now()
                        th = threading.Thread(target=check, args=(f, ch_mov.union if moving else ch_pre.union,
                                                                  ch_mov.boxes if moving else []),
                                              name="post-exec-check", daemon=True)
                        th.start()
            quiet = (self.now() - quiet_since) if quiet_since is not None else 0.0
            if pending is not None and (quiet >= float(p.get("event_quiet_s", 0.15))
                                        or self.now() - pending[2] >= float(p.get("event_max_delay_s", 1.0))):
                self._wake_frame = pending[3]
                return f"event:{pending[0]}", f"p={pending[1]:.2f}"
            if s1_wake:
                if fth is not None and not fth.is_alive():  # a finished fast check: wake if the quiet screen is final
                    fans, ff = fbox.pop("ans", None), fbox.pop("frame", None)
                    fth = None
                    latest = self.stream.latest()
                    still = ff is not None and latest is not None and not self._compare(ff, latest).significant
                    if fans is not None and still and quiet_since is not None and ff.t >= quiet_since:
                        ps = fans.p("settled")
                        if (ps is not None and ps >= wake_tau and (fans.p("loading") or 0.0) < self.rules.tau("loading")
                                and (fans.p("still_changing") or 0.0) < self.rules.tau("still_changing")):
                            return ("s1_settled" if any_change else "s1_no_effect"), f"p={ps:.2f}"
                ready = (any_change and quiet >= min_quiet) or (not any_change and seen_after and el >= noeff_min)
                if f is not None and fth is None and quiet_since is not None and ready and (
                        fast_asked is None or f.idx > fast_asked.idx):
                    fast_asked = f
                    fth = threading.Thread(target=fast, args=(f, ch_pre.union if ch_pre.significant else None),
                                           name="fast-settle-check", daemon=True)
                    fth.start()
            if any_change and quiet >= quiet_s and f is not None:
                same = view_frame is not None and not self._compare(view_frame, f).significant
                busy = view is not None and (view.loading or view.changing is True or view.settled is False)
                if not (same and busy):
                    return "settled_pixel", ""
                if el >= stuck_s and (not view.long_op or el >= long_max):
                    return "stuck", f"same screen judged busy {el:.1f}s after the action"
            if not any_change and seen_after and el >= no_eff:
                return "no_effect_timeout", f"no visible change {el:.1f}s after the action"
            if el >= (long_max if (view is not None and view.long_op) else max_wait):
                return "max_wait", ""

    def _wait_a11y(self, t_done: float, max_wait: float) -> tuple[str, str]:
        """AndroidWorld's stability check: a11y tree identical on 3 consecutive polls 0.5 s apart."""
        prev, same = None, 0
        while self.now() - t_done < max_wait:
            t = self.now()
            try:
                tree = self.env.a11y()
            except Exception:
                tree = None
            same = same + 1 if (tree is not None and tree == prev) else 1
            prev = tree
            if same >= 3:
                return "a11y_stable", ""
            time.sleep(max(0.0, 0.5 - (self.now() - t)))
        return "max_wait", ""

    def _wait_clip(self, pre: Frame, t_done: float, max_wait: float, quiet_s: float, no_eff: float) -> tuple[str, str]:
        import httpx

        from nightwatch.config import env as getenv
        from nightwatch.utils import b64, encode_jpeg

        url = getenv("S1_URL", "http://127.0.0.1:8710").rstrip("/") + "/v1/embed"
        thr = float(self.pcfg.get("clip_dist", 0.02))
        cl = self._clip or httpx.Client(timeout=30)
        self._clip = cl

        def emb(fr: Frame):
            r = cl.post(url, json={"images": [b64(encode_jpeg(fr.rgb, 85, 448))]})
            return np.asarray(r.json()["embeddings"][0])

        e_pre = emb(pre)
        last_idx, prev_e, quiet_since, any_change, seen_after = pre.idx, e_pre, None, False, False
        while True:
            f = self.stream.wait_new(last_idx, timeout=0.2)
            el = self.now() - t_done
            if f is not None:
                last_idx = f.idx
                seen_after = seen_after or f.t - f.grab_s >= t_done
                e = emb(f)
                any_change = any_change or (1 - float(e @ e_pre)) > thr
                moving = (1 - float(e @ prev_e)) > thr
                prev_e = e
                quiet_since = None if moving else (quiet_since if quiet_since is not None else f.t)
                if any_change and quiet_since is not None and f.t - quiet_since >= quiet_s:
                    return "clip_stable", ""
            if not any_change and seen_after and el >= no_eff:
                return "no_effect_timeout", ""
            if el >= max_wait:
                return "max_wait", ""

    # ================================================================== step
    def _step(self, step_no: int) -> tuple[StepRecord, bool, dict]:
        rec = StepRecord(step=step_no, t_obs=self.t_obs, t_obs_final=self.t_obs)
        n0, s0 = self.stream.counters()
        done, info = False, {}
        max_inval = int(self.lim.get("max_invalidations_per_step", 2))
        last_invalid: Act | None = None
        while True:
            snap = self.agent.snapshot()
            t0 = self.now()
            if rec.vlm_calls == 0:
                rec.t_vlm_start = t0
            rec.vlm_calls += 1
            self.vlm_calls += 1
            pred, intr = self._monitored_predict(rec)
            t1 = self.now()
            rec.t_vlm_end = t1
            self._log_llm_call(rec, pred, t0, t1, "interrupted" if intr is not None else ("error" if pred.error else "executed"))
            if pred.t_first_token is not None and rec.t_first_token is None:
                rec.t_first_token = round(pred.t_first_token, 3)
            rec.tokens_estimated |= pred.tokens_estimated
            if intr is not None:  # interrupted (cancelled, or finished-then-discarded when not cancellable)
                self.agent.restore(snap)
                self._seg(rec, "vlm_wasted", t0, t1)
                wasted = pred.output_tokens
                rec.wasted_tokens += wasted
                rec.input_tokens += pred.input_tokens
                rec.interrupted += 1
                rec.t_interrupt = rec.t_interrupt or round(self.last_interrupt_t, 4)
                rec.interrupt_reason = rec.interrupt_reason or intr.reason
                rec.interrupts.append({"t": round(self.last_interrupt_t, 4), "reason": intr.reason, "p": intr.p,
                                       "gen_s": round(t1 - t0, 3), "cancelled": pred.cancelled,
                                       "wasted_tokens": round(wasted, 1)})
                self._event("interrupt", rec, self.obs_frame, self._intr_frame, reason=intr.reason, p=intr.p,
                            gen_s=round(t1 - t0, 3), cancelled=pred.cancelled, wasted_tokens=round(wasted, 1),
                            partial=(pred.response or "")[-600:])
                self._observe(rec)
                rec.obs_count += 1
                if self.vlm_calls >= int(self.lim.get("max_vlm_calls", 200)):
                    break
                continue
            self._seg(rec, "vlm", t0, t1)
            rec.t_obs_final = round(self.t_obs, 4)  # the screenshot this (executed) prediction was based on
            rec.obs_file = self._obs_file
            rec.input_tokens += pred.input_tokens
            rec.output_tokens += pred.output_tokens
            rec.response = (pred.response or "")[-4000:]
            if pred.error:
                rec.error = pred.error
            actions = pred.actions or []
            rec.n_actions = len(actions)
            if last_invalid is not None and actions and parse_action(actions[0], self.space).same_as(last_invalid):
                rec.vetoes += 1  # the agent, on the fresh screen, chose the invalidated action again
            invalidated = False
            for k, action in enumerate(actions):
                act = parse_action(action, self.space)
                dec = None
                if rec.invalidated < max_inval:
                    if self.m.get("validate") == "pusv":
                        dec = self._pusv(act, rec)
                    elif self.m.get("pixel_rules", False) and self.m.get("invalidate", False):
                        dec = self._pixel_pre_exec(act, rec)
                    elif self.m.get("invalidate", False) and self.s1 is not None:
                        dec = self._pre_exec(act, rec)
                if dec is not None:
                    if dec.action == "invalidate":
                        rec.invalidated += 1
                        rec.invalidate_reason = rec.invalidate_reason or dec.reason
                        rec.invalidations.append({"t": round(self.now(), 4), "reason": dec.reason, "p": round(dec.p, 4),
                                                  "action": str(action)[:200], "index": k})
                        self._event("invalidate", rec, self.obs_frame, self._pre_frame, reason=dec.reason,
                                    p=round(dec.p, 4), action=str(action)[:1000], index=k, of=len(actions))
                        if k == 0:
                            if hasattr(self.agent, "on_action_result") and self.m.get("rules", {}).get(
                                    "invalidate_keep_history", True):
                                # keep what the agent saw and wrote (e.g. a code read off a pop-up that is gone now);
                                # only the action is withheld, reported back like a failed action
                                self.agent.on_action_result({"ok": False, "error": (
                                    f"not executed: the screen changed before your action could run ({dec.reason}). "
                                    "Look at the new screenshot and decide again.")})
                            else:
                                self.agent.restore(snap)
                            rec.wasted_tokens += pred.output_tokens
                            rec.output_tokens -= pred.output_tokens
                            # relabel this generation as wasted
                            for s in reversed(rec.segments):
                                if s[0] == "vlm":
                                    s[0] = "vlm_wasted"
                                    break
                            last_invalid = act
                        self._observe(rec)
                        rec.obs_count += 1
                        invalidated = True
                        break
                # ---- dispatch
                self.benign_boxes = []  # an action makes a new context: what was ambient before may be its effect now
                pre = self.stream.latest() or self.obs_frame
                if act.has_target:
                    W, H = self.env.screen_size
                    tb = box_around(min(max(act.x, 0), W - 1), min(max(act.y, 0), H - 1), 60)
                    rec.stale_px = round(self._region_change(self.obs_frame, pre, tb), 4)
                if self.m.get("strip_sleep", False):
                    cap = self.m.get("sleep_cap_s", 1.0)  # null: waits between two actions of one snippet run as written
                    action, cut = strip_agent_sleeps(action, float("inf") if cap is None else float(cap))
                    rec.stripped_sleep_s = round(rec.stripped_sleep_s + cut, 3)
                    if action == "WAIT" and not act.special:  # a snippet that only slept: wait on the screen
                        act = parse_action(action, self.space)
                t_disp = self.now()
                rec.t_dispatch = rec.t_dispatch or round(t_disp, 4)
                info = self.env.execute(action) or {}
                t_done = self.now()
                if hasattr(self.agent, "on_action_result"):
                    self.agent.on_action_result(info)
                self._seg(rec, "exec", t_disp, t_done)
                rec.t_action_done = round(t_done, 4)
                rec.action = str(action)[:2000] if k == 0 else rec.action + "\n" + str(action)[:500]
                rec.act_kind = rec.act_kind or act.kind
                if info.get("done") or act.kind in ("done", "fail"):
                    done = True
                    rec.t_wake, rec.wake_reason = round(t_done, 4), "episode_end"
                    break
                reason, diag = self._wait(act, pre, t_done, rec)
                t_wake = self.now()
                self._seg(rec, "wait", t_done, t_wake)
                rec.t_wake, rec.wake_reason, rec.wake_diag = round(t_wake, 4), reason, diag
                rec.wait_s = round(rec.wait_s + (t_wake - t_done), 4)
                if reason.startswith("event:"):
                    self._event("wake", rec, self.obs_frame, self._wake_frame, event=reason.split(":", 1)[1],
                                p=float(diag[2:]) if diag.startswith("p=") else None, wait_s=round(t_wake - t_done, 3),
                                stripped_sleep_s=rec.stripped_sleep_s, action=str(action)[:1000])
                after = self.stream.latest()
                if after is not None and not act.special:
                    rec.effect_px = round(self._compare(pre, after).frac, 5)
                self._observe(rec)
                self.prev_act, prev = act, self.prev_act
                if prev is not None and act.same_as(prev) and rec.step_type == "normal":
                    rec.step_type = "repeat"
            if invalidated and rec.n_actions and rec.t_dispatch is None:
                if self.vlm_calls >= int(self.lim.get("max_vlm_calls", 200)):
                    break
                continue  # re-predict on the fresh observation
            break
        n1, s1 = self.stream.counters()
        rec.frames_total, rec.diff_frames = n1 - n0, s1 - s0
        rec.step_type = self._step_type(rec)
        return rec, done, info

    def _update_benign(self, ans, boxes) -> None:
        """Small regions System-1 called ambient / trivial (a blinking caret, a spinner) are ignored by stage 1 until
        the page changes or the agent acts again. Only small boxes: a large region learned as "benign" would hide
        the effect of the next action (a 0.8B model calls most changes trivial zero-shot)."""
        nav = ans.probs.get("change_kind", {}).get("navigation", 0.0) >= 0.5 or self.rules.fires(ans, "page_changed")[0]
        if nav:
            self.benign_boxes = []
        elif boxes and ans.benign_change() >= self.rules.benign_tau:
            self.benign_boxes.extend(b for b in boxes if b not in self.benign_boxes
                                     and (b[2] - b[0]) * (b[3] - b[1]) <= self.benign_max_area)

    def _region_change(self, a: Frame, b: Frame, box) -> float:
        sw = a.small.shape[1] / self.env.screen_size[0]
        x0, y0, x1, y1 = (int(v * sw) for v in box)
        ra, rb = a.small[max(0, y0):y1, max(0, x0):x1], b.small[max(0, y0):y1, max(0, x0):x1]
        if ra.size == 0 or ra.shape != rb.shape:
            return 0.0
        return float((np.abs(ra.astype(np.int16) - rb.astype(np.int16)) > self.det.pix_thresh).mean())

    def _step_type(self, rec: StepRecord) -> str:
        th = self.cfg.get("step_types", {})
        if rec.act_kind == "wait":
            return "wait"
        if rec.stale_px is not None and rec.stale_px >= float(th.get("stale_px", 0.05)):
            return "stale"
        if rec.effect_px is not None and rec.act_kind not in ("done", "fail", "") and rec.effect_px < float(th.get("no_effect_frac", 0.0005)):
            return "no_effect"
        if rec.step_type == "repeat":
            return "repeat"
        if not rec.n_actions:
            return "no_action"
        return "normal"

    # ================================================================== episode
    def run(self) -> dict:
        res = {"task_id": self.task.get("id"), "domain": self.task.get("_domain", ""), "method": self.m.name,
               "agent": self.agent.name, "system1": self.s1.name if self.s1 else None, "error": "",
               "instruction": self.instruction[:1000], "repeat": self.task.get("_rep", 0)}
        ecfg = self.cfg.get("episode", {})
        steps_fh = open(self.out / "steps.jsonl", "w", encoding="utf-8")
        t_setup = time.monotonic()
        try:
            self.env.reset(self.task)
            res["setup_s"] = round(time.monotonic() - t_setup, 2)
            time.sleep(float(ecfg.get("settle_after_reset_s", 60)))
            self.agent.reset(vm_ip=self.env.vm_ip)
            self.clock.start()
            self.stream.start()
            if self.kf is not None:
                threading.Thread(target=self._feed_keyframes, name="aoi-keyframes", daemon=True).start()
            noise = self.cfg.get("noise", {})
            if noise.get("enabled") and hasattr(self.env, "controller"):
                from nightwatch.envs.noise import NoiseInjector

                self._noise = NoiseInjector(self.env.controller, noise, self.clock, self.out / "noise.jsonl")
                self._noise.start()
            if self.post == "fixed" and float(self.m.get("first_obs_wait", 0) or 0) > 0:
                time.sleep(float(self.m.get("first_obs_wait")))  # AOI's standard loop waits one interval before step 1
            self._observe()
            step, done, end = 0, False, "max_steps"
            limit = getattr(self.env, "time_limit", None)
            max_steps = int(self.lim.get("max_steps", 50))
            while step < max_steps:
                if self.vlm_calls >= int(self.lim.get("max_vlm_calls", 200)):
                    end = "max_vlm_calls"
                    break
                max_loop_s = self.lim.get("max_loop_s", 3600)
                if max_loop_s is not None and self.now() >= float(max_loop_s):
                    end = "timeout"
                    break
                rec, done, info = self._step(step + 1)
                jline(steps_fh, asdict(rec))
                self._acc(res, rec)
                if rec.t_dispatch is not None or rec.n_actions == 0:
                    step += 1  # interrupted / invalidated generations are not steps
                if hasattr(self.env, "check_done") and self.env.check_done():
                    end = "success"
                    break
                if done:
                    end = "fail" if info.get("fail") or rec.act_kind == "fail" else "done"
                    break
                if limit and self.env.elapsed() > limit:
                    end = "task_time_limit"
                    break
            res["loop_wall_s"] = round(self.now(), 3)
            res["loop_start_unix"] = round(self.clock.epoch0, 4)  # t = 0 of every step / LLM-call timestamp
            res["end_reason"] = end
            res["steps"] = step
        except Exception as e:
            res["error"] = f"{type(e).__name__}: {e}"
            res["traceback"] = traceback.format_exc()
            log.error("episode %s failed: %s", self.task.get("id"), res["traceback"])
            res.setdefault("loop_wall_s", round(self.now(), 3))
        finally:
            if getattr(self, "_noise", None):
                self._noise.stop()
            self._kf_stop.set()
            self.stream.stop()
            if self.s1 is not None and _tracked(self.s1.name):
                self.s1.wait_idle()  # include failures of outstanding watcher calls
            steps_fh.close()
            if self._s1_fh:
                self._s1_fh.close()
            self._ev_fh.close()
        jdump({"columns": ["t", "frac", "significant", "grab_s"], "rows": self.stream.stats}, self.out / "frame_stats.json")
        res["frames"] = {"total": self.stream.idx, "significant": self.stream.n_significant,
                         "grab_errors": self.stream.errors,
                         "fps": round(self.stream.idx / max(res.get("loop_wall_s", 1e-9), 1e-9), 2)}
        if self.s1:
            calls, ms = self.s1.snapshot()
            res["system1_calls"], res["system1_ms"] = calls, {k: round(v, 1) for k, v in ms.items()}
            res["system1_errors"] = self.s1.errors
            res["system1_answer_errors"] = getattr(self.s1, "answer_errors", 0)
            res["system1_skipped"] = getattr(self.s1, "skipped", 0)
        # System-1 decisions, none counted as a step: cancelled generations, withheld actions, early wakes
        res["events"] = {k: self.event_counts.get(k, 0) for k in ("interrupt", "invalidate", "wake")}
        res["stale_frame_refreshes"] = getattr(self, "stale_frame_refreshes", 0)
        if not res["error"]:
            t_eval = time.monotonic()
            time.sleep(float(ecfg.get("settle_before_eval_s", 20)))
            # a watchdog: some OSWorld getters block forever (e.g. pdf_from_url's connect_over_cdp through the AGS
            # a proxy can hang for hours); the episode is then an error to retry and the env is rebuilt
            box: dict = {}

            def evaluate() -> None:
                try:
                    box["score"] = float(self.env.evaluate())
                except Exception as e:
                    box["error"] = f"evaluate: {type(e).__name__}: {e}"

            th = threading.Thread(target=evaluate, name="evaluate", daemon=True)
            th.start()
            limit = float(ecfg.get("eval_timeout_s", 900))
            th.join(limit)
            if th.is_alive():
                res["score"], res["error"], res["env_broken"] = 0.0, f"evaluate: timeout after {limit:.0f}s", True
            elif "error" in box:
                res["score"], res["error"] = 0.0, box["error"]
            else:
                res["score"] = box["score"]
                if getattr(self.env, "last_eval", None) is not None:
                    res["evaluation"] = self.env.last_eval
            res["eval_s"] = round(time.monotonic() - t_eval, 2)
        res["success"] = float(res.get("score", 0.0) or 0.0) >= 1.0
        jdump(res, self.out / "result.json")
        return res

    def _acc(self, res: dict, rec: StepRecord) -> None:
        d = res.setdefault("decomposition", Counter())
        for kind, t0, t1 in rec.segments:
            d[kind] += t1 - t0
        for k in ("vlm_calls", "interrupted", "invalidated", "vetoes", "jev_calls", "keyframes"):
            res[k] = res.get(k, 0) + getattr(rec, k)
        for k in ("wait_s", "jev_ms_total", "jev_ms_critical", "input_tokens", "output_tokens", "wasted_tokens",
                  "stripped_sleep_s"):
            res[k] = round(res.get(k, 0.0) + getattr(rec, k), 3)
        res.setdefault("step_types", Counter())[rec.step_type] += 1
        if rec.wake_reason:
            res.setdefault("wake_reasons", Counter())[rec.wake_reason.split(":")[0] if rec.wake_reason.startswith("event")
                                                      else rec.wake_reason] += 1
        res["interrupt_wasted_s"] = round(res.get("interrupt_wasted_s", 0.0) + sum(i["gen_s"] for i in rec.interrupts), 3)
        res["tokens_estimated"] = res.get("tokens_estimated", False) or rec.tokens_estimated
