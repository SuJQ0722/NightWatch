"""A simulated desktop with real-time dynamics, for offline pipeline checks (no VM, no API, no GPU).

It reproduces the timing failure modes of the proposal in miniature:
  * a click starts a loading phase (spinner, button disabled) of random length; clicks during it are lost;
  * the next page slides in (a short transition: the button moves);
  * the last step is the Wi-Fi example: a success banner that, with probability `p_revert`, flips back to a
    failure state with a short "No network found" toast, after which a Retry button appears;
  * random modal pop-ups (Poisson) that swallow clicks until dismissed;
  * an ambient carousel that never stops changing (pixel-only stability never settles).
Ground truth travels with every screenshot (`gt`) for the simulated agent and simulated System-1.
Pipeline checks only: nothing measured here is a result.
"""

from __future__ import annotations

import math
import random
import threading
import time
import zlib

import numpy as np
from PIL import Image, ImageDraw

from nightwatch.envs.base import Env
from nightwatch.utils import encode_png

W, H = 960, 540


class SimDesktop(Env):
    name = "sim"

    def __init__(self, cfg=None, seed: int = 0):
        cfg = dict(cfg or {})
        self.screen_size = (W, H)
        self.p = {"steps": 3, "load_min": 0.2, "load_max": 1.6, "slide_s": 0.3, "flash_s": 0.9, "p_revert": 0.5,
                  "toast_s": 1.0, "popup_rate_per_min": 4.0, "p_error": 0.15, "error_s": 1.2, "long_op_p": 0.0}
        self.p.update({k: v for k, v in cfg.items() if k in self.p})
        self.base_seed = seed
        self.lock = threading.RLock()
        self.vm_ip = "sim"

    # ------------------------------------------------------------------ task
    def reset(self, task: dict) -> None:
        with self.lock:
            self.rng = random.Random(zlib.crc32(f"{self.base_seed}/{task.get('id', '')}/{task.get('_rep', 0)}".encode()))
            self.t0 = time.monotonic()
            self.K = int(task.get("steps", self.p["steps"]))
            self.page = 0
            self.phase = "ready"  # ready | loading | flash | final | failed
            self.phase_end = math.inf
            self.version = 0
            self.btn = self._new_btn()
            self.prev_btn = self.btn
            self.slide_until = 0.0
            self.popup = None
            self.toast = None  # (text, until, kind)
            self.retry = False
            self.done_flag = None  # "done" | "fail"
            self.done_ok = False
            self.next_popup = self._next_popup_t(0.0)
            self.lost_clicks = 0

    def instruction(self, task: dict) -> str:
        return task.get("instruction", f"Complete all {task.get('steps', self.p['steps'])} steps by clicking the Step "
                                        "buttons in order, then confirm that the connection succeeded.")

    def _now(self) -> float:
        return time.monotonic() - self.t0

    def _next_popup_t(self, t: float) -> float:
        r = self.p["popup_rate_per_min"]
        return t + self.rng.expovariate(r / 60.0) if r > 0 else math.inf

    def _new_btn(self):
        x = self.rng.randint(120, W - 300)
        y = self.rng.randint(140, H - 160)
        return (x, y, x + 180, y + 56)

    # ------------------------------------------------------------------ dynamics
    def _advance(self) -> None:
        t = self._now()
        if self.popup is None and t >= self.next_popup:
            self.popup = (W // 2 - 190, H // 2 - 90, W // 2 + 190, H // 2 + 90)
            self.version += 1
        if self.toast and t >= self.toast[1]:
            self.toast = None
        if t >= self.phase_end:
            if self.phase == "loading":
                if self.page < self.K - 1:
                    if self.rng.random() < self.p["p_error"]:
                        self.toast = ("Request failed, try again", t + self.p["error_s"], "error")
                        self.phase, self.phase_end = "ready", math.inf
                    else:
                        self.page += 1
                        self.prev_btn, self.btn = self.btn, self._new_btn()
                        self.slide_until = t + self.p["slide_s"]
                        self.phase, self.phase_end = "ready", math.inf
                else:
                    self.phase, self.phase_end = "flash", t + self.p["flash_s"]
                self.version += 1
            elif self.phase == "flash":
                if self.rng.random() < self.p["p_revert"] and not self.retry:
                    self.phase, self.phase_end = "failed", math.inf
                    self.toast = ("No network found", t + self.p["toast_s"], "toast")
                    self.retry = True
                else:
                    self.phase, self.phase_end = "final", math.inf
                self.version += 1

    def _btn_now(self, t: float):
        if t >= self.slide_until:
            return self.btn
        a = 1 - (self.slide_until - t) / self.p["slide_s"]
        return tuple(int(p0 + (p1 - p0) * a) for p0, p1 in zip(self.prev_btn, self.btn))

    def _gt(self) -> dict:
        t = self._now()
        btn = self._btn_now(t)
        els = []
        if self.phase in ("ready", "failed"):
            els.append([f"btn{self.page}{'r' if self.retry else ''}", *btn, True])
        elif self.phase == "loading":
            els.append([f"spinner{self.page}", *btn, False])
        if self.popup:
            ok = self._popup_ok()
            els.insert(0, ["popup_ok", *ok, True])
        amb = (W - 170, H - 120, W - 20, H - 20)
        return {"t": round(t, 3), "version": self.version, "page": self.page, "phase": self.phase,
                "loading": self.phase == "loading", "animating": t < self.slide_until or self.phase == "flash",
                "popup": self.popup is not None, "popup_box": list(self.popup) if self.popup else None,
                "toast": bool(self.toast and self.toast[2] == "toast"), "error": bool(self.toast and self.toast[2] == "error"),
                "elements": els, "ambient_box": list(amb), "success_visible": self.phase in ("flash", "final"),
                "final": self.phase == "final", "long_op": False}

    def _popup_ok(self):
        x0, y0, x1, y1 = self.popup
        return (x1 - 120, y1 - 50, x1 - 20, y1 - 14)

    # ------------------------------------------------------------------ rendering
    def _render(self, gt: dict) -> np.ndarray:
        t = gt["t"]
        im = Image.new("RGB", (W, H), (236, 239, 244))
        d = ImageDraw.Draw(im)
        d.rectangle((0, 0, W, 28), fill=(40, 44, 52))
        d.text((10, 8), "Settings - Network", fill=(230, 230, 230))
        d.text((W - 60, 8), f"{int(t) // 60:02d}:{int(t) % 60:02d}", fill=(230, 230, 230))  # clock (benign)
        d.text((40, 50), f"Page {self.page + 1} / {self.K}", fill=(30, 30, 30))
        x0, y0, x1, y1 = self._btn_now(t)
        if self.phase in ("ready", "failed"):
            d.rectangle((x0, y0, x1, y1), fill=(52, 120, 246), outline=(20, 60, 160), width=2)
            label = "Retry" if self.retry and self.page == self.K - 1 else f"Step {self.page + 1}"
            d.text((x0 + 60, y0 + 20), label, fill=(255, 255, 255))
        elif self.phase == "loading":
            d.rectangle((x0, y0, x1, y1), fill=(200, 200, 205))
            cx, cy = (x0 + x1) // 2, (y0 + y1) // 2
            a = (t * 360 * 1.5) % 360
            d.arc((cx - 16, cy - 16, cx + 16, cy + 16), a, a + 270, fill=(80, 80, 80), width=4)
        if self.phase in ("flash", "final"):
            pulse = self.phase == "flash" and int(t * 10) % 2 == 0
            d.rectangle((200, 220, 760, 300), fill=(46, 160, 67), outline=(255, 255, 0) if pulse else (20, 100, 40), width=4)
            d.text((380, 252), "Connected", fill=(255, 255, 255))
        if self.phase == "failed":
            d.text((40, 80), "Status: not connected", fill=(160, 40, 40))
        if self.toast:
            col = (180, 40, 40) if self.toast[2] == "error" else (50, 50, 50)
            d.rectangle((W // 2 - 140, H - 70, W // 2 + 140, H - 30), fill=col)
            d.text((W // 2 - 110, H - 56), self.toast[0], fill=(255, 255, 255))
        ax0, ay0, ax1, ay1 = gt["ambient_box"]  # ambient carousel: always changing, irrelevant
        k = int(t / 0.4) % 4
        d.rectangle((ax0, ay0, ax1, ay1), fill=[(250, 200, 80), (120, 200, 250), (250, 120, 160), (160, 230, 140)][k])
        d.text((ax0 + 30, ay0 + 40), f"AD {k + 1}/4", fill=(20, 20, 20))
        if self.popup:
            px0, py0, px1, py1 = self.popup
            d.rectangle((px0, py0, px1, py1), fill=(255, 255, 255), outline=(60, 60, 60), width=2)
            d.text((px0 + 20, py0 + 20), "Software update available", fill=(0, 0, 0))
            d.text((px0 + 20, py0 + 45), "Restart later?", fill=(60, 60, 60))
            ox0, oy0, ox1, oy1 = self._popup_ok()
            d.rectangle((ox0, oy0, ox1, oy1), fill=(52, 120, 246))
            d.text((ox0 + 38, oy0 + 12), "OK", fill=(255, 255, 255))
        return np.asarray(im)

    # ------------------------------------------------------------------ Env api
    def grab(self):
        with self.lock:
            self._advance()
            gt = self._gt()
            img = self._render(gt)
        return encode_png(img), gt

    def execute(self, action) -> dict:
        from nightwatch.actions import parse_action

        act = parse_action(action)
        with self.lock:
            self._advance()
            t = self._now()
            if act.kind in ("done", "fail"):
                self.done_flag = act.kind
                self.done_ok = act.kind == "done" and self.phase == "final"
                return {"done": True}
            if act.kind != "click" or act.x is None:
                return {}
            x, y = act.x, act.y
            if self.popup:
                ox0, oy0, ox1, oy1 = self._popup_ok()
                if ox0 <= x <= ox1 and oy0 <= y <= oy1:
                    self.popup = None
                    self.next_popup = self._next_popup_t(t)
                    self.version += 1
                else:
                    self.lost_clicks += 1
                return {}
            bx0, by0, bx1, by1 = self._btn_now(t)
            if self.phase in ("ready", "failed") and bx0 <= x <= bx1 and by0 <= y <= by1:
                if self.phase == "failed":
                    self.phase = "ready"
                self.phase = "loading"
                self.phase_end = t + self.rng.uniform(self.p["load_min"], self.p["load_max"])
                self.version += 1
            else:
                self.lost_clicks += 1
        return {}

    def evaluate(self) -> float:
        return 1.0 if self.done_flag == "done" and self.done_ok else 0.0
