"""OSWorld adapter: DesktopEnv from the pinned checkout, with the sleep taken out of `step`.

`execute()` replicates `DesktopEnv.step` (third_party/OSWorld/desktop_env/desktop_env.py) line for line
except for its trailing `time.sleep(pause)` and `_get_obs()`: step counter, action history, the
"environment used" flag, special actions and the controller call. The fixed-sleep baseline then sleeps
`pause` and observes, which is exactly `env.step(action, pause)` (including the extra `pause` sleep that
`step` spends on WAIT); event-driven methods decide when to observe instead.
tests/test_osworld_parity.py checks this replication against the pinned source.
"""

from __future__ import annotations

import logging
from pathlib import Path

from nightwatch.config import DATA_DIR, OSWORLD_DIR, ROOT, env
from nightwatch.envs.base import Env

log = logging.getLogger("nightwatch.osworld")

def client_password(ocfg) -> str:
    """The guest user's sudo password, shared by DesktopEnv (task setup), the AGS provider (sandbox setup,
    which reads AGS_SUDO_PASSWORD) and the agent's prompt. AGS templates differ: the docker-style image
    uses "password", the AWS-style one "osworld-public-evaluation"."""
    pw = ocfg.get("client_password") or ""
    provider = ocfg.get("provider") or env("OSWORLD_PROVIDER", "docker")
    if not pw and provider == "ags":
        pw = env("AGS_SUDO_PASSWORD", "password")
    if not pw:
        pw = "osworld-public-evaluation" if provider == "aws" else "password"  # DesktopEnv's own defaults
    return pw


def task_path(domain: str, example_id: str, examples_dir: str | Path | None = None) -> Path:
    """An OSWorld task file; `examples_dir` points at another task pool with the same layout (e.g. the generated
    training tasks under data/cua_train/tasks), default the benchmark's own examples."""
    base = Path(examples_dir) if examples_dir else OSWORLD_DIR / "evaluation_examples" / "examples"
    if not base.is_absolute():
        base = ROOT / base  # relative to the repo root
    return base / domain / f"{example_id}.json"

class OSWorldEnv(Env):
    name = "osworld"

    def __init__(self, ocfg, need_a11y: bool = False):
        from desktop_env.desktop_env import DesktopEnv

        for name in ("desktopenv", "desktopenv.pycontroller", "desktopenv.controller", "desktopenv.setup"):
            logging.getLogger(name).setLevel(logging.WARNING)  # get_screenshot logs every call at INFO
        provider = ocfg.get("provider") or env("OSWORLD_PROVIDER", "docker")
        path_to_vm = ocfg.get("path_to_vm") or env("OSWORLD_VM_PATH") or None
        if provider == "docker" and not path_to_vm:
            path_to_vm = str(DATA_DIR / "vm" / "Ubuntu.qcow2")
        if provider == "docker" and not Path(path_to_vm).exists():
            raise FileNotFoundError(f"{path_to_vm} not found: run `bash scripts/setup_container.sh vm`")
        w, h = ocfg.get("screen_size", [1920, 1080])
        self.screen_size = (int(w), int(h))
        cache = Path(ocfg.get("cache_dir") or DATA_DIR / "osworld_cache")
        cache.mkdir(parents=True, exist_ok=True)
        self.env = DesktopEnv(provider_name=provider, region=ocfg.get("region") or env("OSWORLD_REGION") or None,
                              path_to_vm=path_to_vm, action_space="pyautogui", cache_dir=str(cache),
                              screen_size=self.screen_size, headless=True, require_a11y_tree=need_a11y,
                              os_type="Ubuntu", client_password=client_password(ocfg),
                              # OSWorld's guest recording-route updater only matters for its screen recording,
                              # which nightwatch never uses; on some AGS templates it aborts the boot
                              recording_update=ocfg.get("recording_update") or env("OSWORLD_RECORDING_UPDATE", "off"))
        self.need_a11y = need_a11y
        self._fast = None
        grab = ocfg.get("ags_grab") or env("OSWORLD_AGS_GRAB", "daemon")  # daemon | run_python (= jpeg) | png
        if provider == "ags" and grab in ("run_python", "jpeg", "daemon"):
            self._fast = AGSFrameGrabber(self.env.provider, mode="daemon" if grab == "daemon" else "run_python",
                                         hz=float(ocfg.get("frames_hz") or 10))

    @property
    def vm_ip(self):
        return self.env.vm_ip

    @property
    def controller(self):
        return self.env.controller

    def reset(self, task: dict) -> None:
        self.env.reset(task_config=task)

    def instruction(self, task: dict) -> str:
        return task["instruction"]

    @property
    def capture_age(self) -> float | None:
        """Seconds from the start of the last frame's capture to the end of its grab, when the source knows it."""
        return self._fast.capture_age if self._fast is not None else None

    def grab(self):
        if self._fast is not None:
            try:
                return self._fast.grab(), None
            except Exception as e:  # fall back to the official endpoint for this frame
                self._fast.capture_age = None  # the fallback screenshot has its own capture time
                log.debug("fast grab failed: %s", e)
        return self.env.controller.get_screenshot(), None

    def windows(self) -> list[str] | None:
        """`wmctrl -l` in the guest (one line per window: id, desktop, host, title); run through OSWorld Server so
        it gets the session's DISPLAY / XAUTHORITY. None when unavailable."""
        try:
            r = self.env.controller.execute_python_command(
                "import subprocess; print(subprocess.run(['wmctrl', '-l'], capture_output=True, text=True).stdout)")
        except Exception as e:
            log.debug("wmctrl failed: %s", e)
            return None
        return [ln for ln in ((r or {}).get("output") or "").splitlines() if ln.strip()] if r else None

    def obs(self) -> dict:
        o = self.env._get_obs()
        o["gt"] = None
        return o

    def a11y(self):
        return self.env.controller.get_accessibility_tree()

    def execute(self, action) -> dict:
        e = self.env
        e._step_no += 1
        e.action_history.append(action)
        e.is_environment_used = True
        info = {}
        special = action in ("WAIT", "FAIL", "DONE") or (isinstance(action, dict) and action.get("action_type") in ("WAIT", "FAIL", "DONE"))
        if special:
            kind = action if isinstance(action, str) else action.get("action_type")
            if kind == "FAIL":
                info = {"fail": True, "done": True}
            elif kind == "DONE":
                info = {"done": True}
            else:
                info = {"wait": True}
            e.controller.execute_action(action)
        elif isinstance(action, str):
            e.controller.execute_python_command(action)
        elif isinstance(action, dict):
            e.controller.execute_python_command(action["command"])
        return info

    def evaluate(self) -> float:
        return float(self.env.evaluate())

    def close(self) -> None:
        try:
            self.env.close()
        except Exception as e:
            log.warning("close failed: %s", e)

class AGSFrameGrabber:
    """Frame-stream screenshots for the AGS (Tencent Cloud Agent Sandbox) provider.

    The official /screenshot is a ~1.7 MB PNG; across regions it takes ~1.2 s even on a kept-alive connection.
    For the *frame stream* only (the agent's observations still use the official PNG via _get_obs) we capture in
    the guest and return a JPEG (~90 KB) on one persistent HTTPS session straight to the sandbox (no local proxy
    hop). Every method gets the same stream. Two modes:

      run_python  one /run_python per frame: a fresh python3 that imports pyautogui and captures (~0.3 s a frame)
      daemon      one capture process in the guest at the stream rate (PIL ImageGrab + JPEG, ~50 ms a frame)
                  writes the newest frame, headed by its sequence number and capture time, to FRAME; each grab
                  reads it through /file (~15 ms, no process start). 10 Hz instead of ~3 Hz.
    """

    SCRIPT = ("import io,base64,pyautogui\n"
              "b=io.BytesIO(); pyautogui.screenshot().convert('RGB').save(b,'JPEG',quality={q})\n"
              "print(base64.b64encode(b.getvalue()).decode())\n")
    FRAME, HEAD = "/tmp/nightwatch_frame.bin", 16  # header: <Qd = sequence number, guest time.time() at capture start
    CLOCK_PROBE = "import time; print(time.time())"
    CLOCK_MAX_RTT_S = 1.0
    DAEMON = """import io, os, struct, time
from PIL import ImageGrab
period, q, seq = {period}, {q}, 0
open("/tmp/nightwatch_grab.pid", "w").write(str(os.getpid()))
while not os.path.exists("/tmp/nightwatch_grab.stop"):
    t0, w = time.monotonic(), time.time()
    b = io.BytesIO(); ImageGrab.grab().convert("RGB").save(b, "JPEG", quality=q)
    seq += 1
    with open("{frame}.tmp", "wb") as f:
        f.write(struct.pack("<Qd", seq, w)); f.write(b.getvalue())
    os.replace("{frame}.tmp", "{frame}")
    time.sleep(max(0.0, period - (time.monotonic() - t0)))
"""
    START = """import os, subprocess, sys, time
try:
    os.kill(int(open("/tmp/nightwatch_grab.pid").read()), 0); print("running")
except Exception:
    for p in ("/tmp/nightwatch_grab.stop", "{frame}"):
        os.path.exists(p) and os.remove(p)
    open("/tmp/nightwatch_grab.py", "w").write({code!r})
    subprocess.Popen([sys.executable, "/tmp/nightwatch_grab.py"], start_new_session=True, stdout=subprocess.DEVNULL,
                     stderr=open("/tmp/nightwatch_grab.err", "w"))
    print("started")
print(time.time())
"""

    def __init__(self, provider, quality: int | None = None, mode: str = "run_python", hz: float = 10.0):
        self.provider = provider
        self.key = None
        self.s = None
        q = int(quality or env("OSWORLD_AGS_JPEG_QUALITY", "85"))
        self.code = self.SCRIPT.format(q=q)
        self.mode = mode
        self.start_code = self.START.format(frame=self.FRAME, code=self.DAEMON.format(
            period=round(1.0 / max(hz, 0.1), 4), q=q, frame=self.FRAME))
        self.seq, self.offset = 0, None
        self.clock_rtt_s = None
        self.capture_age = None  # daemon: seconds from the capture's start to the end of the last grab

    def _session(self):
        """Each reset replaces the sandbox (new host and token): rebuild the session when they change."""
        import requests

        p = self.provider
        key = (p.sandbox.get_host(5000), p.envd_access_token) if p.sandbox is not None else None
        if key is None:
            raise RuntimeError("no sandbox")
        if key != self.key:
            self.s = requests.Session()
            self.s.headers["X-Access-Token"] = key[1]
            if self.mode == "daemon":
                self._start_daemon(f"https://{key[0]}")
            self.key = key  # failed startup must be retried on the next grab
        return self.s, f"https://{key[0]}/run_python"

    def _start_daemon(self, base: str) -> None:
        """Start capture, then calibrate separately: cold startup can take tens of seconds."""
        self.offset, self.clock_rtt_s, self.capture_age = None, None, None
        r = self.s.post(f"{base}/run_python", json={"code": self.start_code}, timeout=30)
        r.raise_for_status()
        out = (r.json().get("output") or "").split()
        self.seq = 0
        self._calibrate_clock(base)
        log.info("AGS capture daemon %s (clock offset %s s, probe RTT %s s)",
                 out[0] if out else "?", self.offset, self.clock_rtt_s)

    def _calibrate_clock(self, base: str) -> None:
        """Use the fastest of three short probes, excluding startup and stalled requests.

        The midpoint estimate has uncertainty at most half the RTT. Frame ages include that uncertainty so
        a pre-observation capture cannot pass the freshness gate merely because the network was asymmetric.
        Without a usable probe, grab() uses synchronous capture instead of trusting an unbounded timestamp.
        """
        import math
        import time

        samples = []
        for _ in range(3):
            try:
                t0, m0 = time.time(), time.monotonic()
                r = self.s.post(f"{base}/run_python", json={"code": self.CLOCK_PROBE}, timeout=2)
                t1, m1 = time.time(), time.monotonic()
                r.raise_for_status()
                guest = float((r.json().get("output") or "").strip())
                rtt = m1 - m0
                if (math.isfinite(guest) and 0 <= rtt <= self.CLOCK_MAX_RTT_S
                        and abs((t1 - t0) - rtt) <= 0.1):
                    samples.append((rtt, guest - (t0 + t1) / 2))
            except Exception as e:
                log.debug("AGS clock probe failed: %s", e)
        if samples:
            self.clock_rtt_s, self.offset = min(samples)
        else:
            self.clock_rtt_s, self.offset = None, None
            log.warning("AGS clock calibration unavailable; using synchronous screenshots for this sandbox")

    def grab(self) -> bytes | None:
        import base64

        self.capture_age = None
        s, url = self._session()
        if self.mode == "daemon" and self.offset is not None:
            return self._grab_daemon(s, url.rsplit("/", 1)[0])
        r = s.post(url, json={"code": self.code}, timeout=30)
        r.raise_for_status()
        out = (r.json().get("output") or "").strip()
        if not out:
            raise RuntimeError("empty frame")
        return base64.b64decode(out.splitlines()[-1])

    def _grab_daemon(self, s, base: str) -> bytes | None:
        """The newest frame the daemon has written that this grabber has not returned yet; None when none arrives
        within 0.3 s (a repeated frame would read as a quiet screen, so the stream skips the slot instead)."""
        import struct
        import time

        deadline = time.monotonic() + 0.3
        while True:
            r = s.post(f"{base}/file", data={"file_path": self.FRAME}, timeout=30)
            if r.status_code == 200 and len(r.content) > self.HEAD:
                seq, w = struct.unpack("<Qd", r.content[:self.HEAD])
                if seq != self.seq:
                    self.seq = seq
                    self.capture_age = max(0.0, time.time() - (w - self.offset) + self.clock_rtt_s / 2)
                    return r.content[self.HEAD:]
            elif r.status_code not in (200, 404):  # 404: the daemon has not written its first frame yet
                r.raise_for_status()
            if time.monotonic() > deadline:
                self.capture_age = None
                return None
            time.sleep(0.01)


def load_tasks(tcfg) -> list[dict]:
    """Task list from an OSWorld meta file ({domain: [ids]}), filtered by domains / ids / limit."""
    import json
    import random

    meta_file = Path(tcfg.get("meta", "evaluation_examples/test_nogdrive.json"))
    if not meta_file.is_absolute():
        meta_file = OSWORLD_DIR / meta_file
    meta = json.loads(meta_file.read_text())
    domains = tcfg.get("domains") or list(meta)
    ids = set(tcfg.get("ids") or [])
    out = []
    for d in domains:
        lst = list(meta.get(d, []))
        if tcfg.get("per_domain"):
            rng = random.Random(int(tcfg.get("seed", 0)))
            lst = rng.sample(lst, min(len(lst), int(tcfg.get("per_domain"))))
        for i in lst:
            if ids and i not in ids:
                continue
            p = task_path(d, i, tcfg.get("examples_dir"))
            t = json.loads(p.read_text())
            t["_domain"] = d
            out.append(t)
    if tcfg.get("sample"):
        out = random.Random(int(tcfg.get("seed", 0))).sample(out, min(len(out), int(tcfg.get("sample"))))
    if tcfg.get("limit"):
        out = out[: int(tcfg.get("limit"))]
    return out
