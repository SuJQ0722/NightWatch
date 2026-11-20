"""Screen frame stream: a background thread grabbing frames at a fixed rate from a ScreenSource.

Every frame gets a stage-1 comparison with the previous frame (`frame.change`), so per-frame
change statistics exist for every method (they feed the measurement study), and waiters/monitors can
block on `wait_new()`. With `record_dir` set, frames are also written to disk (JPEG + frames.jsonl
index) for the 10 Hz recording used in measurement, pilot evaluation and calibration labels.

The stream runs for every method with the same rate, so the screenshot load on the VM server is
identical across methods.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from nightwatch.diff import Change, ChangeDetector
from nightwatch.utils import Clock, decode_image, encode_jpeg, jline, small_gray

log = logging.getLogger("nightwatch.frames")


class ScreenSource(Protocol):
    def grab(self) -> tuple[bytes | None, dict | None]:
        """-> (encoded screenshot, ground truth dict or None). Thread-safe."""


@dataclass
class Frame:
    idx: int
    t: float  # episode clock at the end of the grab
    data: bytes
    small: np.ndarray
    grab_s: float
    gt: dict | None = None
    change: Change = field(default_factory=Change)  # vs previous frame
    _rgb: np.ndarray | None = None

    @property
    def rgb(self) -> np.ndarray:
        if self._rgb is None:
            self._rgb = decode_image(self.data)
        return self._rgb


def frame_from_bytes(data: bytes, t: float, idx: int = -1, gt: dict | None = None, small_width: int = 480) -> Frame:
    f = Frame(idx, t, data, np.zeros((1, 1), np.uint8), 0.0, gt)
    f.small = small_gray(f.rgb, small_width)
    return f


class FrameStream:
    def __init__(self, source: ScreenSource, clock: Clock, detector: ChangeDetector, hz: float = 10.0,
                 ring: int = 120, small_width: int = 480, record_dir: Path | None = None,
                 record_quality: int = 85, record_max_side: int | None = None):
        self.source, self.clock, self.detector = source, clock, detector
        self.period = 1.0 / max(hz, 0.1)
        self.small_width = small_width
        self.ring: deque[Frame] = deque(maxlen=ring)
        self.cond = threading.Condition()
        self.stop_ev = threading.Event()
        self.thread: threading.Thread | None = None
        self.idx = 0
        self.n_significant = 0
        self.errors = 0
        self.stats: list = []  # (t, changed fraction vs previous frame, significant, grab seconds) for every frame
        self.record_dir = Path(record_dir) if record_dir else None
        self.record_quality, self.record_max_side = record_quality, record_max_side
        self._index_fh = None

    # ---------------------------------------------------------------- lifecycle
    def start(self) -> None:
        if self.record_dir:
            (self.record_dir / "frames").mkdir(parents=True, exist_ok=True)
            self._index_fh = open(self.record_dir / "frames.jsonl", "a", encoding="utf-8")
        self.stop_ev.clear()
        self.thread = threading.Thread(target=self._loop, name="frame-stream", daemon=True)
        self.thread.start()
        if self.wait_new(-1, timeout=30) is None:
            log.warning("frame stream: no frame within 30 s")

    def stop(self) -> None:
        self.stop_ev.set()
        if self.thread:
            self.thread.join(timeout=15)
        if self._index_fh:
            self._index_fh.close()
            self._index_fh = None

    def _loop(self) -> None:
        prev: Frame | None = None
        next_t = time.monotonic()
        while not self.stop_ev.is_set():
            t_start = time.monotonic()
            try:
                data, gt = self.source.grab()
            except Exception as e:  # keep streaming through transient VM-server errors
                data, gt = None, None
                self.errors += 1
                if self.errors < 5:
                    log.warning("frame grab failed: %s", e)
            if data:
                try:
                    # a source that captures ahead of the grab (AGS daemon) reports how old the frame is
                    age = getattr(self.source, "capture_age", None)
                    f = Frame(self.idx, self.clock.now(), data, np.zeros((1, 1), np.uint8),
                              age if age is not None else time.monotonic() - t_start, gt)
                    f.small = small_gray(f.rgb, self.small_width)
                    f.change = self.detector.compare(prev.small if prev else None, f.small) if prev else Change()
                    self._publish(f)
                    prev = f
                except Exception as e:
                    self.errors += 1
                    log.warning("bad frame: %s", e)
            next_t = max(next_t + self.period, time.monotonic())
            self.stop_ev.wait(max(0.0, next_t - time.monotonic()))

    def _publish(self, f: Frame) -> None:
        with self.cond:
            self.ring.append(f)
            self.stats.append((round(f.t, 3), round(f.change.frac, 6), int(f.change.significant), round(f.grab_s, 3)))
            self.idx += 1
            if f.change.significant:
                self.n_significant += 1
            self.cond.notify_all()
        if self._index_fh is not None:
            name = f"{f.idx:06d}.jpg"
            (self.record_dir / "frames" / name).write_bytes(
                encode_jpeg(f.rgb, self.record_quality, self.record_max_side))
            jline(self._index_fh, {"idx": f.idx, "t": round(f.t, 4), "file": name, "frac": round(f.change.frac, 6),
                                   "sig": f.change.significant, "boxes": f.change.boxes[:8],
                                   "grab_s": round(f.grab_s, 4), "gt": f.gt})

    # ---------------------------------------------------------------- access
    def latest(self) -> Frame | None:
        with self.cond:
            return self.ring[-1] if self.ring else None

    def wait_new(self, after_idx: int, timeout: float = 1.0) -> Frame | None:
        """Newest frame with idx > after_idx (skips intermediate frames: latest-frame-wins)."""
        deadline = time.monotonic() + timeout
        with self.cond:
            while not self.ring or self.ring[-1].idx <= after_idx:
                rem = deadline - time.monotonic()
                if rem <= 0 or self.stop_ev.is_set():
                    return None
                self.cond.wait(rem)
            return self.ring[-1]

    def frames_since(self, t: float) -> list[Frame]:
        with self.cond:
            return [f for f in self.ring if f.t > t]

    def at_or_before(self, t: float) -> Frame | None:
        with self.cond:
            best = None
            for f in self.ring:
                if f.t <= t:
                    best = f
            return best or (self.ring[0] if self.ring else None)

    def counters(self) -> tuple[int, int]:
        with self.cond:
            return self.idx, self.n_significant
