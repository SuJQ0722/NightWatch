"""Image helpers, clocks and small JSON utilities."""

from __future__ import annotations

import base64
import hashlib
import io
import json
import threading
import time
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


# ------------------------------------------------------------------------------------- images
def decode_image(data: bytes) -> np.ndarray:
    return np.asarray(Image.open(io.BytesIO(data)).convert("RGB"))


def encode_png(img: np.ndarray) -> bytes:
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="PNG", compress_level=1)
    return buf.getvalue()


def encode_jpeg(img: np.ndarray, quality: int = 90, max_side: int | None = None) -> bytes:
    im = Image.fromarray(img)
    if max_side and max(im.size) > max_side:
        s = max_side / max(im.size)
        im = im.resize((max(1, round(im.width * s)), max(1, round(im.height * s))), Image.BILINEAR)
    buf = io.BytesIO()
    im.save(buf, format="JPEG", quality=quality)
    return buf.getvalue()


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode("ascii")


def data_url_jpeg(img: np.ndarray, max_side: int | None = 1280) -> str:
    return "data:image/jpeg;base64," + b64(encode_jpeg(img, max_side=max_side))


def small_gray(img: np.ndarray, width: int = 480) -> np.ndarray:
    """Box-downsampled grayscale copy for the stage-1 pixel diff."""
    im = Image.fromarray(img).convert("L")
    if im.width > width:
        im = im.resize((width, max(1, round(im.height * width / im.width))), Image.BOX)
    return np.asarray(im, dtype=np.uint8)


def clip_box(box, w: int, h: int):
    x0, y0, x1, y1 = box
    return max(0, int(x0)), max(0, int(y0)), min(w, int(x1)), min(h, int(y1))


def crop(img: np.ndarray, box, pad: int = 16, min_side: int = 96) -> np.ndarray:
    """Crop `box` (x0, y0, x1, y1) with padding, grown to at least min_side around its center."""
    h, w = img.shape[:2]
    x0, y0, x1, y1 = box
    x0, y0, x1, y1 = x0 - pad, y0 - pad, x1 + pad, y1 + pad
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    hw, hh = max((x1 - x0) / 2, min_side / 2), max((y1 - y0) / 2, min_side / 2)
    x0, y0, x1, y1 = clip_box((cx - hw, cy - hh, cx + hw, cy + hh), w, h)
    if x1 <= x0 or y1 <= y0:
        return img
    return img[y0:y1, x0:x1]


def box_around(x: float, y: float, half: int) -> tuple[int, int, int, int]:
    return int(x - half), int(y - half), int(x + half), int(y + half)


def draw_marker(img: np.ndarray, x: float, y: float, r: int = 14) -> np.ndarray:
    im = Image.fromarray(img.copy())
    d = ImageDraw.Draw(im)
    for k in (0, 1, 2):
        d.ellipse((x - r - k, y - r - k, x + r + k, y + r + k), outline=(255, 0, 0))
    return np.asarray(im)


def content_hash(img: np.ndarray) -> str:
    return hashlib.blake2b(np.ascontiguousarray(img).tobytes(), digest_size=10).hexdigest()


# ------------------------------------------------------------------------------------- clocks
class Clock:
    """Episode clock: seconds since `start()` on the monotonic clock, plus the wall epoch of t=0."""

    def __init__(self):
        self.start()

    def start(self) -> None:
        self.t0 = time.monotonic()
        self.epoch0 = time.time()

    def now(self) -> float:
        return time.monotonic() - self.t0


def sleep_until(clock: Clock, t: float, stop: threading.Event | None = None) -> None:
    while True:
        rem = t - clock.now()
        if rem <= 0 or (stop is not None and stop.is_set()):
            return
        time.sleep(min(rem, 0.05))


# --------------------------------------------------------------------------------------- json
def jdump(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, ensure_ascii=False, default=_default), encoding="utf-8")
    tmp.replace(path)


def jline(fh, obj) -> None:
    fh.write(json.dumps(obj, ensure_ascii=False, default=_default) + "\n")
    fh.flush()


def read_jsonl(path: Path) -> list[dict]:
    if not Path(path).exists():
        return []
    out = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            try:
                out.append(json.loads(line))
            except json.JSONDecodeError:
                pass  # a line cut by a crash
    return out


def _default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return str(o)
    if isinstance(o, set):
        return sorted(o)
    return str(o)
