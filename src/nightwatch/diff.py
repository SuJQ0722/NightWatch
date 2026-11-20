"""Stage 1: pixel diff gate and change-region localisation (CPU, < 1 ms on 480-px grayscale frames).

`ChangeDetector.compare(a, b)` thresholds |a - b| on box-downsampled grayscale frames, labels the
connected components of the (slightly dilated) change mask and returns their boxes in full-resolution
coordinates. Components that are tiny (text caret, cursor blink) or lie inside known benign regions
(the clock in the top bar) or inside regions System-1 has called benign for this step (ads, videos)
do not count. `significant` is what gates stage 2.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy import ndimage


@dataclass
class Change:
    frac: float = 0.0  # fraction of changed pixels outside ignored regions
    boxes: list = field(default_factory=list)  # [(x0, y0, x1, y1)] full-resolution, largest first
    significant: bool = False

    @property
    def union(self):
        if not self.boxes:
            return None
        xs0, ys0, xs1, ys1 = zip(*self.boxes)
        return min(xs0), min(ys0), max(xs1), max(ys1)

    def near(self, x: float, y: float, radius: float) -> bool:
        return any(x0 - radius <= x <= x1 + radius and y0 - radius <= y <= y1 + radius
                   for x0, y0, x1, y1 in self.boxes)


def _inside(box, region) -> bool:
    x0, y0, x1, y1 = box
    r0, s0, r1, s1 = region
    return x0 >= r0 and y0 >= s0 and x1 <= r1 and y1 <= s1


class ChangeDetector:
    def __init__(self, full_size: tuple[int, int], small_width: int = 480, pix_thresh: int = 16,
                 min_frac: float = 0.0004, min_area: int = 6, benign_regions: list | None = None):
        """full_size = (W, H) of real frames; benign_regions in normalised [x0, y0, x1, y1]."""
        self.W, self.H = full_size
        self.small_width = small_width
        self.scale = self.W / small_width
        self.pix_thresh = pix_thresh
        self.min_frac = min_frac
        self.min_area = min_area
        self.benign = [(r[0] * self.W, r[1] * self.H, r[2] * self.W, r[3] * self.H) for r in (benign_regions or [])]

    def compare(self, a: np.ndarray, b: np.ndarray, ignore: list | None = None) -> Change:
        if a is None or b is None:
            return Change(1.0, [(0, 0, self.W, self.H)], True)
        if a.shape != b.shape:
            return Change(1.0, [(0, 0, self.W, self.H)], True)
        mask = np.abs(a.astype(np.int16) - b.astype(np.int16)) > self.pix_thresh
        if not mask.any():
            return Change()
        lab, n = ndimage.label(ndimage.binary_dilation(mask, iterations=2))
        boxes, kept = [], np.zeros_like(mask)
        ignored = list(self.benign) + list(ignore or [])
        for i, sl in enumerate(ndimage.find_objects(lab), start=1):
            if sl is None:
                continue
            comp = (lab[sl] == i) & mask[sl]
            area = int(comp.sum())
            if area < self.min_area:
                continue
            yy, xx = np.nonzero(comp)  # extent of the changed pixels themselves (not of the dilation)
            ys, xs = sl
            box = ((xs.start + xx.min()) * self.scale, (ys.start + yy.min()) * self.scale,
                   (xs.start + xx.max() + 1) * self.scale, (ys.start + yy.max() + 1) * self.scale)
            if any(_inside(box, r) for r in ignored):
                continue
            boxes.append((area, tuple(int(round(v)) for v in box)))
            kept[sl] |= comp
        frac = float(kept.mean())
        boxes.sort(key=lambda t: -t[0])
        return Change(frac, [b for _, b in boxes], bool(boxes) and frac >= self.min_frac)
