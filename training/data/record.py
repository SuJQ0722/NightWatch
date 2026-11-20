"""Record SynthWeb episodes: headless Chromium, a random interaction policy, 10 Hz frames + exact ground truth.

Runs in the DynaCU env (Playwright).  python -m training.data.record --n 2000 --workers 16 --out data/synthweb/rec
Each episode folder: spec.json, page.html, frames/NNNNNN.jpg, frames.jsonl ({idx, t, file, gt}), actions.jsonl
({t, idx_before, kind, x, y, tid, text}).  The policy clicks visible enabled elements (dialog buttons first when a
dialog is up, with some probability), clicks empty space, types into inputs or idles, at random intervals, so
that every family produces both action-triggered and spontaneous events.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import random
import time
from pathlib import Path

from training.data.synthweb import FAMILIES, generate

SIZES = [(1280, 720), (1280, 720), (1920, 1080), (1440, 900)]


def record_one(seed: int, out: Path, duration: float, hz: float, quality: int, families=None) -> int:
    from playwright.sync_api import sync_playwright

    r = random.Random(seed * 7 + 1)
    w, h = r.choice(SIZES)
    html, spec = generate(seed, families, w, h)
    d = out / f"{seed:07d}"
    if (d / "done").exists():
        return 0
    (d / "frames").mkdir(parents=True, exist_ok=True)
    (d / "page.html").write_text(html, encoding="utf-8")
    (d / "spec.json").write_text(json.dumps(spec), encoding="utf-8")
    fh, ah = open(d / "frames.jsonl", "w"), open(d / "actions.jsonl", "w")
    with sync_playwright() as p:
        b = p.chromium.launch(headless=True, args=["--no-sandbox", "--disable-dev-shm-usage"])
        page = b.new_context(viewport={"width": w, "height": h}).new_page()
        page.goto((d / "page.html").absolute().as_uri())
        t0 = time.monotonic()
        next_act = t0 + r.uniform(0.8, 3.0)
        idx = 0
        while (now := time.monotonic()) - t0 < duration:
            t = round(now - t0, 3)
            name = f"{idx:06d}.jpg"
            (d / "frames" / name).write_bytes(page.screenshot(type="jpeg", quality=quality))
            gt = page.evaluate("window.__gt()")
            fh.write(json.dumps({"idx": idx, "t": t, "file": name, "gt": gt}) + "\n")
            if now >= next_act:
                act = _choose(r, gt, w, h)
                if act:
                    if act["kind"] == "type":
                        page.mouse.click(act["x"], act["y"])
                        page.keyboard.type(act["text"], delay=40)
                    else:
                        page.mouse.click(act["x"], act["y"])
                    ah.write(json.dumps({"t": round(time.monotonic() - t0, 3), "idx_before": idx, **act}) + "\n")
                next_act = time.monotonic() + r.choice([r.uniform(0.3, 1.2), r.uniform(1, 4), r.uniform(3, 7)])
            idx += 1
            time.sleep(max(0.0, 1 / hz - (time.monotonic() - now)))
        b.close()
    fh.close()
    ah.close()
    (d / "done").write_text("ok")
    return idx


def _choose(r: random.Random, gt: dict, w: int, h: int) -> dict | None:
    els = [e for e in gt["elements"] if e[5]]
    u = r.random()
    if gt.get("dialog") and r.random() < 0.6:
        dl = [e for e in els if e[0].startswith("dlg")]
        if dl:
            e = r.choice(dl)
            return {"kind": "click", "x": (e[1] + e[3]) / 2, "y": (e[2] + e[4]) / 2, "tid": e[0], "text": ""}
    if u < 0.62 and els:
        e = r.choice(els)
        if e[0].startswith("in"):
            return {"kind": "type", "x": (e[1] + e[3]) / 2, "y": (e[2] + e[4]) / 2, "tid": e[0], "text": r.choice(["hello", "2026", "test@x.com"])}
        return {"kind": "click", "x": (e[1] + e[3]) / 2 + r.uniform(-4, 4), "y": (e[2] + e[4]) / 2 + r.uniform(-3, 3), "tid": e[0], "text": ""}
    if u < 0.8:
        return {"kind": "click", "x": r.uniform(20, w - 20), "y": r.uniform(70, h - 20), "tid": "", "text": ""}
    return None


def _job(args):
    seed, out, duration, hz, quality, held = args
    fam = None
    try:
        return record_one(seed, Path(out), duration, hz, quality, fam)
    except Exception as e:  # keep the pool going
        print(f"seed {seed} failed: {e}", flush=True)
        return 0


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--duration", type=float, default=40.0)
    ap.add_argument("--hz", type=float, default=10.0)
    ap.add_argument("--quality", type=int, default=85)
    ap.add_argument("--out", default="data/synthweb/rec")
    a = ap.parse_args(argv)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    jobs = [(s, a.out, a.duration, a.hz, a.quality, None) for s in range(a.start, a.start + a.n)]
    t = time.time()
    with mp.get_context("spawn").Pool(a.workers) as pool:
        frames = sum(pool.imap_unordered(_job, jobs))
    print(f"recorded {a.n} episodes, {frames} frames in {time.time() - t:.0f}s -> {a.out} (families: {FAMILIES})")


if __name__ == "__main__":
    main()
