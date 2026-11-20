"""Stage the NightWatch release for the Hugging Face Hub: index files as they are, everything else in ~4 GB tar shards
(the Hub wants < 100k files per repo), the sandbox sudo password redacted from every text file.

    python scripts/lib/hf_pack.py data/hf_upload        # -> data/hf_upload/{model,data,trajectories}

Extracting every shard of a folder in that folder restores the original layout (`for t in *.tar; do tar xf $t; done`).
"""

from __future__ import annotations

import io
import os
import shutil
import sys
import tarfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
TEXT = {".json", ".jsonl", ".log", ".txt", ".md", ".csv", ".yaml", ".yml", ".py"}
SHARD = int(float(os.environ.get("SHARD_GB", "4")) * 2**30)


def _secrets() -> list[bytes]:
    out = []
    for line in (ROOT / ".env").read_text().splitlines():
        k, _, v = line.partition("=")
        if k.strip() in ("AGS_SUDO_PASSWORD",) and len(v.strip().strip("\"'")) >= 4:
            out.append(v.strip().strip("\"'").encode())
    return out


SECRETS = _secrets()


def _clean(p: Path) -> bytes | None:
    if p.suffix not in TEXT:
        return None
    b = p.read_bytes()
    for s in SECRETS:
        b = b.replace(s, b"<SUDO_PASSWORD>")
    return b


def copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    b = _clean(src)
    dst.write_bytes(b) if b is not None else shutil.copy2(src, dst)


def shard(src: Path, out: Path, prefix: str, skip=lambda rel: False) -> int:
    """Tar every file under src (paths relative to src) into out/prefix-0000.tar, ... of about SHARD bytes."""
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(p for p in src.rglob("*") if p.is_file() and not skip(p.relative_to(src)))
    i, size, tar = 0, 0, None
    for p in files:
        if tar is None or size >= SHARD:
            if tar:
                tar.close()
            tar, size = tarfile.open(out / f"{prefix}-{i:04d}.tar", "w"), 0
            i += 1
        rel = str(p.relative_to(src))
        b = _clean(p)
        if b is None:
            tar.add(p, arcname=rel, recursive=False)
            size += p.stat().st_size
        else:
            ti = tar.gettarinfo(p, arcname=rel)
            ti.size = len(b)
            tar.addfile(ti, io.BytesIO(b))
            size += len(b)
    if tar:
        tar.close()
    print(f"{src} -> {out}/{prefix}-*.tar: {len(files)} files, {i} shards", flush=True)
    return i


def main(stage: Path) -> None:
    pyc = lambda rel: "__pycache__" in rel.parts  # noqa: E731
    # model: the trained package (the base model's banner and card dropped) + training record
    m = stage / "model"
    pkg = ROOT / "data/ckpt/d3-lite-cua-sft/package"
    for p in sorted(pkg.rglob("*")):
        rel = p.relative_to(pkg)
        if p.is_file() and not pyc(rel) and rel.parts[0] != "assets" and str(rel) != "README.md":
            copy(p, m / rel)
    for f, to in [("data/ckpt/d3-lite-cua-sft/metrics.json", "training/val_metrics.json"),
                  ("data/ckpt/d3-lite-cua-sft/train_log.jsonl", "training/train_log.jsonl"),
                  ("data/ckpt/d3-lite-cua-sft.log", "training/sft.log")]:
        copy(ROOT / f, m / to)
    # data: desktop (teacher-labelled examples, task pool, raw episode streams) + SynthWeb (examples, recordings)
    d = stage / "data"
    for name in ("examples.jsonl", "stats.json"):
        copy(ROOT / "data/cua_train/ds" / name, d / "desktop" / name)
        copy(ROOT / "data/synthweb/ds" / name, d / "synthweb" / name)
    shard(ROOT / "data/cua_train/ds", d / "desktop", "images", lambda r: r.parts[0] != "images")
    shard(ROOT / "data/synthweb/ds", d / "synthweb", "images", lambda r: r.parts[0] != "img")
    copy(ROOT / "data/cua_train/train_tasks.json", d / "desktop/tasks/train_tasks.json")
    shard(ROOT / "data/cua_train/tasks", d / "desktop/tasks", "task_pool")
    shard(ROOT / "runs/cua_collect", d / "desktop/episodes", "episodes")
    copy(ROOT / "runs/cua_collect.log", d / "desktop/episodes/collect.log")
    shard(ROOT / "data/synthweb/rec", d / "synthweb/recordings", "recordings")
    for name in ("test_ds", "test_rec"):
        shard(ROOT / "data/synthweb" / name, d / f"synthweb/{name}", name)
    # trajectories: every OSWorld / DynaCU run (screenshots, 3 Hz frames, steps, LLM-call timestamps), metrics, logs
    t = stage / "trajectories"
    runs = ROOT / "runs"
    copy(ROOT / "configs/tasks/osworld_100.json", t / "osworld_100_tasks.json")
    for p in sorted(runs.iterdir()):
        if p.is_dir() and (p.name.startswith("osworld_") or p.name.startswith("dynacu_")):
            if p.name.endswith("_metrics") or p.name.endswith("_waste"):
                for f in sorted(p.rglob("*")):
                    if f.is_file():
                        copy(f, t / "metrics" / p.name / f.relative_to(p))
            else:
                shard(p, t / "runs" / p.name, p.name)
        elif p.is_file() and p.suffix in (".log", ".md") and (p.name.startswith("osworld_") or p.name.startswith("dynacu_")):
            copy(p, t / ("metrics" if p.suffix == ".md" else "logs") / p.name)


if __name__ == "__main__":
    main(Path(sys.argv[1]))
