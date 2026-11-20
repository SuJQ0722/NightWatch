"""Shared helpers for the analysis commands."""

from __future__ import annotations

from pathlib import Path

from nightwatch.config import RUNS_DIR


def exp_root(exp: str) -> Path:
    p = Path(exp)
    if p.exists():
        return p
    p = RUNS_DIR / exp
    if not p.exists():
        raise FileNotFoundError(f"no experiment folder {exp!r} (looked in {RUNS_DIR})")
    return p


def episodes(root: Path) -> list[Path]:
    """Episode folders (those with a steps.jsonl) under an experiment."""
    return sorted(p.parent for p in Path(root).rglob("steps.jsonl"))
