"""Settings from `.env` (see `.env.example`) and YAML experiment configs.

Real environment variables win over `.env`; a blank value means "use the default".
Experiment configs are YAML dicts; `load_config` merges `extends:` chains and `--set a.b=value`
overrides and returns a `Cfg` (a dict with attribute access).
"""

from __future__ import annotations

import copy
import json
import os
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]


def load_dotenv(path: Path = ROOT / ".env") -> None:
    if not path.exists():
        return
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip().strip('"').strip("'")
        if value:
            os.environ.setdefault(key.strip(), value)


load_dotenv()


def _ssl_certs() -> None:
    """The conda envs were moved when the project was renamed; OpenSSL keeps its compile-time CA path (the old
    prefix), so point it at the env's own bundle (or certifi) unless the user set SSL_CERT_FILE."""
    import sys

    if os.environ.get("SSL_CERT_FILE"):
        return
    for cand in (Path(sys.prefix) / "ssl" / "cert.pem",):
        if cand.exists():
            os.environ["SSL_CERT_FILE"] = str(cand)
            return
    try:
        import certifi

        os.environ["SSL_CERT_FILE"] = certifi.where()
    except ImportError:
        pass


_ssl_certs()


def _block_gpu_only_kernels() -> None:
    """NIGHTWATCH_NO_FLA=1 hides flash-linear-attention / causal-conv1d from imports (before transformers loads),
    so Qwen3.5 / d3 use the PyTorch reference kernels: the d3 runtime refuses CPU inference when the GPU-only
    kernels are importable. Set it for CPU runs (tests, SMOKE=1); leave it off on GPUs."""
    if os.environ.get("NIGHTWATCH_NO_FLA", "") != "1":
        return
    import importlib.abc
    import sys

    class _Block(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in ("fla", "causal_conv1d"):
                raise ModuleNotFoundError(name)
            return None

    sys.meta_path.insert(0, _Block())


_block_gpu_only_kernels()


def env(key: str, default: str = "") -> str:
    return (os.environ.get(key, "").strip() or default).strip()


DATA_DIR = Path(env("DATA_DIR") or ROOT / "data")
RUNS_DIR = Path(env("RUNS_DIR") or ROOT / "runs")
MODELS_DIR = DATA_DIR / "models"
OSWORLD_DIR = Path(env("OSWORLD_DIR") or ROOT / "third_party" / "OSWorld")
os.environ.setdefault("HF_HOME", str(DATA_DIR / "hf"))
for _k in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_k, env("NIGHTWATCH_CPU_THREADS", "8"))


def model_path(env_key: str, local_name: str, hub_id: str) -> str:
    """explicit env path > data/models/<local_name> > hub id (HF cache)."""
    explicit = env(env_key)
    if explicit:
        return explicit
    local = MODELS_DIR / local_name
    return str(local) if local.exists() else hub_id


class Cfg(dict):
    """Dict with attribute access; nested dicts are wrapped on access."""

    def __getattr__(self, key: str) -> Any:
        try:
            value = self[key]
        except KeyError as e:
            raise AttributeError(key) from e
        return Cfg(value) if isinstance(value, dict) and not isinstance(value, Cfg) else value

    def __setattr__(self, key: str, value: Any) -> None:
        self[key] = value

    def get(self, key: str, default: Any = None) -> Any:  # type: ignore[override]
        value = super().get(key, default)
        return Cfg(value) if isinstance(value, dict) and not isinstance(value, Cfg) else value

    def to_dict(self) -> dict:
        return json.loads(json.dumps(self, default=str))


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def set_path(d: dict, dotted: str, value: Any) -> None:
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = value


def resolve_config_path(name: str | Path) -> Path:
    p = Path(name)
    if p.exists():
        return p
    for cand in (ROOT / "configs" / f"{name}.yaml", ROOT / "configs" / str(name)):
        if cand.exists():
            return cand
    raise FileNotFoundError(f"config {name!r} not found (looked in {ROOT / 'configs'})")


def load_config(name: str | Path, overrides: list[str] | None = None) -> Cfg:
    path = resolve_config_path(name)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    parents = raw.pop("extends", None) or []
    if isinstance(parents, str):
        parents = [parents]
    merged: dict = {}
    for parent in parents:
        merged = deep_merge(merged, dict(load_config(parent)))
    merged = deep_merge(merged, raw)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"override {item!r} is not key=value")
        k, v = item.split("=", 1)
        set_path(merged, k.strip(), yaml.safe_load(v.strip()))
    merged["config_name"] = path.stem
    return Cfg(merged)


def method_config(cfg: Cfg, name: str) -> Cfg:
    """A method = defaults (cfg.method_defaults) <- cfg.methods[name] (which may `base:` another method)."""
    methods = cfg.get("methods", {})
    if name not in methods:
        raise KeyError(f"unknown method {name!r}; known: {', '.join(methods)}")
    chain, cur = [], name
    while cur:
        if cur in chain:
            raise ValueError(f"method base cycle at {cur}")
        chain.append(cur)
        cur = (methods[cur] or {}).get("base")
    out = dict(cfg.get("method_defaults", {}))
    for m in reversed(chain):
        out = deep_merge(out, {k: v for k, v in (methods[m] or {}).items() if k != "base"})
    out["name"] = name
    return Cfg(out)
