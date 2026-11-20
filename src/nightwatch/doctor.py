"""`nightwatch doctor` (what is installed / reachable) and `nightwatch ping` (one real call to each model)."""

from __future__ import annotations

import os
import shutil
import subprocess
import sys
import threading
import time

from nightwatch.config import DATA_DIR, MODELS_DIR, OSWORLD_DIR, ROOT, env

OSWORLD_COMMIT = "044bc33901cb893808e3f91ebc9a17ed8e54b44c"


def _row(ok: bool | None, name: str, detail: str = "") -> bool:
    tag = {True: "\033[32m ok \033[0m", False: "\033[31mFAIL\033[0m", None: "\033[33m -- \033[0m"}[ok]
    print(f"[{tag}] {name:28s} {detail}")
    return ok is not False


def doctor() -> int:
    bad = 0
    print(f"NightWatch at {ROOT}  (python {sys.version.split()[0]}: {sys.executable})")
    # ---- OSWorld checkout and client code
    if (OSWORLD_DIR / "desktop_env").exists():
        head = ""
        if (OSWORLD_DIR / ".git-upstream").exists():
            genv = {**os.environ, "GIT_DIR": str(OSWORLD_DIR / ".git-upstream"), "GIT_WORK_TREE": str(OSWORLD_DIR)}
            head = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, env=genv).stdout.strip()
            bad += not _row(head == OSWORLD_COMMIT, "OSWorld checkout", f"{head[:12]} (pinned {OSWORLD_COMMIT[:12]})")
        elif (OSWORLD_DIR / ".git").exists():
            head = subprocess.run(["git", "-C", str(OSWORLD_DIR), "rev-parse", "HEAD"], capture_output=True, text=True).stdout.strip()
            bad += not _row(head == OSWORLD_COMMIT, "OSWorld checkout", f"{head[:12]} (pinned {OSWORLD_COMMIT[:12]})")
        else:
            _row(None, "OSWorld checkout", f"vendored (upstream {OSWORLD_COMMIT[:12]}); no local .git-upstream")
    else:
        bad += not _row(False, "OSWorld checkout", f"{OSWORLD_DIR} missing")
    try:
        import desktop_env.desktop_env  # noqa: F401
        import mm_agents.agent  # noqa: F401

        _row(True, "OSWorld import", "desktop_env + mm_agents")
    except Exception as e:
        bad += not _row(False, "OSWorld import", f"{type(e).__name__}: {str(e)[:120]} (use .conda/envs/nightwatch/bin/python)")
    # ---- VM and container runtime
    vm = DATA_DIR / "vm" / "Ubuntu.qcow2"
    bad += not _row(vm.exists(), "OSWorld VM image", f"{vm} ({vm.stat().st_size / 2**30:.1f} GB)" if vm.exists()
                    else f"{vm} missing: bash scripts/setup_container.sh vm")
    provider = env("OSWORLD_PROVIDER", "docker")
    if provider == "docker":
        try:
            import docker

            c = docker.from_env()
            c.ping()
            imgs = [t for i in c.images.list() for t in i.tags if "osworld-docker" in t]
            _row(True, "docker daemon", f"reachable; image: {imgs or 'not pulled (setup_container.sh pull)'}")
        except Exception as e:
            bad += not _row(False, "docker daemon", f"{str(e)[:100]} -> add the user to the docker group")
        # the docker daemon (root) passes /dev/kvm into the container, so the device must exist on the host;
        # the user's own access to it does not matter for the docker provider
        kvm = os.path.exists("/dev/kvm")
        bad += not _row(kvm, "/dev/kvm", "present (passed to the container by the docker daemon)" if kvm else
                        "missing: the VM would run without KVM, far too slow for wall-clock numbers")
    else:
        _row(None, "provider", f"{provider} (cloud credentials are checked by OSWorld at start)")
    # ---- models and System-1
    ms = sorted(p.name for p in MODELS_DIR.glob("Intern-Decision-*")) if MODELS_DIR.exists() else []
    bad += not _row(bool(ms), "Intern-Decision weights", ", ".join(ms) or "none: bash scripts/setup_env.sh models")
    url = env("S1_URL", "http://127.0.0.1:8710")
    try:
        import httpx

        info = httpx.get(url + "/health", timeout=3).json()
        _row(True, "perception server", f"{url}: {info}")
    except Exception:
        _row(None, "perception server", f"{url} not running (bash scripts/serve_system1.sh)")
    if shutil.which("nvidia-smi"):
        q = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.used,memory.total", "--format=csv,noheader,nounits"],
                           capture_output=True, text=True).stdout.strip().splitlines()
        free = [(int(i), int(t) - int(u)) for i, u, t in (r.split(", ") for r in q)]
        best = max(free, key=lambda t: t[1]) if free else None
        _row(bool(best and best[1] > 4000), "GPU for System-1",
             f"best free: cuda:{best[0]} {best[1] / 1024:.1f} GB" if best else "no GPU")
    # ---- agent
    key = env("AGENT_API_KEY") or (env("OPENROUTER_API_KEY") if "openrouter" in env("AGENT_BASE_URL", "openrouter") else "")
    bad += not _row(bool(key), "agent API key", f"{env('AGENT_MODEL', 'xiaomi/mimo-v2.6-flash')} @ "
                    f"{env('AGENT_BASE_URL', 'https://openrouter.ai/api/v1')}" + ("" if key else " (no key in .env)"))
    du = shutil.disk_usage(DATA_DIR if DATA_DIR.exists() else ROOT)
    _row(du.free > 60 * 2**30, "disk", f"{du.free / 2**30:.0f} GB free under {DATA_DIR}")
    print(f"\n{bad} blocking problem(s).")
    return 1 if bad else 0


def ping(agent: bool = True, s1: bool = True) -> int:
    rc = 0
    if agent:
        from nightwatch.agents.llm import CallStats, Cancelled, Endpoint, stream_chat

        ep = Endpoint.from_env("AGENT")
        st = CallStats()
        t = time.perf_counter()
        try:
            txt = stream_chat(ep, [{"role": "user", "content": "Reply with the single word OK."}], None, st, max_tokens=256)
            print(f"agent {ep.model}: {txt.strip()[:60]!r} in {time.perf_counter() - t:.2f}s (first token "
                  f"{st.t_first_token and round(st.t_first_token, 2)}s, tokens in/out {st.input_tokens:.0f}/{st.output_tokens:.0f})")
            cancel, st2 = threading.Event(), CallStats()
            threading.Timer(1.0, cancel.set).start()
            t = time.perf_counter()
            try:
                stream_chat(ep, [{"role": "user", "content": "Count from 1 to 400, one number per line."}], cancel, st2,
                            max_tokens=2000)
                print("agent cancel: generation finished before the 1 s cancel (inconclusive)")
            except Cancelled as c:
                print(f"agent cancel: stopped after {time.perf_counter() - t:.2f}s with {len(c.partial)} chars streamed: ok")
        except Exception as e:
            print(f"agent call FAILED: {e}")
            rc = 1
    if s1:
        import numpy as np

        from nightwatch.questions import BANK, build_request
        from nightwatch.system1 import make_system1

        try:
            b = make_system1()
            b.warmup()
            img = np.full((540, 960, 3), 235, np.uint8)
            img2 = img.copy()
            img2[200:340, 300:660] = 255
            img2[200:204, 300:660] = 40
            req = build_request("reasoning", {"seen": img, "now": img2, "crop": img2[180:360, 280:680]}, BANK, "test")
            for _ in range(3):
                a = b.ask(req)
            print(f"System-1 {b.name}: {a.ms:.0f} ms for {len(req.questions)} questions; fire={a.to_dict()['fire']}")
        except Exception as e:
            print(f"System-1 call FAILED: {e}")
            rc = 1
    return rc
