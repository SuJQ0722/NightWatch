"""Experiment orchestration: tasks x methods x repeats, parallel environments, resume.

Output layout:  runs/<experiment>/<method>/r<repeat>/<domain>/<task_id>/{result.json, steps.jsonl,
system1.jsonl, frames/ (when recording)}. A job with an existing result.json is skipped.
Within one (repeat, task) the methods run back to back in a seeded random order on the same worker, so
API-latency drift over the day hits all methods alike (pairing).
"""

from __future__ import annotations

import json
import logging
import multiprocessing as mp
import os
import queue
import random
import signal
import threading
import time
import traceback
import zlib
from pathlib import Path

from nightwatch.config import RUNS_DIR, Cfg, env, method_config
from nightwatch.utils import jdump, jline

log = logging.getLogger("nightwatch.runner")


def make_tasks(cfg: Cfg) -> list[dict]:
    e = cfg.get("env", {})
    if e.get("kind") == "dynacu":
        from nightwatch.envs.dynacu import load_tasks

        return load_tasks(cfg.get("tasks", {}))
    if e.get("kind", "osworld") == "sim":
        n = int(cfg.get("tasks", {}).get("n", 4))
        return [{"id": f"sim-{i:03d}", "_domain": "sim", "steps": int(e.get("sim", {}).get("steps", 3))} for i in range(n)]
    from nightwatch.envs.osworld import load_tasks

    return load_tasks(cfg.get("tasks", {}))


def make_env(cfg: Cfg, worker: int, need_a11y: bool = False):
    e = cfg.get("env", {})
    if e.get("kind") == "dynacu":
        from nightwatch.envs.dynacu import DynaCUEnv

        return DynaCUEnv(e.get("dynacu", {}))
    if e.get("kind", "osworld") == "sim":
        from nightwatch.envs.sim_desktop import SimDesktop

        return SimDesktop(e.get("sim", {}), seed=int(cfg.get("seed", 0)))
    from nightwatch.envs.osworld import OSWorldEnv

    return OSWorldEnv({**e.get("osworld", {}), "frames_hz": cfg.get("frames", {}).get("hz")}, need_a11y=need_a11y)


def make_agent(cfg: Cfg, seed: int):
    a = cfg.get("agent", {})
    if a.get("kind") == "aoi":
        from nightwatch.agents.aoi_agent import AOIAgentAdapter

        return AOIAgentAdapter(a)
    if a.get("kind") == "scripted_osworld":
        from nightwatch.agents.scripted_osworld import ScriptedOSWorldAgent

        return ScriptedOSWorldAgent(float(a.get("latency_s", 2.0)))
    if a.get("kind") == "scripted_web":
        from nightwatch.agents.scripted_web import ScriptedWebAgent

        return ScriptedWebAgent(float(a.get("latency_s", 1.5)), seed=seed)
    if a.get("kind", "prompt") == "sim":
        from nightwatch.agents.sim_agent import SimAgent

        return SimAgent(float(a.get("latency_s", 1.2)), float(a.get("jitter", 0.3)), seed=seed,
                        cancellable=bool(a.get("cancellable", True)))
    if a.get("kind") == "qwen35":
        from nightwatch.agents.qwen35_agent import Qwen35AgentAdapter

        return Qwen35AgentAdapter(a)
    from nightwatch.agents.prompt_agent import PromptAgentAdapter

    from nightwatch.envs.osworld import client_password

    return PromptAgentAdapter(a, client_password=client_password(cfg.get("env", {}).get("osworld", {})))


def job_dir(root: Path, method: str, rep: int, task: dict) -> Path:
    return root / method / f"r{rep}" / task.get("_domain", "x") / str(task["id"])


def _graceful_signals() -> None:
    """SIGTERM / SIGINT end a run through the normal clean-up (`finally: env.close()` kills the sandbox). A process a
    script started in the background ignores SIGINT, and Python's default SIGTERM exits without running `finally`,
    which leaves cloud sandboxes billing until their timeout."""

    def stop(signum, frame):
        raise KeyboardInterrupt(f"signal {signum}")

    for s in (signal.SIGTERM, signal.SIGINT):
        signal.signal(s, stop)


def _worker(wid: int, cfg_d: dict, jobs, root: str, lock) -> None:
    from nightwatch.loop import Episode
    from nightwatch.system1 import Meter, make_system1
    from nightwatch.system1.base import Split

    _graceful_signals()
    logging.basicConfig(level=logging.INFO, format=f"%(asctime)s w{wid} %(name)s %(levelname)s %(message)s")
    cfg = Cfg(cfg_d)
    root_p = Path(root)
    env_obj = None
    s1_cache: dict = {}
    try:
        while True:
            try:
                job = jobs.get(timeout=2)
            except queue.Empty:
                break
            mname, rep, task = job
            task = dict(task, _rep=rep)
            out = job_dir(root_p, mname, rep, task)
            if (out / "result.json").exists():
                continue
            m = method_config(cfg, mname)
            if m.get("native_mode"):
                from nightwatch.dynacu_official import _run_one
                previous_cwd = os.getcwd()
                native_out = root_p / mname / f"r{rep}" / str(task["id"])
                native_out.mkdir(parents=True, exist_ok=True)
                jdump({"pid": os.getpid(), "unix": time.time()}, native_out / "active.json")
                try:
                    _run_one((task["id"], m["native_mode"], rep, cfg.limits.max_steps,
                              str(root_p), bool(m.get("native_no_dom"))))
                finally:
                    os.chdir(previous_cwd)
                continue
            if env_obj is None:
                env_obj = make_env(cfg, wid)
            backend = m.get("system1") or cfg.get("system1", {}).get("backend") or env("SYSTEM1_BACKEND", "http")
            fast = m.get("system1_fast")
            s1 = None
            if m.get("post") == "nightwatch" or m.get("interrupt") or m.get("invalidate"):
                if (backend, fast) not in s1_cache:
                    inner = make_system1(backend)
                    if inner is not None and fast:
                        fast_inner = make_system1(fast)
                        if fast_inner is not None and m.get("system1_fast_max_side") and hasattr(fast_inner, "max_side"):
                            fast_inner.max_side = int(m["system1_fast_max_side"])  # the size its head was trained at
                        inner = Split(inner, fast_inner) if fast_inner is not None else inner
                    if inner is not None:
                        try:
                            inner.warmup()
                        except Exception as e:  # no server: event-driven methods run pixel-only (logged)
                            logging.getLogger("nightwatch.runner").warning("System-1 %s unavailable: %s", backend, e)
                            inner = None
                    s1_cache[(backend, fast)] = inner
                s1 = Meter(s1_cache[(backend, fast)]) if s1_cache[(backend, fast)] is not None else None
            seed = (zlib.crc32(f"{task['id']}/{rep}".encode()) & 0xFFFF) + int(cfg.get("seed", 0))
            agent = make_agent(cfg, seed)
            if hasattr(agent, "bind_env"):
                agent.bind_env(env_obj)
            t0 = time.time()
            try:
                if m.get("episode_loop") == "osworld_qwen35_native_v1":
                    from nightwatch.qwen35_official import run_episode

                    res = run_episode(env_obj, agent, m, cfg, task, out)
                else:
                    res = Episode(env_obj, agent, s1, m, cfg, task, out).run()
            except Exception as e:
                res = {"task_id": task["id"], "method": mname, "error": repr(e), "traceback": traceback.format_exc()}
                if m.get("episode_loop") == "osworld_qwen35_native_v1":
                    out.mkdir(parents=True, exist_ok=True)
                jdump(res, out / "error.json")
                if m.get("episode_loop") == "osworld_qwen35_native_v1":
                    # No scored failure on transport/setup errors. Stop this worker;
                    # resume archives the incomplete native attempt before retrying.
                    raise
            with lock:
                with open(root_p / "progress.jsonl", "a", encoding="utf-8") as fh:
                    jline(fh, {"t": time.strftime("%H:%M:%S"), "worker": wid, "method": mname, "repeat": rep,
                               "task": task["id"], "success": res.get("success"), "loop_s": res.get("loop_wall_s"),
                               "steps": res.get("steps"), "error": res.get("error", ""), "wall": round(time.time() - t0, 1)})
            print(f"[w{wid}] {mname} r{rep} {task.get('_domain')}/{task['id']}: success={res.get('success')} "
                  f"loop={res.get('loop_wall_s')}s steps={res.get('steps')} {res.get('error', '')}", flush=True)
            if res.get("env_broken") and env_obj is not None:  # a call into it never returned: rebuild for the next job
                closer = threading.Thread(target=env_obj.close, daemon=True)
                closer.start()
                closer.join(120)
                env_obj = None
    finally:
        if env_obj is not None:
            env_obj.close()


def run_experiment(cfg: Cfg, methods: list[str] | None = None, repeats: int | None = None,
                   workers: int | None = None, name: str | None = None) -> Path:
    _graceful_signals()
    methods = methods or list(cfg.get("run_methods") or [])
    if not methods:
        raise ValueError("no methods (config run_methods or --methods)")
    for m in methods:
        method_config(cfg, m)  # fail early on unknown names
    repeats = int(repeats or cfg.get("repeats", 1))
    workers = int(workers or cfg.get("workers", 1))
    root = RUNS_DIR / (name or cfg.get("experiment") or cfg.config_name)
    native_qwen = any(method_config(cfg, m).get("episode_loop") == "osworld_qwen35_native_v1" for m in methods)
    if native_qwen:
        from nightwatch.qwen35_official import validate_config

        validate_config(cfg, [method_config(cfg, m) for m in methods], root)
    root.mkdir(parents=True, exist_ok=True)
    tasks = make_tasks(cfg)
    rng = random.Random(int(cfg.get("seed", 0)))
    jobs = []
    for rep in range(repeats):
        for t in tasks:
            order = list(methods)
            rng.shuffle(order)
            jobs.extend((m, rep, t) for m in order)
    def result_path(job):
        m, rep, task = job
        if method_config(cfg, m).get("native_mode"):
            return root / m / f"r{rep}" / str(task["id"]) / "result.json"
        return job_dir(root, m, rep, task) / "result.json"

    todo = [j for j in jobs if not result_path(j).exists()]
    manifest = {"config": cfg.to_dict(), "methods": methods, "repeats": repeats, "n_tasks": len(tasks),
                "agent_model": env("AGENT_MODEL", "xiaomi/mimo-v2.6-flash"), "agent_base_url": env("AGENT_BASE_URL"),
                "system1_backend": cfg.get("system1", {}).get("backend") or env("SYSTEM1_BACKEND", "http"),
                "started": time.strftime("%Y-%m-%d %H:%M:%S"), "host": os.uname().nodename}
    jdump(manifest, root / "manifest.json")
    print(f"[nightwatch run] {root.name}: {len(tasks)} tasks x {len(methods)} methods x {repeats} repeats = {len(jobs)} "
          f"episodes ({len(todo)} to run) on {workers} worker(s)", flush=True)
    if not todo:
        return root
    ctx = mp.get_context("spawn")
    q = ctx.Queue()
    # The feeder may still hold hundreds of queued jobs after a supervised stop.
    # Joining it with no consumers deadlocks interpreter shutdown; workers own cleanup.
    q.cancel_join_thread()
    for j in todo:
        q.put(j)
    lock = ctx.Lock()
    if workers == 1:
        _worker(0, cfg.to_dict(), q, str(root), lock)
    else:
        procs = [ctx.Process(target=_worker, args=(w, cfg.to_dict(), q, str(root), lock)) for w in range(workers)]
        for p in procs:
            p.start()
            time.sleep(float(cfg.get("worker_stagger_s", 5)))  # stagger container starts (port allocation lock)
        for p in procs:
            p.join()
        if native_qwen and any(p.exitcode != 0 for p in procs):
            raise RuntimeError("Native Qwen worker failed; inspect error.json and resume after recovery")
    return root


def load_results(root: Path) -> list[dict]:
    out = []
    for p in Path(root).rglob("result.json"):
        try:
            r = json.loads(p.read_text())
        except json.JSONDecodeError:
            continue
        rel = p.relative_to(root).parts
        r["_method_dir"], r["_repeat"] = rel[0], int(rel[1][1:]) if rel[1].startswith("r") else 0
        r["_dir"] = str(p.parent)
        out.append(r)
    return out
