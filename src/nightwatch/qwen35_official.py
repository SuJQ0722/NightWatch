"""Run Qwen's baseline through the branch's actual OSWorld episode function.

The local run_single_example is identical to dev_djlu/qwen35vl_agent at
bfd62bdc5a3319809a236dc90ddbbaf3cb4b7e06. No NIGHTWATCH Episode is involved.
Only transport (Qwen35AgentAdapter) and screenshot-only recording are adapted.
Native traj.jsonl/result.txt are retained; calls.jsonl also records observations
on empty-action turns, which the upstream action trajectory does not capture.
"""

from __future__ import annotations

import ast
import hashlib
import inspect
import json
import logging
import threading
import time
from pathlib import Path
from types import SimpleNamespace

from nightwatch.utils import jdump, jline

UPSTREAM_COMMIT = "bfd62bdc5a3319809a236dc90ddbbaf3cb4b7e06"
LOOP_SHA256 = "03852c3ba40837a0e5eebee8fdb624cb3c1373e3a67a856462e98ac274967574"
PROTOCOL = "osworld_qwen35_native_v1"


def upstream_loop():
    from lib_run_single import run_single_example

    # Fail before spending API/VM resources if the shared upstream loop changes.
    source = inspect.getsource(run_single_example)
    node = ast.parse(source).body[0]
    digest = hashlib.sha256(ast.get_source_segment(source, node).encode()).hexdigest()
    if digest != LOOP_SHA256:
        raise ValueError("OSWorld run_single_example changed; review Qwen branch parity before running")
    return run_single_example


def validate_config(cfg, methods, root: Path) -> None:
    """Keep native runs separate from the earlier NIGHTWATCH-loop Qwen results."""
    for method in methods:
        if method.get("episode_loop") != PROTOCOL:
            continue
        if cfg.get("env", {}).get("kind") != "osworld" or cfg.get("agent", {}).get("kind") != "qwen35":
            raise ValueError("Qwen native loop requires env.kind=osworld and agent.kind=qwen35")
        if method.get("post") != "fixed" or any(method.get(k) for k in ("interrupt", "invalidate", "strip_sleep")):
            raise ValueError("Native Qwen baseline cannot apply NIGHTWATCH interventions")
        if not cfg.get("frames", {}).get("save_obs") or cfg.get("frames", {}).get("record"):
            raise ValueError("Qwen experiments require save_obs=true and record=false")
        upstream_loop()
        for result in (root / method["name"]).glob("r*/*/*/result.json"):
            if json.loads(result.read_text()).get("protocol") != PROTOCOL:
                raise ValueError(f"Refusing to mix native and previous-loop results: {result}")


class _NoVideoController:
    def __init__(self, controller):
        self._controller = controller

    def __getattr__(self, name):
        return getattr(self._controller, name)

    def start_recording(self):
        pass

    def end_recording(self, *args):
        pass


class _NativeEnv:
    def __init__(self, desktop):
        self._desktop = desktop
        self.controller = _NoVideoController(desktop.controller)
        self.actions = 0
        self.done = False
        self.failed = False

    def __getattr__(self, name):
        return getattr(self._desktop, name)

    def step(self, action, pause):
        # Deliberately call DesktopEnv.step, not OSWorldEnv.execute.
        result = self._desktop.step(action, pause)
        self.actions += 1
        self.done = bool(result[2])
        self.failed = bool(result[3].get("fail")) or action == "FAIL"
        return result


class _NativeAgent:
    def __init__(self, adapter, out: Path):
        self.adapter, self.out = adapter, out
        self.calls = 0
        self.model_seconds = 0.0

    def reset(self, logger=None, vm_ip=None):
        self.adapter.reset(vm_ip=vm_ip)

    def predict(self, instruction, obs):
        self.calls += 1
        stamp = time.time()
        name = f"{self.calls:03d}_{stamp:.6f}.png"
        (self.out / "obs" / name).write_bytes(obs["screenshot"])
        prediction = self.adapter.predict(instruction, obs, threading.Event())
        self.model_seconds += prediction.seconds
        with (self.out / "calls.jsonl").open("a") as f:
            jline(f, {"step": self.calls, "observation_unix": stamp, "obs_file": "obs/" + name,
                      "return_unix": time.time(), "seconds": prediction.seconds,
                      "response": prediction.response, "actions": prediction.actions,
                      "retries": prediction.retries, "http": prediction.extra.get("http", []),
                      **({"images_capped": prediction.extra["images_capped"]} if prediction.extra.get("images_capped") else {})})
        return prediction.response, prediction.actions


def run_episode(env, agent, method, cfg, task, out: Path) -> dict:
    loop = upstream_loop()
    out = Path(out)
    # Preserve unfinished attempts instead of appending a restarted trajectory.
    if out.exists() and any(p.name != "active.json" for p in out.iterdir()):
        archive = out.with_name(out.name + f".incomplete-{time.time_ns()}")
        out.rename(archive)
    (out / "obs").mkdir(parents=True)
    native_env = _NativeEnv(env.env)
    native_agent = _NativeAgent(agent, out)
    args = SimpleNamespace(sleep_after_execution=float(method.get("sleep", 0)), result_dir=str(out))
    ep = agent.agent.endpoint
    jdump({"protocol": PROTOCOL, "upstream_commit": UPSTREAM_COMMIT, "loop_sha256": LOOP_SHA256,
           "max_steps": int(cfg["limits"]["max_steps"]), "sleep_after_execution": args.sleep_after_execution,
           "settle_after_reset_s": 60, "settle_before_eval_s": 20,
           "model": ep.model, "max_tokens": agent.agent.max_tokens,
           "temperature": agent.agent.temperature, "top_p": agent.agent.top_p,
           "extra_body": ep.extra_body, "native_tools": agent.agent.native_tools,
           "max_images": agent.agent.max_images, "http_attempts": ep.retries,
           "frames": {"save_obs": True, "record": False},
           "trajectory_format": "OSWorld traj.jsonl; calls.jsonl includes input observations"}, out / "protocol.json")
    scores = []
    started = time.time()
    log = logging.getLogger(f"desktopenv.example.{task['id']}")
    old_handlers = set(log.handlers)
    try:
        loop(native_agent, native_env, task, int(cfg["limits"]["max_steps"]), task["instruction"],
             args, str(out), scores)
    finally:
        for handler in set(log.handlers) - old_handlers:
            log.removeHandler(handler)
            handler.close()
    score = float(scores[-1])
    result = {"protocol": PROTOCOL, "task_id": task["id"], "domain": task.get("_domain", ""),
              "method": method["name"], "repeat": task.get("_rep", 0), "agent": agent.name,
              "score": score, "success": score >= 1.0, "error": "", "steps": native_agent.calls,
              "actions": native_env.actions, "vlm_calls": native_agent.calls,
              "end_reason": "fail" if native_env.failed else "done" if native_env.done else "max_steps",
              "wall_s": time.time() - started, "model_s": native_agent.model_seconds,
              "timing_note": "Native wall_s includes reset, 60/20s waits and scoring; not NIGHTWATCH loop_wall_s"}
    jdump(result, out / "result.json")
    return result


def summarize_scores(root: Path) -> dict:
    """Report both OSWorld's mean score and the binary completion rate."""
    groups = {}
    for path in sorted(Path(root).glob("*/r*/*/*/result.json")):
        if ".incomplete-" in path.parent.name:
            continue
        result = json.loads(path.read_text())
        key = f"{result['method']}/r{result['repeat']}"
        groups.setdefault(key, []).append(result)
    return {key: {"completed": len(rows), "successes": sum(r.get("success", False) for r in rows),
                  "mean_score_percent": 100 * sum(r.get("score", 0) for r in rows) / len(rows),
                  "success_percent": 100 * sum(r.get("success", False) for r in rows) / len(rows),
                  "errors": sum(bool(r.get("error")) for r in rows)}
            for key, rows in groups.items()}
