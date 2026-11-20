"""Run AOI's DynaCU-Bench harness with MiMo and optional removal of DOM prompt context.

AOI's registry has no MiMo alias, so we add one at run time instead of editing the checkout: the alias
`mimo` points AOI's `LocalVLLMModel(generic)` (an OpenAI-compatible client) at AGENT_BASE_URL / AGENT_MODEL,
with VLLM_API_KEY set from AGENT_API_KEY / OPENROUTER_API_KEY. The upstream checkout is not edited.
The optional --no-dom disables interactive-element extraction before it can reach the agent;
keyframes, visual narration, action execution and benchmark scoring retain their upstream paths.

  python -m nightwatch.dynacu_official --modes standard aoi_visual --task-prefix F- --output-dir runs/dynacu_official
"""

from __future__ import annotations

import argparse
import concurrent.futures
from contextlib import contextmanager, nullcontext
import json
import logging
import multiprocessing
import os
from pathlib import Path
import random
import sys
import time

from nightwatch.config import RUNS_DIR, env
from nightwatch.envs.dynacu import AOI_DIR, SUBSETS


@contextmanager
def pixel_only_context(evaluator, browser_module, audit_path):
    """Suppress the DOM prompt channel and audit actual model inputs, scoped to one episode.

    DOM-based benchmark scoring remains available to the evaluator, never to the agent.
    Each spawned worker runs one episode at a time, so this module binding is process-local.
    """
    original_env = browser_module.BrowserEnvironment
    original_model = evaluator.cu_model
    audit = {"dom_context": False, "suppressed_element_requests": 0, "model_calls": 0,
             "keyframe_images": 0, "dom_prompt_violations": 0}

    class PixelOnlyBrowser(original_env):
        def get_interactive_elements(self):
            audit["suppressed_element_requests"] += 1
            return ""  # Do not call the DOM extractor at all.

    def model(context_text, images, task):
        audit["model_calls"] += 1
        audit["keyframe_images"] += max(0, len(images) - 1)
        violation = "[PAGE ELEMENTS — interactive]" in context_text
        audit["dom_prompt_violations"] += int(violation)
        with audit_path.open("a") as stream:
            stream.write(json.dumps({"call": audit["model_calls"], "context_text": context_text,
                                     "task": task, "image_sizes": [list(im.size) for im in images],
                                     "dom_prompt_violation": violation}) + "\n")
        if violation:
            raise RuntimeError("DOM element context reached a pixel-only model request")
        return original_model(context_text, images, task)

    browser_module.BrowserEnvironment = PixelOnlyBrowser
    evaluator.cu_model = model
    try:
        yield audit
    finally:
        browser_module.BrowserEnvironment = original_env
        evaluator.cu_model = original_model


def _setup():
    os.chdir(AOI_DIR)
    sys.path.insert(0, str(AOI_DIR))
    import aoi.cu_model as cm

    base = env("AGENT_BASE_URL", "https://openrouter.ai/api/v1")
    key = env("AGENT_API_KEY") or (env("OPENROUTER_API_KEY") if "openrouter.ai" in base else env("OPENAI_API_KEY"))
    os.environ.setdefault("VLLM_API_KEY", key or "EMPTY")
    cm.LOCAL_MODELS["mimo"] = (env("AGENT_MODEL", "xiaomi/mimo-v2.6-flash"), "generic", base)


@contextmanager
def saved_screenshots(browser_module, directory):
    """Persist actual native observations and capture times; this is not a video."""
    original = browser_module.BrowserEnvironment
    directory.mkdir(parents=True, exist_ok=True)
    counter = 0

    class RecordedBrowser(original):
        def get_screenshot(self, *args, **kwargs):
            nonlocal counter
            shot = super().get_screenshot(*args, **kwargs)
            if shot is not None:
                stamp = time.time()
                name = f"{counter:05d}_{stamp:.6f}.png"
                counter += 1
                shot.save(directory / name)
                with (directory.parent / "screenshots.jsonl").open("a") as fh:
                    fh.write(json.dumps({"unix": stamp, "file": f"obs/{name}"}) + "\n")
            return shot

    browser_module.BrowserEnvironment = RecordedBrowser
    try:
        yield
    finally:
        browser_module.BrowserEnvironment = original


def _run_one(job):
    from nightwatch.runner import _graceful_signals
    _graceful_signals()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
    task_id, mode, rep, max_steps, output, no_dom = job
    _setup()
    from experiments import browser_eval
    from experiments.browser_eval import BrowserEvaluator, DynaCUBenchV3
    from nightwatch.dynacu_judge import checked_outcome, make_evaluator
    # Upstream defaults to CUDA even in its CPU-only environment. Choose the same
    # automatic device as our AOI adapter, without modifying CLIP or its thresholds.
    import torch
    from aoi.keyframe_extractor import KeyframeExtractor

    class AvailableDeviceKeyframes(KeyframeExtractor):
        def __init__(self, *args, **kwargs):
            kwargs.setdefault("device", "cuda" if torch.cuda.is_available() else "cpu")
            super().__init__(*args, **kwargs)

    browser_eval.KeyframeExtractor = AvailableDeviceKeyframes

    condition = f"{mode}_no_dom" if no_dom else mode
    path = Path(output) / condition / f"r{rep}" / task_id / "result.json"
    if path.exists():
        previous = json.loads(path.read_text())
        if not previous.get("error"):
            return previous
    evaluator = BrowserEvaluator(model_name="mimo", observation_mode=mode, max_steps=max_steps)
    judge = make_evaluator()
    evaluate = judge.evaluate_task
    evidence = {}

    def checked(*args, **kwargs):
        outcome = evaluate(*args, **kwargs)
        evidence.update(checked_outcome(outcome))
        evidence["agent_response"] = kwargs.get("agent_response", "")
        return outcome

    judge.evaluate_task = checked
    evaluator._llm_evaluator = judge
    task = DynaCUBenchV3(html_tasks_dir=AOI_DIR / "benchmark_env/html_tasks").get_task(task_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    audit_path = path.parent / "agent_inputs.jsonl"
    if no_dom:
        audit_path.write_text("")
    with saved_screenshots(browser_eval, path.parent / "obs"):
        context = pixel_only_context(evaluator, browser_eval, audit_path) if no_dom else nullcontext(None)
        with context as audit:
            result = evaluator.run_task(task).to_dict()
    result.update(repeat=rep, evaluation=evidence, condition=condition)
    if audit is not None:
        result["input_audit"] = audit
        if audit["dom_prompt_violations"]:
            result["error"] = "DOM context detected in pixel-only run"
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(result, indent=2))
    temporary.replace(path)
    print(f"[{condition} r{rep}] {task_id}: success={result['success']} error={result.get('error', '')}", flush=True)
    return result


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--modes", nargs="+", default=["standard", "aoi_visual"])
    ap.add_argument("--subset", default="visual", choices=list(SUBSETS))
    ap.add_argument("--task-prefix", default=None)
    ap.add_argument("--max-tasks", type=int, default=None)
    ap.add_argument("--max-steps", type=int, default=50)  # same budget as our harness (AOI default 15)
    ap.add_argument("--task-ids", default=None, help="comma-separated exact task IDs")
    ap.add_argument("--repeats", type=int, default=1)
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--no-dom", action="store_true", help="disable DOM element context; keep visual keyframes and narration")
    ap.add_argument("--output-dir", default=str(RUNS_DIR / "dynacu_official"))
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
    args.output_dir = str(Path(args.output_dir).resolve())
    _setup()  # AOI resolves benchmark_env/html_tasks relative to the repo root
    from experiments import browser_eval as be

    keep = SUBSETS[args.subset]
    ids = set(args.task_ids.split(",")) if args.task_ids else None
    tasks = [t for t in be.DynaCUBenchV3(html_tasks_dir=AOI_DIR / "benchmark_env/html_tasks")
             if keep(t) and (not args.task_prefix or t.task_id.startswith(args.task_prefix))
             and (ids is None or t.task_id in ids)]
    if args.max_tasks:
        tasks = tasks[:args.max_tasks]
    jobs = [(t.task_id, mode, rep, args.max_steps, args.output_dir, args.no_dom)
            for rep in range(args.repeats) for t in tasks for mode in args.modes]
    random.Random(0).shuffle(jobs)
    if args.workers == 1:
        results = [_run_one(job) for job in jobs]
    else:
        with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers,
                mp_context=multiprocessing.get_context("spawn")) as pool:
            results = list(pool.map(_run_one, jobs))
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    (Path(args.output_dir) / "all_results.json").write_text(json.dumps(results, indent=2))
    print(f"done: {sum(r['success'] for r in results)}/{len(results)} passed -> {args.output_dir}")
    if any(r.get("error") for r in results):
        raise SystemExit("Some episodes have infrastructure errors; rerun to retry them.")


if __name__ == "__main__":
    main()
