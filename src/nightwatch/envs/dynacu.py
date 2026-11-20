"""DynaCU-Bench (AOI, arXiv 2606.29472) as an NightWatch environment.

The benchmark code is used unmodified from the pinned checkout third_party/aoi: AOI's
`BrowserEnvironment` (headless Chromium via Playwright, its action grammar and DOM success check),
`DynaCUBenchV3` (tasks) and `LLMEvaluator` (hybrid / LLM-judged tasks).

Playwright's sync API is bound to the thread that started it, while NightWatch grabs frames from a
background thread. So one actor thread owns the browser and every call (screenshot, action, DOM
check) is sent to it through a queue. Calls are therefore serialised, as they would be on one
Chromium page anyway.

Differences from AOI's own loop (experiments/browser_eval.py `standard` mode), shared by every method:
* the task's real-time limit (`duration_s`) bounds the loop, as in AOI;
* success is checked after every step from `window.getTaskResult()` (AOI does the same after a 0.3 s
  pause; we check right after the waiter, which already waited);
* with `native_wait` (fixed-sleep methods) a `wait()` action sleeps inside the environment exactly as in
  AOI (0.5-2 s); event-driven methods treat it as a no-op and let their waiter decide when to look.
Optional `gt_probe` injects a read-only probe (MutationObserver, running animations, visible dialogs /
toasts / spinners) whose output travels with each frame as weak ground truth for analysis.
"""

from __future__ import annotations

import logging
import queue
import sys
import threading
import time

from nightwatch.config import ROOT, env
from nightwatch.envs.base import Env

log = logging.getLogger("nightwatch.dynacu")
AOI_DIR = ROOT / "third_party" / "aoi"

GT_PROBE = r"""
(() => {
  if (window.__nightwatchProbe) return;
  window.__nightwatchProbe = {mut: 0};
  const start = () => new MutationObserver(m => { window.__nightwatchProbe.mut += m.length; })
      .observe(document.documentElement, {subtree: true, childList: true, attributes: true, characterData: true});
  if (document.documentElement) start(); else document.addEventListener('DOMContentLoaded', start);
  window.__nightwatchGT = () => {
    const vis = el => { const r = el.getBoundingClientRect(); const s = getComputedStyle(el);
      return r.width > 4 && r.height > 4 && s.visibility !== 'hidden' && s.display !== 'none' && +s.opacity > 0.05; };
    const q = sel => Array.from(document.querySelectorAll(sel)).filter(vis);
    const cls = re => Array.from(document.querySelectorAll('body *')).filter(e => re.test(e.className && e.className.baseVal === undefined ? e.className : '') && vis(e));
    const box = el => { const r = el.getBoundingClientRect(); return [r.left, r.top, r.right, r.bottom].map(Math.round); };
    const dialogs = q('[role=dialog],[role=alertdialog],dialog[open],.modal,.popup,.overlay').concat(cls(/modal|popup|dialog|overlay|consent/i));
    const toasts = q('[role=alert],[role=status],.toast,.snackbar,.notification').concat(cls(/toast|snackbar|notif|banner/i));
    const spinners = q('[aria-busy=true],.spinner,.loading,.loader,progress').concat(cls(/spinner|loading|loader|progress/i));
    const anims = document.getAnimations ? document.getAnimations().filter(a => a.playState === 'running').length : 0;
    const videos = Array.from(document.querySelectorAll('video')).filter(v => !v.paused && !v.ended).length;
    const p = window.__nightwatchProbe; const mut = p.mut; p.mut = 0;
    return {dialog: dialogs.length > 0, dialog_boxes: dialogs.slice(0, 4).map(box), toast: toasts.length > 0,
            toast_boxes: toasts.slice(0, 4).map(box), loading: spinners.length > 0, animations: anims,
            videos: videos, mutations: mut, url: location.href, title: document.title};
  };
})();
"""


def _aoi_imports():
    if str(AOI_DIR) not in sys.path:
        sys.path.insert(0, str(AOI_DIR))
    from benchmark_env.browser_env import BrowserEnvironment  # noqa: F401
    from dynacubench.tasks_v3 import DynaCUBenchV3  # noqa: F401

    return BrowserEnvironment, DynaCUBenchV3


SUBSETS = {
    "visual": lambda t: t.category.value != "S_static" and "audio" not in [a.value for a in t.axes],
    "static": lambda t: t.category.value == "S_static",
    "audio": lambda t: "audio" in [a.value for a in t.axes],
    "dynamic": lambda t: t.category.value != "S_static",
    "all": lambda t: True,
}


def load_tasks(tcfg) -> list[dict]:
    _, Bench = _aoi_imports()
    bench = Bench(html_tasks_dir=AOI_DIR / "benchmark_env" / "html_tasks")
    subset = tcfg.get("subset", "visual")
    keep = SUBSETS[subset]
    ids = set(tcfg.get("ids") or [])
    cats = tcfg.get("categories")
    out = []
    for t in bench:
        if not keep(t) or (ids and t.task_id not in ids) or (cats and t.category.value[0] not in cats):
            continue
        out.append({"id": t.task_id, "_domain": t.category.value, "instruction": t.instruction,
                    "duration_s": float(t.duration_s), "html_file": t.html_file, "difficulty": t.difficulty.value,
                    "eval_type": t.eval_type.value, "axes": [a.value for a in t.axes]})
    if tcfg.get("limit"):
        out = out[: int(tcfg.get("limit"))]
    return out


class DynaCUEnv(Env):
    name = "dynacu"
    action_space = "aoi"

    def __init__(self, dcfg=None):
        dcfg = dict(dcfg or {})
        self.BrowserEnvironment, Bench = _aoi_imports()
        self.bench = Bench(html_tasks_dir=AOI_DIR / "benchmark_env" / "html_tasks")
        w, h = dcfg.get("screen_size", [1280, 720])
        self.screen_size = (int(w), int(h))
        self.gt_probe = bool(dcfg.get("gt_probe", False))
        self.judge = dcfg.get("judge") or env("DYNACU_JUDGE", "gemini-2.0-flash")
        self.native_wait = True
        self.vm_ip = "chromium"
        self._q: queue.Queue = queue.Queue()
        self._thread: threading.Thread | None = None
        self._env = None
        self.task = None
        self.t_start = None

    # ------------------------------------------------------------------ actor
    def _loop(self) -> None:
        while True:
            fn, box, done = self._q.get()
            if fn is None:
                done.set()
                return
            try:
                box["v"] = fn()
            except Exception as e:  # report to the caller
                box["e"] = e
            done.set()

    def _call(self, fn, timeout: float = 120.0):
        if self._thread is None:
            self._thread = threading.Thread(target=self._loop, daemon=True, name="chromium-actor")
            self._thread.start()
        box: dict = {}
        done = threading.Event()
        self._q.put((fn, box, done))
        if not done.wait(timeout):
            raise TimeoutError("browser call timed out")
        if "e" in box:
            raise box["e"]
        return box.get("v")

    # ------------------------------------------------------------------ Env api
    def reset(self, task: dict) -> None:
        self.task = task
        t = self.bench.get_task(task["id"])
        self._task_obj = t

        def start():
            if self._env is not None:
                self._env.stop()
            e = self.BrowserEnvironment(html_file=t.html_file, width=self.screen_size[0], height=self.screen_size[1],
                                        task_timeout_s=t.duration_s + 10, audio_enabled=False)
            if not e.start():
                raise RuntimeError(f"browser failed to start for {t.task_id}")
            if self.gt_probe:
                e._page.evaluate(GT_PROBE)
            self._env = e
            return True

        self._call(start)
        self.t_start = time.time()

    def instruction(self, task: dict) -> str:
        return task["instruction"]

    @property
    def time_limit(self) -> float | None:
        return float(self.task["duration_s"]) if self.task else None

    def elapsed(self) -> float:
        return time.time() - self.t_start if self.t_start else 0.0

    def grab(self):
        def shot():
            data = self._env._page.screenshot(type="png")
            gt = None
            if self.gt_probe:
                try:
                    gt = self._env._page.evaluate("window.__nightwatchGT ? window.__nightwatchGT() : null")
                except Exception:
                    gt = None
            return data, gt

        return self._call(shot, timeout=30)

    def page_elements(self) -> str:
        return self._call(lambda: self._env.get_interactive_elements())

    def execute(self, action) -> dict:
        from nightwatch.actions import parse_aoi_action

        act = parse_aoi_action(action)
        if act.kind == "done":
            return {"done": True}
        if act.kind == "wait" and not self.native_wait:
            return {"wait": True}
        res = self._call(lambda: self._env.execute_action(str(action)), timeout=180)  # AOI fill() may wait 2 x 30 s
        return {"ok": bool(res.success), "error": res.error or ""}

    def check_done(self) -> bool:
        if self._env is None:
            return False
        _aoi_imports()
        from experiments.browser_eval import validate_dom_result

        _, val = self._call(lambda: self._env.check_success())
        self._last_dom = val
        return bool(validate_dom_result(self._task_obj, val))

    def evaluate(self) -> float:
        from nightwatch.dynacu_judge import checked_outcome, make_evaluator

        _, final = self._call(lambda: self._env.check_success())
        response = self._call(lambda: self._env.get_page_text())
        out = make_evaluator(self.judge).evaluate_task(self._task_obj, dom_result=final, agent_response=response)
        self.last_eval = checked_outcome(out)
        self.last_eval["agent_response"] = response
        # AOI's hybrid pass threshold is 0.5; the generic episode loop expects a binary score.
        return float(out.final_passed)

    def close(self) -> None:
        try:
            if self._env is not None:
                self._call(lambda: self._env.stop())
        except Exception:
            pass
        if self._thread is not None:
            done = threading.Event()
            self._q.put((None, {}, done))
            done.wait(5)
