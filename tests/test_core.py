import numpy as np
import pytest

from nightwatch.actions import parse_action
from nightwatch.analysis.calibrate import ltt_threshold
from nightwatch.analysis.labels import question_label
from nightwatch.config import load_config, method_config
from nightwatch.diff import ChangeDetector
from nightwatch.questions import BANK, ROLES, bank_subset, build_request
from nightwatch.rules import Rules
from nightwatch.system1.base import to_answers
from nightwatch.utils import small_gray


def test_diff_ignores_tiny_and_benign_changes():
    a = np.full((1080, 1920, 3), 200, np.uint8)
    det = ChangeDetector((1920, 1080), benign_regions=[[0.44, 0.0, 0.56, 0.035]])
    b = a.copy()
    b[500:502, 900:903] = 0  # caret blink
    assert not det.compare(small_gray(a), small_gray(b)).significant
    c = a.copy()
    c[5:30, 900:1000] = 0  # top-bar clock
    assert not det.compare(small_gray(a), small_gray(c)).significant
    d = a.copy()
    d[400:600, 800:1100] = 30  # a dialog
    ch = det.compare(small_gray(a), small_gray(d))
    assert ch.significant and ch.near(950, 500, 0)
    x0, y0, x1, y1 = ch.union
    assert 780 <= x0 <= 800 and 1100 <= x1 <= 1120 and 380 <= y0 <= 400


def test_parse_actions():
    a = parse_action("import pyautogui\npyautogui.click(812, 344)")
    assert a.kind == "click" and a.has_target and (a.x, a.y) == (812, 344)
    a = parse_action("pyautogui.moveTo(10, 20)\npyautogui.click()")
    assert a.kind == "click" and a.x == 10
    a = parse_action("pyautogui.click(x=5.5, y=7)")
    assert a.x == 5.5 and a.y == 7
    assert parse_action("pyautogui.typewrite('hello world')").kind == "type"
    assert parse_action("pyautogui.hotkey('ctrl', 's')").text == "ctrl+s"
    assert parse_action("DONE").kind == "done" and parse_action("WAIT").special
    assert parse_action("pyautogui.scroll(-5)").kind == "scroll"


def test_bank_and_requests_fit_intern_decision():
    assert len(BANK) == 16
    img = np.zeros((10, 10, 3), np.uint8)
    for cp, roles in ROLES.items():
        req = build_request(cp, {r: img for r, _ in roles}, BANK, "a task", "click at (1, 2)")
        assert 0 < len(req.questions) <= 16 and len(req.images) <= 8
        for _, text in req.questions:
            assert "{" not in text  # every slot filled
    assert [q.id for q in bank_subset(4)] == ["dialog", "target_present", "effect", "settled"]


def test_rules_and_fire_semantics():
    img = np.zeros((10, 10, 3), np.uint8)
    req = build_request("pre_exec", {r: img for r, _ in ROLES["pre_exec"]}, BANK)
    ans = to_answers(req, {"target_present": {"yes": 0.1}, "dialog": {"yes": 0.05}}, 1.0, "t")
    assert ans.fire["target_present"] == pytest.approx(0.9)  # fire_on = "no"
    r = Rules({"default_tau": 0.5})
    d = r.before_execution(ans)
    assert d.action == "invalidate" and d.reason == "target_present"
    req = build_request("reasoning", {r_: img for r_, _ in ROLES["reasoning"]}, BANK)
    ans = to_answers(req, {"change_kind": {"probs": {"ambient": 0.9, "content": 0.1}}}, 1.0, "t")
    assert r.during_reasoning(ans).action == "ignore"


def test_methods_resolve():
    cfg = load_config("base")
    for m in cfg.methods:
        mc = method_config(cfg, m)
        assert mc.post in ("fixed", "pixel", "a11y", "clip", "nightwatch")
    assert method_config(cfg, "nightwatch_q4").bank_size == 4
    assert method_config(cfg, "nightwatch_no_interrupt").invalidate is True
    assert method_config(cfg, "nightwatch_uncalibrated").rules["calibration"] is None


def test_ltt_controls_false_alarms():
    rng = np.random.default_rng(0)
    y = rng.random(4000) < 0.2
    p = np.clip(np.where(y, rng.normal(0.8, 0.12, 4000), rng.normal(0.25, 0.15, 4000)), 0, 1)
    row = ltt_threshold(p, y, alpha=0.02, beta=0.3, delta=0.1)
    assert row["tau"] is not None and row["guarantee"] == "fpr_and_miss"
    yt = rng.random(20000) < 0.2  # fresh data: the certified threshold keeps FPR <= alpha
    pt = np.clip(np.where(yt, rng.normal(0.8, 0.12, 20000), rng.normal(0.25, 0.15, 20000)), 0, 1)
    assert ((pt >= row["tau"]) & ~yt).sum() / (~yt).sum() <= 0.02
    assert ltt_threshold(p[:5], y[:5], 0.001, 0.1, 0.1)["tau"] is None  # too little data: no certificate


def test_question_labels():
    labels = [("dialog", 5.0, 9.0), ("loading", 1.0, 2.0), ("navigation", 3.0, 3.0)]
    assert question_label("dialog", labels, 4.0, 6.0) is True
    assert question_label("dialog", labels, 5.5, 6.0) is False  # already visible when the agent looked
    assert question_label("page_changed", labels, 2.5, 3.5) is True
    assert question_label("settled", labels, 0.0, 1.5) is False
    assert question_label("target_present", labels, 0, 1) is None


def test_system1_fails_fast_and_cools_down(monkeypatch):
    import time

    import httpx

    from nightwatch.questions import build_request
    from nightwatch.system1.base import Meter
    from nightwatch.system1.remote import HttpS1

    def always_500(request):
        return httpx.Response(500, text="CUDA out of memory")

    monkeypatch.setenv("S1_BUDGET_S", "1.0")
    monkeypatch.setenv("S1_COOLDOWN_S", "30")
    s1 = HttpS1("http://s1.invalid")
    s1.client = httpx.Client(transport=httpx.MockTransport(always_500))
    m = Meter(s1)
    img = np.zeros((64, 64, 3), np.uint8)
    req = build_request("post_exec", {"before": img, "recent": img, "now": img, "crop": img}, BANK, "t")
    t = time.perf_counter()
    assert m.ask(req) is None and m.errors == 1
    assert time.perf_counter() - t < 2.0  # within the budget, not minutes of retries
    t = time.perf_counter()
    assert m.ask(req) is None and m.skipped == 1  # cooling down: no request at all
    assert time.perf_counter() - t < 0.05


def test_strip_trailing_sleep():
    from nightwatch.actions import strip_trailing_sleep

    code = "import pyautogui\nimport time\npyautogui.click(1, 2)\ntime.sleep(0.5)\npyautogui.write('x')\ntime.sleep(5)\n# done\n"
    out, s = strip_trailing_sleep(code)
    assert s == 5.0 and out.endswith("pyautogui.write('x')") and "time.sleep(0.5)" in out  # between-action sleep kept
    assert strip_trailing_sleep("pyautogui.click(1, 2); time.sleep(1)") == ("pyautogui.click(1, 2)", 1.0)
    loop = "for i in range(3):\n    pyautogui.press('down')\n    time.sleep(0.2)"
    assert strip_trailing_sleep(loop) == (loop, 0.0)  # inside a block: untouched
    assert strip_trailing_sleep("import time\ntime.sleep(3)")[1] == 0.0  # a pure wait stays a wait
    assert strip_trailing_sleep("WAIT") == ("WAIT", 0.0)
    cfg = load_config("base")
    assert method_config(cfg, "nightwatch_obs_jev").strip_sleep is True
    assert method_config(cfg, "fixed_default").strip_sleep is False


def test_invalidation_needs_a_higher_bar():
    from nightwatch.questions import Answers

    rules = Rules(load_config("base").method_defaults.rules)
    ans = Answers(fire={"target_occluded": 0.6, "dialog": 0.6})
    assert rules.before_execution(ans).action == "none"  # 0.6 < invalidate_tau 0.8
    assert rules.during_reasoning(ans).action == "interrupt"  # interrupt keeps the 0.5 bar
    ans = Answers(fire={"target_occluded": 0.9})
    assert rules.before_execution(ans).action == "invalidate"


def test_task_sample_is_seeded():
    from nightwatch.envs.osworld import load_tasks

    a = [t["id"] for t in load_tasks({"sample": 20, "seed": 0})]
    assert len(a) == 20 and a == [t["id"] for t in load_tasks({"sample": 20, "seed": 0})]


def test_strip_agent_sleeps():
    from nightwatch.actions import strip_agent_sleeps

    backoff = "import pyautogui\nimport time\ntime.sleep(180)\npyautogui.click(500, 60)\npyautogui.press('enter')\ntime.sleep(20)"
    out, s = strip_agent_sleeps(backoff)
    assert s == 200.0 and "sleep" not in out and out.endswith("pyautogui.press('enter')")
    out, s = strip_agent_sleeps("pyautogui.click(1, 2)\ntime.sleep(3)\npyautogui.write('a')\ntime.sleep(0.5)\npyautogui.press('enter')")
    assert s == 2.0 and "time.sleep(1)" in out and "time.sleep(0.5)" in out  # pacing capped, short pacing kept
    assert strip_agent_sleeps("import time\ntime.sleep(30)") == ("WAIT", 30.0)  # a pure wait waits on the screen
    assert strip_agent_sleeps("DONE") == ("DONE", 0.0)


def test_waste_ledger(tmp_path):
    import json

    from nightwatch.analysis.waste import step_ledger

    ep = tmp_path / "ep"
    ep.mkdir()
    (ep / "result.json").write_text(json.dumps({"method": "m", "task_id": "t", "domain": "d", "loop_wall_s": 9.0}))
    steps = [  # step 1: act at 1.0, screen changes at 1.5 and 2.0, agent looks at 4.0 (idle ~1.67 s)
        {"step": 1, "t_obs": 0.0, "action": "pyautogui.click(1, 1)",
         "segments": [["vlm", 0.0, 1.0], ["exec", 1.0, 1.2], ["wait", 1.2, 4.0], ["obs", 4.0, 4.2]]},
        # step 2: looks at 6.0 while the screen is still changing (5.9, 6.4): premature; then no effect
        {"step": 2, "t_obs": 4.2, "action": "pyautogui.click(2, 2)",
         "segments": [["vlm", 4.2, 5.0], ["exec", 5.0, 5.1], ["wait", 5.1, 6.0], ["obs", 6.0, 6.2]]},
        {"step": 3, "t_obs": 6.2, "action": "pyautogui.click(3, 3)",
         "segments": [["vlm", 6.2, 7.0], ["exec", 7.0, 7.1], ["wait", 7.1, 8.0], ["obs", 8.0, 8.2]]},
    ]
    (ep / "steps.jsonl").write_text("\n".join(json.dumps(s) for s in steps))
    ts = [round(0.1 + i / 3, 4) for i in range(27)]
    sig = {1.5, 2.0, 5.9, 6.4}
    rows = [[t, 0.1 if any(abs(t - x) < 0.17 for x in sig) else 0.0,
             int(any(abs(t - x) < 0.17 for x in sig)), 0.0] for t in ts]
    (ep / "frame_stats.json").write_text(json.dumps({"columns": ["t", "frac", "significant", "grab_s"], "rows": rows}))
    e, r = step_ledger(ep)
    assert abs(r[0]["react"] - (2.0 - 1.0 + 1 / 3)) < 0.2 and abs(r[0]["idle"] - (4.0 - 2.0 - 1 / 3)) < 0.2
    assert r[0]["idle_wait"] > 1.4 and r[0]["effect"] is True
    assert r[1]["premature"] and r[1]["stale"] and 0.2 < r[1]["premature_s"] < 0.7
    assert r[2]["effect"] is False and r[2]["timing_no_effect"]  # acted on a screen that was still changing


def test_aoi_observation_goes_into_the_request_only():
    from nightwatch.agents.prompt_agent import AOI_NARRATION, with_aoi_observation

    shot = {"type": "image_url", "image_url": {"url": "data:image/png;base64,AAA"}}
    hist = [{"role": "system", "content": "sys"},
            {"role": "user", "content": [{"type": "text", "text": "Given the screenshot..."}, shot]}]
    out = with_aoi_observation(hist, [(2.5, b"\xff\xd8kf1"), (0.7, b"\xff\xd8kf2")])
    c = out[-1]["content"]
    assert "2 visual change(s)" in c[0]["text"] and "2.5s" in c[0]["text"]
    assert [x["type"] for x in c] == ["text", "image_url", "image_url", "text", "text", "image_url", "text"]
    assert c[-2] is shot and c[-1]["text"] == AOI_NARRATION  # keyframes before the screenshot, narration last
    assert len(hist[-1]["content"]) == 2  # the agent's history is untouched
    assert [x["type"] for x in with_aoi_observation(hist, [])[-1]["content"]] == ["text", "image_url", "text"]
    assert method_config(load_config("base"), "aoi_visual").keyframes == "aoi"


def test_pusv_layers():
    from nightwatch.loop import pusv_verdict

    rng = np.random.default_rng(0)
    g0 = (rng.random((1080, 1920)) * 40 + 100).astype(np.uint8)
    assert pusv_verdict(g0, g0.copy(), 960, 540, ["0x1 0 host Terminal"], ["0x1 0 host Terminal"]) == ("", 0.0)
    g1 = g0.copy(); g1[500:580, 920:1000] = 255  # something drawn over the click target
    assert pusv_verdict(g0, g1, 960, 540, None, None)[0] == "pusv_l1_ssim"
    g2 = g0.copy(); g2[0:90, 1500:1880] = 255  # a corner banner (380x90), far from the click
    layer, ratio = pusv_verdict(g0, g2, 200, 900, None, None)
    assert layer == "pusv_l2a_global" and 0.01 < ratio < 0.02
    w = ["0x1 0 host Terminal"]
    assert pusv_verdict(g0, g0.copy(), 960, 540, w, w + ["0x2 0 host Security Alert"])[0] == "pusv_l2b_window"
    assert pusv_verdict(g0, g0.copy(), 960, 540, w, w + ["0x2 0 host Files"]) == ("", 0.0)
    assert method_config(load_config("base"), "pusv").validate == "pusv"
