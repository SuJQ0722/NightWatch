"""Training-side logic that needs no model: AOI action grammar, SynthWeb labels, RL rewards and composition."""
import pytest
import torch

from nightwatch.actions import parse_action
from training.rl import offline_rl as rl
from training.data.build import label
from training.data.synthweb import generate


def test_aoi_actions():
    assert parse_action("click(640, 360)", "aoi").has_target
    a = parse_action('fill(#emailInput, "a@b.com")', "aoi")
    assert a.kind == "type" and a.text == "a@b.com"
    assert parse_action("key(Enter)", "aoi").text == "Enter"
    assert parse_action("wait()", "aoi").kind == "wait"
    assert parse_action("done", "aoi").kind == "done"
    assert parse_action("triple_click(10, 20)", "aoi").x == 10


def test_synthweb_generates_and_labels():
    html, spec = generate(3, ["dialog", "toast", "ambient"])
    assert "window.__gt" in html and spec["families"] == ["dialog", "toast", "ambient"]
    base = {"page": 0, "version": 3, "animating": False, "loading": False, "long_op": False, "dialog": False,
            "dialog_box": None, "toast": False, "error": False, "focus": "main", "ambient_boxes": [[0, 0, 100, 30]],
            "elements": [["b0_1", 100, 100, 200, 140, True]]}
    now = dict(base, dialog=True, dialog_box=[0, 0, 1280, 720], version=4, focus="dialog")
    g = {"seen": base, "now": now}
    assert label("dialog", g, {}) is True and label("toast", g, {}) is False
    assert label("target_occluded", {"seen": base, "now_marked": now}, {"point": (150, 120)}) is True
    assert label("target_present", {"seen": base, "now_marked": now}, {"point": (150, 120)}) is False
    moved = dict(base, elements=[["b0_1", 100, 170, 200, 210, True]])
    assert label("target_changed", {"seen": base, "now_marked": moved}, {"point": (150, 120)}) is True
    assert label("change_kind", {"seen": base, "now": now}, {"crop_box": [10, 10, 50, 20]}) == "dialog"
    assert label("change_kind", {"seen": base, "now": base}, {"crop_box": [10, 5, 50, 20]}) in ("ambient", "trivial")


def test_rewards_and_composition():
    c = {"c_early": 4.0, "c_miss": 6.0, "c_stale": 4.0, "c_regen": 1.0, "c_invalid": 7.0, "scale": 5.0}
    ep = {"t_a": 10.0, "t_ready": 11.0, "windows": [], "horizon_t": 18.0}
    assert rl.wake_reward(ep, 11.2, c) == pytest.approx(-1.2)  # ready: only the time spent
    assert rl.wake_reward(ep, 10.5, c) == pytest.approx(-4.5)  # too early: one invalid step
    ep_t = dict(ep, windows=[(10.5, 11.5)])
    assert rl.wake_reward(ep_t, 10.8, c) == pytest.approx(-0.8)  # woke inside the toast window
    assert rl.wake_reward(ep_t, 12.0, c) == pytest.approx(-8.0)  # toast missed
    ie = {"t0": 0.0, "L": 4.0, "t_e": 2.0}
    assert rl.interrupt_reward(ie, 2.3, c) == pytest.approx(-0.3) and rl.interrupt_reward(ie, None, c) == -2.0 - 4.0
    assert rl.interrupt_reward(dict(ie, t_e=None), 1.0, c) == -2.0 and rl.interrupt_reward(dict(ie, t_e=None), None, c) == 0.0
    assert rl.invalidate_reward({"L": 3.0, "stale": True}, False, c) == -7.0
    f = {q: torch.tensor(0.0) for q in rl.WAKE_Q}
    f["settled"] = torch.tensor(1.0)
    assert float(rl.compose("wake", f)) == 1.0
    f["loading"] = torch.tensor(1.0)
    assert float(rl.compose("wake", f)) == 0.0


def test_soft_label_loss():
    import torch

    from training.sft.torch_sft import loss_fn

    lg = torch.tensor([0.3, -0.2])
    assert torch.allclose(loss_fn(lg, 1, "soft_ce_brier"), loss_fn(lg, [0.0, 1.0], "soft_ce_brier"))  # one-hot == index
    lo = loss_fn(torch.tensor([0.0, 1.0]), [0.27, 0.73], "soft_ce_brier")  # p(yes) = 0.73: matches the target
    hi = loss_fn(torch.tensor([1.0, 0.0]), [0.27, 0.73], "soft_ce_brier")
    assert lo < hi
