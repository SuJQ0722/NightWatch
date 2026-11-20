"""Check the real branch loop's semantics without model requests or VMs."""

import json
from types import SimpleNamespace

import pytest

pytest.importorskip("lib_run_single")

from nightwatch.agents.base import Prediction
from nightwatch.config import load_config, method_config
from nightwatch import qwen35_official as native


class Desktop:
    vm_ip = "test"

    def __init__(self, done_after=None):
        self.controller = SimpleNamespace(
            start_recording=lambda: pytest.fail("native recording must remain off"),
            end_recording=lambda *a: pytest.fail("native recording must remain off"))
        self.actions = []
        self.done_after = done_after

    def reset(self, task_config):
        self.task = task_config

    def _get_obs(self):
        return {"screenshot": b"test PNG"}

    def step(self, action, pause):
        self.actions.append((action, pause))
        return self._get_obs(), 0, len(self.actions) == self.done_after, {}

    def evaluate(self):
        return 0.5


class Agent:
    name = "test_qwen"

    def __init__(self, actions):
        self.actions = actions
        self.agent = SimpleNamespace(endpoint=SimpleNamespace(model="qwen-test", extra_body={}, retries=5),
                                     max_tokens=32768, temperature=0, top_p=0.9,
                                     native_tools=True, max_images=4)

    def reset(self, vm_ip=None):
        assert vm_ip == "test"

    def predict(self, instruction, obs, cancel):
        return Prediction(response="test", actions=self.actions, seconds=0.1)


def run(tmp_path, monkeypatch, actions, done_after=None):
    import lib_run_single

    sleeps = []
    monkeypatch.setattr(lib_run_single.time, "sleep", sleeps.append)
    cfg = load_config("osworld_qwen35", ["limits.max_steps=2"])
    desktop = Desktop(done_after)
    result = native.run_episode(SimpleNamespace(env=desktop), Agent(actions), method_config(cfg, "fixed_0"),
                                cfg, {"id": "task", "_domain": "test", "instruction": "do it"}, tmp_path / "episode")
    return result, desktop, sleeps


def test_native_loop_calls_env_step_and_counts_predictions(tmp_path, monkeypatch):
    result, desktop, sleeps = run(tmp_path, monkeypatch, ["first", "second"])
    assert result["steps"] == 2
    assert desktop.actions == [("first", 0), ("second", 0)] * 2
    assert sleeps == [60, 20]
    assert result["score"] == 0.5 and result["success"] is False
    assert result["end_reason"] == "max_steps"
    assert (tmp_path / "episode/result.txt").read_text().strip() == "0.5"
    assert len((tmp_path / "episode/traj.jsonl").read_text().splitlines()) == 4
    assert len(list((tmp_path / "episode/obs").glob("*.png"))) == 2


def test_empty_actions_still_spend_steps_and_keep_observations(tmp_path, monkeypatch):
    result, desktop, _ = run(tmp_path, monkeypatch, [])
    assert result["steps"] == 2 and desktop.actions == []
    assert len((tmp_path / "episode/calls.jsonl").read_text().splitlines()) == 2


def test_done_stops_remaining_actions(tmp_path, monkeypatch):
    result, desktop, _ = run(tmp_path, monkeypatch, ["DONE", "never"], done_after=1)
    assert result["steps"] == 1 and result["end_reason"] == "done"
    assert desktop.actions == [("DONE", 0)]


def test_native_resume_rejects_old_loop_results(tmp_path):
    cfg = load_config("osworld_qwen35")
    result = tmp_path / "fixed_0/r0/test/task/result.json"
    result.parent.mkdir(parents=True)
    result.write_text(json.dumps({"score": 1}))
    with pytest.raises(ValueError, match="mix native"):
        native.validate_config(cfg, [method_config(cfg, "fixed_0")], tmp_path)


def test_mimo_and_interventions_keep_existing_loop(tmp_path):
    mimo = load_config("base")
    assert method_config(mimo, "fixed_0").get("episode_loop") is None
    qwen = load_config("osworld_qwen35")
    ours = method_config(qwen, "nightwatch_d3lite_settle_s1wake")
    assert ours.get("episode_loop") is None
    native.validate_config(qwen, [method_config(qwen, "fixed_0"), ours], tmp_path)
    with pytest.raises(ValueError, match="requires"):
        native.validate_config(mimo, [method_config(qwen, "fixed_0")], tmp_path)


def test_failed_attempt_is_preserved(tmp_path, monkeypatch):
    out = tmp_path / "episode"
    out.mkdir()
    (out / "traj.jsonl").write_text("old attempt\n")
    run(tmp_path, monkeypatch, [])
    archives = list(tmp_path.glob("episode.incomplete-*"))
    assert len(archives) == 1
    assert (archives[0] / "traj.jsonl").read_text() == "old attempt\n"


def test_request_failure_is_not_a_scored_episode(tmp_path, monkeypatch):
    def fail(*args):
        raise ConnectionError("endpoint disconnected after retries")

    monkeypatch.setattr(Agent, "predict", fail)
    with pytest.raises(ConnectionError):
        run(tmp_path, monkeypatch, [])
    assert not (tmp_path / "episode/result.json").exists()
    assert not (tmp_path / "episode/result.txt").exists()
    assert len(list((tmp_path / "episode/obs").glob("*.png"))) == 1


def test_native_results_cannot_be_read_as_zero_waste(tmp_path, monkeypatch):
    from nightwatch.analysis.metrics import episode_row

    run(tmp_path, monkeypatch, [])
    with pytest.raises(ValueError, match="timing/waste"):
        episode_row(tmp_path / "episode")
