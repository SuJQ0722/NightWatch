"""A disabled wall-time cap still enforces the prediction-step budget."""
import pytest

from nightwatch.config import load_config, method_config
from nightwatch.envs.sim_desktop import SimDesktop
from nightwatch.agents.sim_agent import SimAgent
from nightwatch.loop import Episode


@pytest.mark.parametrize("cap,steps,reason", [(None, 1, "max_steps"), (0, 0, "timeout")])
def test_optional_wall_time_cap(tmp_path, cap, steps, reason):
    cfg = load_config("smoke", ["limits.max_steps=1", f"limits.max_loop_s={cap if cap is not None else 'null'}"])
    env = SimDesktop({"popup_rate_per_min": 0})
    ep = Episode(env, SimAgent(latency_s=.05, jitter=0), None, method_config(cfg, "fixed_0"),
                 cfg, {"id": "budget", "_domain": "sim", "steps": 3}, tmp_path)
    result = ep.run()
    assert result["error"] == ""
    assert result["steps"] == steps
    assert result["end_reason"] == reason
