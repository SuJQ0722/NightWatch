"""End-to-end on the simulated desktop (real time, about a minute)."""
import json

import pytest

from nightwatch.config import load_config
from nightwatch.runner import run_experiment


@pytest.mark.slow
def test_smoke_end_to_end(tmp_path, monkeypatch):
    import nightwatch.runner as runner

    monkeypatch.setattr(runner, "RUNS_DIR", tmp_path)
    cfg = load_config("smoke", ["tasks.n=1", "limits.max_loop_s=40"])
    root = run_experiment(cfg, ["fixed_0.5", "pixel_stable", "nightwatch"], workers=3, name="t")
    for m in ("fixed_0.5", "pixel_stable", "nightwatch"):
        d = root / m / "r0" / "sim" / "sim-000"
        res = json.loads((d / "result.json").read_text())
        assert not res["error"], res.get("traceback")
        steps = [json.loads(line) for line in (d / "steps.jsonl").read_text().splitlines()]
        assert steps and all("segments" in s and "wake_reason" in s for s in steps)
        assert (d / "frame_stats.json").exists()
        if m == "nightwatch":
            assert res["system1_calls"].get("post_exec", 0) > 0
    from nightwatch.analysis.measure import measure
    from nightwatch.analysis.report import build_report

    assert "fixed_0.5" in measure(root)
    monkeypatch.setattr("nightwatch.analysis.report.RUNS_DIR", tmp_path)
    assert "nightwatch" in build_report(["t"])
