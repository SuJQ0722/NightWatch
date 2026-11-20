from types import SimpleNamespace

import pytest

from nightwatch.dynacu_judge import checked_outcome
from nightwatch.envs.dynacu import DynaCUEnv


def test_judge_failure_is_not_a_benchmark_score():
    outcome = SimpleNamespace(llm_result=SimpleNamespace(error="no credits"))
    with pytest.raises(RuntimeError, match="no credits"):
        checked_outcome(outcome)


@pytest.mark.parametrize("passed,expected", [(True, 1.0), (False, 0.0)])
def test_hybrid_uses_official_binary_outcome(monkeypatch, passed, expected):
    import nightwatch.dynacu_judge as judge

    outcome = SimpleNamespace(
        final_score=0.7, final_passed=passed, llm_result=None,
        to_dict=lambda: {"final_score": 0.7, "final_passed": passed},
    )
    monkeypatch.setattr(judge, "make_evaluator", lambda model: SimpleNamespace(evaluate_task=lambda *a, **kw: outcome))
    desktop = DynaCUEnv.__new__(DynaCUEnv)
    desktop.judge = "test"
    desktop._task_obj = object()
    desktop._call = lambda fn: fn()
    desktop._env = SimpleNamespace(check_success=lambda: (True, "submitted"), get_page_text=lambda: "answer")
    assert desktop.evaluate() == expected
    assert desktop.last_eval["final_score"] == 0.7
    assert desktop.last_eval["agent_response"] == "answer"


def test_missing_judge_cannot_fall_back_to_dom(monkeypatch):
    import nightwatch.dynacu_judge as judge

    def unavailable(*a, **kw):
        raise RuntimeError("judge unavailable")

    monkeypatch.setattr(judge, "make_evaluator", unavailable)
    desktop = DynaCUEnv.__new__(DynaCUEnv)
    desktop.judge = "test"
    desktop._task_obj = object()
    desktop._call = lambda fn: fn()
    desktop._env = SimpleNamespace(check_success=lambda: (True, "submitted"), get_page_text=lambda: "answer")
    with pytest.raises(RuntimeError, match="judge unavailable"):
        desktop.evaluate()
