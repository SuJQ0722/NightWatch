import json
from types import SimpleNamespace

import pytest

from nightwatch.dynacu_official import pixel_only_context


def test_no_dom_blocks_extraction_preserves_images_and_scoring(tmp_path):
    class Browser:
        def get_interactive_elements(self):
            raise AssertionError("DOM extractor must not run")

        def check_success(self):
            return True, "correct"

    images = [SimpleNamespace(size=(1280, 720)) for _ in range(3)]
    received = []

    def original_model(text, frames, task):
        received.append((text, frames, task))
        return "NARRATION: visible code\nACTION: click(10, 20)"

    evaluator = SimpleNamespace(cu_model=original_model)
    module = SimpleNamespace(BrowserEnvironment=Browser)
    output = tmp_path / "inputs.jsonl"
    with pixel_only_context(evaluator, module, output) as audit:
        browser = module.BrowserEnvironment()
        assert browser.get_interactive_elements() == ""
        assert browser.check_success() == (True, "correct")
        assert evaluator.cu_model("[KEYFRAMES]", images, "task").endswith("click(10, 20)")
        assert received == [("[KEYFRAMES]", images, "task")]
        assert audit["suppressed_element_requests"] == 1
        assert audit["keyframe_images"] == 2
        assert audit["dom_prompt_violations"] == 0
    assert module.BrowserEnvironment is Browser
    assert evaluator.cu_model is original_model
    record = json.loads(output.read_text())
    assert record["image_sizes"] == [[1280, 720]] * 3


def test_dom_violation_never_reaches_model_and_scope_restores(tmp_path):
    original_env = type("Browser", (), {})

    def original_model(*args):
        pytest.fail("Request with DOM context must not reach model")

    evaluator = SimpleNamespace(cu_model=original_model)
    module = SimpleNamespace(BrowserEnvironment=original_env)
    with pytest.raises(RuntimeError, match="DOM element context"):
        with pixel_only_context(evaluator, module, tmp_path / "inputs.jsonl") as audit:
            evaluator.cu_model("[PAGE ELEMENTS — interactive]\ninput id=secret", [], "task")
    assert audit["dom_prompt_violations"] == 1
    assert evaluator.cu_model is original_model
    assert module.BrowserEnvironment is original_env
