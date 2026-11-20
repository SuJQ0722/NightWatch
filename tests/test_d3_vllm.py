import json

import httpx
import numpy as np
import pytest

from nightwatch.questions import BY_ID, Request
from nightwatch.system1.d3_vllm import D3VllmS1


def test_native_probabilities_preserve_image_order_and_fire_polarity():
    backend = D3VllmS1()
    questions = [BY_ID["target_present"], BY_ID["change_kind"]]
    probabilities = {k: 1.0 if k == "dialog" else 0.0 for k in questions[1].options}
    def handle(request):
        body = json.loads(request.content)
        assert body["model"] == "d3-fp8"
        assert len(body["images"]) == 2 and body["images"][0] != body["images"][1]
        assert body["questions"]["change_kind"]["criteria"] == questions[1].options
        return httpx.Response(200, json={"answers": {
            "target_present": {"type": "noul", "probabilities": {"yes": 0.8, "no": 0.2}},
            "change_kind": {"type": "choice", "probabilities": probabilities}}})
    backend.client = httpx.Client(transport=httpx.MockTransport(handle))
    images = [np.full((64, 64, 3), v, dtype=np.uint8) for v in (0, 255)]
    answer = backend.ask(Request("pre_exec", "state", images, [(q, q.id) for q in questions]))
    assert answer.fire["target_present"] == pytest.approx(0.2)
    assert answer.probs["change_kind"] == probabilities


@pytest.mark.parametrize("answers", [{}, {"loading": {"type": "noul", "probabilities": {"yes": 0.3}}},
    {"loading": {"type": "noul", "probabilities": {"yes": 1.2, "no": -0.2}}},
    {"loading": {"type": "noul", "probabilities": {"yes": 0.3, "no": 0.3}}}])
def test_incomplete_or_invalid_answers_fail_instead_of_becoming_negative_evidence(answers):
    backend = D3VllmS1()
    backend.client = httpx.Client(transport=httpx.MockTransport(
        lambda request: httpx.Response(200, json={"answers": answers})))
    with pytest.raises(RuntimeError):
        backend.ask(Request("post_exec", "state", [], [(BY_ID["loading"], "Loading?")]))
