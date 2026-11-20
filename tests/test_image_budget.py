from nightwatch.agents.prompt_agent import fit_image_budget, with_aoi_observation


def _turn(role, n_img=1):
    if role == "assistant":
        return {"role": "assistant", "content": [{"type": "text", "text": "ok"}]}
    return {"role": "user", "content": [{"type": "text", "text": "Given the screenshot"}] +
            [{"type": "image_url", "image_url": {"url": "data:x"}}] * n_img}


def _count(msgs):
    return sum(p.get("type") == "image_url" for m in msgs if isinstance(m.get("content"), list) for p in m["content"])


def _history():  # system + 3 history turns + current: 4 images, as OSWorld's PromptAgent sends
    msgs = [{"role": "system", "content": "sys"}]
    for _ in range(3):
        msgs += [_turn("user"), _turn("assistant")]
    return msgs + [_turn("user")]


def test_noop_without_limit_or_keyframes():
    msgs = _history()
    assert fit_image_budget(msgs, [], None) == (msgs, [])
    assert fit_image_budget(msgs, [], 4)[0] == msgs


def test_keyframes_take_priority_over_old_history():
    kfs = [(float(i), b"jpg") for i in range(5)]
    out = with_aoi_observation(_history(), kfs, 4)
    assert _count(out) == 4  # current + 3 most recent keyframes, no history images
    assert "3 visual change(s)" in out[-1]["content"][0]["text"]
    out = with_aoi_observation(_history(), kfs[:1], 4)
    assert _count(out) == 4  # current + 1 keyframe + 2 newest history screenshots
    assert _count(out[1:2]) == 0 and _count(out[3:4]) == 1 and _count(out[5:6]) == 1
    assert out[1]["content"][1] == {"type": "text", "text": "(earlier screenshot omitted)"}


def test_unlimited_aoi_unchanged():
    kfs = [(float(i), b"jpg") for i in range(5)]
    assert _count(with_aoi_observation(_history(), kfs)) == 9
