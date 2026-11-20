import io
import threading

import pytest
from PIL import Image

pytest.importorskip("mm_agents.qwen35vl_agent")

from nightwatch.agents import qwen35_agent as q  # noqa: E402


def screenshot():
    b = io.BytesIO()
    Image.new("RGB", (1920, 1080), (200, 200, 200)).save(b, "PNG")
    return b.getvalue()


REPLY = ("Action: Click the middle.\n<tool_call>\n<function=computer_use>\n<parameter=action>\nleft_click\n</parameter>\n"
         "<parameter=coordinate>\n[500, 500]\n</parameter>\n</function>\n</tool_call>")


@pytest.fixture
def adapter(monkeypatch):
    monkeypatch.setenv("AGENT_MAX_IMAGES", "4")
    seen = []

    def fake(ep, messages, cancel, stats, **kw):
        seen.append((messages, kw))
        return REPLY

    monkeypatch.setattr(q, "stream_chat", fake)
    a = q.Qwen35AgentAdapter({})
    a.reset()
    a.seen = seen
    return a


def test_relative_click_maps_to_the_screen(adapter):
    p = adapter.predict("Click.", {"screenshot": screenshot()}, threading.Event())
    assert p.actions == ["pyautogui.click(960, 540)"]
    assert adapter.seen[-1][1] == {"max_tokens": 32768, "temperature": 0.0, "top_p": 0.9}


def test_native_tool_call_reaches_the_parser(adapter):
    from nightwatch.agents.llm import tool_calls_as_xml

    adapter.agent.native_tools = True
    xml = "Action: go." + tool_calls_as_xml(
        [{"function": {"name": "computer_use", "arguments": '{"action": "left_click", "coordinate": [500, 500]}'}}])
    adapter.agent.call_llm = lambda payload, model: xml
    assert adapter.predict("Click.", {"screenshot": screenshot()}, threading.Event()).actions == ["pyautogui.click(960, 540)"]


def test_tools_are_sent_only_when_asked(adapter):
    obs = {"screenshot": screenshot()}
    adapter.predict("Click.", obs, threading.Event())
    assert "tools" not in adapter.seen[-1][1]
    adapter.agent.native_tools = True
    adapter.predict("Click.", obs, threading.Event())
    assert adapter.seen[-1][1]["tools"][0]["function"]["name"] == "computer_use"


def test_image_limit_and_history_restore(adapter):
    obs = {"screenshot": screenshot()}
    for _ in range(3):
        adapter.predict("Click.", obs, threading.Event())
    snap = adapter.snapshot()
    for _ in range(3):
        adapter.predict("Click.", obs, threading.Event())
    parts = [p for m in adapter.seen[-1][0] if isinstance(m["content"], list) for p in m["content"]]
    assert sum(p.get("type") == "image_url" for p in parts) == 4
    adapter.restore(snap)
    assert len(adapter.agent.screenshots) == len(adapter.agent.responses) == len(adapter.agent.actions) == 3


def test_cancel_at_folding_boundary_restores_the_next_prompt(adapter, monkeypatch):
    from nightwatch.agents.llm import Cancelled

    adapter.agent.endpoint.api_key = "test"
    adapter.agent.image_max = 2
    adapter.agent.fold_size = 1
    obs = {"screenshot": screenshot()}
    for _ in range(2):
        adapter.predict("Click.", obs, threading.Event())
    snap = adapter.snapshot()
    requests = []

    def cancel_once(ep, messages, cancel, stats, **kw):
        requests.append(messages)
        if len(requests) == 1:
            raise Cancelled("", "")
        return REPLY

    monkeypatch.setattr(q, "stream_chat", cancel_once)
    assert adapter.predict("Click.", obs, threading.Event()).cancelled
    assert adapter.agent.folded_prefix_k == 1
    adapter.restore(snap)
    assert adapter.agent.folded_prefix_k == 0
    assert len(adapter.agent.responses) == len(adapter.agent.screenshots) == 2
    assert not adapter.predict("Click.", obs, threading.Event()).cancelled
    assert requests[0] == requests[1]  # no phantom WAIT or lost screenshot after an interrupt


def test_branch_config_does_not_inherit_mimo_provider_or_thinking(monkeypatch):
    from nightwatch.config import load_config

    monkeypatch.setenv("AGENT_EXTRA_BODY", '{"provider":{"order":["xiaomi"]},"reasoning":{"effort":"high"}}')
    monkeypatch.setenv("AGENT_RETRIES", "9")
    cfg = load_config("osworld_qwen35")
    a = q.Qwen35AgentAdapter(cfg["agent"])
    assert a.agent.endpoint.extra_body == {}
    assert a.agent.endpoint.retries == 5
    assert a.agent.max_tokens == 32768
    assert a.agent.temperature == 0


def test_obs_size_absolute_maps_back_to_the_screen(monkeypatch):
    reply = ("Action: go.\n<tool_call>\n<function=computer_use>\n<parameter=action>\nleft_click\n</parameter>\n"
             "<parameter=coordinate>\n[640, 352]\n</parameter>\n</function>\n</tool_call>")
    seen = []

    def fake(ep, messages, cancel, stats, **kw):
        seen.append(messages)
        return reply

    monkeypatch.setattr(q, "stream_chat", fake)
    a = q.Qwen35AgentAdapter({"obs_size": [1280, 720], "coordinate_type": "absolute"})
    a.reset()
    p = a.predict("Click.", {"screenshot": screenshot()}, threading.Event())
    # 1280x720 is sent as 1280x704 (factor 32); (640, 352) is the centre of that image, i.e. (960, 540) on the screen
    assert p.actions == ["pyautogui.click(960, 540)"]
    assert "The screen's resolution is 1280x704" in seen[0][0]["content"][0]["text"]


def test_aoi_keyframes_go_before_the_current_screenshot(adapter):
    adapter.predict("Click.", {"screenshot": screenshot(), "aoi_keyframes": [(0.5, b"\xff\xd8jpg")]}, threading.Event())
    last = adapter.seen[-1][0][-1]["content"]
    kinds = [p.get("type") for p in last]
    assert kinds.count("image_url") == 2 and "KEYFRAMES" in last[0]["text"]
    assert last[-1]["text"].endswith("then give your Action and tool call as usual.")



@pytest.mark.parametrize("params,expected", [
    ("<parameter=keys>\n[\"ctrl\", \"s\"]\n</parameter>", ['pyautogui.hotkey("ctrl", "s")']),
    ("<parameter=text>\nReturn\n</parameter>", ['pyautogui.press("Return")']),
    ("<parameter=text>\nctrl+alt+e\n</parameter>", ['pyautogui.hotkey("ctrl", "alt", "e")']),
    ("<parameter=text>\n+\n</parameter>", ['pyautogui.press("+")']),
    ("<parameter=text>\nctrl++\n</parameter>", ['pyautogui.hotkey("ctrl", "+")']),
    ("<parameter=text>\nctrl+shift++\n</parameter>", ['pyautogui.hotkey("ctrl", "shift", "+")']),
    ("", []),
])
def test_key_actions(params, expected):
    from mm_agents.qwen35vl_agent import Qwen35VLAgent

    reply = f"Action: k.\n<tool_call>\n<function=computer_use>\n<parameter=action>\nkey\n</parameter>\n{params}\n</function>\n</tool_call>"
    assert Qwen35VLAgent().parse_response(reply, 1920, 1080, 1920, 1088)[1] == expected


def test_cap_image_bytes_drops_oldest_history_only():
    from nightwatch.agents.prompt_agent import cap_image_bytes

    img = lambda n: {"type": "image_url", "image_url": {"url": "data:image/png;base64," + "A" * n}}  # noqa: E731
    msgs = [{"role": "system", "content": [{"type": "text", "text": "s"}]}]
    msgs += [{"role": "user", "content": [img(100)]}, {"role": "assistant", "content": "a"}] * 3
    msgs += [{"role": "user", "content": [img(100), {"type": "text", "text": "now"}]}]
    same, n = cap_image_bytes(msgs, 10_000)
    assert same == msgs and n == 0
    out, n = cap_image_bytes(msgs, 300)
    assert n == 2
    kinds = [[p["type"] for p in m["content"]] for m in out if m["role"] == "user"]
    assert kinds == [["text"], ["text"], ["image_url"], ["image_url", "text"]]
    out, n = cap_image_bytes(msgs, 10)  # never drops the current turn
    assert out[-1] == msgs[-1] and n == 3
