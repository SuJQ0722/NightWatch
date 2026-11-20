"""Streaming + mid-generation cancel against a local fake OpenAI-compatible SSE server, and the OSWorld
PromptAgent subclass end to end (when the OSWorld checkout is importable)."""
import json
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest

from nightwatch.agents.llm import CallStats, Cancelled, Endpoint, stream_chat

REPLY = "The dialog is open.\n```python\npyautogui.click(100, 200)\n```"


class _H(BaseHTTPRequestHandler):
    delay = 0.02

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream")
        self.end_headers()
        time.sleep(0.3)  # "prefill"
        try:
            for ch in REPLY if body.get("max_tokens", 0) < 5000 else REPLY * 200:
                self.wfile.write(f"data: {json.dumps({'choices': [{'delta': {'content': ch}}]})}\n\n".encode())
                self.wfile.flush()
                time.sleep(self.delay)
            self.wfile.write(b'data: {"choices": [], "usage": {"prompt_tokens": 50, "completion_tokens": 20}}\n\n')
            self.wfile.write(b"data: [DONE]\n\n")
        except (BrokenPipeError, ConnectionResetError):
            pass


@pytest.fixture(scope="module")
def server():
    srv = ThreadingHTTPServer(("127.0.0.1", 0), _H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}/v1"
    srv.shutdown()


def test_stream_and_cancel(server):
    ep = Endpoint(server, "fake", "k", retries=1)
    st = CallStats()
    assert stream_chat(ep, [{"role": "user", "content": "x"}], None, st) == REPLY
    assert st.output_tokens == 20 and st.t_first_token >= 0.25
    cancel, st2 = threading.Event(), CallStats()
    threading.Timer(0.5, cancel.set).start()
    t = time.perf_counter()
    with pytest.raises(Cancelled):
        stream_chat(ep, [{"role": "user", "content": "x"}], cancel, st2, max_tokens=9000)
    assert time.perf_counter() - t < 1.5 and st2.estimated
    cancel3 = threading.Event()  # cancel during "prefill", before any byte
    threading.Timer(0.1, cancel3.set).start()
    t = time.perf_counter()
    with pytest.raises(Cancelled):
        stream_chat(ep, [{"role": "user", "content": "x"}], cancel3, CallStats())
    assert time.perf_counter() - t < 0.6


def test_prompt_agent_cancel_and_restore(server, monkeypatch):
    pytest.importorskip("mm_agents.agent")
    from nightwatch.agents.prompt_agent import PromptAgentAdapter
    from nightwatch.config import Cfg
    from nightwatch.envs.sim_desktop import SimDesktop

    monkeypatch.setenv("AGENT_BASE_URL", server)
    monkeypatch.setenv("AGENT_API_KEY", "k")
    monkeypatch.setenv("AGENT_MODEL", "fake")
    a = PromptAgentAdapter(Cfg({"max_trajectory_length": 3}))
    a.reset()
    env = SimDesktop()
    env.reset({"id": "t"})
    obs = env.obs()
    pred = a.predict("do it", obs, threading.Event())
    assert pred.actions == ["pyautogui.click(100, 200)"] and not pred.cancelled
    snap = a.snapshot()
    cancel = threading.Event()
    threading.Timer(0.4, cancel.set).start()
    a.agent.max_tokens = 9000
    t = time.perf_counter()
    pred = a.predict("do it", obs, cancel)
    assert pred.cancelled
    assert time.perf_counter() - t < 1.5  # no "empty response" sleep-and-retry after a cancel
    a.restore(snap)
    assert len(a.agent.thoughts) == len(a.agent.actions) == len(a.agent.observations) == 1
