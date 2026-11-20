"""Clock calibration must not turn a slow daemon launch into permanently stale frames."""
import base64
import struct
from types import SimpleNamespace

import pytest

from nightwatch.envs.osworld import AGSFrameGrabber, OSWorldEnv


class Clock:
    now = 1000.0

    def time(self):
        return self.now


class Reply:
    status_code = 200

    def __init__(self, output="", content=b""):
        self.output, self.content = output, content

    def raise_for_status(self):
        pass

    def json(self):
        return {"output": self.output}


class Session:
    def __init__(self, clock, skew, probes):
        self.clock, self.skew, self.probes = clock, skew, iter(probes)
        self.headers = {}
        self.starts = 0

    def post(self, url, **kwargs):
        if url.endswith("/file"):
            capture_time = self.clock.now + self.skew - 0.1
            return Reply(content=struct.pack("<Qd", 1, capture_time) + b"jpeg")
        code = kwargs["json"]["code"]
        if code == AGSFrameGrabber.CLOCK_PROBE:
            delay = next(self.probes)
            stamp = self.clock.now + delay / 2 + self.skew
            self.clock.now += delay
            return Reply(str(stamp))
        if "subprocess.Popen" in code:
            self.starts += 1
            self.clock.now += 30  # reproduces the observed 15-second midpoint error
            return Reply(f"started\n{self.clock.now + self.skew}")
        return Reply(base64.b64encode(b"synchronous screenshot").decode())


@pytest.fixture
def clock(monkeypatch):
    import time

    c = Clock()
    monkeypatch.setattr(time, "time", c.time)
    monkeypatch.setattr(time, "monotonic", c.time)
    return c


@pytest.mark.parametrize("skew", [-7.0, 0.0, 12.0])
def test_slow_start_does_not_pollute_calibration(clock, skew):
    grabber = AGSFrameGrabber(None, mode="daemon")
    grabber.s = Session(clock, skew, [.2, .04, .1])
    grabber._start_daemon("https://guest")
    assert grabber.offset == pytest.approx(skew)
    assert grabber.clock_rtt_s == pytest.approx(.04)
    assert grabber._grab_daemon(grabber.s, "https://guest") == b"jpeg"
    # Includes the 20 ms uncertainty, not half of the 30-second startup.
    assert grabber.capture_age == pytest.approx(.12)


def test_stalled_probe_is_not_used(clock):
    grabber = AGSFrameGrabber(None, mode="daemon")
    grabber.s = Session(clock, 3.0, [30, .08, .04])
    grabber._calibrate_clock("https://guest")
    assert grabber.offset == pytest.approx(3.0)
    assert grabber.clock_rtt_s == pytest.approx(.04)


def test_no_reliable_clock_uses_synchronous_capture(clock, monkeypatch):
    grabber = AGSFrameGrabber(None, mode="daemon")
    grabber.s = Session(clock, 0, [2, 2, 2])
    grabber._start_daemon("https://guest")
    assert grabber.offset is None
    monkeypatch.setattr(grabber, "_session", lambda: (grabber.s, "https://guest/run_python"))
    grabber.capture_age = 15  # cannot survive a fallback
    assert grabber.grab() == b"synchronous screenshot"
    assert grabber.capture_age is None


def test_failed_startup_is_retried(clock, monkeypatch):
    import requests

    provider = SimpleNamespace(sandbox=SimpleNamespace(get_host=lambda port: "guest"), envd_access_token="test")
    grabber = AGSFrameGrabber(provider, mode="daemon")
    session = Session(clock, 0, [.04] * 3)
    monkeypatch.setattr(requests, "Session", lambda: session)
    original = grabber._start_daemon
    attempts = []

    def start(base):
        attempts.append(base)
        if len(attempts) == 1:
            raise ConnectionError("startup disconnected")
        original(base)

    monkeypatch.setattr(grabber, "_start_daemon", start)
    with pytest.raises(ConnectionError):
        grabber._session()
    assert grabber.key is None
    grabber._session()
    assert len(attempts) == 2
    assert grabber.offset == pytest.approx(0)


def test_official_screenshot_does_not_inherit_old_capture_age():
    def fail():
        raise ConnectionError("frame download failed")

    env = OSWorldEnv.__new__(OSWorldEnv)
    env._fast = SimpleNamespace(capture_age=15, grab=fail)
    env.env = SimpleNamespace(controller=SimpleNamespace(get_screenshot=lambda: b"fresh png"))
    assert env.grab() == (b"fresh png", None)
    assert env.capture_age is None
