import io
from types import SimpleNamespace

from PIL import Image

from nightwatch.frames import frame_from_bytes
from nightwatch.loop import Episode


def png():
    b = io.BytesIO()
    Image.new("RGB", (64, 36)).save(b, "PNG")
    return b.getvalue()


class Stream:
    def __init__(self, frames):
        self.frames = frames

    def latest(self):
        return self.frames[-1] if self.frames else None

    def wait_new(self, after_idx, timeout=1.0):
        return None


def episode(frames, shots):
    ctl = SimpleNamespace(get_screenshot=lambda: shots.append(1) or png())
    return SimpleNamespace(stream=Stream(frames), obs_frame=SimpleNamespace(t=10.0), cfg={},
                           env=SimpleNamespace(controller=ctl, grab=lambda: (_ for _ in ()).throw(AssertionError("stream"))),
                           now=lambda: 12.0)


def frame(t, age):
    f = frame_from_bytes(png(), t)
    f.grab_s = age
    return f


def test_frame_captured_after_the_observation_is_used():
    f = frame(10.5, 0.1)
    shots = []
    assert Episode._fresh_frame(episode([f], shots)) is f and shots == []


def test_late_return_of_an_old_capture_is_refused():
    shots = []  # returned at 10.2 s, captured at 9.0 s, observation at 10.0 s
    out = Episode._fresh_frame(episode([frame(10.2, 1.2)], shots))
    assert shots == [1] and out.t == 12.0
