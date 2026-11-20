from nightwatch.agents.prompt_agent import rescale_coordinates as r


def test_positional_and_keyword_clicks():
    assert r("pyautogui.click(662, 298)", 1.5, 1.5) == "pyautogui.click(993, 447)"
    assert r("pyautogui.click(x=100, y=200, clicks=2)", 1.5, 1.5) == "pyautogui.click(x=150, y=300, clicks=2)"
    assert r("pyautogui.dragTo(100, 200, duration=0.5)", 2, 3) == "pyautogui.dragTo(200, 600, duration=0.5)"


def test_tuple_negative_and_multiline():
    assert r("pyautogui.click((10, 20))", 1.5, 1.5) == "pyautogui.click((15, 30))"
    assert r("pyautogui.moveRel(-10, 4)", 1.5, 1.5) == "pyautogui.moveRel(-15, 6)"
    code = "import pyautogui\ntime.sleep(0.5)\npyautogui.click(10, 20)\npyautogui.typewrite('点 10, 20')\npyautogui.click(2, 4)"
    assert r(code, 2, 2) == code.replace("click(10, 20)", "click(20, 40)").replace("click(2, 4)", "click(4, 8)")


def test_scroll_keeps_its_amount():
    assert r("pyautogui.scroll(-5, x=100, y=200)", 2, 2) == "pyautogui.scroll(-5, x=200, y=400)"
    assert r("pyautogui.scroll(-5)", 2, 2) == "pyautogui.scroll(-5)"


def test_left_alone():
    for code in ("WAIT", "DONE", "FAIL", "pyautogui.click(821 * sx, 223 * sy)", "pyautogui.press('enter')",
                 "pyautogui.click(", "pyautogui.click()"):
        assert r(code, 1.5, 1.5) == code


class Recorder:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        if name in ("size", "position"):
            return lambda: (1920, 1080) if name == "size" else (500, 400)
        if name.startswith("locate"):
            return lambda *a, **k: (900, 700)
        return lambda *a, **k: self.calls.append((name, a, k))


def run(code, sx=1.5, sy=1.5, **env):
    p = Recorder()
    exec(r(code, sx, sy), {"pyautogui": p, **env})
    return [(n, a, k) for n, a, k in p.calls]


def test_variables_and_expressions():
    assert run("x, y = 100, 200\npyautogui.click(x, y)") == [("click", (150, 300), {})]
    assert run("i = 1\npyautogui.moveTo(774 - i*18, 329, duration=0.15)") == [("moveTo", (1134, 494), {"duration": 0.15})]
    assert run("p = [10]\npyautogui.click(x=p[0], y=40)", 2, 3) == [("click", (), {"x": 20, "y": 120})]
    assert run("pyautogui.click(None, None)") == [("click", (None, None), {})]
    assert run("x = None\npyautogui.click(x)") == [("click", (None,), {})]


def test_screen_space_names_are_left_alone_but_literals_still_map():
    code = "p = pyautogui.locateCenterOnScreen('ok.png')\nx, y = p\npyautogui.click(x, y)\npyautogui.click(100, 200)"
    assert run(code) == [("click", (900, 700), {}), ("click", (150, 300), {})]
    assert run("x, y = pyautogui.position()\npyautogui.moveTo(x + 5, y)") == [("moveTo", (505, 400), {})]


def test_point_arguments():
    out = r("pyautogui.click(*cells[0])", 1.5, 1.5)
    assert out.startswith("pyautogui.click(*[_v if _v is None else int(round(_v * _s)) for _v, _s in zip(tuple(cells[0])")
    cells = [(100, 200)]
    calls = []

    class P:
        @staticmethod
        def click(*a, **k):
            calls.append(a)
    exec(out, {"pyautogui": P, "cells": cells})
    assert calls == [(150, 300)]
    out = r("pyautogui.click(pyautogui.center(pos))", 2, 2)
    assert "zip(tuple(_p)[:2], (2, 2))" in out and out.endswith("(pyautogui.center(pos)))")


def test_loop_over_points_runs():
    code = "for x, y in [(656,167),(729,239)]:\n    pyautogui.click(x, y)"
    calls = []

    class P:
        @staticmethod
        def click(*a, **k):
            calls.append(a)
    exec(r(code, 1.5, 1.5), {"pyautogui": P})
    assert calls == [(984, 250), (1094, 358)]


def test_single_argument_number_or_point():
    calls = []

    class P:
        @staticmethod
        def click(*a, **k):
            calls.append(a)
    exec(r("x = 100\npyautogui.click(x)", 1.5, 1.5), {"pyautogui": P})
    exec(r("p = (10, 20)\npyautogui.click(p)", 1.5, 1.5), {"pyautogui": P})
    assert calls == [(150,), ([15, 30],)]


def test_screen_space_values_left_alone():
    code = "p = pyautogui.locateCenterOnScreen('ok.png')\npyautogui.click(p)"
    assert r(code, 1.5, 1.5) == code


def test_reassignment_follows_source_order():
    assert run("x, y = pyautogui.position()\nx, y = 100, 200\npyautogui.click(x, y)") == [("click", (150, 300), {})]
    assert run("x, y = 100, 200\npyautogui.click(x, y)\nx, y = pyautogui.position()\npyautogui.click(x, y)") == \
        [("click", (150, 300), {}), ("click", (500, 400), {})]
    assert run("x, y = pyautogui.position()\nx += 10\npyautogui.moveTo(x, y)") == [("moveTo", (510, 400), {})]


def test_none_inside_unpacked_and_point_arguments():
    assert run("point = (None, 200)\npyautogui.click(*point)") == [("click", (None, 300), {})]
    assert run("point = (None, 200)\npyautogui.click(point)") == [("click", ([None, 300],), {})]


def test_is_noop_code():
    from nightwatch.agents.prompt_agent import is_noop_code
    assert is_noop_code("# done")
    assert is_noop_code("\n# step 1: click\n   # then save\n")
    assert not is_noop_code("# click\npyautogui.click(1, 2)")
    assert not is_noop_code("DONE") and not is_noop_code("WAIT") and not is_noop_code("")
    assert not is_noop_code(None)
