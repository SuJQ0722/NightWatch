import pytest

agent = pytest.importorskip("mm_agents.agent")
parse = agent.parse_code_from_string


def block(code):
    return "Reflection: it didn't open; try again.\n```python\n" + code + "\n```"


def test_semicolon_in_string_is_kept():
    assert parse(block("pyautogui.write('git status; git branch\\n')")) == ["pyautogui.write('git status; git branch\\n')"]


def test_statements_are_still_split():
    assert parse(block("import pyautogui; pyautogui.click(1, 2);")) == ["import pyautogui\npyautogui.click(1, 2)"]
    assert parse(block("if a:\n    b(); c()")) == ["if a:\n    b()\n    c()"]


def test_comment_and_triple_quote():
    assert parse(block("a()  # x; y\nb('''p; q''')")) == ["a()  # x; y\nb('''p; q''')"]


def test_special_codes():
    assert parse("DONE;") == ["DONE"]
    assert parse("```\nWAIT\n```") == ["WAIT"]
    assert parse(block("pyautogui.click(1, 2)\nDONE")) == ["pyautogui.click(1, 2)", "DONE"]
