"""OSWorldEnv.execute must replicate DesktopEnv.step minus the trailing sleep and observation."""
import re

import pytest

from nightwatch.config import OSWORLD_DIR

SRC = OSWORLD_DIR / "desktop_env" / "desktop_env.py"


@pytest.mark.skipif(not SRC.exists(), reason="OSWorld checkout missing (scripts/setup_env.sh osworld)")
def test_step_body_is_what_execute_replicates():
    src = SRC.read_text()
    body = src[src.index("    def step(self, action, pause=2):"):src.index("    def evaluate(self):")]
    code = [ln.strip() for ln in body.splitlines() if ln.strip() and not ln.strip().startswith(("#", "logger.", "reward", "done", "info", "\"\"\""))]
    expected_effects = ["self._step_no += 1", "self.action_history.append(action)", "self.is_environment_used = True",
                        "self.controller.execute_action(action)", "self.controller.execute_python_command(command)",
                        "time.sleep(pause)", "observation = self._get_obs()"]
    for e in expected_effects:
        assert any(e in c for c in code), f"DesktopEnv.step changed: {e!r} not found; re-check OSWorldEnv.execute"
    calls = [c for c in code if re.search(r"self\.\w+(\.\w+)*\(", c) and "isinstance" not in c and "type(" not in c]
    known = ("execute_action", "execute_python_command", "action_history.append", "_get_obs")
    assert all(any(k in c for k in known) for c in calls), f"new side effect in DesktopEnv.step: {calls}"
