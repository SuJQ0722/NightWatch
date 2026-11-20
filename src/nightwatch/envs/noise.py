"""Live noise injection for OSWorld (optional dynamic setting, a `noise:` block in the config).

A background thread injects modal dialogs (zenity) and desktop notifications (notify-send) into the VM
at Poisson times while the agent works, in the spirit of MobileBench-OL / D-GARA but at run time.
Injection times are logged to noise.jsonl in the episode folder: they are free ground-truth labels for
the dialog / toast questions (analysis/labels.py). Untested on a real VM so far.
"""

from __future__ import annotations

import json
import random
import threading

from nightwatch.utils import Clock, jline

MESSAGES = {
    "dialog": ["Software Updater: updated software is available. Install now?",
               "Authentication required to change system settings.",
               "The application is not responding. Force quit?"],
    "notification": ["Low disk space on Filesystem root", "Network connection lost", "Backup completed"],
}


class NoiseInjector:
    def __init__(self, controller, ncfg, clock: Clock, log_path, seed: int = 0):
        self.c, self.cfg, self.clock = controller, ncfg, clock
        self.rng = random.Random(seed)
        self.stop_ev = threading.Event()
        self.fh = open(log_path, "a", encoding="utf-8")
        self.th = threading.Thread(target=self._loop, daemon=True, name="noise")

    def start(self) -> None:
        self.th.start()

    def stop(self) -> None:
        self.stop_ev.set()
        self.th.join(timeout=5)
        self.fh.close()

    def _loop(self) -> None:
        rate = float(self.cfg.get("rate_per_min", 2.0)) / 60.0
        kinds = list(self.cfg.get("kinds", ["dialog", "notification"]))
        while not self.stop_ev.wait(self.rng.expovariate(rate)):
            kind = self.rng.choice(kinds)
            msg = self.rng.choice(MESSAGES[kind])
            if kind == "dialog":
                argv = ["zenity", "--question", "--title=System", f"--text={msg}",
                        f"--timeout={int(self.cfg.get('dialog_timeout_s', 15))}"]
            else:
                argv = ["notify-send", "-t", "4000", "System", msg]
            cmd = f"import subprocess; subprocess.Popen({json.dumps(argv)})"
            t = self.clock.now()
            try:
                self.c.execute_python_command(cmd)
                ok = True
            except Exception:
                ok = False
            jline(self.fh, {"t": round(t, 3), "t_done": round(self.clock.now(), 3), "kind": kind, "msg": msg, "ok": ok})
