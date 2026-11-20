"""Training task pool for desktop (CUA) data: generated instructions on OSWorld's own Ubuntu image, disjoint from
every OSWorld evaluation task.

Each domain gets tasks written by the agent LLM (AGENT_* endpoint) that start from a freshly launched application
(the launch steps OSWorld itself uses for that app) with no benchmark files, pages or profiles. Instructions are
dropped when they are too close to any OSWorld instruction (difflib ratio >= --max-sim). The evaluator is
`infeasible` (instant, never blocks): only the screen states the agent passes through are used, not success.

    python -m training.data.cua_tasks --per-domain 40 --out data/cua_train/tasks
writes <out>/<domain>/<id>.json and <out>/../train_tasks.json ({domain: [ids]}), the layout `tasks.meta` /
`tasks.examples_dir` read.
"""

from __future__ import annotations

import argparse
import difflib
import json
import re
import uuid
from pathlib import Path

from nightwatch.agents.llm import CallStats, Endpoint, stream_chat
from nightwatch.config import OSWORLD_DIR

CHROME = [{"type": "launch", "parameters": {"command": ["google-chrome", "--remote-debugging-port=1337"]}},
          {"type": "launch", "parameters": {"command": ["socat", "tcp-listen:9222,fork", "tcp:localhost:1337"]}}]
LAUNCH = {
    "chrome": CHROME,
    "gimp": [{"type": "launch", "parameters": {"command": ["gimp"]}}],
    "libreoffice_calc": [{"type": "launch", "parameters": {"command": ["libreoffice", "--calc"]}}],
    "libreoffice_impress": [{"type": "launch", "parameters": {"command": ["libreoffice", "--impress"]}}],
    "libreoffice_writer": [{"type": "launch", "parameters": {"command": ["libreoffice", "--writer"]}}],
    "thunderbird": [{"type": "launch", "parameters": {"command": ["/usr/bin/thunderbird"]}}],
    "vlc": [{"type": "launch", "parameters": {"command": "VLC_VERBOSE=-1 vlc --no-audio --no-video-title-show",
                                              "shell": True}}],
    "vs_code": [{"type": "launch", "parameters": {"command": ["code"]}}],
    "os": [],
    "multi_apps": [],
}
APP = {
    "chrome": "Google Chrome (already open on the start page you choose)",
    "gimp": "GIMP (just launched, no image open)",
    "libreoffice_calc": "LibreOffice Calc (just launched, empty spreadsheet)",
    "libreoffice_impress": "LibreOffice Impress (just launched, new presentation wizard or empty deck)",
    "libreoffice_writer": "LibreOffice Writer (just launched, empty document)",
    "thunderbird": "Thunderbird (just launched; no mail account may be configured)",
    "vlc": "VLC media player (just launched, nothing playing)",
    "vs_code": "Visual Studio Code (just launched, no folder open)",
    "os": "the Ubuntu 22.04 GNOME desktop itself: Settings, Files, Terminal, the app grid",
    "multi_apps": "several desktop applications together (the agent opens them itself)",
}
PROMPT = """Write {n} different tasks for a computer-use agent on Ubuntu 22.04 (GNOME), using {app}.
Rules:
- Each task starts from that state, with no pre-existing user files: create any content the task needs.
- 5 to 20 GUI steps each; realistic everyday work; use menus, dialogs, settings panes, file pickers, tabs, page
  loads or progress where natural.
- No logins, accounts, purchases, payments, personal data, downloads of large files, or system-breaking changes.
- Vary the features used; do not repeat a task idea.{extra}
Return ONLY a JSON list of objects with the key "instruction"{keys}. No commentary."""


def _ask(ep: Endpoint, prompt: str) -> list[dict]:
    """The tasks of one reply. Objects are read one by one, so a reply cut off by max_tokens still yields every
    task it completed (long instructions used to lose the whole batch)."""
    txt = stream_chat(ep, [{"role": "user", "content": prompt}], None, CallStats(), max_tokens=12000, temperature=1.0)
    out = []
    for m in re.finditer(r"\{[^{}]*\}", txt, re.S):
        try:
            out.append(json.loads(m.group(0)))
        except json.JSONDecodeError:
            continue
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-domain", type=int, default=40)
    ap.add_argument("--batch", type=int, default=20)
    ap.add_argument("--out", default="data/cua_train/tasks")
    ap.add_argument("--max-sim", type=float, default=0.6, help="drop instructions this close to an OSWorld task")
    ap.add_argument("--domains", default=None, help="comma separated; default all. Tops a domain up to --per-domain")
    a = ap.parse_args()
    ep = Endpoint.from_env("AGENT")
    bench = []
    for f in (OSWORLD_DIR / "evaluation_examples" / "examples").rglob("*.json"):
        try:
            bench.append(json.loads(f.read_text())["instruction"].lower())
        except Exception:
            pass
    out = Path(a.out)
    meta, dropped = {}, 0
    doms = [d.strip() for d in a.domains.split(",")] if a.domains else list(LAUNCH)
    for dom in doms:
        steps = LAUNCH[dom]
        seen = [json.loads(f.read_text())["instruction"].lower() for f in (out / dom).glob("*.json")] if (out / dom).is_dir() else []
        tries = 0
        while len(seen) < a.per_domain and tries < 6:
            tries += 1
            extra = (" Give each chrome task a public, login-free start_url (a well-known site such as Wikipedia, "
                     "a public documentation page, a news or weather site)." if dom == "chrome" else "")
            keys = ' and "start_url"' if dom == "chrome" else ""
            try:
                items = _ask(ep, PROMPT.format(n=a.batch, app=APP[dom], extra=extra, keys=keys))
            except Exception as e:
                print(f"{dom}: generation failed: {e}")
                continue
            for it in items:
                ins = str(it.get("instruction", "")).strip()
                low = ins.lower()
                if len(ins) < 20 or any(difflib.SequenceMatcher(None, low, s).ratio() > 0.85 for s in seen):
                    continue
                if any(difflib.SequenceMatcher(None, low, b).ratio() >= a.max_sim for b in bench):
                    dropped += 1
                    continue
                config = [dict(s) for s in steps]
                if dom == "chrome":
                    url = str(it.get("start_url") or "https://www.wikipedia.org")
                    config.append({"type": "chrome_open_tabs", "parameters": {"urls_to_open": [url]}})
                tid = str(uuid.uuid4())
                task = {"id": tid, "snapshot": dom, "instruction": ins, "source": "nightwatch-generated", "config": config,
                        "related_apps": [dom], "evaluator": {"func": "infeasible"}, "proxy": False}
                (out / dom).mkdir(parents=True, exist_ok=True)
                (out / dom / f"{tid}.json").write_text(json.dumps(task, ensure_ascii=False, indent=2))
                meta.setdefault(dom, []).append(tid)
                seen.append(low)
                if len(seen) >= a.per_domain:
                    break
        print(f"{dom}: {len(seen)} tasks", flush=True)
    meta = {d.name: sorted(f.stem for f in d.glob("*.json")) for d in sorted(out.iterdir()) if d.is_dir()}
    (out.parent / "train_tasks.json").write_text(json.dumps(meta, indent=2))
    print(f"total {sum(map(len, meta.values()))} tasks, {dropped} dropped as too close to OSWorld this run; "
          f"meta {out.parent / 'train_tasks.json'}", flush=True)


if __name__ == "__main__":
    main()
