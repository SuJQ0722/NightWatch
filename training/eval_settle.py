"""Offline check of the fast stability tier (post_cfg.s1_wake) on settle_build examples: would System-1 wake at the
right moments? Posts every example (fast questions only) to perception server(s), then for a grid of wake_tau
reports, per decision kind, how many final screens it wakes on and how many wakes come too early (the screen
changes again within the horizon).

    python -m training.eval_settle --ds data/cua_train/settle_ds --urls http://127.0.0.1:8722,http://127.0.0.1:8762 \
        [--split val] [--out runs/settle_eval_zeroshot.json]
"""

from __future__ import annotations

import argparse
import base64
import json
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx

from nightwatch.questions import BY_ID

TAUS = (0.5, 0.7, 0.8, 0.9, 0.95, 0.98)


def ask(client: httpx.Client, url: str, ds: Path, e: dict) -> dict:
    body = {"checkpoint": e["checkpoint"], "state": e["state"],
            "images": [base64.b64encode((ds / p).read_bytes()).decode() for p in e["images"]],
            "questions": [{"id": q, "kind": BY_ID[q].kind, "text": t, "options": BY_ID[q].options} for q, t in e["questions"]]}
    r = client.post(url + "/v1/ask", json=body)
    r.raise_for_status()
    raw = r.json()
    return {q: v.get("yes") for q, v in raw["raw"].items()} | {"ms": raw.get("ms")}


def report(rows: list[dict], loading_tau: float = 0.5, changing_tau: float = 0.5) -> dict:
    out = {}
    groups = defaultdict(list)
    for r in rows:
        m = r["meta"]
        groups[f"{m['kind']}_{m['k']}"].append(r)
    for g, rs in sorted(groups.items()):
        n, fin = len(rs), sum(r["final"] for r in rs)
        res = {"n": n, "final": fin, "changes": n - fin, "always_wake_too_early": round((n - fin) / n, 4)}
        for tau in TAUS:
            wake = [r for r in rs if (r["p"].get("settled") or 0) >= tau and (r["p"].get("loading") or 0) < loading_tau
                    and (r["p"].get("still_changing") or 0) < changing_tau]
            early = sum(not r["final"] for r in wake)
            res[f"tau{tau}"] = {"wakes_on_final": round(sum(r["final"] for r in wake) / max(fin, 1), 4),
                                "too_early_per_moment": round(early / n, 4),
                                "caught_changes": round(1 - early / max(n - fin, 1), 4)}
        out[g] = res
    return out


def main(argv=None) -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds", default="data/cua_train/settle_ds")
    ap.add_argument("--urls", default="http://127.0.0.1:8722,http://127.0.0.1:8762")
    ap.add_argument("--split", default=None, help="train / val (default: all)")
    ap.add_argument("--out", default=None)
    ap.add_argument("--workers", type=int, default=4)
    a = ap.parse_args(argv)
    ds = Path(a.ds)
    ex = [json.loads(l) for l in open(ds / "examples.jsonl")]
    ex = [e for e in ex if a.split is None or e["split"] == a.split]
    urls = a.urls.split(",")
    client = httpx.Client(timeout=120)

    def one(i_e):
        i, e = i_e
        p = ask(client, urls[i % len(urls)], ds, e)
        return {"id": e["id"], "split": e["split"], "meta": e["meta"], "final": e["labels"]["settled"], "p": p}

    with ThreadPoolExecutor(a.workers) as pool:
        rows = list(pool.map(one, enumerate(ex)))
    rep = report(rows)
    ms = sorted(r["p"]["ms"] for r in rows if r["p"].get("ms") is not None)
    rep["_latency_ms_median"] = ms[len(ms) // 2] if ms else None
    print(json.dumps(rep, indent=1))
    if a.out:
        Path(a.out).write_text(json.dumps({"report": rep, "rows": rows}))


if __name__ == "__main__":
    main()
