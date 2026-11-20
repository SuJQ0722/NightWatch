"""Command line:  python -m nightwatch <command>

  doctor                       environment, OSWorld checkout, VM image, docker/KVM, models, keys, servers
  ping [--agent] [--s1]        one real agent call (streaming + a cancel) and one System-1 call
  serve [--backend intern]     perception server (run in the System-1 env; scripts/serve_system1.sh)
  run --config C [--methods a,b] [--repeats N] [--workers W] [--name X] [--set k=v ...]
  measure --exp E              C1 measurement study from logs (+ recordings) -> runs/E/measure/
  pilot --exp E [--every 0.5]  System-1 event detection on recorded frames -> runs/E/pilot/
  label --exp E                export labelling sheets (labels.csv template + contact sheets)
  calibrate --exp E            Learn-then-Test thresholds -> data/calibration/thresholds.json
  metrics --exps a,b           score / effective steps / latency / waste / events / cost per method -> runs/<b>/metrics/
  waste --exps a,b             wasted-time ledger (idle / premature / no-effect, per step) -> runs/<b>/waste/
  report [--exps a,b]          tables and figures -> runs/report/
"""

from __future__ import annotations

import argparse
import logging
import sys


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(prog="nightwatch")
    sub = ap.add_subparsers(dest="cmd", required=True)
    sub.add_parser("doctor")
    p = sub.add_parser("ping")
    p.add_argument("--agent", action="store_true")
    p.add_argument("--s1", action="store_true")
    p = sub.add_parser("serve")
    p.add_argument("--backend", default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--host", default="127.0.0.1")
    p = sub.add_parser("run")
    p.add_argument("--config", required=True)
    p.add_argument("--methods", default=None)
    p.add_argument("--repeats", type=int, default=None)
    p.add_argument("--workers", type=int, default=None)
    p.add_argument("--name", default=None)
    p.add_argument("--set", action="append", default=[], dest="overrides")
    for name in ("measure", "pilot", "label", "calibrate"):
        p = sub.add_parser(name)
        p.add_argument("--exp", required=True, help="experiment folder under runs/ (or a path)")
        if name == "pilot":
            p.add_argument("--every", type=float, default=0.5, help="seconds between evaluated frames")
            p.add_argument("--backend", default=None)
            p.add_argument("--limit", type=int, default=None, help="max frames per episode")
        if name == "calibrate":
            p.add_argument("--false-per-min", type=float, default=1.0)
            p.add_argument("--miss", type=float, default=0.2)
            p.add_argument("--delta", type=float, default=0.1)
            p.add_argument("--test-domains", default=None, help="held-out OSWorld domains, comma separated")
            p.add_argument("--out", default=None)
    p = sub.add_parser("metrics")
    p.add_argument("--exps", required=True, help="comma-separated experiment folders compared on their common tasks")
    p.add_argument("--out", default=None)
    p = sub.add_parser("waste")
    p.add_argument("--exps", required=True, help="comma-separated experiment folders compared on their common tasks")
    p.add_argument("--out", default=None)
    p = sub.add_parser("report")
    p.add_argument("--exps", default=None, help="comma-separated experiment folders (default: all under runs/)")
    p.add_argument("--baseline", default="fixed_default")
    p.add_argument("--margin", type=float, default=0.02, help="non-inferiority margin on success rate")
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")

    if args.cmd == "doctor":
        from nightwatch.doctor import doctor

        sys.exit(doctor())
    if args.cmd == "ping":
        from nightwatch.doctor import ping

        sys.exit(ping(agent=args.agent or not args.s1, s1=args.s1 or not args.agent))
    if args.cmd == "serve":
        from nightwatch.server import serve

        serve(args.backend, args.host, args.port)
        return
    if args.cmd == "run":
        from nightwatch.config import load_config
        from nightwatch.runner import run_experiment

        cfg = load_config(args.config, args.overrides)
        methods = args.methods.split(",") if args.methods else None
        run_experiment(cfg, methods, args.repeats, args.workers, args.name)
        return
    from nightwatch.analysis import common

    if args.cmd == "metrics":
        from nightwatch.analysis.metrics import metrics

        print(metrics([e.strip() for e in args.exps.split(",") if e.strip()], args.out))
        return
    if args.cmd == "waste":
        from nightwatch.analysis.waste import waste

        print(waste([e.strip() for e in args.exps.split(",") if e.strip()], args.out))
        return
    if args.cmd == "report":
        from nightwatch.analysis.report import build_report

        print(build_report(args.exps.split(",") if args.exps else None, args.baseline, args.margin))
        return
    root = common.exp_root(args.exp)
    if args.cmd == "measure":
        from nightwatch.analysis.measure import measure

        print(measure(root))
    elif args.cmd == "pilot":
        from nightwatch.analysis.pilot import pilot

        print(pilot(root, args.every, args.backend, args.limit))
    elif args.cmd == "label":
        from nightwatch.analysis.labels import export_sheets

        print(export_sheets(root))
    elif args.cmd == "calibrate":
        from nightwatch.analysis.calibrate import calibrate

        print(calibrate(root, args.false_per_min, args.miss, args.delta,
                        args.test_domains.split(",") if args.test_domains else None, args.out))


if __name__ == "__main__":
    main()
