#!/usr/bin/env bash
# Serve the trained watcher for the `nightwatch_full` method: four perception servers on one GPU,
# full checks on 8782 / 8783 and the fast settledness tier on 8772 / 8773 (configs/base.yaml,
# nightwatch_d3lite_settle_s1wake). One process per replica: concurrent requests in one server slow each other down.
#
#   bash scripts/serve_watcher.sh                     # data/models/NightWatch-0.8B on GPU 0
#   WATCHER_GPU=1 bash scripts/serve_watcher.sh
#   WATCHER=data/models/d3-lite FULL_PORTS="8812 8813" FAST_PORTS="8822 8823" bash scripts/serve_watcher.sh
#                                                     # the untuned d3-lite arm of configs/dynacu_ablations.yaml
#   bash scripts/serve_watcher.sh stop
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
FULL_PORTS="${FULL_PORTS:-8782 8783}"
FAST_PORTS="${FAST_PORTS:-8772 8773}"
if [[ "${1:-}" == stop ]]; then
  for port in $FULL_PORTS $FAST_PORTS; do
    f="runs/system1_server_$port.pid"
    [[ -f "$f" ]] && kill "$(cat "$f")" 2>/dev/null && echo "stopped $port"; rm -f "$f"
  done
  exit 0
fi
WATCHER="${WATCHER:-data/models/NightWatch-0.8B}"
[[ -e "$WATCHER" ]] || { echo "$WATCHER missing: MODELS=nightwatch bash scripts/setup_env.sh models" >&2; exit 1; }
for port in $FULL_PORTS $FAST_PORTS; do
  S1_SERVER_BACKEND=d3 D3_MODEL="$WATCHER" S1_PORT="$port" S1_GPU="${WATCHER_GPU:-0}" D3_DEVICE=cuda:0 \
    D3_BATCH=4 D3_LM_BATCH=4 S1_GPU_MEM_FRACTION="${S1_GPU_MEM_FRACTION:-0.10}" S1_WARMUP=0 \
    bash scripts/serve_system1.sh --background
done
