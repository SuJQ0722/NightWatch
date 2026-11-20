#!/usr/bin/env bash
# Start the perception server (System-1 shared by all harness processes) and wait until it is healthy.
#
#   bash scripts/serve_system1.sh                   # foreground; backend S1_SERVER_BACKEND (default intern)
#   bash scripts/serve_system1.sh --background      # detach, log to runs/system1_server.log, print the PID
#   S1_GPU=3 INTERN_DECISION_SIZE=2B bash scripts/serve_system1.sh
#   S1_SERVER_BACKEND=d3 D3_MODEL=data/ckpt/d3-mini-rl/package S1_PORT=8731 bash scripts/serve_system1.sh --background
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
load_env() { local k v; while IFS='=' read -r k v; do [[ -z "${!k:-}" && -n "$v" ]] && export "$k=$v"; done \
  < <(grep -E '^[A-Z_][A-Z0-9_]*=' "$1" | sed 's/\r$//'); return 0; }
[[ -f .env ]] && load_env .env
SPY="${S1_PYTHON:-$ROOT/.conda/envs/nightwatch-s1/bin/python}"
[[ -x "$SPY" ]] || { echo "System-1 env missing: bash scripts/setup_env.sh s1 models (or set S1_PYTHON)" >&2; exit 1; }
PORT="${S1_PORT:-8710}"
URL="http://127.0.0.1:$PORT"
export PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" HF_HOME="$ROOT/data/hf"
[[ -n "${S1_GPU:-}" ]] && export CUDA_VISIBLE_DEVICES="$S1_GPU" INTERN_DEVICE="${INTERN_DEVICE:-cuda:0}"
if curl -fs "$URL/health" >/dev/null 2>&1; then echo "already running: $(curl -fs "$URL/health")"; exit 0; fi
if [[ "${1:-}" == --background ]]; then
  mkdir -p runs
  nohup "$SPY" -m nightwatch serve --port "$PORT" > "runs/system1_server_$PORT.log" 2>&1 &
  pid=$!
  for _ in $(seq 1 600); do
    curl -fs "$URL/health" >/dev/null 2>&1 && { echo "perception server up (pid $pid): $(curl -fs "$URL/health")"; echo "$pid" > "runs/system1_server_$PORT.pid"; exit 0; }
    kill -0 "$pid" 2>/dev/null || { echo "server died; tail runs/system1_server.log:" >&2; tail -20 "runs/system1_server_$PORT.log" >&2; exit 1; }
    sleep 1
  done
  echo "server not healthy after 10 min (see runs/system1_server_$PORT.log)" >&2; exit 1
fi
exec "$SPY" -m nightwatch serve --port "$PORT"
