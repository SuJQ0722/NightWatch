#!/usr/bin/env bash
# Start one perception server per decision model (port convention of configs/base.yaml):
#   zero-shot 871x, SFT 872x, SFT+RL 873x;  x = 1 d3-mini (4B), 2 d3-lite (0.8B), 3 Intern-Decision-0.8B
#
#   bash scripts/serve_models.sh                 # zero-shot servers for the three models
#   STAGE=sft bash scripts/serve_models.sh       # data/ckpt/<model>-sft/package
#   STAGE=rl  bash scripts/serve_models.sh       # data/ckpt/<model>-rl/package
#   S1_GPUS="0 1 2" bash scripts/serve_models.sh # one GPU per server (default: auto / CPU)
#   S1_MODELS="d3-mini d3-lite" S1_GPUS="0,2 3,4" S1_GPU_MEM_FRACTION=0.85 bash scripts/serve_models.sh
#                                                # 2 replicas per model, one per GPU: the second replica listens on
#                                                # port + 40 (d3-mini 8711 + 8751); methods list both URLs
#   bash scripts/serve_models.sh stop
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
STAGE="${STAGE:-zero}"
MODELS=(d3-mini d3-lite Intern-Decision-0.8B)   # ports x=1,2,3; missing packages are skipped
case "$STAGE" in zero) BASE=8710 ;; sft) BASE=8720 ;; rl) BASE=8730 ;; *) echo "STAGE=zero|sft|rl" >&2; exit 1 ;; esac
if [[ "${1:-}" == stop ]]; then
  for f in runs/system1_server_87*.pid; do [[ -f "$f" ]] && kill "$(cat "$f")" 2>/dev/null && echo "stopped $f"; rm -f "$f"; done
  exit 0
fi
read -r -a GPUS <<< "${S1_GPUS:-}"
for i in "${!MODELS[@]}"; do
  m="${MODELS[$i]}"
  [[ -n "${S1_MODELS:-}" && " ${S1_MODELS} " != *" $m "* ]] && continue
  path="data/models/$m"
  [[ "$STAGE" != zero ]] && path="data/ckpt/$m-$STAGE/package"
  [[ -e "$path" ]] || { echo "skip $m ($path missing)"; continue; }
  IFS=, read -r -a gl <<< "${GPUS[$i]:-}"
  [[ ${#gl[@]} -eq 0 ]] && gl=("")
  for j in "${!gl[@]}"; do
    gpu="${gl[$j]}"; port=$((BASE + i + 1 + 40 * j))
    if [[ "$m" == d3* ]]; then
      S1_SERVER_BACKEND=d3 D3_MODEL="$path" S1_PORT=$port S1_GPU="$gpu" D3_DEVICE="${gpu:+cuda:0}" bash scripts/serve_system1.sh --background
    else
      S1_SERVER_BACKEND=intern INTERN_DECISION_PATH="$path" S1_PORT=$port S1_GPU="$gpu" bash scripts/serve_system1.sh --background
    fi
  done
done
