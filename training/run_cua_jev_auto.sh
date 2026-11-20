#!/usr/bin/env bash
# Unattended CUA-Jev run that never leaves our GPUs idle (a shared machine: an idle GPU gets taken, and GPU jobs
# get killed from outside):
#   1. on every free GPU of ALLOWED_GPUS: a zero-shot d3-lite server holding S1_GPU_RESERVE of it (ports 8712+40*j)
#      plus a signal canary (scripts/sig_canary.py: logs who sends the kill, runs/sig_canary_<gpu>.log)
#   2. wait for the desktop dataset (stats.json, written last by training/data/cua_build.py)
#   3. SFT on two of our GPUs (their servers stopped first); killed -> re-hold, wait, resume from resume.pt
#   4. serve the SFT package on the training GPUs (8722 / 8762) and run the 20-task eval
# Every job runs in its own session, so killing one process group does not take down this supervisor.
#
#   setsid nohup bash training/run_cua_jev_auto.sh > runs/cua_jev_auto.log 2>&1 &
set -uo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
ALLOWED_GPUS="${ALLOWED_GPUS:-0,1,2,3,4,6,7}"
CUA_DS="${CUA_DS:-data/cua_train/ds}"
TRIES="${TRIES:-30}"
CPY="$ROOT/.conda/envs/nightwatch-train/bin/python"
export S1_GPU_MEM_FRACTION="${S1_GPU_MEM_FRACTION:-0.85}" S1_GPU_RESERVE="${S1_GPU_RESERVE:-0.8}" \
    TRAIN_GPU_RESERVE="${TRAIN_GPU_RESERVE:-0.8}"
say() { printf '\n[%(%F %T)T] %s\n' -1 "$*"; }
port_of() { echo $((8712 + 40 * $1)); }   # by GPU index: 0 -> 8712, 1 -> 8752, ...
used_mib() { nvidia-smi -i "$1" --query-gpu=memory.used --format=csv,noheader,nounits | tr -d ' '; }
ours() { [[ -f runs/system1_server_$(port_of "$1").pid ]] && kill -0 "$(cat "runs/system1_server_$(port_of "$1").pid")" 2>/dev/null; }
free_gpus() { local g; for g in ${ALLOWED_GPUS//,/ }; do (( $(used_mib "$g") < 1000 )) && printf '%s ' "$g"; done; }
hold() {   # server + canary on every free allowed GPU
  local g
  for g in $(free_gpus); do
    S1_SERVER_BACKEND=d3 D3_MODEL=data/models/d3-lite S1_PORT="$(port_of "$g")" S1_GPU="$g" D3_DEVICE=cuda:0 \
        setsid bash scripts/serve_system1.sh --background || say "server on GPU $g failed"
    CUDA_VISIBLE_DEVICES="$g" setsid nohup "$CPY" scripts/sig_canary.py "runs/sig_canary_$g.log" >/dev/null 2>&1 < /dev/null &
  done
}
release() {   # stop our server on GPU $1
  local f="runs/system1_server_$(port_of "$1").pid" pid
  [[ -f $f ]] || return 0
  pid=$(cat "$f"); kill "$pid" 2>/dev/null
  while kill -0 "$pid" 2>/dev/null; do sleep 2; done
  rm -f "$f"
}
held() { local g; for g in ${ALLOWED_GPUS//,/ }; do ours "$g" && printf '%s ' "$g"; done; }

say "hold free GPUs of $ALLOWED_GPUS"
hold
say "holding: $(held)"
nvidia-smi --query-gpu=index,memory.used --format=csv,noheader

say "wait for $CUA_DS/stats.json"
until [[ -f $CUA_DS/stats.json ]]; do sleep 30; done

ok=0
for try in $(seq "$TRIES"); do
  hold   # re-take anything freed (after a kill)
  read -r -a mine <<< "$(held)"
  if (( ${#mine[@]} < 2 )); then say "try $try: only ${#mine[@]} GPU(s) held, waiting"; sleep 60; continue; fi
  GPUS="${mine[-2]},${mine[-1]}"
  say "try $try: SFT on GPUs $GPUS (held: ${mine[*]})"
  for g in ${GPUS//,/ }; do release "$g"; done
  if STAGES=sft GPUS="$GPUS" setsid --wait bash training/train_cua_jev.sh; then ok=1; break; fi
  say "try $try: SFT died; canary logs:"; cat runs/sig_canary_*.log 2>/dev/null | grep got | tail -5
  sleep 30
done
if (( ok )); then
  say "SFT done: serve the package on $GPUS and run the 20 tasks"
  STAGES="serve eval" GPUS="$GPUS" bash training/train_cua_jev.sh
  say "eval done"
else
  say "SFT gave up after $TRIES tries"
fi
hold
