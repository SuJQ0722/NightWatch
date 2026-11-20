#!/usr/bin/env bash
# "CUA Jev": distil hosted Jev (+ hindsight gold from the frame stream) into a d3 decision model on desktop states,
# mixed with SynthWeb's exact-label web data. RL is not part of this script.
#
#   bash training/train_cua_jev.sh                                  # all stages
#   STAGES="build sft" bash training/train_cua_jev.sh              # some stages
#   MODEL=d3-mini GPUS=3,4 bash training/train_cua_jev.sh
#
# Stages:
#   tasks    generated training task pool (training/data/cua_tasks.py; disjoint from every OSWorld task)
#   collect  MiMo on that pool with the frame stream + screenshots recorded (configs/cua_collect.yaml, AGS)
#   build    teacher-labelled desktop examples (training/data/cua_build.py; OPENROUTER_API_KEY for Jev)
#   synth    SynthWeb record + build (training/train.sh record build), the exact-label web data
#   sft      LoRA SFT on both, data parallel over GPUS (training/sft/torch_sft.py: soft CE + Brier, temperature refit)
#   serve    the trained package as two perception servers (ports 8722 / 8762: the nightwatch_d3lite_sft method)
#   eval     OSWorld 100 tasks x 3 with the trained model (scripts/run_osworld.sh; METHOD, default nightwatch_d3lite_sft)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
STAGES="${STAGES:-tasks collect build synth sft serve eval}"
MODEL="${MODEL:-d3-lite}"
GPUS="${GPUS:-3,4}"
CUA_RUNS="${CUA_RUNS:-runs/cua_collect}"
CUA_DS="${CUA_DS:-data/cua_train/ds}"
SYN_DS="${SYN_DS:-data/synthweb/ds}"
MIX="${MIX:-$CUA_DS,$SYN_DS:${SYN_CAP:-15000}}"      # all desktop examples + a capped SynthWeb share
CK="${CK:-data/ckpt/$MODEL-cua-sft}"
PY="$ROOT/.conda/envs/nightwatch/bin/python"
TPY="$ROOT/.conda/envs/nightwatch-train/bin/python"
export PYTHONPATH="$ROOT:$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" HF_HOME="$ROOT/data/hf"
has() { [[ " $STAGES " == *" $1 "* ]]; }
say() { printf '\n\033[35m### %s\033[0m\n' "$*"; }

if has tasks; then
  say "training task pool"
  "$PY" -m training.data.cua_tasks --per-domain "${PER_DOMAIN:-40}" --out data/cua_train/tasks
fi
if has collect; then
  say "collect desktop episodes (${CUA_WORKERS:-12} sandboxes)"
  "$PY" -m nightwatch run --config cua_collect --name "$(basename "$CUA_RUNS")" --workers "${CUA_WORKERS:-12}"
fi
if has build; then
  say "teacher-labelled desktop examples -> $CUA_DS"
  "$PY" -m training.data.cua_build --runs "$CUA_RUNS" --out "$CUA_DS" --workers "${BUILD_WORKERS:-12}" \
      --per-episode "${PER_EPISODE:-60}"
fi
if has synth; then
  say "SynthWeb record + build"
  N_REC="${N_REC:-1200}" REC_SECONDS="${REC_SECONDS:-30}" STAGES="record build" MODELS="$MODEL" bash training/train.sh
fi
if has sft; then
  n=$(awk -F, '{print NF}' <<< "$GPUS")
  say "SFT $MODEL on GPUs $GPUS ($n processes): $MIX"
  CUDA_VISIBLE_DEVICES="$GPUS" "$(dirname "$TPY")/torchrun" --nproc_per_node "$n" -m training.sft.torch_sft \
      --family d3 --model "data/models/$MODEL" --data "$MIX" --out "$CK" --max-side "${MAX_SIDE:-768}" \
      --epochs "${EPOCHS:-1}" --lr "${LR:-1e-4}" --lora-r "${LORA_R:-16}" --accum "${ACCUM:-8}" \
      --questions-per-example "${QPE:-4}" --val-limit "${VAL_LIMIT:-800}" --eval-every "${EVAL_EVERY:-400}" \
      2>&1 | tee -a "$CK.log"
fi
if has serve; then
  say "serve $CK/package on ports 8722 / 8762"
  IFS=, read -r -a gl <<< "$GPUS"
  for j in 0 1; do
    S1_SERVER_BACKEND=d3 D3_MODEL="$CK/package" S1_PORT=$((8722 + 40 * j)) S1_GPU="${gl[$j]:-${gl[0]}}" D3_DEVICE=cuda:0 \
        S1_GPU_MEM_FRACTION="${S1_GPU_MEM_FRACTION:-0.85}" bash scripts/serve_system1.sh --background
  done
fi
if has eval; then
  E_ARGS=(METHODS="${METHOD:-nightwatch_d3lite_sft}" PREFIX="${E_PREFIX:-osworld_eval}" WORKERS="${E_WORKERS:-5}")
  say "OSWorld 100 x 3: ${METHOD:-nightwatch_d3lite_sft}"
  env "${E_ARGS[@]}" bash scripts/run_osworld.sh start
  while env "${E_ARGS[@]}" bash scripts/run_osworld.sh status | grep -q " running"; do sleep 300; done
  env "${E_ARGS[@]}" bash scripts/run_osworld.sh status
  env "${E_ARGS[@]}" bash scripts/run_osworld.sh metrics
fi

