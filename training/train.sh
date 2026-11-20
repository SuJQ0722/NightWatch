#!/usr/bin/env bash
# One-click training of the decision models (System-1): SynthWeb data -> SFT -> RL -> offline eval -> serve.
# Needs `bash scripts/setup_env.sh s1 models dynacu`. GPUs strongly recommended (CPU works only for tiny smoke runs).
#
#   bash training/train.sh                                    # all stages, all three models
#   MODELS="d3-mini" STAGES="sft rl eval" bash training/train.sh
#   N_REC=200 STAGES="record build" bash training/train.sh    # a small dataset first
#   NGPU=4 bash training/train.sh                             # data-parallel SFT (torchrun)
#   SMOKE=1 bash training/train.sh                            # CPU pipeline check: tiny data, 2 steps, small images
#
# Stages:
#   record   SynthWeb pages in headless Chromium, random interactions, 10 Hz frames + exact ground truth
#            (N_REC episodes x REC_SECONDS s, REC_WORKERS browsers)            -> data/synthweb/rec
#   build    checkpoint-shaped examples with exact labels; splits by seed; HOLDOUT families -> data/synthweb/ds
#   sft      per model: LoRA SFT on the question bank, temperature refit       -> data/ckpt/<model>-sft/package
#   rl       per model: decision RL (exact expected return, KL to SFT)         -> data/ckpt/<model>-rl/package
#   eval     per model x {zero, sft, rl}: decision metrics on the test split   -> data/ckpt/eval/<model>-<stage>.json
#   serve    perception servers for the trained packages (STAGE=rl ports 873x), then run e.g.
#            MODELS_STAGE=rl bash scripts/run_dynacu.sh
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
load_env() { local k v; while IFS='=' read -r k v; do [[ -z "${!k:-}" && -n "$v" ]] && export "$k=$v"; done \
  < <(grep -E '^[A-Z_][A-Z0-9_]*=' "$1" | sed 's/\r$//'); return 0; }
[[ -f .env ]] && load_env .env
STAGES="${STAGES:-record build sft rl eval}"
MODELS="${MODELS:-d3-mini d3-lite}"   # Decision 3.0 family (Intern-Decision-0.8B still supported: add it to MODELS)
SPY="${S1_PYTHON:-$ROOT/.conda/envs/nightwatch-s1/bin/python}"
DE="$ROOT/.conda/envs/nightwatch-dynacu"
DPY="$DE/bin/python"
# $ROOT: the `training` package; $ROOT/src: nightwatch
export PYTHONPATH="$ROOT:$ROOT/src${PYTHONPATH:+:$PYTHONPATH}" HF_HOME="$ROOT/data/hf"
REC="${REC_DIR:-data/synthweb/rec}"; DS="${DS_DIR:-data/synthweb/ds}"; CK="${CKPT_DIR:-data/ckpt}"
DEVICE="${DEVICE:-$("$SPY" -c 'import torch; print("cuda" if torch.cuda.is_available() else "cpu")')}"
[[ "$DEVICE" == cpu ]] && export NIGHTWATCH_NO_FLA=1
NGPU="${NGPU:-1}"
if [[ "${SMOKE:-0}" == 1 ]]; then
  REC=data/synthweb/smoke_rec; DS=data/synthweb/smoke_ds; CK=data/ckpt/smoke
  N_REC=4; REC_SECONDS=20; DEVICE=cpu; NGPU=1
  export NIGHTWATCH_NO_FLA=1   # CPU: hide the GPU-only kernels from the d3 runtime
  SFT_ARGS="--max-steps 2 --accum 2 --limit 8 --val-limit 6 --max-side 448 --questions-per-example 2"
  RL_ARGS="--episodes 4 --accum 2 --max-frames 3 --max-side 448"
  EVAL_ARGS="--episodes 6 --max-frames 3 --max-side 448 --split train"
fi
has() { [[ " $STAGES " == *" $1 "* ]]; }
say() { printf '\n\033[35m### %s\033[0m\n' "$*"; }
family() { [[ "$1" == d3* ]] && echo d3 || echo intern; }

if has record; then
  say "record SynthWeb (${N_REC:-3000} episodes)"
  LD_LIBRARY_PATH="$DE/lib" PLAYWRIGHT_BROWSERS_PATH="$ROOT/data/ms-playwright" "$DPY" -m training.data.record \
      --n "${N_REC:-3000}" --duration "${REC_SECONDS:-40}" --workers "${REC_WORKERS:-16}" --out "$REC"
fi
if has build; then
  say "build examples (holdout families: ${HOLDOUT:-shift})"
  "$DPY" -m training.data.build --rec "$REC" --out "$DS" --holdout-families "${HOLDOUT:-shift}"
fi
for m in $MODELS; do
  fam=$(family "$m")
  if has sft; then
    say "SFT $m ($fam, $DEVICE x $NGPU)"
    if (( NGPU > 1 )); then
      "$(dirname "$SPY")/torchrun" --nproc_per_node "$NGPU" -m training.sft.torch_sft --family "$fam" --model "data/models/$m" \
          --data "$DS" --out "$CK/$m-sft" --device cuda ${SFT_ARGS:-}
    else
      "$SPY" -m training.sft.torch_sft --family "$fam" --model "data/models/$m" --data "$DS" --out "$CK/$m-sft" --device "$DEVICE" ${SFT_ARGS:-}
    fi
  fi
  if has rl; then
    say "RL $m"
    "$SPY" -m training.rl.offline_rl --family "$fam" --model "$CK/$m-sft/package" --rec "$REC" --out "$CK/$m-rl" --device "$DEVICE" ${RL_ARGS:-}
  fi
  if has eval; then
    for st in zero sft rl; do
      pkg="data/models/$m"; [[ $st != zero ]] && pkg="$CK/$m-$st/package"
      [[ -e "$pkg" ]] || continue
      say "eval $m [$st] on the test split"
      mkdir -p "$CK/eval"
      "$SPY" -m training.rl.offline_rl --family "$fam" --model "$pkg" --rec "$REC" --out "$CK/eval/$m-$st" --device "$DEVICE" \
          --eval-only --split test --episodes "${EVAL_EPISODES:-600}" ${EVAL_ARGS:-}
    done
  fi
done
if has serve; then
  say "serve trained packages (STAGE=rl)"
  STAGE=rl bash scripts/serve_models.sh
fi
