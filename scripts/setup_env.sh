#!/usr/bin/env bash
# Environment setup (one click, idempotent): Python envs, the pinned OSWorld checkout, model weights.
# Containers / the VM image are a separate script: scripts/setup_container.sh.
#
#   bash scripts/setup_env.sh                    # osworld main s1 models dynacu
#   bash scripts/setup_env.sh s1 models          # only some components
#   bash scripts/setup_env.sh dynacu             # DynaCU-Bench: AOI checkout, browser env, Chromium (no docker/VM)
#   DYNACU_AUDIO=1 bash scripts/setup_env.sh dynacu   # + Whisper large-v3 for the audio categories
#   bash scripts/setup_env.sh vllm qwen          # local agent: vLLM env (build picked from the driver's CUDA) +
#                                                # Qwen3.8-27B-FP8 weights (QWEN_MODEL)
#   S1_PYTHON=/path/to/python bash scripts/setup_env.sh s1   # reuse an env that already has torch + transformers 5.x
#   MODELS="d3-mini d3-lite nightwatch" bash scripts/setup_env.sh models   # decision models (default; nightwatch =
#                                                # the trained watcher (nullptr04/NightWatch-0.8B);
#                                                # add Intern-Decision-0.8B if wanted)
#
# Finished components leave data/.done_<component>; delete it to redo. Everything lives under this repo:
# .conda/envs, .cache/pip, data/models, data/hf.
#   .conda/envs/nightwatch     harness: python 3.12, OSWorld requirements (desktop_env, mm_agents, evaluators), nightwatch
#   .conda/envs/nightwatch-s1  System-1 + training: torch 2.9.1 (cu128), transformers 5.17.0 (d3 and Intern-Decision), peft
#   .conda/envs/nightwatch-dynacu  DynaCU: AOI requirements, Playwright Chromium (data/ms-playwright), PulseAudio, CPU torch
#   third_party/aoi       19PINE-AI/aoi pinned (AOI_COMMIT below): DynaCU-Bench tasks, browser env, AOI baselines
#   third_party/OSWorld   xlang-ai/OSWorld pinned (OSWORLD_COMMIT below), on nightwatch's sys.path via osworld.pth
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
COMPONENTS="${*:-osworld main s1 models dynacu}"
DATA="$ROOT/data"; ENVS="$ROOT/.conda/envs"; CACHE="$ROOT/.cache"
mkdir -p "$DATA/models" "$DATA/hf" "$ENVS" "$CACHE/pip" "$CACHE/conda_pkgs" "$CACHE/tmp" "$ROOT/third_party"
export PIP_CACHE_DIR="$CACHE/pip" CONDA_PKGS_DIRS="$CACHE/conda_pkgs" TMPDIR="$CACHE/tmp" HF_HOME="$DATA/hf"
load_env() { local k v; while IFS='=' read -r k v; do [[ -z "${!k:-}" && -n "$v" ]] && export "$k=$v"; done \
  < <(grep -E '^[A-Z_][A-Z0-9_]*=' "$1" | sed 's/\r$//'); return 0; }
[[ -f .env ]] && load_env .env
# HF_HOME moves into the repo, so a token from `hf auth login` (~/.cache/huggingface/token) is no longer found
[[ -z "${HF_TOKEN:-}" && ! -f "$HF_HOME/token" && -f "$HOME/.cache/huggingface/token" ]] \
  && export HF_TOKEN_PATH="$HOME/.cache/huggingface/token"
OSWORLD_COMMIT=044bc33901cb893808e3f91ebc9a17ed8e54b44c
AOI_COMMIT=5ee56ceee768fe6a3f3cfa9aaab6c81dfa31407d
say() { printf '\n\033[36m=== %s\033[0m\n' "$*"; }
has() { [[ " $COMPONENTS " == *" $1 "* ]]; }
done_() { [[ -f "$DATA/.done_$1" ]]; }
mark() { touch "$DATA/.done_$1"; }
pipi() { "$1" -m pip install -q --no-warn-script-location "${@:2}"; }
command -v conda >/dev/null || { echo "conda not found on PATH (install miniconda first)" >&2; exit 1; }
PY="$ENVS/nightwatch/bin/python"
SPY="${S1_PYTHON:-$ENVS/nightwatch-s1/bin/python}"

# --------------------------------------------------------------------------------------- osworld
if has osworld && ! done_ osworld; then
  say "OSWorld checkout (third_party/OSWorld @ ${OSWORLD_COMMIT:0:12})"
  OSW="$ROOT/third_party/OSWorld"
  [[ -d "$OSW/desktop_env" ]] || { echo "third_party/OSWorld is vendored; re-clone nightwatch" >&2; exit 1; }
  if [[ -d "$OSW/.git-upstream" ]]; then
    head="$(GIT_DIR="$OSW/.git-upstream" GIT_WORK_TREE="$OSW" git rev-parse HEAD)"
    [[ "$head" == "$OSWORLD_COMMIT" ]] || { echo "OSWorld upstream HEAD ${head:0:12} != pin ${OSWORLD_COMMIT:0:12}" >&2; exit 1; }
  elif [[ -d "$OSW/.git" ]]; then
    head="$(git -C "$OSW" rev-parse HEAD)"
    [[ "$head" == "$OSWORLD_COMMIT" ]] || { echo "OSWorld HEAD ${head:0:12} != pin ${OSWORLD_COMMIT:0:12}" >&2; exit 1; }
  fi
  mark osworld
fi

# ------------------------------------------------------------------------------------------ main
if has main && ! done_ main; then
  say "harness env (.conda/envs/nightwatch: python 3.12 + OSWorld requirements + nightwatch)"
  [[ -d "$ROOT/third_party/OSWorld" ]] || { echo "run the osworld component first" >&2; exit 1; }
  [[ -x "$PY" ]] || conda create -y -q -p "$ENVS/nightwatch" -c conda-forge --override-channels python=3.12 git
  pipi "$PY" --upgrade pip wheel
  pipi "$PY" -r "$ROOT/third_party/OSWorld/requirements.txt"
  pipi "$PY" "e2b-code-interpreter==2.9.2"   # the AGS sandbox provider (not in OSWorld's requirements)
  pipi "$PY" -e "$ROOT[report,dev]"
  SITE=$("$PY" -c "import sysconfig; print(sysconfig.get_paths()['purelib'])")
  echo "$ROOT/third_party/OSWorld" > "$SITE/osworld.pth"   # import desktop_env / mm_agents from the checkout
  "$PY" -c "import desktop_env.desktop_env, desktop_env.providers.ags.provider, e2b_code_interpreter, mm_agents.agent, \
nightwatch.loop, nightwatch.agents.prompt_agent; print('harness env ok')"
  mark main
fi

# -------------------------------------------------------------------------------------------- s1
if has s1 && ! done_ s1; then
  if [[ -n "${S1_PYTHON:-}" ]]; then
    say "System-1 env: reusing $S1_PYTHON (needs torch + transformers 5.x; nightwatch runs from src/ via PYTHONPATH)"
    "$SPY" -c "import torch, transformers; print('torch', torch.__version__, 'transformers', transformers.__version__)"
  else
    say "System-1 env (.conda/envs/nightwatch-s1: torch 2.9.1 cu128, transformers 5.17.0)"
    [[ -x "$SPY" ]] || conda create -y -q -p "$ENVS/nightwatch-s1" -c conda-forge --override-channels python=3.12
    pipi "$SPY" --upgrade pip
    pipi "$SPY" "torch==2.9.1" "torchvision==0.24.1" --index-url https://download.pytorch.org/whl/cu128
    # transformers 5.17.0: required by the d3 runtime; Intern-Decision gives identical outputs (checked: max diff 2e-6)
    pipi "$SPY" "transformers==5.17.0" accelerate safetensors huggingface_hub pillow numpy scipy pyyaml httpx \
         einops opencv-python-headless peft matplotlib pandas pytest
    pipi "$SPY" "flash-linear-attention==0.5.2" || echo "flash-linear-attention unavailable: d3 uses the slower reference kernels"
    pipi "$SPY" -e "$ROOT" --no-deps
    "$SPY" -c "import torch, transformers, nightwatch.server; print('s1 env ok, cuda:', torch.cuda.is_available())"
  fi
  mark s1
fi

# ---------------------------------------------------------------------------------------- models
fetch() {  # hub_id local_name
  local id="$1" name="$2" snap d="$DATA/models/$2"
  [[ -e "$d/.fetched" ]] && { echo "have $name"; return 0; }
  # a config.json from anything but an (interruptible) local_dir download: a linked cache or a copied package
  [[ -e "$d/config.json" && ! -d "$d/.cache/huggingface/download" ]] && { echo "have $name"; return 0; }
  for snap in "${HF_LOCAL_CACHE:-$HOME/.cache/huggingface}/hub/models--${id//\//--}/snapshots/"*/; do
    if [[ -f "$snap/config.json" ]]; then ln -sfn "$snap" "$DATA/models/$name"; echo "linked $id from $snap"; return 0; fi
  done
  PYTHONPATH= "${FETCH_PY:-$SPY}" -c "from huggingface_hub import snapshot_download as d; d('$id', local_dir='$DATA/models/$name')" \
    || { echo "download of $id failed (private repos need HF_TOKEN in .env with access)" >&2; return 1; }
  touch "$d/.fetched"
}
if has models && ! done_ models; then
  MODELS_LIST="${MODELS:-d3-mini d3-lite nightwatch}"
  say "models (data/models): $MODELS_LIST, CLIP ViT-B/32"
  [[ -x "$SPY" ]] || { echo "run the s1 component first (it provides huggingface_hub)" >&2; exit 1; }
  for m in $MODELS_LIST; do
    case "$m" in
      d3*) fetch "vllm-sr/$m" "$m" ;;
      nightwatch) fetch "${NIGHTWATCH_MODEL_REPO:-nullptr04/NightWatch-0.8B}" NightWatch-0.8B ;;
      Intern-Decision-*) fetch "internlm/$m" "$m" ;;
      *) fetch "$m" "${m##*/}" ;;
    esac
  done
  fetch openai/clip-vit-base-patch32 clip-vit-base-patch32
  mark models
fi

# ---------------------------------------------------------------------------------------- dynacu
DPY="$ENVS/nightwatch-dynacu/bin/python"
if has dynacu && ! done_ dynacu; then
  say "DynaCU-Bench (third_party/aoi @ ${AOI_COMMIT:0:12}, .conda/envs/nightwatch-dynacu, Chromium)"
  AOI="$ROOT/third_party/aoi"
  [[ -d "$AOI/.git" ]] || git clone -q https://github.com/19PINE-AI/aoi.git "$AOI"
  git -C "$AOI" fetch -q origin || true
  git -C "$AOI" checkout -q "$AOI_COMMIT"
  # headless Chromium needs atk/at-spi/Xcomposite/Xdamage/cups (missing on this host, no root): take them from conda-forge
  [[ -x "$DPY" ]] || conda create -y -q -p "$ENVS/nightwatch-dynacu" -c conda-forge --override-channels python=3.12 \
      pulseaudio atk at-spi2-atk at-spi2-core xorg-libxcomposite xorg-libxdamage libcups nss ffmpeg git
  pipi "$DPY" --upgrade pip
  pipi "$DPY" torch torchvision --index-url https://download.pytorch.org/whl/cpu   # AOI's CLIP keyframe filter only
  pipi "$DPY" -r "$AOI/requirements.txt"
  pipi "$DPY" "av<14"   # faster-whisper 1.2 passes metadata_errors= to av.open, removed in PyAV 14
  pipi "$DPY" -e "$ROOT[report,dev]"
  PLAYWRIGHT_BROWSERS_PATH="$DATA/ms-playwright" "$DPY" -m playwright install chromium
  LD_LIBRARY_PATH="$ENVS/nightwatch-dynacu/lib" PLAYWRIGHT_BROWSERS_PATH="$DATA/ms-playwright" "$DPY" - <<'PYEOF'
import sys; sys.path.insert(0, "third_party/aoi")
from benchmark_env.browser_env import BrowserEnvironment
e = BrowserEnvironment("F_E1_toast_error_fix.html", audio_enabled=False); assert e.start()
print("dynacu ok: page", e.get_screenshot().size, "dom", e.check_success()); e.stop()
PYEOF
  if [[ "${DYNACU_AUDIO:-0}" == 1 ]]; then
    say "Whisper large-v3 (audio categories)"
    HF_HOME="$DATA/hf" "$DPY" -c "from faster_whisper import WhisperModel; WhisperModel('large-v3', device='cpu', compute_type='int8')"
  fi
  mark dynacu
fi

# ------------------------------------------------------------------------------------------ vllm
VPY="$ENVS/nightwatch-vllm/bin/python"
if has vllm && ! done_ vllm; then
  # PyPI wheels are built for CUDA 13 (driver >= 580); a 12.x driver gets the official cu129 build
  VLLM_VERSION="${VLLM_VERSION:-0.30.0}"
  DRIVER_CUDA="$(nvidia-smi 2>/dev/null | grep -o 'CUDA Version: [0-9]*' | grep -o '[0-9]*$' || echo 12)"
  say "vLLM $VLLM_VERSION env (.conda/envs/nightwatch-vllm, driver CUDA $DRIVER_CUDA): the local agent model"
  [[ -x "$VPY" ]] || conda create -y -q -p "$ENVS/nightwatch-vllm" -c conda-forge --override-channels python=3.12
  pipi "$VPY" --upgrade pip
  if (( DRIVER_CUDA >= 13 )); then
    pipi "$VPY" "vllm==$VLLM_VERSION"
  else
    pipi "$VPY" torch==2.13.0 torchvision torchaudio --index-url https://download.pytorch.org/whl/cu129
    pipi "$VPY" "vllm @ https://github.com/vllm-project/vllm/releases/download/v$VLLM_VERSION/vllm-$VLLM_VERSION%2Bcu129-cp38-abi3-manylinux_2_28_x86_64.whl" \
        --extra-index-url https://download.pytorch.org/whl/cu129
    # vLLM pulls torchcodec from PyPI, a CUDA 13 build (libnvrtc.so.13); it only decodes media on the CPU: CPU build
    pipi "$VPY" --no-deps --force-reinstall "torchcodec==0.17.0+cpu" --index-url https://download.pytorch.org/whl/cpu
  fi
  PYTHONPATH= "$VPY" -c "import torch, vllm; assert torch.cuda.is_available(), 'no GPU visible'; \
print('vllm', vllm.__version__, 'torch', torch.__version__, 'cuda', torch.version.cuda, torch.cuda.get_device_name(0))"
  PYTHONPATH= "$VPY" -c "import torchcodec.decoders; print('torchcodec', __import__('torchcodec').__version__, 'ok')"
  mark vllm
fi

# ------------------------------------------------------------------------------------------ qwen
if has qwen; then   # weights of the local agent model (no marker: a finished download is detected by its config)
  QWEN_MODEL="${QWEN_MODEL:-Qwen/Qwen3.8-27B-FP8}"
  say "agent model $QWEN_MODEL -> data/models/${QWEN_MODEL##*/}"
  [[ -x "$VPY" ]] || { echo "run the vllm component first (it provides huggingface_hub)" >&2; exit 1; }
  FETCH_PY="$VPY" fetch "$QWEN_MODEL" "${QWEN_MODEL##*/}"
fi
say "environment setup done: $COMPONENTS"
echo "next: bash scripts/setup_container.sh   (OSWorld: docker image + VM image + one test container)"
echo "      bash scripts/run_dynacu.sh        (DynaCU-Bench needs no container)"
