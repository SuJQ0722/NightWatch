#!/usr/bin/env bash
# Container / VM setup for OSWorld (one click, idempotent). Separate from the Python environment
# (scripts/setup_env.sh), which must have run first for `smoke`.
#
#   bash scripts/setup_container.sh            # check -> pull -> vm -> smoke
#   bash scripts/setup_container.sh check      # docker access, /dev/kvm, disk (prints the admin commands if needed)
#   bash scripts/setup_container.sh pull       # docker pull happysixd/osworld-docker
#   bash scripts/setup_container.sh vm         # download + unzip Ubuntu.qcow2 into data/vm (12.3 GB zip, resumable)
#   bash scripts/setup_container.sh smoke      # start one OSWorld container, measure screenshot latency / fps, stop it
#   bash scripts/setup_container.sh status     # running OSWorld containers
#   bash scripts/setup_container.sh stop       # stop + remove all OSWorld containers (orphans after a crash)
#
# Provider: OSWORLD_PROVIDER = docker (default) | ags.
#   ags  Tencent Cloud Agent Sandbox (E2B-compatible): `check` verifies E2B_API_KEY / E2B_DOMAIN / AGS_TEMPLATE
#        are set (never printed), `ags` installs the cookbook's provider into our OSWorld pin + the e2b SDK,
#        `smoke` creates one sandbox, measures screenshots and kills it. No docker / KVM / VM image needed.
#        bash scripts/setup_container.sh check ags smoke
# OSWorld's DockerProvider starts one container per
# environment itself (ports 5000/8006/9222/8080 upward); this script prepares the image, the VM disk and
# verifies that a container boots. For cloud providers (aws, ...) only `check` applies.
# HF_ENDPOINT=https://hf-mirror.com switches the VM download to the mirror (same as OSWorld).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
load_env() { local k v; while IFS='=' read -r k v; do [[ -z "${!k:-}" && -n "$v" ]] && export "$k=$v"; done \
  < <(grep -E '^[A-Z_][A-Z0-9_]*=' "$1" | sed 's/\r$//'); return 0; }
[[ -f .env ]] && load_env .env
PROVIDER="${OSWORLD_PROVIDER:-docker}"
if [[ "$PROVIDER" == ags ]]; then STEPS="${*:-check ags smoke}"; else STEPS="${*:-check pull vm smoke}"; fi
IMAGE="${OSWORLD_DOCKER_IMAGE:-happysixd/osworld-docker}"
VMDIR="$ROOT/data/vm"
URL="https://huggingface.co/datasets/xlangai/ubuntu_osworld/resolve/main/Ubuntu.qcow2.zip"
[[ "${HF_ENDPOINT:-}" == *hf-mirror.com* ]] && URL="${URL/huggingface.co/hf-mirror.com}"
PY="$ROOT/.conda/envs/nightwatch/bin/python"
say() { printf '\n\033[36m=== %s\033[0m\n' "$*"; }
has() { [[ " $STEPS " == *" $1 "* ]]; }
ME="$(id -un)"
mkdir -p "$VMDIR"

docker_ok() { docker info >/dev/null 2>&1; }

# ------------------------------------------------------------------------------------------- check
if has check; then
  say "check (provider: $PROVIDER)"
  fail=0
  if [[ "$PROVIDER" == docker ]]; then
    if ! command -v docker >/dev/null; then
      echo "FAIL docker CLI not installed. An admin needs to install Docker Engine (https://docs.docker.com/engine/install/)."
      fail=1
    elif docker_ok; then
      echo "ok   docker daemon reachable as $ME ($(docker version --format '{{.Server.Version}}' 2>/dev/null))"
    else
      echo "FAIL $ME cannot talk to the docker daemon: $(docker info 2>&1 | grep -m1 -i -E 'permission|error|cannot' || true)"
      echo "     Ask an admin for ONE of:"
      echo "       sudo usermod -aG docker $ME            # then log out and back in (or: newgrp docker)"
      echo "       rootless docker for $ME: needs uidmap + /etc/subuid,/etc/subgid entries, then dockerd-rootless-setuptool.sh install"
      echo "     (podman also works through its docker-compatible socket: export DOCKER_HOST=unix://\$XDG_RUNTIME_DIR/podman/podman.sock)"
      fail=1
    fi
    if [[ -e /dev/kvm ]]; then
      echo "ok   /dev/kvm present (the docker daemon passes it into the container; $ME's own access is not needed for docker)"
    else
      echo "FAIL /dev/kvm missing: the VM would be emulated (TCG), which is far too slow for wall-clock measurements."
      echo "     Enable virtualisation (VT-x / AMD-V) in the BIOS or use a cloud provider (OSWORLD_PROVIDER=aws)."
      fail=1
    fi
  elif [[ "$PROVIDER" == ags ]]; then
    for k in E2B_API_KEY E2B_DOMAIN AGS_TEMPLATE; do
      if [[ -n "${!k:-}" ]]; then echo "ok   $k is set"; else echo "FAIL $k is empty (.env)"; fail=1; fi
    done
    [[ -n "${E2B_DOMAIN:-}" ]] && echo "     domain: $E2B_DOMAIN"
  else
    echo "--   provider $PROVIDER: OSWorld reads its credentials from the environment (see third_party/OSWorld/desktop_env/providers/$PROVIDER)."
  fi
  free_gb=$(df -BG --output=avail "$VMDIR" | tail -1 | tr -dc 0-9)
  if (( free_gb < 60 )); then echo "FAIL only ${free_gb} GB free under $VMDIR (need ~40 GB: zip + qcow2)"; fail=1
  else echo "ok   ${free_gb} GB free under $VMDIR"; fi
  if (( fail )) && [[ " $STEPS " != " check " ]]; then
    echo; echo "Container prerequisites missing. Continuing with the steps that do not need the daemon (vm download)."
    STEPS="${STEPS/pull/}"; STEPS="${STEPS/smoke/}"
  fi
fi

# --------------------------------------------------------------------------------------------- ags
AGS_COOKBOOK_COMMIT=20f7b6dcd2e143ee073b6159016966a014b3cd89
if has ags; then
  say "AGS provider (ags-cookbook @ ${AGS_COOKBOOK_COMMIT:0:12}) into third_party/OSWorld + e2b SDK"
  CB="$ROOT/third_party/ags-cookbook"
  [[ -d "$CB/.git" ]] || git clone -q https://github.com/TencentCloudAgentRuntime/ags-cookbook.git "$CB"
  git -C "$CB" fetch -q origin || true
  git -C "$CB" checkout -q "$AGS_COOKBOOK_COMMIT"
  [[ -x "$PY" ]] || { echo "harness env missing: bash scripts/setup_env.sh" >&2; exit 1; }
  "$PY" -m pip install -q --no-warn-script-location "e2b==2.46.0" "e2b-code-interpreter==2.9.2" "aiohttp>=3.9" python-dotenv
  "$PY" "$ROOT/scripts/lib/patch_osworld_ags.py" "$ROOT/third_party/OSWorld" "$CB/examples/osworld-ags/overlay/OSWorld"
  "$PY" -c "from desktop_env.providers import create_vm_manager_and_provider as f; import desktop_env.providers.ags.provider; print('ags provider importable')"
fi

# -------------------------------------------------------------------------------------------- pull
if has pull && [[ "$PROVIDER" == docker ]]; then
  say "docker pull $IMAGE"
  docker image inspect "$IMAGE" >/dev/null 2>&1 && echo "image present" || docker pull "$IMAGE"
fi

# ---------------------------------------------------------------------------------------------- vm
if has vm && [[ "$PROVIDER" == docker ]]; then
  say "OSWorld Ubuntu VM image -> $VMDIR/Ubuntu.qcow2"
  if [[ -f "$VMDIR/Ubuntu.qcow2" && -f "$VMDIR/.unzipped" ]]; then
    echo "present ($(du -h "$VMDIR/Ubuntu.qcow2" | cut -f1))"
  else
    curl -fL --progress-bar --retry 20 --retry-delay 10 -C - -o "$VMDIR/Ubuntu.qcow2.zip" "$URL"
    want=12273896463
    have=$(stat -c %s "$VMDIR/Ubuntu.qcow2.zip")
    [[ "$URL" == *huggingface.co* && "$have" != "$want" ]] && { echo "size $have != $want: rerun to resume" >&2; exit 1; }
    (cd "$VMDIR" && unzip -o -q Ubuntu.qcow2.zip)
    touch "$VMDIR/.unzipped"
    [[ "${KEEP_VM_ZIP:-0}" == 1 ]] || rm -f "$VMDIR/Ubuntu.qcow2.zip"
    echo "ok: $(du -h "$VMDIR/Ubuntu.qcow2" | cut -f1)"
  fi
fi

# ------------------------------------------------------------------------------------------- smoke
if has smoke; then
  say "smoke: boot one OSWorld environment, measure screenshots, shut it down"
  [[ -x "$PY" ]] || { echo "harness env missing: bash scripts/setup_env.sh" >&2; exit 1; }
  "$PY" - <<'PYEOF'
import json, time, statistics
from nightwatch.config import DATA_DIR, load_config
from nightwatch.envs.osworld import OSWorldEnv
cfg = load_config("base")
t0 = time.time()
e = OSWorldEnv(cfg.env.osworld)
boot = time.time() - t0
lat = []
for _ in range(30):
    t = time.time(); img, _ = e.grab(); lat.append(time.time() - t)
    assert img and img[:4] == b"\x89PNG", "screenshot failed"
e.execute("import pyautogui; pyautogui.moveTo(10, 10)")
out = {"boot_s": round(boot, 1), "vm_ip": e.vm_ip, "screenshot_ms_p50": round(1000 * statistics.median(lat), 1),
       "screenshot_ms_max": round(1000 * max(lat), 1), "max_fps": round(1 / statistics.median(lat), 1)}
e.close()
(DATA_DIR / "container_smoke.json").write_text(json.dumps(out, indent=2))
print(json.dumps(out, indent=2))
if out["max_fps"] < 5:
    print("NOTE: screenshots are slower than 5 fps; the frame stream will run below frames.hz (achieved fps is logged)")
PYEOF
fi

# ------------------------------------------------------------------------------------- status/stop
if has status; then
  docker ps --filter "ancestor=$IMAGE" --format 'table {{.ID}}\t{{.Status}}\t{{.Ports}}'
fi
if has stop; then
  ids=$(docker ps -aq --filter "ancestor=$IMAGE")
  [[ -n "$ids" ]] && docker rm -f $ids || echo "no OSWorld containers"
fi
say "container setup done: $STEPS"
