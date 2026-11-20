#!/usr/bin/env bash
# DynaCU-Bench, 49 dynamic tasks x 3 repeats (after `bash scripts/setup_env.sh dynacu`; no docker / VM needed).
#
#   bash scripts/run_dynacu.sh                         # check smoke serve main metrics
#   STAGES="check smoke" bash scripts/run_dynacu.sh    # offline: real Chromium + task pages, scripted agent, no key
#   STAGES="serve ablations" bash scripts/run_dynacu.sh
#
# Stages:
#   check      Chromium starts, a task page loads, DOM check works
#   smoke      configs/dynacu_smoke.yaml: 3 tasks, scripted agent (pipeline check only, never a result)
#   serve      the watcher servers (scripts/serve_watcher.sh), stopped again at exit
#   main       configs/dynacu.yaml: Vanilla (standard) and AOI in AOI's own harness, PUSV and NightWatch -> runs/dynacu
#   ablations  configs/dynacu_ablations.yaml -> runs/dynacu_ablations (watcher_d3_27b: see that file)
#   metrics    success, time breakdown and waste of `main` -> runs/dynacu_metrics
# Knobs: WORKERS (default 3), METHODS (comma-separated, overrides a stage's methods).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
load_env() { local k v; while IFS='=' read -r k v; do [[ -z "${!k:-}" && -n "$v" ]] && export "$k=$v"; done \
  < <(grep -E '^[A-Z_][A-Z0-9_]*=' "$1" | sed 's/\r$//'); return 0; }
[[ -f .env ]] && load_env .env
# the DynaCU request body (provider pin without the OSWorld reasoning setting), see .env.example
[[ -n "${DYNACU_AGENT_EXTRA_BODY:-}" ]] && export AGENT_EXTRA_BODY="$DYNACU_AGENT_EXTRA_BODY"
E="$ROOT/.conda/envs/nightwatch-dynacu"
PY="$E/bin/python"
[[ -x "$PY" ]] || { echo "DynaCU env missing: bash scripts/setup_env.sh dynacu" >&2; exit 1; }
export LD_LIBRARY_PATH="$E/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}" PLAYWRIGHT_BROWSERS_PATH="$ROOT/data/ms-playwright"
export PATH="$E/bin:$PATH" HF_HOME="$ROOT/data/hf" PYTHONPATH="$ROOT/src${PYTHONPATH:+:$PYTHONPATH}"
STAGES="${STAGES:-check smoke serve main metrics}"
WORKERS="${WORKERS:-3}"
has() { [[ " $STAGES " == *" $1 "* ]]; }
say() { printf '\n\033[35m### %s\033[0m\n' "$*"; }
cleanup() { [[ "${SERVED:-0}" == 1 ]] && bash scripts/serve_watcher.sh stop >/dev/null || true; }
trap cleanup EXIT
run() {
  local extra=()
  [[ -n "${METHODS:-}" ]] && extra+=(--methods "$METHODS")
  "$PY" -m nightwatch run --config "$1" --workers "$WORKERS" "${extra[@]}"
}

if has check; then
  say "check: Chromium + one task page"
  (cd third_party/aoi && "$PY" - <<'PYEOF'
import sys; sys.path.insert(0, ".")
from benchmark_env.browser_env import BrowserEnvironment
from dynacubench.tasks_v3 import DynaCUBenchV3
b = DynaCUBenchV3(); t = b.get_task("F-E1")
e = BrowserEnvironment(t.html_file, audio_enabled=False); assert e.start(), "Chromium failed to start"
print("ok:", t.task_id, e.get_screenshot().size, e.check_success()); e.stop()
PYEOF
  )
fi
if has smoke; then
  say "smoke (scripted agent; pipeline check only)"
  "$PY" -m nightwatch run --config dynacu_smoke
fi
if has serve && ! curl -fs http://127.0.0.1:8782/health >/dev/null 2>&1; then
  say "watcher servers"
  bash scripts/serve_watcher.sh && SERVED=1
fi
if has main; then
  say "main: 49 tasks x 3"
  run dynacu
fi
if has ablations; then
  say "ablations: 49 tasks x 3"
  run dynacu_ablations
fi
if has metrics; then
  say metrics
  "$PY" -m nightwatch metrics --exps dynacu --out runs/dynacu_metrics
fi
