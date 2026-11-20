#!/usr/bin/env bash
# OSWorld-Verified, 100 tasks x 3 repeats (configs/osworld.yaml): Vanilla (fixed_0), AOI, PUSV and NightWatch.
# One experiment per method, so each can be started, stopped and resumed on its own; episodes with a
# result.json are skipped on a restart. Needs scripts/setup_env.sh, the OSWorld provider in .env
# (docker or ags, see .env.example) and, for NightWatch, the watcher servers (scripts/serve_watcher.sh).
#
#   bash scripts/run_osworld.sh start     # launch every method in the background
#   bash scripts/run_osworld.sh status    # finished / 300 per method, errors, running or not
#   bash scripts/run_osworld.sh retry     # set aside episodes that failed at setup, relaunch (resume)
#   bash scripts/run_osworld.sh metrics   # success, time breakdown, waste, interventions, cost -> runs/osworld_metrics
#   bash scripts/run_osworld.sh waste     # the wasted-time ledger alone -> runs/osworld_waste
#   bash scripts/run_osworld.sh stop      # SIGTERM to each run's process group: workers close their sandboxes
#                                         # on the way out (never kill -9: AGS sandboxes would leak)
#
#   METHODS="fixed_0 aoi_visual pusv nightwatch_full"   # default
#   WORKERS=6                                           # environments per method (AGS quota: METHODS x WORKERS)
#   SET="frames.record=true"                            # extra --set overrides (space-separated)
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
PY="${NIGHTWATCH_PYTHON:-$ROOT/.conda/envs/nightwatch/bin/python}"
METHODS="${METHODS:-fixed_0 aoi_visual pusv nightwatch_full}"
WORKERS="${WORKERS:-6}"
PREFIX="${PREFIX:-osworld}"
TOTAL=300
EXTRA=()
for kv in ${SET:-}; do EXTRA+=(--set "$kv"); done
mkdir -p runs
exp() { echo "${PREFIX}_$1"; }
running() { [[ -f "runs/$(exp "$1").pid" ]] && kill -0 "$(cat "runs/$(exp "$1").pid")" 2>/dev/null; }
# result.json files of an experiment (setup failures set aside by `retry` excluded); counts never fail under pipefail
results() { find "runs/$1" -name result.json -not -path '*_setup_failed*' 2>/dev/null || true; }
count() { local f; f=$(results "$1"); [[ -z "$f" ]] && { echo 0; return; }; grep -l "$2" $f 2>/dev/null | wc -l || true; }
exps() { for m in $METHODS; do printf '%s,' "$(exp "$m")"; done | sed 's/,$//'; }

launch() {
  local m="$1" e
  e="$(exp "$m")"
  if running "$m"; then echo "$e already running (pid $(cat "runs/$e.pid"))"; return; fi
  # own process group (setsid): `stop` signals the runner and its spawn workers together
  setsid nohup "$PY" -m nightwatch run --config osworld --name "$e" --methods "$m" --workers "$WORKERS" \
      "${EXTRA[@]}" >> "runs/$e.log" 2>&1 &
  echo $! > "runs/$e.pid"
  echo "started $e (pid $!, log runs/$e.log)"
}

case "${1:-status}" in
  start)
    for m in $METHODS; do launch "$m"; done ;;
  status)
    for m in $METHODS; do
      e="$(exp "$m")"
      n=$(results "$e" | wc -l)
      err=$(count "$e" '"error": "[^"]')
      ok=$(count "$e" '"success": true')
      printf '%-28s %4d / %d done, %3d passed, %3d setup errors, %s\n' "$e" "$n" "$TOTAL" "$ok" "$err" \
        "$(running "$m" && echo running || echo stopped)"
    done ;;
  retry)
    for m in $METHODS; do
      e="$(exp "$m")"
      if running "$m"; then echo "$e is running: retry after it stops"; continue; fi
      files=$(results "$e")
      for f in $( [[ -n "$files" ]] && grep -l '"error": "[^"]' $files 2>/dev/null || true ); do
        d="$(dirname "$f")"; dest="runs/$e/_setup_failed/${d#runs/$e/}"
        mkdir -p "$(dirname "$dest")"; rm -rf "$dest"; mv "$d" "$dest"; echo "set aside $d"
      done
      launch "$m"
    done ;;
  metrics)
    "$PY" -m nightwatch metrics --exps "$(exps)" --out "runs/${PREFIX}_metrics" ;;
  waste)
    "$PY" -m nightwatch waste --exps "$(exps)" --out "runs/${PREFIX}_waste" ;;
  stop)
    for m in $METHODS; do
      running "$m" || continue
      pid="$(cat "runs/$(exp "$m").pid")"
      # workers close their sandboxes on the first SIGTERM; the runner can miss one while it waits on them: repeat
      for i in $(seq 1 6); do
        pgrep -g "$pid" >/dev/null || break
        kill -TERM -- "-$pid" 2>/dev/null || true
        sleep 10
      done
      if pgrep -g "$pid" >/dev/null; then echo "$(exp "$m"): still running after 60 s (check runs/$(exp "$m").log)"
      else echo "stopped $(exp "$m")"; fi
    done ;;
  *) echo "usage: $0 start|status|retry|metrics|waste|stop" >&2; exit 1 ;;
esac
