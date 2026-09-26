#!/usr/bin/env bash
# Auto-restarting supervisor for a pool runner (scripts/run_weekend5.py by
# default; set RUNNER / RESULTS_CSV for another batch, e.g. run_weekend6.py).
#
# This shared 8xH200 node has a documented recurring freeze (PROJECT_STATUS.md,
# 2026-07-01 and again 2026-08-05): all pool GPUs drop to 0% utilization with
# memory still held, log files go idle, and the training processes never
# recover on their own. Previously this required a human (or another session)
# to notice the box was unresponsive and manually kill + relaunch. This script
# automates that: it launches run_weekend5.py, watches pool GPU utilization,
# and if every pool GPU has read 0% for STALL_MINUTES straight (well past the
# ~20-25 min multi-model-loading phase, so this does not fire on a legitimate
# cold start), it kills the run and relaunches. run_weekend5.py / do_job are
# already idempotent (skip on existing slot_metrics.json, reuse existing Stage-1
# checkpoints), so a restart just resumes.
#
# Usage:
#   tmux new -s batch5
#   source .venv/bin/activate && export $(grep -v '^#' .env | xargs)
#   export TMPDIR=$PWD/.tmp
#   export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#   POOL_GPUS=0,1,2,3,4,5,6,7 SAVE_TOP_K=1 bash scripts/weekend_watchdog.sh
#   RUNNER=scripts/run_weekend6.py RESULTS_CSV=outputs/weekend6_results.csv \
#       POOL_GPUS=0,1,2,3,4,5,6,7 SAVE_TOP_K=1 NUM_WORKERS=2 bash scripts/weekend_watchdog.sh
#
# Several pools of one batch can run side by side: give each a TAG (names its
# logs) and its own JOB_ONLY / JOB_SKIP. A stall restart only kills the process
# tree of its own runner.
set -uo pipefail
cd "$(dirname "$0")/.."

RUNNER="${RUNNER:-scripts/run_weekend5.py}"
RESULTS_CSV="${RESULTS_CSV:-outputs/weekend_results.csv}"
TAG="${TAG:+_$TAG}"
RUNNER_LOG="logs/weekend/$(basename "$RUNNER" .py | sed 's/run_weekend/runner/')${TAG}.log"
POOL_GPUS="${POOL_GPUS:-0,1,2,3,4,5,6,7}"
POLL_SECONDS="${POLL_SECONDS:-300}"        # check every 5 min
STALL_MINUTES="${STALL_MINUTES:-40}"       # 0% util this long => declare a stall
GRACE_MINUTES="${GRACE_MINUTES:-30}"       # no stall checks for this long after a (re)launch
MAX_RESTARTS="${MAX_RESTARTS:-6}"          # give up after this many restarts of either kind
LOG="logs/weekend/watchdog${TAG}.log"
mkdir -p logs/weekend
STALL_POLLS=$(( STALL_MINUTES * 60 / POLL_SECONDS ))

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*" | tee -a "$LOG"; }

remaining_jobs() {
    RUNNER="$RUNNER" RESULTS_CSV="$RESULTS_CSV" python3 - << 'PY'
import csv, importlib.util, os, sys
sys.path.insert(0, "scripts")
spec = importlib.util.spec_from_file_location("m", os.environ["RUNNER"])
m = importlib.util.module_from_spec(spec)
spec.loader.exec_module(m)
done = set()
try:
    for r in csv.DictReader(open(os.environ["RESULTS_CSV"])):
        # "skip" = metrics already existed when the runner was (re)launched
        if r.get("status", "") in ("ok", "skip"):
            done.add(r["name"])
except FileNotFoundError:
    pass
only = set(filter(None, os.environ.get("JOB_ONLY", "").split(",")))
skip = set(filter(None, os.environ.get("JOB_SKIP", "").split(",")))
jobs = [j["name"] for j in m.JOBS if (not only or j["name"] in only) and j["name"] not in skip]
print(sum(1 for name in jobs if name not in done))
PY
}

pool_all_idle() {
    # true iff EVERY pool GPU currently reports 0% utilization
    local util
    util=$(nvidia-smi --query-gpu=index,utilization.gpu --format=csv,noheader,nounits \
           | awk -F', *' -v gpus="$POOL_GPUS" '
             BEGIN { n=split(gpus, want, ","); for (i=1;i<=n;i++) w[want[i]]=1 }
             ($1 in w) { print $2 }')
    for u in $util; do
        [ "$u" -ne 0 ] && return 1
    done
    return 0
}

launch() {
    log "Launching $RUNNER on GPUs [$POOL_GPUS] (save_top_k=${SAVE_TOP_K:-default}, num_workers=${NUM_WORKERS:-default})"
    POOL_GPUS="$POOL_GPUS" SAVE_TOP_K="${SAVE_TOP_K:-}" NUM_WORKERS="${NUM_WORKERS:-}" \
        python3 "$RUNNER" >> "$RUNNER_LOG" 2>&1 &
    RUNNER_PID=$!
    LAST_LAUNCH=$(date +%s)
    log "$RUNNER started, pid=$RUNNER_PID"
}

descendants() {
    local child
    for child in $(ps -o pid= --ppid "$1"); do
        echo "$child"
        descendants "$child"
    done
}

kill_all() {
    log "Killing orchestrator (pid=$RUNNER_PID) and its training processes"
    local tree
    tree="$RUNNER_PID $(descendants "$RUNNER_PID")"
    kill $tree 2>/dev/null
    sleep 10
    kill -9 $tree 2>/dev/null
    sleep 5
}

restarts=0
launch
idle_polls=0

while true; do
    sleep "$POLL_SECONDS"

    remaining=$(remaining_jobs)
    if [ "$remaining" -eq 0 ]; then
        log "All jobs have ok status in the CSV. Done."
        break
    fi

    if ! kill -0 "$RUNNER_PID" 2>/dev/null; then
        restarts=$(( restarts + 1 ))
        if [ "$restarts" -gt "$MAX_RESTARTS" ]; then
            log "Orchestrator exited with $remaining job(s) remaining after $MAX_RESTARTS relaunches; giving up (jobs failing permanently?)."
            break
        fi
        log "Orchestrator exited (pid=$RUNNER_PID gone) with $remaining job(s) still remaining. Relaunch #$restarts."
        launch
        idle_polls=0
        continue
    fi

    elapsed_since_launch=$(( $(date +%s) - LAST_LAUNCH ))
    if [ "$elapsed_since_launch" -lt $(( GRACE_MINUTES * 60 )) ]; then
        continue  # still inside the model-loading grace period, don't stall-check yet
    fi

    if pool_all_idle; then
        idle_polls=$(( idle_polls + 1 ))
        log "Pool GPUs [$POOL_GPUS] all at 0% util (idle_polls=$idle_polls/$STALL_POLLS)"
    else
        idle_polls=0
    fi

    if [ "$idle_polls" -ge "$STALL_POLLS" ]; then
        restarts=$(( restarts + 1 ))
        log "STALL DETECTED: ${STALL_MINUTES}min of 0% pool util. Restart #$restarts."
        if [ "$restarts" -gt "$MAX_RESTARTS" ]; then
            log "Exceeded MAX_RESTARTS=$MAX_RESTARTS. Giving up; needs human attention."
            break
        fi
        kill_all
        launch
        idle_polls=0
    fi
done

log "Watchdog exiting. Remaining jobs: $(remaining_jobs)"
