#!/usr/bin/env bash
# Overnight end-to-end run: metrics (step 2) -> training (step 3) -> plots (step 4).
# Resilient by design — does NOT abort on a single failure, so you wake up to
# whatever completed. Every step is timestamped and logged.
#
# Recommended (survives SSH disconnect):
#   tmux new -s xlift           # then run the line below inside tmux
#   XLIFT_BACKEND=qwen bash run_all.sh
#   # detach with Ctrl-b then d ; reattach later with: tmux attach -t xlift
#
# Or with nohup:
#   nohup env XLIFT_BACKEND=qwen bash run_all.sh > results/overnight.out 2>&1 &
#
# Tunables (override on the command line, e.g. MAX_TASKS=40 ROLLOUTS=5):
MAX_TASKS="${MAX_TASKS:-40}"
ROLLOUTS="${ROLLOUTS:-5}"
GEPA_GENS="${GEPA_GENS:-3}"
STEPS="${STEPS:-200}"
export XLIFT_BACKEND="${XLIFT_BACKEND:-qwen}"

# Interpreter (boxes often have python3 but no `python`)
PY="$(command -v python3 || command -v python || true)"
if [ -z "$PY" ]; then echo "No python3/python on PATH"; exit 1; fi
export PY

mkdir -p results/overnight
RUN_LOG="results/overnight/run_$(date +%Y%m%d_%H%M%S).log"

log() { echo "[$(date '+%H:%M:%S')] $*" | tee -a "$RUN_LOG"; }

log "================ xLift overnight run ================"
log "backend=$XLIFT_BACKEND  max_tasks=$MAX_TASKS  rollouts=$ROLLOUTS  gepa_gens=$GEPA_GENS  steps=$STEPS"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader 2>&1 | tee -a "$RUN_LOG" || true

# --- Pre-flight: make sure all 5 cohorts + eval set exist ---
if [ ! -f results/cohorts/eval_set.json ] || \
   [ ! -f results/cohorts/mixed.json ] || \
   [ ! -f results/cohorts/weak_verifier.json ]; then
  log "Cohorts missing or incomplete — building all 5 (step data)..."
  "$PY" run_experiment.py --step data --shortcut >> "$RUN_LOG" 2>&1
fi

# --- STEP 2: metrics, one cohort at a time so a failure in one keeps the others ---
log ">>> STEP 2: metrics (5 cohorts: frontier easy hard mixed weak_verifier)"
for c in frontier easy hard mixed weak_verifier; do
  log "  metrics: $c cohort ..."
  "$PY" run_experiment.py --step metrics --cohort "$c" \
      --max-tasks "$MAX_TASKS" --rollouts "$ROLLOUTS" --gepa-gens "$GEPA_GENS" \
      >> "results/overnight/metrics_$c.log" 2>&1
  rc=$?
  if [ $rc -eq 0 ]; then log "  metrics: $c DONE"; else log "  metrics: $c FAILED (rc=$rc) — see results/overnight/metrics_$c.log"; fi
  bash save_results.sh "results: metrics $c" >> "$RUN_LOG" 2>&1   # loss-proof after each cohort
done

# --- STEP 3: training (auto-adapts to GPU count via train_parallel.sh) ---
log ">>> STEP 3: training (train_parallel.sh, $STEPS steps/cohort)"
bash train_parallel.sh "$STEPS" >> "results/overnight/train.log" 2>&1
rc=$?
if [ $rc -eq 0 ]; then log "  training DONE"; else log "  training FAILED/partial (rc=$rc) — see results/overnight/train.log and results/train_*.log"; fi
bash save_results.sh "results: training done" >> "$RUN_LOG" 2>&1

# --- STEP 4: plots (only useful if at least metrics + some training landed) ---
log ">>> STEP 4: visualize"
"$PY" run_experiment.py --step visualize >> "results/overnight/visualize.log" 2>&1
rc=$?
if [ $rc -eq 0 ]; then log "  plots DONE -> results/plots/"; else log "  visualize FAILED (rc=$rc) — see results/overnight/visualize.log"; fi

# --- Dashboard (self-contained HTML, always safe to build) ---
log ">>> Building dashboard"
"$PY" eval/dashboard.py >> "results/overnight/dashboard.log" 2>&1 \
  && log "  dashboard DONE -> results/dashboard.html" \
  || log "  dashboard FAILED — see results/overnight/dashboard.log"
bash save_results.sh "results: final (metrics+train+dashboard)" >> "$RUN_LOG" 2>&1

# --- Morning summary ---
log "================ SUMMARY ================"
"$PY" run_experiment.py --step status 2>&1 | tee -a "$RUN_LOG" || true
log "xLift scores:  results/xlift_scores/*.json"
log "Train lifts:   results/grpo/*/lift_result.json"
log "Plots:         results/plots/*.png"
log "Full log:      $RUN_LOG"
log "================ DONE ================"
