#!/usr/bin/env bash
# Train all 5 cohorts across available GPUs. Auto-adapts to GPU count:
#   - 2+ GPUs: interleaved waves so GPUs stay busy
#   - 1 GPU:   sequential (frontier first — most important result)
# Each cohort writes to results/grpo/<cohort>/ — no collisions.
#
# Usage:  bash train_parallel.sh [STEPS]   (STEPS defaults to 200)
#
# Prereq: run `python run_experiment.py --step data --shortcut` once first so the
# cohorts AND held-out eval_set.json exist (single-process, avoids any race).
set -euo pipefail

STEPS="${1:-200}"

# Interpreter (boxes often have python3 but no `python`); allow override via $PY
PY="${PY:-$(command -v python3 || command -v python || true)}"
if [ -z "$PY" ]; then echo "No python3/python on PATH"; exit 1; fi

if [ ! -f results/cohorts/eval_set.json ]; then
  echo "eval_set.json missing — run: $PY run_experiment.py --step data --shortcut"
  exit 1
fi

# Count visible GPUs
NGPU=$("$PY" -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)
echo "Detected $NGPU GPU(s). Training 5 cohorts ($STEPS steps each)."
mkdir -p results

run() {  # run <gpu_id> <cohort>
  local gpu="$1" cohort="$2"
  CUDA_VISIBLE_DEVICES="$gpu" "$PY" run_experiment.py \
    --step train --cohort "$cohort" --steps "$STEPS" \
    > "results/train_$cohort.log" 2>&1
}

# Priority order: frontier first (headline result), then hard/easy, then mixed/weak_verifier
if [ "$NGPU" -ge 2 ]; then
  echo "Wave 1: frontier->GPU0, hard->GPU1"
  run 0 frontier & run 1 hard & wait

  echo "Wave 2: easy->GPU0, mixed->GPU1"
  run 0 easy & run 1 mixed & wait

  echo "Wave 3: weak_verifier->GPU0 (the reward-hacking exhibit)"
  run 0 weak_verifier
else
  for c in frontier hard easy mixed weak_verifier; do
    echo "Training $c (log: results/train_$c.log)"
    run 0 "$c"
  done
fi

echo ""
echo "All cohorts finished. Lift results:"
for c in frontier hard easy mixed weak_verifier; do
  printf "  %-14s " "$c"
  "$PY" -c "
import json, sys
try:
    d = json.load(open('results/grpo/$c/lift_result.json'))
    print(f'baseline={d[\"baseline_accuracy\"]:.1%}  post={d[\"post_training_accuracy\"]:.1%}  lift={d[\"actual_lift\"]:+.1%}')
except Exception as e:
    print(f'no result ({e})')
" 2>/dev/null || echo "no result"
done
