#!/usr/bin/env bash
# Train all 3 cohorts across available GPUs. Auto-adapts to GPU count:
#   - 3+ GPUs: all 3 cohorts at once (one per GPU)
#   - 2 GPUs:  wave 1 = frontier(GPU0) + hard(GPU1), wave 2 = easy(GPU0)
#   - 1 GPU:   sequential
# Each cohort writes to results/grpo/<cohort>/ — no collisions.
#
# Usage:  bash train_parallel.sh [STEPS]   (STEPS defaults to 200)
#
# Prereq: run `python run_experiment.py --step data --shortcut` once first so the
# cohorts AND held-out eval_set.json exist (single-process, avoids any race).
set -euo pipefail

STEPS="${1:-200}"

if [ ! -f results/cohorts/eval_set.json ]; then
  echo "eval_set.json missing — run: python run_experiment.py --step data --shortcut"
  exit 1
fi

# Count visible GPUs
NGPU=$(python -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)
echo "Detected $NGPU GPU(s). Training 3 cohorts ($STEPS steps each)."
mkdir -p results

run() {  # run <gpu_id> <cohort>
  CUDA_VISIBLE_DEVICES="$1" python run_experiment.py --step train --cohort "$2" --steps "$STEPS" \
    > "results/train_$2.log" 2>&1
}

if [ "$NGPU" -ge 3 ]; then
  run 0 easy & run 1 frontier & run 2 hard & wait
elif [ "$NGPU" -eq 2 ]; then
  echo "Wave 1: frontier->GPU0, hard->GPU1 (logs: results/train_frontier.log, train_hard.log)"
  run 0 frontier & run 1 hard & wait
  echo "Wave 2: easy->GPU0 (log: results/train_easy.log)"
  run 0 easy
else
  for c in frontier hard easy; do echo "Training $c (log: results/train_$c.log)"; run 0 "$c"; done
fi

echo "All cohorts finished. Lift results:"
for c in easy frontier hard; do
  printf "  %-9s " "$c"
  python -c "import json;d=json.load(open('results/grpo/$c/lift_result.json'));print(f'baseline={d[\"baseline_accuracy\"]:.1%} post={d[\"post_training_accuracy\"]:.1%} lift={d[\"actual_lift\"]:+.1%}')" 2>/dev/null || echo "no result"
done
