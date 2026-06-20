#!/usr/bin/env bash
# Train all 3 cohorts in PARALLEL, one per GPU (use on a multi-GPU box, e.g. 3-4x A100).
# Each cohort is an independent job writing to results/grpo/<cohort>/ — no collisions.
#
# Usage:
#   bash train_parallel.sh [STEPS]
#   STEPS defaults to 200.
#
# Prereq: run `python run_experiment.py --step data --shortcut` once first so the
# cohorts AND the held-out eval_set.json exist (single-process, avoids any race).
set -euo pipefail

STEPS="${1:-200}"

if [ ! -f results/cohorts/eval_set.json ]; then
  echo "eval_set.json missing — run: python run_experiment.py --step data --shortcut"
  exit 1
fi

echo "Launching 3 cohorts in parallel ($STEPS steps each), one per GPU..."

CUDA_VISIBLE_DEVICES=0 python run_experiment.py --step train --cohort easy     --steps "$STEPS" > results/train_easy.log     2>&1 &
PID_EASY=$!
CUDA_VISIBLE_DEVICES=1 python run_experiment.py --step train --cohort frontier --steps "$STEPS" > results/train_frontier.log 2>&1 &
PID_FRONTIER=$!
CUDA_VISIBLE_DEVICES=2 python run_experiment.py --step train --cohort hard     --steps "$STEPS" > results/train_hard.log     2>&1 &
PID_HARD=$!

echo "  easy     -> GPU0 (pid $PID_EASY)     log: results/train_easy.log"
echo "  frontier -> GPU1 (pid $PID_FRONTIER) log: results/train_frontier.log"
echo "  hard     -> GPU2 (pid $PID_HARD)     log: results/train_hard.log"
echo "Tail a log with:  tail -f results/train_frontier.log"

wait $PID_EASY $PID_FRONTIER $PID_HARD
echo "All three cohorts finished. Lift results:"
for c in easy frontier hard; do
  echo "  $c: $(cat results/grpo/$c/lift_result.json 2>/dev/null | python -c 'import sys,json; d=json.load(sys.stdin); print(f"baseline={d[\"baseline_accuracy\"]:.1%} post={d[\"post_training_accuracy\"]:.1%} lift={d[\"actual_lift\"]:+.1%}")' 2>/dev/null || echo 'no result')"
done
