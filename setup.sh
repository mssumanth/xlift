#!/usr/bin/env bash
# One-command bootstrap for a fresh H100 box (e.g. Prime Intellect).
# Usage:
#   bash setup.sh
#   # then edit .env to add ANTHROPIC_API_KEY (and HF_TOKEN if needed)
set -euo pipefail

echo "==> Python venv"
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip

echo "==> Installing requirements (this pulls torch/transformers/trl — a few minutes)"
pip install -r requirements.txt

if [ ! -f .env ]; then
  cp .env.example .env
  echo "==> Created .env from template. EDIT IT NOW to add your ANTHROPIC_API_KEY."
  echo "    BASE_MODEL defaults to Qwen/Qwen2.5-1.5B-Instruct"
fi

echo "==> Building cohorts (MATH difficulty shortcut)"
python run_experiment.py --step data --shortcut

cat <<'EOF'

============================================================
Setup complete. Next steps:

  1. Make sure .env has ANTHROPIC_API_KEY set.

  2. SMOKE TEST (2 min) — confirms the Qwen backend loads & runs:
       XLIFT_BACKEND=qwen python run_experiment.py --step metrics --smoke

  3. FULL METRICS (Step 2):
       XLIFT_BACKEND=qwen python run_experiment.py --step metrics \
         --max-tasks 40 --rollouts 5 --gepa-gens 3

  4. TRAIN (Step 3) — GRPO per cohort:
       python run_experiment.py --step train --cohort easy
       python run_experiment.py --step train --cohort frontier
       python run_experiment.py --step train --cohort hard

  5. VISUALIZE (Step 4):
       python run_experiment.py --step visualize
============================================================
EOF
