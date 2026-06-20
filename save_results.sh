#!/usr/bin/env bash
# Loss-proof the result artifacts so a box termination can't wipe them.
# Saves ONLY the lightweight outputs (JSON scores, lift results, plots, dashboard,
# logs) — never the multi-GB GRPO model checkpoints. Safe to run anytime / repeatedly.
#
#   bash save_results.sh ["commit message"]
#
# Does two things:
#   1) writes results_bundle.tar.gz  (always works — scp it off the box)
#   2) git commit + push             (best-effort — survives box death if auth is set)
set -uo pipefail
shopt -s nullglob

MSG="${1:-results checkpoint $(date -u +%FT%TZ)}"

paths=(
  results/cohorts/*.json
  results/xlift_scores/*.json
  results/grpo/*/lift_result.json
  results/plots/*.png
  results/dashboard.html
  results/overnight/*.log
)
if [ ${#paths[@]} -eq 0 ]; then
  echo "[save_results] nothing to save yet"
  exit 0
fi

# 1) Always-works artifact you can scp down even with no git auth.
tar czf results_bundle.tar.gz "${paths[@]}" 2>/dev/null \
  && echo "[save_results] wrote results_bundle.tar.gz ($(du -h results_bundle.tar.gz | cut -f1))"

# 2) Best-effort commit + push so the numbers survive the box being terminated.
git config --local user.email >/dev/null 2>&1 || git config --local user.email "xlift-overnight@local"
git config --local user.name  >/dev/null 2>&1 || git config --local user.name  "xlift overnight"
git add -f "${paths[@]}"
if git diff --cached --quiet; then
  echo "[save_results] no new changes to commit"
  exit 0
fi
git commit -q -m "$MSG"
if git push -q 2>/dev/null; then
  echo "[save_results] pushed results to GitHub ✓"
else
  echo "[save_results] PUSH FAILED (no git auth on this box). Results are saved locally + in"
  echo "               results_bundle.tar.gz. To make them survive termination, EITHER:"
  echo "                 set a token once:  git remote set-url origin https://<TOKEN>@github.com/mssumanth/xlift.git"
  echo "                 or scp the bundle: scp -P <port> -i <key> root@<ip>:$(pwd)/results_bundle.tar.gz ."
fi
