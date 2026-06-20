#!/usr/bin/env bash
# Bootstrap for a Prime Intellect / GPU box (Ubuntu image with CUDA torch preinstalled).
#
#   bash setup.sh
#   # then edit .env to add ANTHROPIC_API_KEY (and HF_TOKEN if you have one)
#
# Notes:
#  - By default we install into the EXISTING python env so the box's CUDA-matched
#    torch is kept (a fresh venv would force a slow torch reinstall and can break CUDA).
#  - vLLM is NOT installed by default — the overnight flow (run_experiment.py +
#    models/backend.py uses transformers, grpo_train uses TRL). vLLM is only needed
#    for the optional `xlift/` package (python -m xlift.cli). Set INSTALL_VLLM=1 for it.
#  - On a laptop with no system torch, set USE_VENV=1 to get an isolated env.
set -uo pipefail

USE_VENV="${USE_VENV:-0}"
INSTALL_VLLM="${INSTALL_VLLM:-0}"

if [ "$USE_VENV" = "1" ]; then
  echo "==> Creating isolated venv (.venv)"
  python3 -m venv .venv
  source .venv/bin/activate
fi

PY=python
$PY -m pip install --upgrade pip >/dev/null

echo "==> Checking for preinstalled CUDA torch..."
if $PY -c "import torch; assert torch.cuda.is_available()" 2>/dev/null; then
  echo "    OK — $($PY -c 'import torch;print("torch", torch.__version__, "| CUDA", torch.cuda.is_available(), "| GPUs", torch.cuda.device_count())')"
  HAVE_TORCH=1
else
  echo "    No CUDA torch found — it will be installed from requirements (slow)."
  HAVE_TORCH=0
fi

echo "==> Installing Python requirements"
# Build a filtered requirements list: drop torch if already present (keep the
# box's CUDA build) and drop vllm unless explicitly requested.
TMP_REQ="$(mktemp)"
grep -vE '^\s*#' requirements.txt | while IFS= read -r line; do
  pkg_lower="$(echo "$line" | tr '[:upper:]' '[:lower:]')"
  case "$pkg_lower" in
    torch*)  [ "$HAVE_TORCH" = "1" ] && continue ;;
    vllm*)   [ "$INSTALL_VLLM" != "1" ] && continue ;;
  esac
  echo "$line"
done > "$TMP_REQ"

echo "    Installing:"; sed 's/^/      /' "$TMP_REQ"
$PY -m pip install -r "$TMP_REQ"
RC=$?
rm -f "$TMP_REQ"
if [ $RC -ne 0 ]; then
  echo "!! pip install failed (rc=$RC). Fix the error above, then re-run setup.sh."
  exit $RC
fi

if [ ! -f .env ]; then
  cp .env.example .env
  echo "==> Created .env from template. EDIT IT to add your ANTHROPIC_API_KEY."
fi

echo "==> Building cohorts (MATH difficulty shortcut)"
$PY run_experiment.py --step data --shortcut

cat <<'EOF'

============================================================
Setup complete. Next steps:

  1. Edit .env  ->  ANTHROPIC_API_KEY=sk-ant-...   (HF_TOKEN optional)

  2. Confirm GPUs:
       python -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"

  3. SMOKE TEST (2 min):
       XLIFT_BACKEND=qwen python run_experiment.py --step metrics --smoke

  4. OVERNIGHT (metrics + train + plots), inside tmux so it survives disconnect:
       tmux new -s xlift
       XLIFT_BACKEND=qwen bash run_all.sh
       # detach: Ctrl-b then d   |   reattach: tmux attach -t xlift
============================================================
EOF
