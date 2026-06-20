#!/usr/bin/env bash
# Bootstrap for a Prime Intellect / GPU box (Ubuntu image with CUDA torch preinstalled).
#
#   bash setup.sh
#   # then edit .env to add ANTHROPIC_API_KEY (and HF_TOKEN if you have one)
#
# Notes:
#  - Uses `uv` if available (much faster installs); falls back to pip otherwise.
#    Set USE_UV=0 to force pip, or ALLOW_UV_INSTALL=1 to auto-install uv if missing.
#  - By default we install into the EXISTING python env so the box's CUDA-matched
#    torch is kept (a fresh venv would force a slow torch reinstall and can break CUDA).
#  - vLLM is NOT installed by default — the overnight flow (run_experiment.py +
#    models/backend.py uses transformers, grpo_train uses TRL). vLLM is only needed
#    for the optional `xlift/` package (python -m xlift.cli). Set INSTALL_VLLM=1 for it.
#  - On a laptop with no system torch, set USE_VENV=1 to get an isolated env.
set -uo pipefail

USE_VENV="${USE_VENV:-0}"
INSTALL_VLLM="${INSTALL_VLLM:-0}"
USE_UV="${USE_UV:-1}"
ALLOW_UV_INSTALL="${ALLOW_UV_INSTALL:-0}"

# Detect the Python interpreter (boxes often have python3 but no `python`).
PY="$(command -v python3 || command -v python || true)"
if [ -z "$PY" ]; then
  echo "!! No python3/python found on PATH. Install Python first."; exit 1
fi
echo "==> Using interpreter: $PY ($($PY --version 2>&1))"

# Detect uv (fast installer). Optionally install it.
UV="$(command -v uv || true)"
if [ -z "$UV" ] && [ "$USE_UV" = "1" ] && [ "$ALLOW_UV_INSTALL" = "1" ]; then
  echo "==> Installing uv ..."
  curl -LsSf https://astral.sh/uv/install.sh | sh && UV="$HOME/.local/bin/uv"
fi
if [ "$USE_UV" = "1" ] && [ -n "$UV" ]; then
  echo "==> Using uv for installs: $("$UV" --version)"
else
  UV=""
  echo "==> uv not used; falling back to pip"
fi

# Install a requirements file with uv (fast) or pip (fallback), into the right env.
pip_install() {  # pip_install <reqfile>
  if [ -n "$UV" ]; then
    if [ "$USE_VENV" = "1" ]; then "$UV" pip install -r "$1"; else "$UV" pip install --python "$PY" -r "$1"; fi
  else
    "$PY" -m pip install -r "$1"
  fi
}

if [ "$USE_VENV" = "1" ]; then
  echo "==> Creating isolated venv (.venv)"
  if [ -n "$UV" ]; then "$UV" venv .venv; else "$PY" -m venv .venv; fi
  source .venv/bin/activate
  PY=python   # inside the venv, `python` exists
fi

[ -z "$UV" ] && "$PY" -m pip install --upgrade pip >/dev/null

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
  # strip inline comments and surrounding whitespace
  line="$(echo "$line" | sed 's/[[:space:]]*#.*$//' | sed 's/[[:space:]]*$//;s/^[[:space:]]*//')"
  [ -z "$line" ] && continue
  pkg_lower="$(echo "$line" | tr '[:upper:]' '[:lower:]')"
  case "$pkg_lower" in
    torchvision*) ;;                                  # keep torchvision (matched before torch*)
    torch*)  [ "$HAVE_TORCH" = "1" ] && continue ;;
    vllm*)   [ "$INSTALL_VLLM" != "1" ] && continue ;;
  esac
  echo "$line"
done > "$TMP_REQ"

echo "    Installing:"; sed 's/^/      /' "$TMP_REQ"
pip_install "$TMP_REQ"
RC=$?
rm -f "$TMP_REQ"
if [ $RC -ne 0 ]; then
  echo "!! install failed (rc=$RC). Fix the error above, then re-run setup.sh."
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
       python3 -c "import torch; print(torch.cuda.is_available(), torch.cuda.device_count())"

  3. SMOKE TEST (2 min):
       XLIFT_BACKEND=qwen python3 run_experiment.py --step metrics --smoke

  4. OVERNIGHT (metrics + train + plots), inside tmux so it survives disconnect:
       tmux new -s xlift
       XLIFT_BACKEND=qwen bash run_all.sh
       # detach: Ctrl-b then d   |   reattach: tmux attach -t xlift

  (installs use uv when available; USE_UV=0 forces pip, ALLOW_UV_INSTALL=1 installs uv)
============================================================
EOF
