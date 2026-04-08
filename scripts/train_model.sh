#!/usr/bin/env bash
# Install training dependencies and run multimodal DL training (train_dl.py).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==> Train MultimodalPDNet (repo: $ROOT)"

py() {
  if command -v python >/dev/null 2>&1; then
    python "$@"
  elif command -v python3 >/dev/null 2>&1; then
    python3 "$@"
  else
    echo "Missing Python. Install Python 3 and ensure 'python' (or 'python3') is on PATH." >&2
    exit 1
  fi
}

if [[ ! -d venv ]]; then
  py -m venv venv
fi

# shellcheck source=/dev/null
if [[ -f venv/bin/activate ]]; then
  source venv/bin/activate
elif [[ -f venv/Scripts/activate ]]; then
  # Windows venv layout (works in Git Bash / MSYS2)
  source venv/Scripts/activate
else
  echo "Could not find venv activation script under venv/ (expected bin/activate or Scripts/activate)." >&2
  exit 1
fi

python -m pip install --upgrade pip -q
python -m pip install -r requirements.txt -r requirements-ml.txt -q

echo "==> python train_dl.py $*"
exec python train_dl.py "$@"
