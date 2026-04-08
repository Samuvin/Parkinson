#!/usr/bin/env bash
# Install app + ML + Jupyter deps and open the multimodal staff demo notebook.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
NB="${ROOT}/notebooks/parkinson_multimodal_staff_demo.ipynb"

echo "==> Notebook environment (repo: $ROOT)"

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
python -m pip install \
  -r requirements.txt \
  -r requirements-ml.txt \
  -r requirements-notebooks.txt \
  -q

if [[ ! -f "$NB" ]]; then
  echo "Missing notebook: $NB" >&2
  exit 1
fi

echo "==> Launching Jupyter with $NB"
exec python -m jupyter notebook "$NB" --notebook-dir="$ROOT/notebooks"
