#!/usr/bin/env bash
# Install app + ML + Jupyter deps and open the multimodal staff demo notebook.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
NB="${ROOT}/notebooks/parkinson_multimodal_staff_demo.ipynb"

echo "==> Notebook environment (repo: $ROOT)"

if [[ ! -d venv ]]; then
  python3 -m venv venv
fi
# shellcheck source=/dev/null
source venv/bin/activate

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
