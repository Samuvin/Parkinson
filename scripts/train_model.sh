#!/usr/bin/env bash
# Install training dependencies and run multimodal DL training (train_dl.py).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==> Train MultimodalPDNet (repo: $ROOT)"

if [[ ! -d venv ]]; then
  python3 -m venv venv
fi
# shellcheck source=/dev/null
source venv/bin/activate

python -m pip install --upgrade pip -q
python -m pip install -r requirements.txt -r requirements-ml.txt -q

echo "==> python train_dl.py $*"
exec python train_dl.py "$@"
