#!/usr/bin/env bash
# Run the Flask + Waitress web app from the repo root (same stack as ./start.sh).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==> Parkinson web app (repo: $ROOT)"

if [[ ! -d venv ]]; then
  python3 -m venv venv
fi
# shellcheck source=/dev/null
source venv/bin/activate

python -m pip install --upgrade pip -q
python -m pip install -r requirements.txt -r requirements-ml.txt -q

echo "==> Starting http://0.0.0.0:${PORT:-8000}"
exec python wsgi.py
