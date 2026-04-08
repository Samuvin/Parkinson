#!/usr/bin/env bash
# Run the Flask + Waitress web app from the repo root (same stack as ./start.sh).
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

echo "==> Parkinson web app (repo: $ROOT)"

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

echo "==> Starting http://0.0.0.0:${PORT:-8000}"
exec python wsgi.py
