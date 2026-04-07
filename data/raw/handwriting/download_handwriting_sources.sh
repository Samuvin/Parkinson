#!/usr/bin/env bash
# Download handwriting-related public archives (fixed HTTPS URLs only).
# UCI 395: Parkinson Disease Spiral Drawings (pen traces + readme).
# Optional: set DOWNLOAD_KAGGLE_HANDWRITING=1 to also fetch Kaggle spiral images
# (requires ~/.kaggle/kaggle.json and the kaggle CLI).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"
UCI395_ZIP_URL="https://archive.ics.uci.edu/static/public/395/parkinson+disease+spiral+drawings+using+digitized+graphics+tablet.zip"
DEST_DIR="$ROOT/uci_spiral"
TMP_ZIP="$ROOT/.uci395_spiral_tmp.zip"

mkdir -p "$DEST_DIR"
curl -fsSL -A "Mozilla/5.0 (research; +https://github.com)" "$UCI395_ZIP_URL" -o "$TMP_ZIP"
unzip -qo "$TMP_ZIP" -d "$DEST_DIR"
rm -f "$TMP_ZIP"

echo "OK: UCI spiral traces extracted under $DEST_DIR"
echo "Tabular summary: python scripts/uci_spiral_traces_to_csv.py"

if [ "${DOWNLOAD_KAGGLE_HANDWRITING:-}" = "1" ]; then
  KDIR="$ROOT/kaggle_spiral"
  mkdir -p "$KDIR"
  if ! command -v kaggle >/dev/null 2>&1; then
    echo "ERROR: kaggle CLI not found; install kaggle or skip Kaggle mirror." >&2
    exit 1
  fi
  kaggle datasets download -d team-ai/parkinson-disease-spiral-drawings -p "$KDIR" --unzip
  echo "OK: Kaggle spiral images under $KDIR"
fi
