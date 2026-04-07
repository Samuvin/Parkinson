#!/usr/bin/env bash
# Gait raw mirrors: no single UCI curl URL matches this repo's gait CSV contract.
# This script optionally downloads the Kaggle tabular mirror (credentials required).
# For project-shaped gait_data.csv (ten summary features), run:
#   python scripts/build_handwriting_gait_from_public_sources.py
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"

if [ "${DOWNLOAD_KAGGLE_GAIT:-}" != "1" ]; then
  echo "Skip Kaggle download (set DOWNLOAD_KAGGLE_GAIT=1 to enable)."
  echo "See $ROOT/GAIT_DATASETS_SOURCES.txt for URLs and CLI examples."
  exit 0
fi

KDIR="$ROOT/kaggle_mirror"
mkdir -p "$KDIR"
if ! command -v kaggle >/dev/null 2>&1; then
  echo "ERROR: kaggle CLI not found." >&2
  exit 1
fi
kaggle datasets download -d zarif98sjs/gait-in-parkinsons-disease -p "$KDIR" --unzip
echo "OK: Kaggle gait mirror under $KDIR"
