#!/usr/bin/env bash
# Refresh speech-related CSVs under this directory (fixed HTTPS URLs only).
set -euo pipefail

ROOT="$(cd "$(dirname "$0")" && pwd)"

curl -fsSL "https://raw.githubusercontent.com/SagarBapodara/Parkison-Disease-Detection-using-Machine-Learning/main/Data/parkinsons.csv" \
  -o "$ROOT/parkinsons.csv"

curl -fsSL "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/telemonitoring/parkinsons_updrs.data" \
  -o "$ROOT/parkinsons_telemonitoring.csv"

curl -fsSL "https://archive.ics.uci.edu/ml/machine-learning-databases/00489/ReplicatedAcousticFeatures-ParkinsonDatabase.csv" \
  -o "$ROOT/voice_uci489_replicated_acoustic.csv"

curl -fsSL "https://raw.githubusercontent.com/adachille/parkinsons-detector/master/data/multiple-sound-recording/train_data.csv" \
  -o "$ROOT/voice_uci301_multiple_sound_train.csv"

curl -fsSL "https://raw.githubusercontent.com/adachille/parkinsons-detector/master/data/disease-classification/pd_speech_features.csv" \
  -o "$ROOT/voice_uci470_pd_classification_features.csv"

echo "OK: wrote speech CSVs under $ROOT"
wc -l "$ROOT"/*.csv 2>/dev/null | tail -20
