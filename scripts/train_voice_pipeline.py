#!/usr/bin/env python3
"""
Optional speech CSV download (fixed HTTPS URLs only).

Multimodal training uses ``train_dl.py`` (1D CNN / SE-ResNet encoders + attention fusion + dense head),
not tabular LR/SVM.
"""

from __future__ import annotations

import argparse
import ssl
import sys
import urllib.request
from pathlib import Path
from typing import List, Tuple

# Fixed URL → filename under data/raw/speech/ (not user-controlled).
SPEECH_DOWNLOADS: List[Tuple[str, str]] = [
    (
        "https://raw.githubusercontent.com/SagarBapodara/Parkison-Disease-Detection-using-Machine-Learning/main/Data/parkinsons.csv",
        "parkinsons.csv",
    ),
    (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/parkinsons/telemonitoring/parkinsons_updrs.data",
        "parkinsons_telemonitoring.csv",
    ),
    (
        "https://archive.ics.uci.edu/ml/machine-learning-databases/00489/ReplicatedAcousticFeatures-ParkinsonDatabase.csv",
        "voice_uci489_replicated_acoustic.csv",
    ),
    (
        "https://raw.githubusercontent.com/adachille/parkinsons-detector/master/data/multiple-sound-recording/train_data.csv",
        "voice_uci301_multiple_sound_train.csv",
    ),
    (
        "https://raw.githubusercontent.com/adachille/parkinsons-detector/master/data/disease-classification/pd_speech_features.csv",
        "voice_uci470_pd_classification_features.csv",
    ),
]


def download_speech_csvs(speech_dir: Path) -> None:
    """Download speech-related CSVs into ``speech_dir`` using verified TLS."""
    import certifi

    speech_dir.mkdir(parents=True, exist_ok=True)
    ctx = ssl.create_default_context(cafile=certifi.where())
    for url, name in SPEECH_DOWNLOADS:
        dest = speech_dir / name
        req = urllib.request.Request(url, headers={"User-Agent": "ParkinsonTrainVoicePipeline/1.0"})
        print(f"Downloading {name} …")
        with urllib.request.urlopen(req, context=ctx, timeout=120) as resp:
            dest.write_bytes(resp.read())
        print(f"  → {dest}")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Download speech CSVs (optional). Train with train_dl.py, not LR/SVM."
    )
    parser.add_argument(
        "--download-speech",
        action="store_true",
        help="Refresh speech CSVs under data/raw/speech/ from fixed public URLs.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))

    if not args.download_speech:
        print("This script only downloads speech CSVs when you pass --download-speech.")
        print("Train the multimodal deep model with:  python train_dl.py")
        return 0

    download_speech_csvs(root / "data" / "raw" / "speech")
    print("Done. Next: python train_dl.py")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
