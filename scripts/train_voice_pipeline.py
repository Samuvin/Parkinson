#!/usr/bin/env python3
"""
Voice (tabular) training pipeline: optional CSV refresh, train LR+SVM, export reports.

Uses fixed HTTPS URLs only (same sources as data/raw/speech/download_speech_csvs.sh).
"""

from __future__ import annotations

import argparse
import json
import ssl
import sys
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

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


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    if isinstance(value, (np.floating, np.float32, np.float64)):
        return float(value)
    if isinstance(value, (np.integer, np.int32, np.int64)):
        return int(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


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
    parser = argparse.ArgumentParser(description="Train tabular voice models and export reports.")
    parser.add_argument(
        "--download-speech",
        action="store_true",
        help="Refresh speech CSVs under data/raw/speech/ from fixed public URLs.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    sys.path.insert(0, str(root))

    if args.download_speech:
        download_speech_csvs(root / "data" / "raw" / "speech")

    from src.utils.config import ensure_dir_exists, get_reports_dir
    from train import run_voice_training

    ensure_dir_exists(get_reports_dir())
    result = run_voice_training(save_preprocessed=True)

    lr_coef = np.asarray(result.lr_model.model.coef_).ravel()
    metrics_path = get_reports_dir() / "voice_metrics.json"
    payload: Dict[str, Any] = {
        "feature_names": result.feature_names,
        "best_model_name": result.best_model_name,
        "logistic_regression": _json_safe(result.lr_metrics),
        "svm": _json_safe(result.svm_metrics),
        "logistic_regression_coefficients": [
            [name, float(c)] for name, c in zip(result.feature_names, lr_coef)
        ],
    }
    metrics_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Wrote {metrics_path}")

    npz_path = get_reports_dir() / "voice_test_predictions.npz"
    np.savez_compressed(
        npz_path,
        X_test=result.X_test,
        y_test=result.y_test,
        y_pred_lr=result.y_pred_lr,
        y_proba_lr=result.y_proba_lr,
        y_pred_svm=result.y_pred_svm,
        y_proba_svm=result.y_proba_svm,
        lr_coef=lr_coef,
    )
    print(f"Wrote {npz_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
