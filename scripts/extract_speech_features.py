"""
extract_speech_features.py
==========================
Build ``data/raw/speech/parkinsons.csv`` from two UCI speech datasets:

  - UCI 174 (parkinsons.csv)          — 195 rows, all 22 MDVP features
  - UCI 301 (voice_uci301_*.csv)      — 1 040 rows, 29 acoustic columns

Both datasets use acoustic features recorded from sustained phonation.
UCI 301 uses different column names; this script maps them to the canonical
22 MDVP feature names used throughout the project.

Usage
-----
    python scripts/extract_speech_features.py [--data-dir data/raw]

Dataset download
----------------
UCI 174  — already bundled: data/raw/speech/parkinsons.csv
UCI 301  — already bundled: data/raw/speech/voice_uci301_multiple_sound_train.csv
    Mirror: https://raw.githubusercontent.com/adachille/parkinsons-detector/
            master/data/multiple-sound-recording/train_data.csv
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]

# ---------------------------------------------------------------------------
# The 22 MDVP feature names used across the whole project
# ---------------------------------------------------------------------------
MDVP_FEATURES = [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ",
    "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer(dB)",
    "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA",
    "NHR", "HNR", "RPDE", "DFA", "spread1", "spread2", "D2", "PPE",
]

# ---------------------------------------------------------------------------
# UCI 301 column mapping
# ---------------------------------------------------------------------------
# UCI 301 train_data.csv columns (29 total):
#   subject_id, jitter_local, jitter_local_absolute, jitter_rap, jitter_ppq5,
#   jitter_ddp, shimmer_local, shimmer_local_db, shimmer_apq3, shimmer_apq5,
#   shimmer_apq11, shimmer_data, AC, NTH, HTN, median_pitch, mean_pitch,
#   standard_dev_pitch, min_pitch, max_pitch, num_pulses, num_periods,
#   mean_period, standard_dev_period, frac_locally_unvoiced_frames,
#   num_voice_breaks, degree_of_voice_breaks, UPDRS, class_info

_UCI301_MAP = {
    "median_pitch":           "MDVP:Fo(Hz)",
    "max_pitch":              "MDVP:Fhi(Hz)",
    "min_pitch":              "MDVP:Flo(Hz)",
    "jitter_local":           "MDVP:Jitter(%)",
    "jitter_local_absolute":  "MDVP:Jitter(Abs)",
    "jitter_rap":             "MDVP:RAP",
    "jitter_ppq5":            "MDVP:PPQ",
    "jitter_ddp":             "Jitter:DDP",
    "shimmer_local":          "MDVP:Shimmer",
    "shimmer_local_db":       "MDVP:Shimmer(dB)",
    "shimmer_apq3":           "Shimmer:APQ3",
    "shimmer_apq5":           "Shimmer:APQ5",
    "shimmer_apq11":          "MDVP:APQ",
    "shimmer_data":           "Shimmer:DDA",
    "NTH":                    "NHR",
    "HTN":                    "HNR",
    "class_info":             "status",
}

# These 6 MDVP features have no direct UCI-301 counterpart; they are filled
# with per-column medians computed from the UCI-174 dataset.
_UCI301_MISSING = ["RPDE", "DFA", "spread1", "spread2", "D2", "PPE"]


def _load_uci174(speech_dir: Path) -> pd.DataFrame:
    """Load UCI-174 parkinsons.csv — already has all 22 MDVP columns."""
    path = speech_dir / "parkinsons.csv"
    if not path.is_file():
        raise FileNotFoundError(f"UCI-174 not found: {path}")
    df = pd.read_csv(path)
    df["status"] = df["status"].astype(int)
    print(f"  UCI-174: {len(df)} rows  ({path.name})")
    return df[MDVP_FEATURES + ["status"]].copy()


def _load_uci301(speech_dir: Path, medians_174: pd.Series) -> pd.DataFrame | None:
    """
    Load UCI-301 and map its 29 columns to the 22 MDVP feature names.

    The 6 features absent in UCI-301 (RPDE, DFA, spread1, spread2, D2, PPE)
    are filled with per-column medians from UCI-174 so the model does not
    receive entirely meaningless zeros for those positions.
    """
    candidates = [
        speech_dir / "voice_uci301_multiple_sound_train.csv",
        speech_dir / "uci301" / "train_data.csv",
        speech_dir / "uci301" / "train_data.txt",
    ]
    path = next((p for p in candidates if p.is_file()), None)
    if path is None:
        print("  UCI-301: not found — skipping (place train_data.csv under data/raw/speech/)")
        return None

    raw = pd.read_csv(path)
    df = raw.rename(columns=_UCI301_MAP)

    # Build output with all 22 MDVP features
    out = pd.DataFrame(index=df.index)
    for feat in MDVP_FEATURES:
        if feat in df.columns:
            out[feat] = pd.to_numeric(df[feat], errors="coerce")
        else:
            # Fill with UCI-174 median
            out[feat] = medians_174[feat] if feat in medians_174.index else 0.0

    out["status"] = (pd.to_numeric(df["status"], errors="coerce") > 0).astype(int)

    # Drop rows where any mapped column is NaN
    out = out.dropna(subset=MDVP_FEATURES)
    print(f"  UCI-301: {len(out)} rows mapped  ({path.name})")
    return out[MDVP_FEATURES + ["status"]].copy()


def build_speech_csv(data_dir: Path, output_path: Path) -> None:
    """Merge UCI-174 and UCI-301 into a single enriched speech CSV."""
    speech_dir = data_dir / "speech"

    df174 = _load_uci174(speech_dir)
    medians_174 = df174[MDVP_FEATURES].median()

    df301 = _load_uci301(speech_dir, medians_174)

    frames = [df174]
    if df301 is not None:
        frames.append(df301)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=MDVP_FEATURES)
    combined["status"] = combined["status"].astype(int)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output_path, index=False)

    n_pd = int(combined["status"].sum())
    n_hc = len(combined) - n_pd
    print(f"\nOutput: {output_path}")
    print(f"  Rows : {len(combined)}  (PD={n_pd}, HC={n_hc})")
    print(f"  Cols : {list(combined.columns)}")


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--data-dir",
        default=str(_REPO_ROOT / "data" / "raw"),
        help="Root data directory (default: data/raw)",
    )
    p.add_argument(
        "--output",
        default=None,
        help="Output CSV path (default: <data-dir>/speech/parkinsons.csv)",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    data_dir = Path(args.data_dir)
    output_path = (
        Path(args.output) if args.output
        else data_dir / "speech" / "parkinsons.csv"
    )
    print("=== Speech Feature Extractor ===")
    try:
        build_speech_csv(data_dir, output_path)
    except FileNotFoundError as exc:
        print(f"\nERROR: {exc}", file=sys.stderr)
        sys.exit(1)
