"""
pipelines/handwriting_pipeline.py
==================================
Extracts 10 handwriting features from PaHaW pre-computed CSVs and writes
``data/processed/handwriting/handwriting_data.csv``.

Raw data location: data/raw/handwriting/000XX__N_1.csv  (597 files, 8 tasks × ~75 subjects)

Each PaHaW CSV contains ~978 pre-computed statistical features of the raw
digitised pen-movement signals (x, y, pressure, azimuth, altitude, velocity,
displacement, …).  We select / derive the 10 clinically meaningful features
that match ``config/multimodal_features.yaml``:

  Feature name          PaHaW source column(s)
  ──────────────────────────────────────────────────────────────────────────
  tremor_power          pressure_variance
  spiral_irregularity   azimuth_variance            (pen-angle rotation spread)
  stroke_smoothness     1 / (1 + velocity_variance) (inverted velocity jitter)
  contour_complexity    displacement_variance
  tremor_frequency      pressure_Number_of_changing_point
  pen_up_ratio          ratio_of_in_air_time
  mean_stroke_width     displacement_mean
  drawing_speed_proxy   velocity_mean
  line_waviness         y_variance                  (vertical-position spread)
  fluency_score         velocity_mean / (velocity_standard_deviation + 1e-6)
  ──────────────────────────────────────────────────────────────────────────

All 8 tasks are used (not just the spiral task) so that each subject
contributes up to 8 rows, raising the output from 75 to ~597 rows.

Label: ``label`` column (last) in each PaHaW file — 1 = PD, 0 = HC.
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

HW_FEATURES = [
    "tremor_power",
    "spiral_irregularity",
    "stroke_smoothness",
    "contour_complexity",
    "tremor_frequency",
    "pen_up_ratio",
    "mean_stroke_width",
    "drawing_speed_proxy",
    "line_waviness",
    "fluency_score",
]

# Direct PaHaW column → target feature name.
# stroke_smoothness and fluency_score are computed from velocity columns.
_PAHAW_MAP = {
    "pressure_variance":               "tremor_power",
    "azimuth_variance":                "spiral_irregularity",
    "displacement_variance":           "contour_complexity",
    "pressure_Number_of_changing_point": "tremor_frequency",
    "ratio_of_in_air_time":            "pen_up_ratio",
    "displacement_mean":               "mean_stroke_width",
    "velocity_mean":                   "drawing_speed_proxy",
    "y_variance":                      "line_waviness",
}


def _load_pahaw_file(path: Path) -> Optional[pd.Series]:
    """Load one PaHaW CSV row and return a Series with 10 features + label."""
    try:
        df = pd.read_csv(path, header=0)
    except Exception:
        return None

    if df.empty or df.shape[0] < 1:
        return None

    row = df.iloc[0]
    out: dict = {}

    # Direct column mappings
    for src, dst in _PAHAW_MAP.items():
        val = pd.to_numeric(row.get(src, np.nan), errors="coerce")
        out[dst] = float(val) if pd.notna(val) else np.nan

    # stroke_smoothness = 1 / (1 + velocity_variance)
    vel_var = pd.to_numeric(row.get("velocity_variance", np.nan), errors="coerce")
    out["stroke_smoothness"] = (1.0 / (1.0 + float(vel_var))
                                if pd.notna(vel_var) else np.nan)

    # fluency_score = velocity_mean / (velocity_standard_deviation + 1e-6)
    vel_mean = pd.to_numeric(row.get("velocity_mean",                np.nan), errors="coerce")
    vel_std  = pd.to_numeric(row.get("velocity_standard_deviation",  np.nan), errors="coerce")
    if pd.notna(vel_mean) and pd.notna(vel_std):
        out["fluency_score"] = float(vel_mean) / (float(vel_std) + 1e-6)
    else:
        out["fluency_score"] = np.nan

    # label
    label_val = pd.to_numeric(row.get("label", np.nan), errors="coerce")
    out["status"] = int(label_val) if pd.notna(label_val) else np.nan

    return pd.Series(out)


def run(raw_dir: Path, processed_dir: Path) -> Path:
    """Extract handwriting features from PaHaW CSVs and save processed CSV."""
    hw_raw_dir = Path(raw_dir) / "handwriting"
    out_dir = Path(processed_dir) / "handwriting"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "handwriting_data.csv"

    print("[Handwriting Pipeline]")

    # Use all task files (tasks 1-8) for every subject — excludes Jupyter
    # checkpoint artefacts (*-checkpoint.csv).
    all_task_files = sorted(
        f for f in hw_raw_dir.glob("*__*_1.csv")
        if "checkpoint" not in f.name
    )

    if not all_task_files:
        print("  No PaHaW CSV files found. Ensure data/raw/handwriting/000XX__N_1.csv exist.")
        return out_path

    print(f"  Using all tasks: {len(all_task_files)} files")

    records = []
    n_ok = n_fail = 0

    for f in all_task_files:
        row = _load_pahaw_file(f)
        if row is None or row[HW_FEATURES].isna().all():
            n_fail += 1
            continue
        records.append(row)
        n_ok += 1

    if not records:
        print("  No valid handwriting records extracted.")
        return out_path

    df = pd.DataFrame(records)

    # Fill any remaining NaNs with column medians
    for col in HW_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    df = df.dropna(subset=HW_FEATURES)
    df["status"] = df["status"].fillna(df["status"].median()).astype(int)

    # Clip extreme outliers (> 99.9th / < 0.1th percentile)
    for col in HW_FEATURES:
        cap   = df[col].quantile(0.999)
        floor = df[col].quantile(0.001)
        df[col] = df[col].clip(lower=floor, upper=cap)

    df = df[HW_FEATURES + ["status"]].reset_index(drop=True)
    df.to_csv(out_path, index=False)

    n_pd = int(df["status"].sum())
    n_hc = len(df) - n_pd
    print(f"  Extracted: {n_ok} ok, {n_fail} skipped")
    print(f"  → {out_path}  ({len(df)} rows: PD={n_pd}, HC={n_hc})\n")
    return out_path


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir",       default="data/raw")
    p.add_argument("--processed-dir", default="data/processed")
    args = p.parse_args()
    run(Path(args.raw_dir), Path(args.processed_dir))
