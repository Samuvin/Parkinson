"""
pipelines/handwriting_pipeline.py
==================================
Extracts 10 handwriting features from PaHaW pre-computed CSVs and writes
``data/processed/handwriting/handwriting_data.csv``.

Raw data location: data/raw/handwriting/000XX__N_1.csv
  Task 8 (spiral drawing) files: 000XX__8_1.csv  ← primary source
  Fallback: all tasks aggregated if task-8 files absent.

The PaHaW CSVs already contain 1000+ pre-computed statistical features.
We select/derive the 10 most clinically meaningful ones (matching
config/multimodal_features.yaml):

  tremor_power         ← pressure_variance
  spiral_irregularity  ← angle_data_with_1_variance   (local angle variance)
  stroke_smoothness    ← 1 / (1 + velocity_variance)  (inverted variance)
  contour_complexity   ← displacement_variance
  tremor_frequency     ← pressure_Number_of_changing_point
  pen_up_ratio         ← ratio_of_in_air_time
  mean_stroke_width    ← displacement_mean
  drawing_speed_proxy  ← velocity_mean
  line_waviness        ← angle_data_with_50_variance   (broader angle variance)
  fluency_score        ← velocity_mean / (velocity_standard_deviation + 1e-6)

Label: last column ``label`` in each PaHaW file (1 = PD, 0 = HC).
"""
from __future__ import annotations

import re
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

# PaHaW source column → target feature name
# (stroke_smoothness and fluency_score are computed, not direct copies)
_PAHAW_MAP = {
    "pressure_variance":               "tremor_power",
    "angle_data_with_1_variance":      "spiral_irregularity",
    "displacement_variance":           "contour_complexity",
    "pressure_Number_of_changing_point": "tremor_frequency",
    "ratio_of_in_air_time":            "pen_up_ratio",
    "displacement_mean":               "mean_stroke_width",
    "velocity_mean":                   "drawing_speed_proxy",
    "angle_data_with_50_variance":     "line_waviness",
}

_SPIRAL_TASK = "8"   # Archimedean spiral — most diagnostically relevant


def _load_pahaw_file(path: Path) -> Optional[pd.Series]:
    """Load one PaHaW CSV and return a Series with 10 features + label."""
    try:
        df = pd.read_csv(path, header=0)
    except Exception:
        return None

    if df.empty or df.shape[0] < 1:
        return None

    row = df.iloc[0]
    out: dict = {}

    # Direct mappings
    for src, dst in _PAHAW_MAP.items():
        val = pd.to_numeric(row.get(src, np.nan), errors="coerce")
        out[dst] = float(val) if pd.notna(val) else np.nan

    # stroke_smoothness = 1 / (1 + velocity_variance)
    vel_var = pd.to_numeric(row.get("velocity_variance", np.nan), errors="coerce")
    out["stroke_smoothness"] = 1.0 / (1.0 + float(vel_var)) if pd.notna(vel_var) else np.nan

    # fluency_score = velocity_mean / (velocity_standard_deviation + 1e-6)
    vel_mean = pd.to_numeric(row.get("velocity_mean", np.nan), errors="coerce")
    vel_std  = pd.to_numeric(row.get("velocity_standard_deviation", np.nan), errors="coerce")
    if pd.notna(vel_mean) and pd.notna(vel_std):
        out["fluency_score"] = float(vel_mean) / (float(vel_std) + 1e-6)
    else:
        out["fluency_score"] = np.nan

    # label
    label_val = pd.to_numeric(row.get("label", np.nan), errors="coerce")
    out["status"] = int(label_val) if pd.notna(label_val) else np.nan

    return pd.Series(out)


def run(raw_dir: Path, processed_dir: Path) -> Path:
    """
    Extract handwriting features from raw PaHaW CSVs and save processed CSV.
    Returns the output path.
    """
    hw_raw_dir = Path(raw_dir) / "handwriting"
    out_dir = Path(processed_dir) / "handwriting"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "handwriting_data.csv"

    print("[Handwriting Pipeline]")

    # Prefer spiral task files (task 8): 000XX__8_1.csv
    spiral_pattern = re.compile(r"^\d{5}__8_1\.csv$")
    spiral_files = [f for f in hw_raw_dir.glob("*__8_1.csv")
                    if spiral_pattern.match(f.name)]

    if spiral_files:
        task_files = spiral_files
        print(f"  Using spiral task (task 8): {len(task_files)} files")
    else:
        # Fall back to ALL subject task files
        task_files = [f for f in hw_raw_dir.glob("*__*_1.csv")]
        print(f"  Spiral files not found — using all task files: {len(task_files)} files")

    if not task_files:
        print("  No PaHaW CSV files found. Ensure data/raw/handwriting/000XX__N_1.csv exist.")
        return out_path

    records = []
    n_ok = n_fail = 0

    for f in sorted(task_files):
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

    # Fill any remaining NaNs with column medians (robust to occasional missing cols)
    for col in HW_FEATURES:
        if col in df.columns:
            df[col] = df[col].fillna(df[col].median())

    df = df.dropna(subset=HW_FEATURES)
    df["status"] = df["status"].fillna(df["status"].median()).astype(int)

    # Clip extreme outliers (> 99.9th percentile) to keep features in range
    for col in HW_FEATURES:
        cap = df[col].quantile(0.999)
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
