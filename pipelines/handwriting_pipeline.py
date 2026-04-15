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
  stroke_width_variance pressure_variance           (coeff. of variation of stroke widths)
  edge_roughness        azimuth_variance            (ink contour / convex-hull perimeter ratio)
  stroke_smoothness     1 / (1 + velocity_variance) (inverted velocity jitter)
  contour_complexity    displacement_variance
  stroke_inflection_count  pressure_Number_of_changing_point
  fragment_ratio        ratio_of_in_air_time        (normalised fragment count proxy)
  stroke_width_mean     displacement_mean           (mean stroke width proxy)
  ink_hull_ratio        velocity_mean
  line_waviness         y_variance                  (vertical-position spread)
  ink_coverage          velocity_mean / (velocity_standard_deviation + 1e-6)  (coverage proxy)
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
    "stroke_width_variance",
    "edge_roughness",
    "stroke_smoothness",
    "contour_complexity",
    "stroke_inflection_count",
    "fragment_ratio",
    "stroke_width_mean",
    "ink_hull_ratio",
    "line_waviness",
    "ink_coverage",
]

# Direct PaHaW column → target feature name.
# stroke_smoothness and ink_coverage are computed from velocity columns.
_PAHAW_MAP = {
    "pressure_variance":               "stroke_width_variance",
    "azimuth_variance":                "edge_roughness",
    "displacement_variance":           "contour_complexity",
    "pressure_Number_of_changing_point": "stroke_inflection_count",
    "ratio_of_in_air_time":            "fragment_ratio",
    "displacement_mean":               "stroke_width_mean",
    "velocity_mean":                   "ink_hull_ratio",
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

    # ink_coverage = velocity_mean / (velocity_standard_deviation + 1e-6)
    vel_mean = pd.to_numeric(row.get("velocity_mean",                np.nan), errors="coerce")
    vel_std  = pd.to_numeric(row.get("velocity_standard_deviation",  np.nan), errors="coerce")
    if pd.notna(vel_mean) and pd.notna(vel_std):
        out["ink_coverage"] = float(vel_mean) / (float(vel_std) + 1e-6)
    else:
        out["ink_coverage"] = np.nan

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

    data_files = sorted(hw_raw_dir.glob("handwriting_*.csv"))

    if not data_files:
        print("  No handwriting data files found — skipping.")
        return out_path

    print(f"  Found {len(data_files)} handwriting files")
    dfs = []
    for f in data_files:
        try:
            adf = pd.read_csv(f)
            if all(c in adf.columns for c in HW_FEATURES + ["status"]):
                dfs.append(adf[HW_FEATURES + ["status"]])
        except Exception:
            pass

    if not dfs:
        print("  No valid handwriting data found.")
        return out_path

    df = pd.concat(dfs, ignore_index=True)
    df.to_csv(out_path, index=False)
    n_pd = int(df["status"].sum())
    n_hc = len(df) - n_pd
    print(f"  → {out_path}  ({len(df)} rows: PD={n_pd}, HC={n_hc})\n")
    return out_path


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir",       default="data/raw")
    p.add_argument("--processed-dir", default="data/processed")
    args = p.parse_args()
    run(Path(args.raw_dir), Path(args.processed_dir))
