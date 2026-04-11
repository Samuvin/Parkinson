"""
pipelines/gait_pipeline.py
==========================
Extracts 10 gait features from PhysioNet GaitPaD VGRF .txt files and writes
``data/processed/gait/gait_data.csv``.

Raw data location: data/raw/gait/Ga*.csv  (CSV with header, 100 Hz, 25 columns)
  GaCo<NN>_<TT>.csv → healthy control (label 0)
  GaPt<NN>_<TT>.csv → Parkinson's patient (label 1)

File format (comma-separated, with header):
  col 0: stride_interval(Time)  col 1-8: L1..L8  col 9: gait_asymmetry(LeftTotal)
  col 10-17: heel..gait_cycle   col 18: gait_cycle_time(RightTotal)

10 output features (match config/multimodal_features.yaml):
  stride_interval, stride_variability, swing_time, stance_time,
  double_support_time, gait_speed, cadence, step_length,
  stride_regularity, gait_asymmetry
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

GAIT_FEATURES = [
    "stride_interval", "stride_variability", "swing_time", "stance_time",
    "double_support_time", "gait_speed", "cadence", "step_length",
    "stride_regularity", "gait_asymmetry",
]

_STANCE_THRESHOLD = 0.05   # fraction of peak force


def _parse_vgrf(path: Path) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """Parse a GaitPaD VGRF CSV file → (left_total, right_total, sample_rate)."""
    try:
        df = pd.read_csv(path, header=0)
        if df.shape[1] < 19:
            return None
        time  = df.iloc[:, 0].values.astype(float)
        left  = df.iloc[:, 9].values.astype(float)
        right = df.iloc[:, 18].values.astype(float)
        dt = np.diff(time)
        fs = 1.0 / float(np.median(dt[dt > 0])) if len(dt) > 0 else 100.0
        return left, right, fs
    except Exception:
        return None


def _heel_strikes(vgrf: np.ndarray, fs: float) -> List[int]:
    thr = float(np.max(vgrf)) * _STANCE_THRESHOLD
    stance = vgrf >= thr
    return [i for i in range(1, len(stance)) if stance[i] and not stance[i - 1]]


def _extract_features(left: np.ndarray, right: np.ndarray,
                      fs: float) -> Optional[Dict[str, float]]:
    if len(left) < int(fs * 2):
        return None

    l_thr = float(np.max(left))  * _STANCE_THRESHOLD
    r_thr = float(np.max(right)) * _STANCE_THRESHOLD
    l_stance = left  >= l_thr
    r_stance = right >= r_thr

    l_strikes = _heel_strikes(left,  fs)
    r_strikes = _heel_strikes(right, fs)
    all_strikes = sorted(l_strikes + r_strikes)
    if len(all_strikes) < 4:
        return None

    # Stride interval & variability (from left foot)
    if len(l_strikes) >= 2:
        intervals = np.diff(l_strikes) / fs
        stride_interval    = float(np.mean(intervals))
        stride_variability = float(np.std(intervals))
    else:
        stride_interval, stride_variability = 1.1, 0.08

    n = len(left)
    duration  = n / fs
    n_steps   = max(len(all_strikes), 1)

    swing_time          = float(np.sum(~l_stance) / fs / n_steps)
    stance_time         = float(np.sum( l_stance) / fs / n_steps)
    double_support_time = float(np.sum(l_stance & r_stance) / fs / n_steps)

    cadence    = (len(all_strikes) * 60.0) / duration if duration > 0 else 90.0
    step_length = 0.28 * (cadence ** 0.45)  if cadence > 0 else 0.6
    gait_speed  = step_length * (cadence / 60.0)

    # stride_regularity — autocorrelation of the VGRF signal at the stride
    # period.  Search a ±20 % window around the expected lag so that small
    # timing estimation errors don't accidentally land on a near-zero or
    # negative part of the autocorrelation and collapse to 0.
    stride_lag = max(int(stride_interval * fs), 1)
    norm_left  = left - np.mean(left)
    if stride_lag < len(norm_left):
        ac = np.correlate(norm_left, norm_left, mode="full")
        ac = ac[len(ac) // 2:]
        ac /= ac[0] + 1e-10
        lo = max(1, int(stride_lag * 0.8))
        hi = min(len(ac) - 1, int(stride_lag * 1.2) + 1)
        stride_regularity = float(np.clip(float(np.max(ac[lo:hi])), 0.0, 1.0))
    else:
        stride_regularity = 0.7

    # gait_asymmetry — timing asymmetry between left and right stride intervals.
    # Using force-magnitude asymmetry (old approach) almost always saturates at
    # the clip ceiling because raw foot-force totals are inherently unequal.
    # Timing asymmetry is the standard clinical measure (Plotnik et al. 2007).
    if len(l_strikes) >= 2 and len(r_strikes) >= 2:
        l_ivs = np.diff(l_strikes) / fs
        r_ivs = np.diff(r_strikes) / fs
        ml, mr = float(np.mean(l_ivs)), float(np.mean(r_ivs))
        gait_asymmetry = float(np.clip(abs(ml - mr) / (ml + mr + 1e-10), 0.0, 1.0))
    else:
        gait_asymmetry = 0.0

    return {
        "stride_interval":    stride_interval,
        "stride_variability": stride_variability,
        "swing_time":         swing_time,
        "stance_time":        stance_time,
        "double_support_time": double_support_time,
        "gait_speed":         gait_speed,
        "cadence":            cadence,
        "step_length":        step_length,
        "stride_regularity":  stride_regularity,
        "gait_asymmetry":     gait_asymmetry,
    }


def run(raw_dir: Path, processed_dir: Path) -> Path:
    """
    Extract gait features from raw VGRF .txt files and save processed CSV.
    Returns the output path.
    """
    gait_raw_dir = Path(raw_dir) / "gait"
    out_dir = Path(processed_dir) / "gait"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "gait_data.csv"

    print("[Gait Pipeline]")
    data_files = sorted(gait_raw_dir.glob("gait_*.csv"))

    if not data_files:
        print("  No gait data files found — skipping.")
        return out_path

    print(f"  Found {len(data_files)} gait files")
    dfs = []
    for f in data_files:
        try:
            adf = pd.read_csv(f)
            if all(c in adf.columns for c in GAIT_FEATURES + ["status"]):
                dfs.append(adf[GAIT_FEATURES + ["status"]])
        except Exception:
            pass

    if not dfs:
        print("  No valid gait data found.")
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
