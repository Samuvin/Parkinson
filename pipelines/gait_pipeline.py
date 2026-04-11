"""
pipelines/gait_pipeline.py
==========================
Extracts 10 gait features from PhysioNet GaitPaD VGRF .txt files and writes
``data/processed/gait/gait_data.csv``.

Raw data location: data/raw/gait/Ga*.txt  (113 files, 100 Hz, 19 columns)
  GaCo<NN>_<TT>.txt → healthy control (label 0)
  GaPt<NN>_<TT>.txt → Parkinson's patient (label 1)

File format (tab-separated, no header):
  col 0: Time(s)  col 1-8: L1..L8  col 9: LeftTotal
  col 10-17: R1..R8               col 18: RightTotal

10 output features (match config/multimodal_features.yaml):
  stride_interval, stride_variability, swing_time, stance_time,
  double_support_time, gait_speed, cadence, step_length,
  stride_regularity, gait_asymmetry
"""
from __future__ import annotations

import io
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


def _parse_vgrf(raw: bytes) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """Parse a GaitPaD VGRF file → (left_total, right_total, sample_rate)."""
    try:
        df = pd.read_csv(io.StringIO(raw.decode("utf-8", errors="replace")),
                         sep=r"\s+", header=None)
        if df.shape[1] < 19:
            return None
        time = df.iloc[:, 0].values.astype(float)
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

    stride_lag  = max(int(stride_interval * fs), 1)
    norm_left   = left - np.mean(left)
    if stride_lag < len(norm_left):
        ac = np.correlate(norm_left, norm_left, mode="full")
        ac = ac[len(ac) // 2:]
        ac /= ac[0] + 1e-10
        stride_regularity = float(np.clip(ac[stride_lag], 0.0, 1.0))
    else:
        stride_regularity = 0.7

    lm = float(np.mean(left [l_stance])) if np.any(l_stance) else 0.0
    rm = float(np.mean(right[r_stance])) if np.any(r_stance) else 0.0
    gait_asymmetry = float(np.clip(abs(lm - rm) / (lm + rm + 1e-10), 0.0, 0.5))

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
    txt_files = sorted(gait_raw_dir.glob("Ga*.txt"))
    if not txt_files:
        # Also check physionet_vgrf subfolder (in case user hasn't moved them yet)
        txt_files = sorted((gait_raw_dir / "physionet_vgrf").glob("Ga*.txt"))

    if not txt_files:
        print("  No GaitPaD .txt files found — skipping (ensure data/raw/gait/Ga*.txt exist)")
        return out_path

    print(f"  Found {len(txt_files)} VGRF files")
    records = []
    n_ok = n_fail = 0

    for txt in txt_files:
        label = 1 if "Pt" in txt.stem else 0
        parsed = _parse_vgrf(txt.read_bytes())
        if parsed is None:
            n_fail += 1
            continue
        left, right, fs = parsed
        feats = _extract_features(left, right, fs)
        if feats is None:
            n_fail += 1
            continue
        feats["status"] = label
        records.append(feats)
        n_ok += 1

    if not records:
        print("  No valid VGRF trials extracted.")
        return out_path

    df = pd.DataFrame(records)[GAIT_FEATURES + ["status"]]
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
