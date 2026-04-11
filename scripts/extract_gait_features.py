"""
extract_gait_features.py
========================
Download the PhysioNet GaitPaD VGRF dataset and build
``data/raw/gait/gait_data.csv`` with 10 clinically-grounded gait features.

PhysioNet GaitPaD (gaitpdb)
---------------------------
  93 subjects: 47 idiopathic PD + 46 healthy controls.
  Each trial file is a tab-separated time-series at ~100 Hz with 19 columns:
    Time  L1..L8  LeftTotal  R1..R8  RightTotal

  URL base: https://physionet.org/files/gaitpdb/1.0.0/
  No login required for individual file access.

  NOTE: Python's urllib silently returns empty content because PhysioNet uses
  HTTP→HTTPS redirects + HTTP/2.  This script uses ``curl -sL`` (via
  subprocess) which follows redirects correctly.

10 gait features extracted (match ``config/multimodal_features.yaml``):
  stride_interval      stride_variability   swing_time
  stance_time          double_support_time  gait_speed
  cadence              step_length          stride_regularity
  gait_asymmetry

Usage
-----
    python3 scripts/extract_gait_features.py [--data-dir data/raw]
    python3 scripts/extract_gait_features.py --no-download   # keep existing CSV
"""

import argparse
import io
import re
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
_PHYSIONET_BASE = "https://physionet.org/files/gaitpdb/1.0.0/"

GAIT_FEATURES = [
    "stride_interval", "stride_variability", "swing_time", "stance_time",
    "double_support_time", "gait_speed", "cadence", "step_length",
    "stride_regularity", "gait_asymmetry",
]

_VGRF_THRESHOLD_FRAC = 0.05   # fraction of peak force below which foot is airborne


# ---------------------------------------------------------------------------
# HTTP with curl (follows redirects and HTTP/2)
# ---------------------------------------------------------------------------

def _curl_get(url: str, timeout: int = 30) -> Optional[bytes]:
    """Fetch URL bytes using ``curl -sL``; returns None on failure."""
    try:
        r = subprocess.run(
            ["curl", "-sL", "--max-time", str(timeout), url],
            capture_output=True,
            timeout=timeout + 5,
        )
        if r.returncode == 0 and r.stdout:
            return r.stdout
    except Exception:
        pass
    return None


def _curl_get_text(url: str, timeout: int = 30) -> Optional[str]:
    raw = _curl_get(url, timeout)
    return raw.decode("utf-8", errors="replace") if raw else None


# ---------------------------------------------------------------------------
# Discover all VGRF trial filenames from the PhysioNet index page
# ---------------------------------------------------------------------------

def _list_vgrf_files() -> List[Tuple[str, int]]:
    """
    Scrape the PhysioNet GaitPaD index page and return a list of
    (filename, label) tuples where label=0 for controls, 1 for patients.

    Filename convention:
      GaCo<NN>_<TT>.txt  → control  (label 0)
      GaPt<NN>_<TT>.txt  → patient  (label 1)
    """
    print(f"  Fetching file index from {_PHYSIONET_BASE} …")
    html = _curl_get_text(_PHYSIONET_BASE, timeout=30)
    if not html:
        print("  Could not reach PhysioNet index page.")
        return []

    files = re.findall(r'href="(Ga(?:Co|Pt)\d+_\d+\.txt)"', html)
    result = []
    for fname in files:
        if "Co" in fname:
            result.append((fname, 0))
        else:
            result.append((fname, 1))
    print(f"  Found {len(result)} trial files ({sum(1 for _,l in result if l==0)} control, "
          f"{sum(1 for _,l in result if l==1)} patient)")
    return result


# ---------------------------------------------------------------------------
# VGRF parsing and feature extraction
# ---------------------------------------------------------------------------

def _parse_vgrf(raw: bytes) -> Optional[Tuple[np.ndarray, np.ndarray, float]]:
    """
    Parse a GaitPaD VGRF file and return (left_total, right_total, fs).

    File format (tab-separated, no header):
      col 0  : Time (s)
      col 1-8: Left force sensors L1..L8
      col 9  : LeftTotal
      col 10-17: Right force sensors R1..R8
      col 18 : RightTotal
    """
    try:
        text = raw.decode("utf-8", errors="replace")
        buf = io.StringIO(text)
        df = pd.read_csv(buf, sep=r"\s+", header=None)
        if df.shape[1] < 19:
            return None
        time_col = df.iloc[:, 0].values.astype(float)
        left_total = df.iloc[:, 9].values.astype(float)
        right_total = df.iloc[:, 18].values.astype(float)
        dt = np.diff(time_col)
        fs = 1.0 / float(np.median(dt[dt > 0])) if len(dt) > 0 else 100.0
        return left_total, right_total, fs
    except Exception:
        return None


def _heel_strikes(vgrf: np.ndarray, fs: float) -> List[int]:
    """Rising-edge threshold crossings in VGRF = heel strikes."""
    thr = float(np.max(vgrf)) * _VGRF_THRESHOLD_FRAC
    in_stance = vgrf >= thr
    return [i for i in range(1, len(in_stance)) if in_stance[i] and not in_stance[i - 1]]


def _compute_gait_features(
    left: np.ndarray,
    right: np.ndarray,
    fs: float,
) -> Optional[Dict[str, float]]:
    """Extract 10 gait features from left/right VGRF arrays."""
    if len(left) < int(fs * 2):
        return None

    left_thr = float(np.max(left)) * _VGRF_THRESHOLD_FRAC
    right_thr = float(np.max(right)) * _VGRF_THRESHOLD_FRAC
    left_stance = left >= left_thr
    right_stance = right >= right_thr

    left_strikes = _heel_strikes(left, fs)
    right_strikes = _heel_strikes(right, fs)
    all_strikes = sorted(left_strikes + right_strikes)

    if len(all_strikes) < 4:
        return None

    # --- Stride interval and variability ---
    if len(left_strikes) >= 2:
        li = np.diff(left_strikes) / fs
        stride_interval = float(np.mean(li))
        stride_variability = float(np.std(li))
    else:
        stride_interval, stride_variability = 1.1, 0.08

    # --- Temporal phase times ---
    n = len(left)
    duration = n / fs
    n_steps = max(len(all_strikes), 1)

    swing_time = float(np.sum(~left_stance) / fs / n_steps)
    stance_time = float(np.sum(left_stance) / fs / n_steps)
    double_support_time = float(np.sum(left_stance & right_stance) / fs / n_steps)

    # --- Cadence and speed ---
    cadence = (len(all_strikes) * 60.0) / duration if duration > 0 else 90.0
    step_length = 0.28 * (cadence ** 0.45) if cadence > 0 else 0.6
    gait_speed = step_length * (cadence / 60.0)

    # --- Stride regularity ---
    stride_lag = max(int(stride_interval * fs), 1)
    norm_left = left - np.mean(left)
    if stride_lag < len(norm_left):
        ac = np.correlate(norm_left, norm_left, mode="full")
        ac = ac[len(ac) // 2:]
        ac /= ac[0] + 1e-10
        stride_regularity = float(np.clip(ac[stride_lag], 0.0, 1.0))
    else:
        stride_regularity = 0.7

    # --- Gait asymmetry (L vs R contact force) ---
    lm = float(np.mean(left[left_stance])) if np.any(left_stance) else 0.0
    rm = float(np.mean(right[right_stance])) if np.any(right_stance) else 0.0
    gait_asymmetry = float(np.clip(abs(lm - rm) / (lm + rm + 1e-10), 0.0, 0.5))

    return {
        "stride_interval": stride_interval,
        "stride_variability": stride_variability,
        "swing_time": swing_time,
        "stance_time": stance_time,
        "double_support_time": double_support_time,
        "gait_speed": gait_speed,
        "cadence": cadence,
        "step_length": step_length,
        "stride_regularity": stride_regularity,
        "gait_asymmetry": gait_asymmetry,
    }


# ---------------------------------------------------------------------------
# Main download and build logic
# ---------------------------------------------------------------------------

def _download_and_extract(gait_dir: Path) -> List[Dict]:
    """
    Download all PhysioNet GaitPaD VGRF trial files, save raw copies, and
    extract 10 gait features per trial.
    """
    vgrf_dir = gait_dir
    vgrf_dir.mkdir(parents=True, exist_ok=True)

    trial_files = _list_vgrf_files()
    if not trial_files:
        return []

    records = []
    n_ok = 0
    n_fail = 0

    for i, (fname, label) in enumerate(trial_files, 1):
        local_path = vgrf_dir / fname

        # Use cached file if already downloaded
        if local_path.is_file() and local_path.stat().st_size > 10_000:
            raw = local_path.read_bytes()
        else:
            url = _PHYSIONET_BASE + fname
            raw = _curl_get(url, timeout=30)
            if raw and len(raw) > 10_000:
                local_path.write_bytes(raw)
            else:
                n_fail += 1
                continue

        parsed = _parse_vgrf(raw)
        if parsed is None:
            n_fail += 1
            continue

        left, right, fs = parsed
        feats = _compute_gait_features(left, right, fs)
        if feats is None:
            n_fail += 1
            continue

        feats["status"] = label
        records.append(feats)
        n_ok += 1

        if i % 20 == 0 or i == len(trial_files):
            print(f"  [{i}/{len(trial_files)}] extracted={n_ok}  skipped={n_fail}", flush=True)

    print(f"  PhysioNet: {n_ok} trials extracted, {n_fail} skipped")
    return records


def build_gait_csv(data_dir: Path, output_path: Path) -> None:
    gait_dir = data_dir / "gait"

    print("  Downloading PhysioNet GaitPaD VGRF files …")
    records = _download_and_extract(gait_dir)

    existing_path = gait_dir / "gait_data.csv"

    if records:
        df_new = pd.DataFrame(records)[GAIT_FEATURES + ["status"]]

        # Merge with existing data if it has different (non-PhysioNet) rows
        if existing_path.is_file() and str(existing_path) != str(output_path):
            df_old = pd.read_csv(existing_path)
            if all(c in df_old.columns for c in GAIT_FEATURES + ["status"]):
                df_combined = pd.concat(
                    [df_old[GAIT_FEATURES + ["status"]], df_new],
                    ignore_index=True,
                ).drop_duplicates()
                print(f"  Merged existing ({len(df_old)}) + PhysioNet ({len(df_new)}) rows")
            else:
                df_combined = df_new
        else:
            df_combined = df_new

        output_path.parent.mkdir(parents=True, exist_ok=True)
        df_combined.to_csv(output_path, index=False)
        n_pd = int(df_combined["status"].sum())
        n_hc = len(df_combined) - n_pd
        print(f"\nOutput: {output_path}")
        print(f"  Rows : {len(df_combined)}  (PD={n_pd}, HC={n_hc})")

    else:
        print("  No PhysioNet records extracted — keeping existing gait_data.csv")
        if existing_path.is_file():
            df = pd.read_csv(existing_path)
            print(f"  Existing: {existing_path} ({len(df)} rows)")
        else:
            print("  WARNING: No gait data found.", file=sys.stderr)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data-dir", default=str(_REPO_ROOT / "data" / "raw"))
    p.add_argument("--output", default=None)
    p.add_argument("--no-download", action="store_true",
                   help="Skip download; use existing gait_data.csv as-is")
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    data_dir = Path(args.data_dir)
    output_path = (
        Path(args.output) if args.output
        else data_dir / "gait" / "gait_data.csv"
    )
    print("=== Gait Feature Extractor (PhysioNet GaitPaD) ===")
    if args.no_download:
        existing = data_dir / "gait" / "gait_data.csv"
        df = pd.read_csv(existing)
        print(f"  Using existing: {existing} ({len(df)} rows)")
    else:
        build_gait_csv(data_dir, output_path)
