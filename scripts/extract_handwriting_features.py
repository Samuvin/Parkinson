"""
extract_handwriting_features.py
================================
Build ``data/raw/handwriting/handwriting_data.csv`` from UCI-395 spiral traces.

UCI-395 — Parkinson Disease Spiral Drawings Using Digitized Graphics Tablet
---------------------------------------------------------------------------
  Landing : https://archive.ics.uci.edu/dataset/395/
            parkinson+disease+spiral+drawings+using+digitized+graphics+tablet
  ZIP URL : https://archive.ics.uci.edu/static/public/395/
            parkinson+disease+spiral+drawings+using+digitized+graphics+tablet.zip
  License : CC BY 4.0

  After extraction the layout is:
    data/raw/handwriting/uci_spiral/hw_dataset/     <- original dataset
    data/raw/handwriting/uci_spiral/new_dataset/    <- extended dataset

  Each subject folder contains per-task trace files (semicolon-delimited):
    X ; Y ; Z ; Pressure ; GripAngle ; Timestamp ; TestID

  Z ≈ 0 means the pen is lifted (air-move); Z > 0 means pen-on-paper.
  PD subjects are in sub-folders labelled "PD" or similar; controls in "HC".

10 features extracted from each trace — identical names to those produced by
``common/image_processing.py`` at inference time (training ↔ inference
consistency):

  tremor_power         — FFT peak amplitude of radial deviation from ideal spiral
  spiral_irregularity  — variance of inter-ring gap distances
  stroke_smoothness    — mean absolute curvature of the drawn path
  contour_complexity   — path_length² / bounding area
  tremor_frequency     — dominant radial-oscillation frequency (real Hz via Timestamp)
  pen_up_ratio         — fraction of total duration with pen lifted (Z near 0)
  mean_stroke_width    — mean Pressure (direct digitizer value, proxy for width)
  drawing_speed_proxy  — total arc length / bounding-box diagonal
  line_waviness        — RMS perpendicular deviation from principal axis
  fluency_score        — smoothness × (1 − tremor) × arc_coverage

Usage
-----
    python scripts/extract_handwriting_features.py [--data-dir data/raw]
    python scripts/extract_handwriting_features.py --no-download  # use existing
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import fft as sp_fft

_REPO_ROOT = Path(__file__).resolve().parents[1]

HANDWRITING_FEATURES = [
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

_PEN_UP_THRESHOLD = 1.0   # Z value below which pen is considered lifted


# ---------------------------------------------------------------------------
# Trace file parsing
# ---------------------------------------------------------------------------

def _parse_trace_file(path: Path) -> Optional[pd.DataFrame]:
    """
    Read a UCI-395 trace file (semicolon-separated) and return a DataFrame
    with columns [X, Y, Z, Pressure, GripAngle, Timestamp].

    Returns None if the file cannot be parsed or has fewer than 20 points.
    """
    try:
        df = pd.read_csv(path, sep=";", header=None, skipinitialspace=True)
    except Exception:
        return None

    # Drop last column if it's TestID (non-numeric / all same value)
    if df.shape[1] >= 7:
        df = df.iloc[:, :7]
    if df.shape[1] < 6:
        return None

    df.columns = ["X", "Y", "Z", "Pressure", "GripAngle", "Timestamp"] + (
        ["TestID"] if df.shape[1] == 7 else []
    )

    for col in ["X", "Y", "Z", "Pressure", "GripAngle", "Timestamp"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    df = df.dropna(subset=["X", "Y", "Z", "Pressure", "Timestamp"])
    return df if len(df) >= 20 else None


# ---------------------------------------------------------------------------
# Feature extraction from trace data
# ---------------------------------------------------------------------------

def _tremor_power(x: np.ndarray, y: np.ndarray) -> float:
    """FFT peak amplitude of radial deviation from ideal Archimedean spiral."""
    cx, cy = float(np.mean(x)), float(np.mean(y))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    if r.max() < 1e-6:
        return 0.5

    r_norm = r / r.max()
    angles = np.arctan2(y - cy, x - cx)
    order = np.argsort(angles)
    r_sorted = r_norm[order]

    idx = np.arange(len(r_sorted))
    if len(idx) > 1:
        coeffs = np.polyfit(idx, r_sorted, 1)
        residuals = r_sorted - np.polyval(coeffs, idx)
    else:
        residuals = r_sorted - np.mean(r_sorted)

    spectrum = np.abs(sp_fft.rfft(residuals))
    peak = float(np.max(spectrum[1:])) / (len(residuals) + 1e-10)
    return float(np.clip(peak, 0.0, 1.0))


def _spiral_irregularity(x: np.ndarray, y: np.ndarray) -> float:
    """Variance of inter-ring gap distances from radial distance histogram."""
    cx, cy = float(np.mean(x)), float(np.mean(y))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    if r.max() < 1e-6:
        return 0.3

    hist, edges = np.histogram(r, bins=max(20, len(r) // 50))
    ring_gaps = np.diff(edges[np.where(hist > np.mean(hist))[0]])
    if len(ring_gaps) < 2:
        return 0.3
    return float(np.clip(np.std(ring_gaps) / (np.mean(ring_gaps) + 1e-10), 0.0, 2.0))


def _stroke_smoothness(x: np.ndarray, y: np.ndarray) -> float:
    """Mean absolute curvature of the traced path (lower = smoother)."""
    if len(x) < 3:
        return 0.5
    angles = []
    for i in range(1, len(x) - 1):
        v1 = np.array([x[i] - x[i - 1], y[i] - y[i - 1]])
        v2 = np.array([x[i + 1] - x[i], y[i + 1] - y[i]])
        cross = v1[0] * v2[1] - v1[1] * v2[0]
        dot = v1[0] * v2[0] + v1[1] * v2[1]
        angles.append(abs(np.arctan2(cross, dot)))
    return float(np.clip(np.mean(angles) / np.pi, 0.0, 1.0)) if angles else 0.5


def _contour_complexity(x: np.ndarray, y: np.ndarray) -> float:
    """Path length² / bounding area (isoperimetric-like measure)."""
    if len(x) < 2:
        return 1.0
    dx = np.diff(x)
    dy = np.diff(y)
    path_len = float(np.sum(np.sqrt(dx ** 2 + dy ** 2)))
    width = float(x.max() - x.min())
    height = float(y.max() - y.min())
    area = max(width * height, 1e-6)
    return float(np.clip(path_len ** 2 / area, 1.0, 200.0))


def _tremor_frequency(
    x: np.ndarray, y: np.ndarray, timestamps: np.ndarray
) -> float:
    """Dominant radial-oscillation frequency in real Hz using Timestamp."""
    cx, cy = float(np.mean(x)), float(np.mean(y))
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
    if r.max() < 1e-6 or len(r) < 10:
        return 5.0

    # Sort by angle to get a sweep signal
    angles = np.arctan2(y - cy, x - cx)
    order = np.argsort(angles)
    r_signal = r[order] / (r.max() + 1e-10)
    ts_sorted = timestamps[order]

    # Estimate duration for Hz calculation
    duration = float(ts_sorted.max() - ts_sorted.min())
    if duration < 1e-6:
        duration = 1.0

    spectrum = np.abs(sp_fft.rfft(r_signal))
    if len(spectrum) < 2:
        return 5.0

    dominant_bin = int(np.argmax(spectrum[1:]) + 1)
    freq_hz = dominant_bin / duration
    return float(np.clip(freq_hz, 0.0, 30.0))


def _pen_up_ratio(z: np.ndarray, timestamps: np.ndarray) -> float:
    """Fraction of total trace duration with pen lifted (Z < threshold)."""
    total_duration = float(timestamps.max() - timestamps.min())
    if total_duration < 1e-6:
        return float(np.mean(z < _PEN_UP_THRESHOLD))
    # Compute duration-weighted pen-up ratio
    dt = np.abs(np.diff(timestamps))
    pen_up = z[:-1] < _PEN_UP_THRESHOLD
    pen_up_duration = float(np.sum(dt[pen_up]))
    return float(np.clip(pen_up_duration / total_duration, 0.0, 1.0))


def _mean_stroke_width(pressure: np.ndarray, z: np.ndarray) -> float:
    """Mean Pressure on pen-down points (direct digitizer measurement)."""
    on_paper = z >= _PEN_UP_THRESHOLD
    if not np.any(on_paper):
        return float(np.mean(pressure))
    return float(np.mean(pressure[on_paper]))


def _drawing_speed_proxy(x: np.ndarray, y: np.ndarray) -> float:
    """Total arc length divided by bounding-box diagonal."""
    if len(x) < 2:
        return 0.5
    dx = np.diff(x)
    dy = np.diff(y)
    arc_len = float(np.sum(np.sqrt(dx ** 2 + dy ** 2)))
    diagonal = float(
        np.sqrt((x.max() - x.min()) ** 2 + (y.max() - y.min()) ** 2)
    ) + 1e-10
    return float(np.clip(arc_len / diagonal, 0.0, 20.0))


def _line_waviness(x: np.ndarray, y: np.ndarray) -> float:
    """RMS perpendicular deviation from the principal axis of the path."""
    if len(x) < 5:
        return 0.3
    pts = np.column_stack([x.astype(float), y.astype(float)])
    mean_pt = pts.mean(axis=0)
    centred = pts - mean_pt
    _, _, vt = np.linalg.svd(centred, full_matrices=False)
    axis = vt[0]
    perp = centred - (centred @ axis)[:, None] * axis
    rms = float(np.sqrt(np.mean(np.sum(perp ** 2, axis=1))))
    # Normalise by bounding diagonal
    diagonal = float(
        np.sqrt((x.max() - x.min()) ** 2 + (y.max() - y.min()) ** 2)
    ) + 1e-10
    return float(np.clip(rms / diagonal, 0.0, 1.0))


def _fluency_score(
    stroke_smoothness: float,
    tremor_power: float,
    x: np.ndarray,
    y: np.ndarray,
    z: np.ndarray,
) -> float:
    """Composite: (1−smoothness) × (1−tremor) × arc_coverage."""
    on_paper = z >= _PEN_UP_THRESHOLD
    if not np.any(on_paper):
        coverage = 0.0
    else:
        # Arc coverage: ink arc / total bounding diagonal
        xp, yp = x[on_paper], y[on_paper]
        if len(xp) < 2:
            coverage = 0.0
        else:
            dx = np.diff(xp)
            dy = np.diff(yp)
            arc = float(np.sum(np.sqrt(dx ** 2 + dy ** 2)))
            diag = float(
                np.sqrt((x.max() - x.min()) ** 2 + (y.max() - y.min()) ** 2)
            ) + 1e-10
            coverage = float(np.clip(arc / diag, 0.0, 1.0))

    smoothness_score = 1.0 - float(stroke_smoothness)
    return float(np.clip(smoothness_score * (1.0 - float(tremor_power)) * coverage, 0.0, 1.0))


def extract_features_from_trace(df: pd.DataFrame) -> Dict[str, float]:
    """Compute all 10 handwriting features from a parsed trace DataFrame."""
    x = df["X"].values
    y = df["Y"].values
    z = df["Z"].values
    pressure = df["Pressure"].values
    ts = df["Timestamp"].values

    tp = _tremor_power(x, y)
    ss = _stroke_smoothness(x, y)

    return {
        "tremor_power":        tp,
        "spiral_irregularity": _spiral_irregularity(x, y),
        "stroke_smoothness":   ss,
        "contour_complexity":  _contour_complexity(x, y),
        "tremor_frequency":    _tremor_frequency(x, y, ts),
        "pen_up_ratio":        _pen_up_ratio(z, ts),
        "mean_stroke_width":   _mean_stroke_width(pressure, z),
        "drawing_speed_proxy": _drawing_speed_proxy(x, y),
        "line_waviness":       _line_waviness(x, y),
        "fluency_score":       _fluency_score(ss, tp, x, y, z),
    }


# ---------------------------------------------------------------------------
# Dataset discovery
# ---------------------------------------------------------------------------

def _find_trace_files(uci_dir: Path) -> List[Tuple[Path, int]]:
    """
    Walk the UCI-395 extraction directory and return (path, label) pairs.

    Label 0 = healthy control (directory name contains 'HC', 'healthy',
               'control', 'Co' — case-insensitive)
    Label 1 = Parkinson's patient (directory name contains 'PD', 'patient',
               'parkinson' — case-insensitive)
    """
    pairs: List[Tuple[Path, int]] = []
    healthy_kw = {"hc", "healthy", "control", "co", "norm"}
    pd_kw = {"pd", "patient", "parkinson", "pt"}

    for p in sorted(uci_dir.rglob("*.txt")):
        # Determine label from parent directory names
        label = None
        for part in p.parts:
            lower = part.lower()
            if any(kw in lower for kw in pd_kw):
                label = 1
                break
            if any(kw in lower for kw in healthy_kw):
                label = 0
                break
        # If no label could be determined, skip
        if label is None:
            continue
        pairs.append((p, label))

    return pairs


def _process_uci395(uci_dir: Path) -> List[Dict]:
    """Extract features from all UCI-395 trace files in *uci_dir*."""
    pairs = _find_trace_files(uci_dir)
    if not pairs:
        print(f"  No labelled trace files found under {uci_dir}")
        return []

    records = []
    skipped = 0
    for path, label in pairs:
        df = _parse_trace_file(path)
        if df is None:
            skipped += 1
            continue
        try:
            feats = extract_features_from_trace(df)
        except Exception:
            skipped += 1
            continue
        feats["status"] = label
        records.append(feats)

    print(f"  UCI-395: {len(records)} traces extracted, {skipped} skipped")
    return records


# ---------------------------------------------------------------------------
# Main build logic
# ---------------------------------------------------------------------------

def _download_uci395(handwriting_dir: Path) -> bool:
    """Run the existing download script to fetch UCI-395 spiral traces."""
    script = handwriting_dir / "download_handwriting_sources.sh"
    if not script.is_file():
        print(f"  Download script not found: {script}")
        return False
    print(f"  Running {script} …")
    try:
        result = subprocess.run(
            ["bash", str(script)],
            cwd=str(handwriting_dir),
            capture_output=True,
            text=True,
            timeout=120,
        )
        if result.returncode == 0:
            print("  Download complete.")
            return True
        print(f"  Download script exited {result.returncode}: {result.stderr[:200]}")
        return False
    except subprocess.TimeoutExpired:
        print("  Download timed out.")
        return False
    except Exception as exc:
        print(f"  Download error: {exc}")
        return False


def build_handwriting_csv(
    data_dir: Path, output_path: Path, no_download: bool = False
) -> None:
    """Build handwriting_data.csv from UCI-395 traces."""
    hw_dir = data_dir / "handwriting"
    uci_dir = hw_dir / "uci_spiral"

    # Download if not already present
    if not uci_dir.is_dir() and not no_download:
        _download_uci395(hw_dir)

    records: List[Dict] = []
    if uci_dir.is_dir():
        records = _process_uci395(uci_dir)

    if not records:
        print("  No UCI-395 data available. Falling back to existing handwriting_data.csv …")
        existing = hw_dir / "handwriting_data.csv"
        if existing.is_file():
            df_old = pd.read_csv(existing)

            # Map old column positions to new feature names directly to avoid
            # conflicts when both old and new names coexist (e.g. 'tremor_power').
            # Old column order: mean_pressure, pressure_variation, mean_velocity,
            #   velocity_variation, mean_acceleration, penup_time_ratio,
            #   mean_stroke_length, writing_tempo, tremor_power, fluency_score
            OLD_TO_NEW = {
                "mean_pressure":    "tremor_power",
                "pressure_variation": "spiral_irregularity",
                "mean_velocity":    "stroke_smoothness",
                "velocity_variation": "contour_complexity",
                "mean_acceleration": "tremor_frequency",
                "penup_time_ratio": "pen_up_ratio",
                "mean_stroke_length": "mean_stroke_width",
                "writing_tempo":    "drawing_speed_proxy",
                "tremor_power":     "line_waviness",   # old tremor_power → waviness
                "fluency_score":    "fluency_score",
            }

            # Build new dataframe column by column to avoid pandas duplicate renames
            df_out = pd.DataFrame(index=df_old.index)
            used_old_cols: set = set()
            for old_col, new_col in OLD_TO_NEW.items():
                if old_col in df_old.columns and old_col not in used_old_cols:
                    df_out[new_col] = df_old[old_col].values
                    used_old_cols.add(old_col)

            # Fill any still-missing target features with 0
            for col in HANDWRITING_FEATURES:
                if col not in df_out.columns:
                    df_out[col] = 0.0

            df_out["status"] = df_old["status"].values if "status" in df_old.columns else 0

            df_out = df_out[HANDWRITING_FEATURES + ["status"]]
            df_out.to_csv(output_path, index=False)
            print(f"  Kept existing (renamed): {output_path} ({len(df_out)} rows)")
        else:
            print(
                "  WARNING: No handwriting data available.\n"
                "  Run: bash data/raw/handwriting/download_handwriting_sources.sh\n"
                "  Or place UCI-395 trace files under data/raw/handwriting/uci_spiral/",
                file=sys.stderr,
            )
        return

    df = pd.DataFrame(records)[HANDWRITING_FEATURES + ["status"]]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)

    n_pd = int(df["status"].sum())
    n_hc = len(df) - n_pd
    print(f"\nOutput: {output_path}")
    print(f"  Rows : {len(df)}  (PD={n_pd}, HC={n_hc})")
    print(f"  Cols : {HANDWRITING_FEATURES + ['status']}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

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
        help="Output CSV path (default: <data-dir>/handwriting/handwriting_data.csv)",
    )
    p.add_argument(
        "--no-download",
        action="store_true",
        help="Skip UCI-395 download; use existing files only",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    data_dir = Path(args.data_dir)
    output_path = (
        Path(args.output) if args.output
        else data_dir / "handwriting" / "handwriting_data.csv"
    )
    print("=== Handwriting Feature Extractor ===")
    build_handwriting_csv(data_dir, output_path, no_download=args.no_download)
