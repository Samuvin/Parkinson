"""
pipelines/speech_pipeline.py
============================
Extracts 22 MDVP speech features from raw CSV datasets and writes
``data/processed/speech/parkinsons.csv``.

Sources used (in data/raw/speech/):
  - parkinsons.csv                         UCI-174  (195 rows,  all 22 MDVP features)
  - voice_uci301_multiple_sound_train.csv  UCI-301  (1040 rows, 29 acoustic cols → mapped)
  - voice_uci470_pd_classification_features.csv  UCI-470 (757 rows → mapped)
  - voice_uci489_replicated_acoustic.csv   UCI-489  (240 rows, 120 HC + 120 PD → mapped)

  parkinsons_telemonitoring.csv is intentionally skipped: it contains only
  Parkinson's patients (motor_UPDRS > 0 for every row), so including it
  without healthy controls would severely bias the training distribution.

Output columns: 22 MDVP feature names + ``status`` (0=healthy, 1=PD)
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

MDVP_FEATURES = [
    "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)",
    "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ",
    "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer(dB)",
    "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA",
    "NHR", "HNR", "RPDE", "DFA", "spread1", "spread2", "D2", "PPE",
]

# ── UCI-301 ──────────────────────────────────────────────────────────────────
_UCI301_MAP = {
    "median_pitch":          "MDVP:Fo(Hz)",
    "max_pitch":             "MDVP:Fhi(Hz)",
    "min_pitch":             "MDVP:Flo(Hz)",
    "jitter_local":          "MDVP:Jitter(%)",
    "jitter_local_absolute": "MDVP:Jitter(Abs)",
    "jitter_rap":            "MDVP:RAP",
    "jitter_ppq5":           "MDVP:PPQ",
    "jitter_ddp":            "Jitter:DDP",
    "shimmer_local":         "MDVP:Shimmer",
    "shimmer_local_db":      "MDVP:Shimmer(dB)",
    "shimmer_apq3":          "Shimmer:APQ3",
    "shimmer_apq5":          "Shimmer:APQ5",
    "shimmer_apq11":         "MDVP:APQ",
    "shimmer_data":          "Shimmer:DDA",
    "NTH":                   "NHR",
    "HTN":                   "HNR",
    "class_info":            "status",
}
_UCI301_MISSING = ["RPDE", "DFA", "spread1", "spread2", "D2", "PPE"]

# ── UCI-470 ──────────────────────────────────────────────────────────────────
_UCI470_MAP = {
    "locPctJitter":               "MDVP:Jitter(%)",
    "locAbsJitter":               "MDVP:Jitter(Abs)",
    "rapJitter":                  "MDVP:RAP",
    "ppq5Jitter":                 "MDVP:PPQ",
    "ddpJitter":                  "Jitter:DDP",
    "locShimmer":                 "MDVP:Shimmer",
    "locDbShimmer":               "MDVP:Shimmer(dB)",
    "apq3Shimmer":                "Shimmer:APQ3",
    "apq5Shimmer":                "Shimmer:APQ5",
    "apq11Shimmer":               "MDVP:APQ",
    "ddaShimmer":                 "Shimmer:DDA",
    "meanNoiseToHarmHarmonicity": "NHR",
    "meanHarmToNoiseHarmonicity": "HNR",
    "RPDE":                       "RPDE",
    "DFA":                        "DFA",
    "PPE":                        "PPE",
    "class":                      "status",
}
_UCI470_MISSING = ["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "spread1", "spread2", "D2"]

# ── UCI-489 ──────────────────────────────────────────────────────────────────
# Columns Jitter:DDP, Shimmer:DDA and NHR are not present; derived below.
# HNR25 is the 25-Hz-band HNR, closest to the wideband HNR in MDVP.
# Pitch features and spread1/spread2/D2 are absent; filled from UCI-174 medians.
_UCI489_MAP = {
    "Jitter_rel":  "MDVP:Jitter(%)",
    "Jitter_abs":  "MDVP:Jitter(Abs)",
    "Jitter_RAP":  "MDVP:RAP",
    "Jitter_PPQ":  "MDVP:PPQ",
    "Shim_loc":    "MDVP:Shimmer",
    "Shim_dB":     "MDVP:Shimmer(dB)",
    "Shim_APQ3":   "Shimmer:APQ3",
    "Shim_APQ5":   "Shimmer:APQ5",
    "Shi_APQ11":   "MDVP:APQ",
    "HNR25":       "HNR",
    "RPDE":        "RPDE",
    "DFA":         "DFA",
    "PPE":         "PPE",
    "Status":      "status",
}
_UCI489_MISSING = ["MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)", "spread1", "spread2", "D2"]


# ─────────────────────────────────────────────────────────────────────────────

def _load_uci174(speech_dir: Path) -> pd.DataFrame:
    path = speech_dir / "parkinsons.csv"
    if not path.is_file():
        raise FileNotFoundError(f"UCI-174 not found: {path}")
    df = pd.read_csv(path)
    df["status"] = df["status"].astype(int)
    print(f"  UCI-174: {len(df)} rows")
    return df[MDVP_FEATURES + ["status"]].copy()


def _load_uci301(speech_dir: Path, medians: pd.Series) -> Optional[pd.DataFrame]:
    path = speech_dir / "voice_uci301_multiple_sound_train.csv"
    if not path.is_file():
        print("  UCI-301: not found — skipping")
        return None

    raw = pd.read_csv(path)
    df = raw.rename(columns=_UCI301_MAP)
    out = pd.DataFrame(index=df.index)
    for feat in MDVP_FEATURES:
        if feat in df.columns:
            out[feat] = pd.to_numeric(df[feat], errors="coerce")
        else:
            out[feat] = float(medians.get(feat, 0.0))
    out["status"] = (pd.to_numeric(df.get("status", 0), errors="coerce") > 0).astype(int)
    out = out.dropna(subset=MDVP_FEATURES)
    print(f"  UCI-301: {len(out)} rows mapped")
    return out[MDVP_FEATURES + ["status"]]


def _load_uci470(speech_dir: Path, medians: pd.Series) -> Optional[pd.DataFrame]:
    path = speech_dir / "voice_uci470_pd_classification_features.csv"
    if not path.is_file():
        print("  UCI-470: not found — skipping")
        return None

    # Row 0 is a category header; row 1 is the actual column header
    raw = pd.read_csv(path, header=1)
    df = raw.rename(columns=_UCI470_MAP)
    out = pd.DataFrame(index=df.index)
    for feat in MDVP_FEATURES:
        if feat in df.columns:
            out[feat] = pd.to_numeric(df[feat], errors="coerce")
        else:
            out[feat] = float(medians.get(feat, 0.0))
    out["status"] = pd.to_numeric(df.get("status", 0), errors="coerce").fillna(0).astype(int)
    out = out.dropna(subset=MDVP_FEATURES)
    print(f"  UCI-470: {len(out)} rows mapped")
    return out[MDVP_FEATURES + ["status"]]


def _load_uci489(speech_dir: Path, medians: pd.Series) -> Optional[pd.DataFrame]:
    """UCI-489: 240 rows, perfectly balanced (120 HC, 120 PD).
    Derives Jitter:DDP = 3×RAP, Shimmer:DDA = 3×APQ3, NHR ≈ 1/HNR.
    """
    path = speech_dir / "voice_uci489_replicated_acoustic.csv"
    if not path.is_file():
        print("  UCI-489: not found — skipping")
        return None

    raw = pd.read_csv(path)
    df = raw.rename(columns=_UCI489_MAP)
    out = pd.DataFrame(index=df.index)

    for feat in MDVP_FEATURES:
        if feat in df.columns:
            out[feat] = pd.to_numeric(df[feat], errors="coerce")
        elif feat in _UCI489_MISSING:
            out[feat] = float(medians.get(feat, 0.0))
        else:
            out[feat] = np.nan

    # Derive missing columns
    if out["MDVP:RAP"].notna().any():
        out["Jitter:DDP"] = out["MDVP:RAP"] * 3.0
    if out["Shimmer:APQ3"].notna().any():
        out["Shimmer:DDA"] = out["Shimmer:APQ3"] * 3.0
    # NHR ≈ noise-to-harmonics; approximate as 1/HNR where HNR > 0
    hnr = pd.to_numeric(out.get("HNR", pd.Series(dtype=float)), errors="coerce")
    out["NHR"] = (1.0 / (hnr + 1e-10)).where(hnr > 0, other=float(medians.get("NHR", 0.02)))

    out["status"] = pd.to_numeric(df.get("status", 0), errors="coerce").fillna(0).astype(int)
    out = out.dropna(subset=MDVP_FEATURES)
    print(f"  UCI-489: {len(out)} rows mapped (HC={int((out['status']==0).sum())}, "
          f"PD={int((out['status']==1).sum())})")
    return out[MDVP_FEATURES + ["status"]]


# ─────────────────────────────────────────────────────────────────────────────

def run(raw_dir: Path, processed_dir: Path) -> Path:
    """Extract speech features from raw CSV datasets and save processed CSV."""
    speech_dir = Path(raw_dir) / "speech"
    out_dir = Path(processed_dir) / "speech"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "parkinsons.csv"

    print("[Speech Pipeline]")
    df174 = _load_uci174(speech_dir)
    medians = df174[MDVP_FEATURES].median()

    frames = [df174]
    for loader in (_load_uci301, _load_uci470, _load_uci489):
        result = loader(speech_dir, medians)
        if result is not None:
            frames.append(result)

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.dropna(subset=MDVP_FEATURES)
    combined["status"] = combined["status"].astype(int)
    combined = combined.drop_duplicates()

    combined.to_csv(out_path, index=False)
    n_pd = int(combined["status"].sum())
    n_hc = len(combined) - n_pd
    print(f"  → {out_path}  ({len(combined)} rows: PD={n_pd}, HC={n_hc})\n")
    return out_path


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument("--raw-dir",       default="data/raw")
    p.add_argument("--processed-dir", default="data/processed")
    args = p.parse_args()
    run(Path(args.raw_dir), Path(args.processed_dir))
