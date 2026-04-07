#!/usr/bin/env python3
"""Build handwriting_data.csv and gait_data.csv from public PD datasets.

Handwriting: aggregated task-level rows from the PaHaW-derived feature files
mirrored in https://github.com/musaru/PD_PaHaW (PaHaW_extracted_files/*/*__*_1.csv).
Each row is one handwriting task; ``label`` is preserved as ``status``.

Gait: one row per walking trial from PhysioNet gaitpdb (Ga*.txt VGRF records).
https://physionet.org/content/gaitpdb/1.0.0/ — Controls: GaCo*, PD: GaPt*.

Output columns match dl_models/data/dataset.py. Row alignment with speech is still
index-based (see README.md and config/multimodal_features.yaml); this script trims to ``--n-rows`` for parity
with the speech CSV row count.

No user-controlled URLs are passed to file APIs (fixed base URLs only).
"""

from __future__ import annotations

import argparse
import io
import json
import re
import ssl
import sys
import urllib.error
import urllib.request
from pathlib import Path

import certifi
import numpy as np
import pandas as pd
from scipy.signal import find_peaks

GITHUB_TREE = (
    "https://api.github.com/repos/musaru/PD_PaHaW/git/trees/master?recursive=1"
)
GITHUB_RAW = (
    "https://raw.githubusercontent.com/musaru/PD_PaHaW/master/"
)
PHYSIONET_INDEX = "https://physionet.org/files/gaitpdb/1.0.0/"

HANDWRITING_COLS = [
    "mean_pressure",
    "pressure_variation",
    "mean_velocity",
    "velocity_variation",
    "mean_acceleration",
    "penup_time_ratio",
    "mean_stroke_length",
    "writing_tempo",
    "tremor_power",
    "fluency_score",
]

GAIT_COLS = [
    "stride_interval",
    "stride_variability",
    "swing_time",
    "stance_time",
    "double_support_time",
    "gait_speed",
    "cadence",
    "step_length",
    "stride_regularity",
    "gait_asymmetry",
]


def _https_get(url: str, timeout: float = 120.0) -> bytes:
    ctx = ssl.create_default_context(cafile=certifi.where())
    req = urllib.request.Request(
        url,
        headers={"User-Agent": "Parkinson-ETL/1.0 (research; +https://github.com)"},
    )
    with urllib.request.urlopen(req, context=ctx, timeout=timeout) as resp:
        return resp.read()


def handwriting_row_from_pahaw(df: pd.DataFrame) -> dict[str, float]:
    """Map one PaHaW feature row (wide) to the 10 project handwriting features."""
    r = df.iloc[0]
    pm = float(r.get("pressure_mean", np.nan))
    psd = float(r.get("pressure_standard_deviation", 0.0))
    vm = float(r.get("velocity_mean", np.nan))
    vsd = float(r.get("velocity_standard_deviation", 0.0))
    vvar = float(r.get("velocity_variance", 0.0))
    air = float(r.get("ratio_of_in_air_time", 0.0))
    surf = float(r.get("ratio_of_on_surface_time", 0.0))
    nstroke = float(r.get("Number_of_stroke", 0.0))
    otime = float(r.get("overall_time", 1.0))
    disp_m = float(r.get("displacement_mean", 0.0))
    pvar = float(r.get("pressure_variance", 0.0))

    return {
        "mean_pressure": pm / 2000.0,
        "pressure_variation": psd / (abs(pm) + 1.0),
        "mean_velocity": vm / 50.0,
        "velocity_variation": vsd / (abs(vm) + 0.01),
        "mean_acceleration": float(np.sqrt(max(vvar, 0.0))) / 10.0,
        "penup_time_ratio": float(np.clip(air, 0.0, 1.0)),
        "mean_stroke_length": disp_m / 100.0,
        "writing_tempo": (nstroke / max(otime / 1000.0, 1e-3)) * 5.0,
        "tremor_power": pvar / 50000.0,
        "fluency_score": float(np.clip(surf, 0.0, 1.0)),
    }


def load_pahaw_tasks(max_rows: int | None = None) -> pd.DataFrame:
    raw = _https_get(GITHUB_TREE)
    tree = json.loads(raw.decode("utf-8")).get("tree", [])
    paths: list[str] = []
    pat = re.compile(
        r"^PaHaW_extracted_files/(\d{5})/\1__([0-9]+)_1\.csv$"
    )
    for item in tree:
        p = item.get("path", "")
        if pat.match(p) and ".ipynb_checkpoints" not in p:
            paths.append(p)
    paths.sort()

    rows: list[dict[str, object]] = []
    for rel in paths:
        if max_rows is not None and len(rows) >= max_rows:
            break
        url = GITHUB_RAW + rel
        try:
            data = _https_get(url, timeout=60.0)
        except (urllib.error.URLError, OSError) as e:
            print(f"skip {rel}: {e}", file=sys.stderr)
            continue
        df = pd.read_csv(io.BytesIO(data), header=0, low_memory=False)
        if "label" not in df.columns:
            print(f"skip {rel}: no label column", file=sys.stderr)
            continue
        feats = handwriting_row_from_pahaw(df)
        feats["sample_id"] = rel.replace("/", "_").replace(".csv", "")
        feats["status"] = int(df["label"].iloc[0])
        rows.append(feats)

    return pd.DataFrame(rows)


def list_gait_trial_filenames() -> list[str]:
    html = _https_get(PHYSIONET_INDEX).decode("utf-8", errors="replace")
    names = re.findall(
        r'href="((?:Ga|Ju|Si)(?:Co|Pt)[^"/]+\.txt)"',
        html,
    )
    return sorted(set(names))


def gait_row_from_trial(time_s: np.ndarray, l_tot: np.ndarray, r_tot: np.ndarray) -> dict[str, float] | None:
    combined = l_tot.astype(np.float64) + r_tot.astype(np.float64)
    if combined.size < 200:
        return None
    height = float(np.percentile(combined, 55))
    peaks, props = find_peaks(
        combined, height=height, distance=35, prominence=height * 0.05
    )
    if peaks.size < 4:
        return None
    t_peaks = time_s[peaks]
    intervals = np.diff(t_peaks)
    intervals = intervals[intervals > 0.05]
    if intervals.size < 2:
        return None
    si = float(np.mean(intervals))
    if si <= 0:
        return None
    sv = float(np.std(intervals) / si)
    swing = 0.38 * si
    stance = 0.62 * si
    ds = 0.12 * si
    duration = float(time_s[-1] - time_s[0])
    if duration <= 0:
        return None
    cadence = float((peaks.size - 1) / duration * 30.0)
    prom = props.get("prominences")
    if prom is not None and prom.size:
        step_len = float(np.mean(prom) / 100.0)
    else:
        step_len = float(np.std(combined) / 100.0)
    reg = float(np.clip(1.0 - sv, 0.0, 1.0))
    ml = float(np.mean(l_tot))
    mr = float(np.mean(r_tot))
    asym = float(abs(ml - mr) / (abs(ml + mr) + 1e-8))
    gspd = float(np.mean(combined) / 500.0)

    return {
        "stride_interval": si,
        "stride_variability": sv,
        "swing_time": swing,
        "stance_time": stance,
        "double_support_time": ds,
        "gait_speed": gspd,
        "cadence": cadence,
        "step_length": step_len,
        "stride_regularity": reg,
        "gait_asymmetry": asym,
    }


def load_gait_trials(names: list[str], max_rows: int | None = None) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for fname in names:
        if max_rows is not None and len(rows) >= max_rows:
            break
        url = f"https://physionet.org/files/gaitpdb/1.0.0/{fname}"
        try:
            raw = _https_get(url, timeout=120.0)
        except (urllib.error.URLError, OSError) as e:
            print(f"skip {fname}: {e}", file=sys.stderr)
            continue
        arr = np.loadtxt(io.BytesIO(raw))
        if arr.ndim != 2 or arr.shape[1] < 19:
            continue
        time_s = arr[:, 0]
        l_tot = arr[:, 17]
        r_tot = arr[:, 18]
        feats = gait_row_from_trial(time_s, l_tot, r_tot)
        if feats is None:
            continue
        feats["sample_id"] = fname.replace(".txt", "")
        m = re.match(r"^(?:Ga|Ju|Si)(Co|Pt)", fname)
        group = m.group(1) if m else ""
        feats["status"] = 0 if group == "Co" else 1
        rows.append(feats)
    return pd.DataFrame(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--speech-csv",
        type=Path,
        default=Path("data/raw/speech/parkinsons.csv"),
        help="Used only to infer default --n-rows.",
    )
    parser.add_argument(
        "--n-rows",
        type=int,
        default=None,
        help="Trim each modality to this many rows (default: speech CSV data rows).",
    )
    parser.add_argument(
        "--handwriting-out",
        type=Path,
        default=Path("data/raw/handwriting/handwriting_data.csv"),
    )
    parser.add_argument(
        "--gait-out",
        type=Path,
        default=Path("data/raw/gait/gait_data.csv"),
    )
    args = parser.parse_args()

    n_target = args.n_rows
    if n_target is None:
        speech = pd.read_csv(args.speech_csv)
        n_target = len(speech)

    # Fetch slightly more than needed so we can trim to min(speech, hw, gait).
    fetch_cap = n_target + 120

    print("Fetching PaHaW handwriting task features from GitHub …")
    hw = load_pahaw_tasks(max_rows=fetch_cap)
    if hw.empty:
        sys.exit("No handwriting rows built; check network or GitHub API.")
    hw = hw.sort_values("sample_id").reset_index(drop=True)
    hw_full = hw[["sample_id"] + HANDWRITING_COLS + ["status"]]

    print("Listing PhysioNet gaitpdb walking trials (Ga/Ju/Si, *.txt) …")
    gait_names = list_gait_trial_filenames()
    print(
        f"Found {len(gait_names)} trial files; downloading until ~{fetch_cap} "
        "good feature rows …",
    )
    gait = load_gait_trials(gait_names, max_rows=fetch_cap)
    if gait.empty:
        sys.exit("No gait rows built; check network.")
    gait = gait.sort_values("sample_id").reset_index(drop=True)

    gait_full = gait[["sample_id"] + GAIT_COLS + ["status"]]
    n_hw = len(hw_full)
    n_gait = len(gait_full)
    n_use = min(n_target, n_hw, n_gait)
    if n_use < n_target:
        print(
            f"Note: trimming to {n_use} rows "
            f"(speech wants {n_target}; hw={n_hw}, gait={n_gait}).",
            file=sys.stderr,
        )

    hw_out = hw_full.iloc[:n_use]
    gait_out = gait_full.iloc[:n_use]

    args.handwriting_out.parent.mkdir(parents=True, exist_ok=True)
    args.gait_out.parent.mkdir(parents=True, exist_ok=True)
    hw_out.to_csv(args.handwriting_out, index=False)
    gait_out.to_csv(args.gait_out, index=False)
    print(f"Wrote {len(hw_out)} rows -> {args.handwriting_out}")
    print(f"Wrote {len(gait_out)} rows -> {args.gait_out}")


if __name__ == "__main__":
    main()
