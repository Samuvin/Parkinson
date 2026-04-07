#!/usr/bin/env python3
"""Summarize UCI dataset 395 pen traces into one CSV (+ optional project-shaped columns).

Reads text files under the unpacked UCI spiral archive (see
data/raw/handwriting/download_handwriting_sources.sh). Each line:
X;Y;Z;Pressure;GripAngle;Timestamp;Test ID (see uci_spiral/readme.txt).

Output is for research / schema inspection; ``--approx-project-cols`` adds rough
analogues of dl_models.dataset.HANDWRITING_FEATURE_NAMES (not identical to PaHaW mapping).

No user-controlled URLs; only local directory paths from argparse defaults.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

# Mirrors dl_models/dataset.py HANDWRITING_FEATURE_NAMES (avoid importing torch).
PROJECT_HW_NAMES = [
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


def _read_trace(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep=r"\s*;\s*",
        header=None,
        engine="python",
        names=[
            "X",
            "Y",
            "Z",
            "Pressure",
            "GripAngle",
            "Timestamp",
            "TestID",
        ],
    )


def _status_from_relpath(rel: str) -> int:
    p = rel.replace("\\", "/").lower()
    if "/control/" in p or "/healthy/" in p:
        return 0
    return 1


def _row_from_trace(path: Path, root: Path) -> dict[str, float | int | str]:
    df = _read_trace(path)
    if len(df) < 3:
        raise ValueError(f"too few points in {path}")

    rel = str(path.relative_to(root))
    status = _status_from_relpath(rel)
    x = df["X"].to_numpy(dtype=np.float64)
    y = df["Y"].to_numpy(dtype=np.float64)
    pr = df["Pressure"].to_numpy(dtype=np.float64)
    ts = df["Timestamp"].to_numpy(dtype=np.float64)
    test_id = int(df["TestID"].mode().iloc[0])

    dts = np.diff(ts)
    dx = np.diff(x)
    dy = np.diff(y)
    valid_dt = dts > 1e-9
    vx = np.where(valid_dt, dx / np.maximum(dts, 1e-12), 0.0)
    vy = np.where(valid_dt, dy / np.maximum(dts, 1e-12), 0.0)
    spd = np.sqrt(vx * vx + vy * vy)
    if spd.size == 0:
        spd = np.array([0.0])

    path_len = float(np.sum(np.sqrt(dx * dx + dy * dy)))
    duration = float(ts.max() - ts.min())
    mean_pr = float(np.mean(pr))
    std_pr = float(np.std(pr))

    if spd.size > 1:
        dspd = np.diff(spd)
        dt2 = dts[1:]
        valid_a = (dt2 > 1e-9) & np.isfinite(dspd)
        accel = np.abs(dspd[valid_a] / dt2[valid_a]) if valid_a.any() else np.array([0.0])
    else:
        accel = np.array([0.0])

    row: dict[str, float | int | str] = {
        "sample_id": rel,
        "status": status,
        "test_id": test_id,
        "n_points": len(df),
        "duration_ms": duration,
        "path_length_px": path_len,
        "mean_pressure_raw": mean_pr,
        "std_pressure_raw": std_pr,
        "mean_speed_px_per_ms": float(np.mean(spd)),
        "std_speed_px_per_ms": float(np.std(spd)),
        "mean_abs_accel": float(np.mean(accel)) if accel.size else 0.0,
        "penup_ratio_pressure_le0": float(np.mean(pr <= 0.0)),
    }
    return row


def _approx_project_columns(base: dict[str, float | int | str]) -> dict[str, float]:
    mp = float(base["mean_pressure_raw"])
    vm = float(base["mean_speed_px_per_ms"])
    vs = float(base["std_speed_px_per_ms"])
    vvar = vs**2
    penup = float(base["penup_ratio_pressure_le0"])
    path_len = float(base["path_length_px"])
    dur_s = max(float(base["duration_ms"]) / 1000.0, 1e-6)
    n_pts = int(base["n_points"])

    # Heuristic scaling aligned loosely with scripts/build_handwriting_gait_from_public_sources.py
    surf = float(np.clip(1.0 - penup, 0.0, 1.0))
    return {
        "mean_pressure": mp / 2000.0,
        "pressure_variation": float(base["std_pressure_raw"]) / (abs(mp) + 1.0),
        "mean_velocity": vm / 50.0,
        "velocity_variation": vs / (abs(vm) + 0.01),
        "mean_acceleration": float(np.sqrt(max(vvar, 0.0))) / 10.0,
        "penup_time_ratio": float(np.clip(penup, 0.0, 1.0)),
        "mean_stroke_length": path_len / 100.0,
        "writing_tempo": (n_pts / dur_s) * 5.0 * 1e-4,
        "tremor_power": float(base["std_pressure_raw"]) ** 2 / 50000.0,
        "fluency_score": surf,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("data/raw/handwriting/uci_spiral"),
        help="Directory containing hw_dataset/ and/or new_dataset/.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/raw/handwriting/uci_spiral/uci_spiral_traces_summary.csv"),
        help="Output CSV path.",
    )
    parser.add_argument(
        "--approx-project-cols",
        action="store_true",
        help=f"Append heuristic columns named {PROJECT_HW_NAMES} (train_dl hints only).",
    )
    args = parser.parse_args()

    root = args.root.resolve()
    if not root.is_dir():
        raise SystemExit(f"Missing UCI spiral directory: {root} — run download_handwriting_sources.sh")

    txt_files = list(root.rglob("*.txt"))
    txt_files = [p for p in txt_files if p.name.lower() != "readme.txt"]
    if not txt_files:
        raise SystemExit(f"No trace .txt files under {root}")

    rows: list[dict[str, float | int | str]] = []
    for p in sorted(txt_files):
        try:
            rows.append(_row_from_trace(p, root))
        except (ValueError, pd.errors.ParserError) as e:
            print(f"skip {p}: {e}")

    out_df = pd.DataFrame(rows)
    if args.approx_project_cols:
        approx = pd.DataFrame([_approx_project_columns(r) for r in rows])
        out_df = pd.concat([out_df.reset_index(drop=True), approx], axis=1)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.out, index=False)
    print(f"Wrote {len(out_df)} rows -> {args.out}")


if __name__ == "__main__":
    main()
