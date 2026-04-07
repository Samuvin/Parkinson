#!/usr/bin/env python3
"""
Write short synthetic stereo-safe mono WAVs for notebooks under data/examples/speech/.

Signals are **not** real speech or patient data—only for librosa plots and pipeline smoke tests.
Regenerate after deleting: python scripts/generate_example_speech_wavs.py
"""

from pathlib import Path

import numpy as np
import soundfile as sf


def _normalize(y: np.ndarray) -> np.ndarray:
    y = np.asarray(y, dtype=np.float32)
    peak = np.max(np.abs(y)) + 1e-9
    return (y / peak * 0.9).astype(np.float32)


def main() -> int:
    root = Path(__file__).resolve().parents[1]
    out_dir = root / "data" / "examples" / "speech"
    out_dir.mkdir(parents=True, exist_ok=True)

    sr = 22_050
    duration_s = 2.5
    n = int(sr * duration_s)
    t = np.linspace(0.0, duration_s, n, endpoint=False, dtype=np.float64)

    f0 = 185.0
    # "Healthy" toy: stable harmonic stack + tiny noise
    healthy = (
        0.45 * np.sin(2 * np.pi * f0 * t)
        + 0.22 * np.sin(2 * np.pi * 2 * f0 * t)
        + 0.08 * np.sin(2 * np.pi * 3 * f0 * t)
        + 0.015 * np.random.default_rng(42).standard_normal(n)
    )
    healthy = _normalize(healthy)

    # "PD-like" toy: amplitude tremor + stronger jitter (noise) + mild F0 modulation (not clinical)
    rng = np.random.default_rng(7)
    tremor = 1.0 + 0.18 * np.sin(2 * np.pi * 5.2 * t)
    f0_wobble = f0 + 12.0 * np.sin(2 * np.pi * 3.1 * t)
    phase = 2 * np.pi * np.cumsum(f0_wobble) / sr
    pd_like = tremor * (
        0.5 * np.sin(phase)
        + 0.2 * np.sin(2 * phase)
        + 0.12 * np.sin(3 * phase)
    )
    pd_like += 0.055 * rng.standard_normal(n)
    pd_like = _normalize(pd_like)

    healthy_path = out_dir / "healthy_example.wav"
    pd_path = out_dir / "parkinson_example.wav"
    sf.write(str(healthy_path), healthy, sr, subtype="PCM_16")
    sf.write(str(pd_path), pd_like, sr, subtype="PCM_16")
    print(f"Wrote {healthy_path} ({len(healthy)} samples @ {sr} Hz)")
    print(f"Wrote {pd_path} ({len(pd_like)} samples @ {sr} Hz)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
