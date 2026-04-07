"""
PyTorch Dataset for multimodal Parkinson's Disease tabular data.

Feature column names and CSV paths are defined in ``config/multimodal_features.yaml``
(override with ``load_all_modalities(..., feature_spec_path=...)``).
"""

from __future__ import annotations

import copy
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import Dataset

logger = logging.getLogger(__name__)

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_SPEC_PATH = _REPO_ROOT / "config" / "multimodal_features.yaml"


def _embedded_default_spec() -> dict[str, Any]:
    return {
        "label_column": "status",
        "csv_paths": {
            "speech": "speech/parkinsons.csv",
            "handwriting": "handwriting/handwriting_data.csv",
            "gait": "gait/gait_data.csv",
        },
        "speech_features": [
            "MDVP:Fo(Hz)", "MDVP:Fhi(Hz)", "MDVP:Flo(Hz)",
            "MDVP:Jitter(%)", "MDVP:Jitter(Abs)", "MDVP:RAP", "MDVP:PPQ",
            "Jitter:DDP", "MDVP:Shimmer", "MDVP:Shimmer(dB)",
            "Shimmer:APQ3", "Shimmer:APQ5", "MDVP:APQ", "Shimmer:DDA",
            "NHR", "HNR", "RPDE", "DFA", "spread1", "spread2", "D2", "PPE",
        ],
        "handwriting_features": [
            "mean_pressure", "pressure_variation", "mean_velocity",
            "velocity_variation", "mean_acceleration", "penup_time_ratio",
            "mean_stroke_length", "writing_tempo", "tremor_power",
            "fluency_score",
        ],
        "gait_features": [
            "stride_interval", "stride_variability", "swing_time",
            "stance_time", "double_support_time", "gait_speed",
            "cadence", "step_length", "stride_regularity",
            "gait_asymmetry",
        ],
    }


def load_multimodal_feature_spec(
    feature_spec_path: str | Path | None = None,
) -> dict[str, Any]:
    """Load YAML spec for CSV paths and feature column lists.

    If *feature_spec_path* is omitted or the file is missing, uses embedded
    defaults (same lists as the repository YAML).
    """
    if feature_spec_path is not None:
        p = Path(feature_spec_path)
        if not p.is_file():
            raise FileNotFoundError(f"Multimodal feature spec not found: {p.resolve()}")
        with open(p) as f:
            spec = yaml.safe_load(f)
    else:
        if _DEFAULT_SPEC_PATH.is_file():
            with open(_DEFAULT_SPEC_PATH) as f:
                spec = yaml.safe_load(f)
        else:
            spec = _embedded_default_spec()
    if not isinstance(spec, dict):
        raise ValueError("multimodal feature spec must be a YAML mapping")
    out = copy.deepcopy(spec)
    for key in ("speech_features", "handwriting_features", "gait_features", "csv_paths", "label_column"):
        if key not in out:
            raise ValueError(f"multimodal feature spec missing required key: {key!r}")
    for m in ("speech", "handwriting", "gait"):
        if m not in out["csv_paths"]:
            raise ValueError(f"csv_paths missing {m!r}")
    return out


# Module-level names for imports (from default spec path or embedded fallback).
_SPEC0 = load_multimodal_feature_spec(None)
SPEECH_FEATURE_NAMES: list[str] = list(_SPEC0["speech_features"])
HANDWRITING_FEATURE_NAMES: list[str] = list(_SPEC0["handwriting_features"])
GAIT_FEATURE_NAMES: list[str] = list(_SPEC0["gait_features"])


class MultimodalPDDataset(Dataset):
    """Multimodal Parkinson's Disease dataset.

    Each sample contains speech, handwriting, and gait feature
    vectors plus a binary label (0 = healthy, 1 = PD).
    """

    def __init__(
        self,
        speech_features: np.ndarray,
        handwriting_features: np.ndarray,
        gait_features: np.ndarray,
        labels: np.ndarray,
    ) -> None:
        assert len(speech_features) == len(labels)
        assert len(handwriting_features) == len(labels)
        assert len(gait_features) == len(labels)

        self.speech = torch.tensor(speech_features, dtype=torch.float32)
        self.handwriting = torch.tensor(handwriting_features, dtype=torch.float32)
        self.gait = torch.tensor(gait_features, dtype=torch.float32)
        self.labels = torch.tensor(labels, dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "speech": self.speech[idx],
            "handwriting": self.handwriting[idx],
            "gait": self.gait[idx],
            "label": self.labels[idx],
        }


def load_modality_csv(
    path: str | Path,
    feature_columns: list[str],
    label_column: str = "status",
) -> tuple[np.ndarray, np.ndarray]:
    """Load a single modality CSV and return (features, labels)."""
    df = pd.read_csv(path)
    features = df[feature_columns].values.astype(np.float64)
    labels = df[label_column].values.astype(np.int64)
    logger.info(
        "Loaded %s: %d samples, %d features, PD=%d, Healthy=%d",
        Path(path).name, len(df), features.shape[1],
        int(labels.sum()), int((labels == 0).sum()),
    )
    return features, labels


def load_all_modalities(
    data_dir: str | Path,
    feature_spec_path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load speech, handwriting, and gait CSVs using the multimodal feature spec.

    Paths in the spec are joined with *data_dir*. Row alignment follows the
    minimum length across modalities; labels come from the speech CSV.
    """
    spec = load_multimodal_feature_spec(feature_spec_path)
    data_dir = Path(data_dir)
    lc = str(spec["label_column"])
    cp = spec["csv_paths"]

    speech_feats, speech_labels = load_modality_csv(
        data_dir / cp["speech"],
        list(spec["speech_features"]),
        label_column=lc,
    )
    hw_feats, _ = load_modality_csv(
        data_dir / cp["handwriting"],
        list(spec["handwriting_features"]),
        label_column=lc,
    )
    gait_feats, _ = load_modality_csv(
        data_dir / cp["gait"],
        list(spec["gait_features"]),
        label_column=lc,
    )

    n = min(len(speech_feats), len(hw_feats), len(gait_feats))
    if n < len(speech_feats):
        logger.warning(
            "Modality sizes differ; truncating to %d samples.", n,
        )

    return (
        speech_feats[:n],
        hw_feats[:n],
        gait_feats[:n],
        speech_labels[:n],
    )
