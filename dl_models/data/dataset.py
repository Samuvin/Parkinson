"""
PyTorch Dataset for multimodal Parkinson's Disease tabular data.

Feature column names and CSV paths are defined in ``config/multimodal_features.yaml``
(override with ``load_all_modalities(..., feature_spec_path=...)``).
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import yaml
from torch.utils.data import Dataset

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
            "stroke_width_variance", "edge_roughness", "stroke_smoothness",
            "contour_complexity", "stroke_inflection_count", "fragment_ratio",
            "stroke_width_mean", "ink_hull_ratio", "line_waviness",
            "ink_coverage",
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
    cp = out["csv_paths"]
    # Accept either a single combined CSV or three separate modality CSVs.
    if "combined" not in cp and not all(m in cp for m in ("speech", "handwriting", "gait")):
        raise ValueError(
            "csv_paths must contain either 'combined' or all of 'speech', 'handwriting', 'gait'"
        )
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
    return features, labels


def load_all_modalities(
    data_dir: str | Path,
    feature_spec_path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load speech, handwriting, and gait features using the multimodal feature spec.

    Supports two CSV layouts controlled by ``csv_paths`` in the spec:

    * **Single combined CSV** (``csv_paths.combined``): one file with all
      modality columns side-by-side.  Preferred — run
      ``scripts/build_combined_dataset.py`` to create it.
    * **Separate modality CSVs** (``csv_paths.speech/handwriting/gait``):
      legacy layout; rows are aligned by index.

    Returns ``(speech_feats, hw_feats, gait_feats, labels)`` as float64/int64
    NumPy arrays of equal length.
    """
    spec = load_multimodal_feature_spec(feature_spec_path)
    data_dir = Path(data_dir)
    lc = str(spec["label_column"])
    cp = spec["csv_paths"]

    if "combined" in cp:
        # ------------------------------------------------------------------ #
        # Single-file path: slice columns from one combined CSV
        # ------------------------------------------------------------------ #
        df = pd.read_csv(data_dir / cp["combined"])
        speech_feats = df[list(spec["speech_features"])].values.astype(np.float64)
        hw_feats = df[list(spec["handwriting_features"])].values.astype(np.float64)
        gait_feats = df[list(spec["gait_features"])].values.astype(np.float64)
        labels = df[lc].values.astype(np.int64)
        return speech_feats, hw_feats, gait_feats, labels

    # ---------------------------------------------------------------------- #
    # Legacy three-file path
    # ---------------------------------------------------------------------- #
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
    return (
        speech_feats[:n],
        hw_feats[:n],
        gait_feats[:n],
        speech_labels[:n],
    )


# --------------------------------------------------------------------------- #
#  ModalityMatchedDataset                                                       #
# --------------------------------------------------------------------------- #

class ModalityMatchedDataset(Dataset):
    """Multimodal dataset for independent per-modality data pools.

    Speech (1 990 rows), handwriting (75 rows), and gait (113 rows) come from
    different studies and cannot be row-aligned.  This dataset uses gait as
    the anchor (smallest, sets ``__len__``); for each gait sample it picks a
    speech and handwriting sample whose **label matches** the gait label.

    Modes
    -----
    dynamic=True  (training)
        Partners are re-drawn randomly on every ``__getitem__`` call so the
        model sees different combinations each epoch.

    dynamic=False  (evaluation)
        Partners are pre-assigned once at construction with a fixed RNG seed
        for deterministic, reproducible metrics.

    Args:
        speech_features: Float array ``[N_s, speech_dim]``.
        speech_labels: Int array ``[N_s]``.
        hw_features: Float array ``[N_h, hw_dim]``.
        hw_labels: Int array ``[N_h]``.
        gait_features: Float array ``[N_g, gait_dim]``  (anchor).
        gait_labels: Int array ``[N_g]``.
        dynamic: ``True`` for training, ``False`` for eval (pre-paired).
        seed: RNG seed used when ``dynamic=False``.
    """

    def __init__(
        self,
        speech_features: np.ndarray,
        speech_labels: np.ndarray,
        hw_features: np.ndarray,
        hw_labels: np.ndarray,
        gait_features: np.ndarray,
        gait_labels: np.ndarray,
        dynamic: bool = True,
        seed: int = 42,
    ) -> None:
        self.dynamic = dynamic

        self._gait_feats = torch.tensor(gait_features, dtype=torch.float32)
        self._gait_labels = np.asarray(gait_labels, dtype=np.int64)

        self._speech_feats = torch.tensor(speech_features, dtype=torch.float32)
        self._hw_feats = torch.tensor(hw_features, dtype=torch.float32)

        speech_labels_arr = np.asarray(speech_labels, dtype=np.int64)
        hw_labels_arr = np.asarray(hw_labels, dtype=np.int64)

        classes = sorted(set(int(c) for c in np.unique(gait_labels)))
        _all_speech = np.arange(len(speech_labels_arr))
        _all_hw = np.arange(len(hw_labels_arr))

        self._speech_by_class: dict[int, np.ndarray] = {}
        self._hw_by_class: dict[int, np.ndarray] = {}
        for c in classes:
            s_pool = np.where(speech_labels_arr == c)[0]
            self._speech_by_class[c] = s_pool if len(s_pool) else _all_speech
            h_pool = np.where(hw_labels_arr == c)[0]
            self._hw_by_class[c] = h_pool if len(h_pool) else _all_hw

        if not dynamic:
            rng = np.random.default_rng(seed)
            n = len(self._gait_labels)
            self._speech_assigned = np.empty(n, dtype=np.int64)
            self._hw_assigned = np.empty(n, dtype=np.int64)
            for i, lbl in enumerate(self._gait_labels):
                c = int(lbl)
                self._speech_assigned[i] = rng.choice(self._speech_by_class[c])
                self._hw_assigned[i] = rng.choice(self._hw_by_class[c])

    def __len__(self) -> int:
        return len(self._gait_labels)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        gait = self._gait_feats[idx]
        label = int(self._gait_labels[idx])

        if self.dynamic:
            s_pool = self._speech_by_class[label]
            h_pool = self._hw_by_class[label]
            s_idx = int(s_pool[torch.randint(len(s_pool), (1,)).item()])
            h_idx = int(h_pool[torch.randint(len(h_pool), (1,)).item()])
        else:
            s_idx = int(self._speech_assigned[idx])
            h_idx = int(self._hw_assigned[idx])

        return {
            "speech": self._speech_feats[s_idx],
            "handwriting": self._hw_feats[h_idx],
            "gait": gait,
            "label": torch.tensor(float(label), dtype=torch.float32),
        }


def load_modalities_separate(
    data_dir: str | Path,
    feature_spec_path: str | Path | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load each modality independently without truncating to min length.

    Unlike ``load_all_modalities``, the three modalities are returned as
    separate (features, labels) pairs and are **not** row-aligned.

    Returns
    -------
    speech_feats, speech_labels, hw_feats, hw_labels, gait_feats, gait_labels
    """
    spec = load_multimodal_feature_spec(feature_spec_path)
    data_dir = Path(data_dir)
    lc = str(spec["label_column"])
    cp = spec["csv_paths"]

    speech_feats, speech_labels = load_modality_csv(
        data_dir / cp["speech"], list(spec["speech_features"]), label_column=lc,
    )
    hw_feats, hw_labels = load_modality_csv(
        data_dir / cp["handwriting"], list(spec["handwriting_features"]), label_column=lc,
    )
    gait_feats, gait_labels = load_modality_csv(
        data_dir / cp["gait"], list(spec["gait_features"]), label_column=lc,
    )
    return speech_feats, speech_labels, hw_feats, hw_labels, gait_feats, gait_labels
