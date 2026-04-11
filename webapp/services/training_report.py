"""Load config-driven feature spec + saved training metrics for reporting (no user input in paths)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


def _is_under(parent: Path, child: Path) -> bool:
    """Return True if *child* resolves inside *parent* (path traversal guard)."""
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def load_training_report(repo_root: Optional[Path] = None) -> dict[str, Any]:
    """
    Build a JSON-serialisable report: multimodal feature config, data paths,
    training metrics file, and URLs for plot artifacts.

    Paths come only from ``config/project.yaml`` and the referenced
    ``multimodal_features.yaml`` — not from request parameters.
    """
    root = repo_root if repo_root is not None else _repo_root()
    project_yaml = root / "config" / "project.yaml"
    out: dict[str, Any] = {
        "success": True,
        "pipeline": {
            "steps": [
                {
                    "order": 1,
                    "title": "Define features (config)",
                    "detail": (
                        "Column names and CSV locations live in "
                        "config/multimodal_features.yaml, referenced by "
                        "data.multimodal_features_config in config/project.yaml."
                    ),
                },
                {
                    "order": 2,
                    "title": "Prepare tabular data",
                    "detail": (
                        "Place aligned speech, handwriting, and gait CSVs under "
                        "data/raw/ (paths from the YAML). For raw audio / images / video, "
                        "extract features with the shared code under common/ so vectors "
                        "match those column definitions."
                    ),
                },
                {
                    "order": 3,
                    "title": "Train the model",
                    "detail": (
                        "Run train_dl.py — loads the three CSVs, fits StandardScaler per "
                        "modality on the train split, trains MultimodalPDNet, and writes "
                        "models/multimodal_pdnet.pt, dl_scalers.joblib, dl_model_metrics.json, "
                        "and plots."
                    ),
                },
                {
                    "order": 4,
                    "title": "Serve inference",
                    "detail": (
                        "The web app loads the checkpoint and scalers; POST /api/detect "
                        "expects the same feature geometry (22 + 10 + 10)."
                    ),
                },
            ],
        },
        "feature_spec": {},
        "training_artifacts": {},
        "artifact_urls": {},
        "errors": [],
    }

    if not project_yaml.is_file():
        out["success"] = False
        out["errors"].append(f"Missing {project_yaml}")
        return out

    with open(project_yaml) as f:
        project_cfg = yaml.safe_load(f) or {}

    data_cfg = project_cfg.get("data") or {}
    dl_cfg = project_cfg.get("deep_learning") or {}
    raw_dir = root / str(data_cfg.get("raw_dir", "data/raw"))
    mm_rel = data_cfg.get("multimodal_features_config", "config/multimodal_features.yaml")
    mm_path = root / str(mm_rel)
    save_dir = root / str(dl_cfg.get("save_dir", "models"))

    out["feature_spec"]["config_paths"] = {
        "project_yaml": str(project_yaml.relative_to(root)),
        "multimodal_yaml": str(mm_path.relative_to(root)) if mm_path.is_file() else str(mm_rel),
        "raw_dir": str(raw_dir.relative_to(root)) if _is_under(root, raw_dir) else str(raw_dir),
    }

    if not mm_path.is_file():
        out["errors"].append(f"Multimodal spec not found: {mm_path}")
        out["feature_spec"]["modalities"] = {}
    else:
        with open(mm_path) as f:
            spec = yaml.safe_load(f) or {}
        label_col = spec.get("label_column", "status")
        csv_paths = spec.get("csv_paths") or {}
        resolved_csv: dict[str, str] = {}
        for modality, rel in csv_paths.items():
            if not isinstance(rel, str) or ".." in rel.replace("\\", "/"):
                out["errors"].append(f"Skipped unsafe csv path for {modality!r}")
                continue
            full = (raw_dir / rel).resolve()
            if not _is_under(root, full):
                out["errors"].append(f"CSV path escapes repo root: {modality}")
                continue
            resolved_csv[modality] = str(full.relative_to(root))

        speech_names = list(spec.get("speech_features") or [])
        hw_names = list(spec.get("handwriting_features") or [])
        gait_names = list(spec.get("gait_features") or [])

        out["feature_spec"] = {
            **out["feature_spec"],
            "label_column": label_col,
            "csv_paths_relative_to_raw_dir": dict(csv_paths) if isinstance(csv_paths, dict) else {},
            "csv_paths_resolved_under_repo": resolved_csv,
            "modalities": {
                "speech": {"feature_count": len(speech_names), "feature_names": speech_names},
                "handwriting": {"feature_count": len(hw_names), "feature_names": hw_names},
                "gait": {"feature_count": len(gait_names), "feature_names": gait_names},
            },
        }

    metrics_path = save_dir / "dl_model_metrics.json"
    model_path = save_dir / "multimodal_pdnet.pt"
    scalers_path = save_dir / "dl_scalers.joblib"

    metrics_obj: Optional[dict[str, Any]] = None
    if metrics_path.is_file() and _is_under(root, metrics_path):
        try:
            with open(metrics_path) as f:
                metrics_obj = json.load(f)
        except (json.JSONDecodeError, OSError) as e:
            out["errors"].append(f"Could not read metrics JSON: {e}")
    elif not metrics_path.is_file():
        out["errors"].append(
            "No dl_model_metrics.json yet — run: python train_dl.py",
        )

    out["training_artifacts"] = {
        "save_dir": str(save_dir.relative_to(root)) if _is_under(root, save_dir) else str(save_dir),
        "metrics_file": str(metrics_path.relative_to(root)) if _is_under(root, metrics_path) else "dl_model_metrics.json",
        "metrics_loaded": metrics_obj is not None,
        "model_weights_exist": model_path.is_file() and _is_under(root, model_path),
        "scalers_exist": scalers_path.is_file() and _is_under(root, scalers_path),
        "metrics": metrics_obj,
    }

    plot_names = [
        "dl_roc_curve.png",
        "dl_confusion_matrix.png",
        "dl_training_curves.png",
    ]
    out["artifact_urls"] = {
        name: f"/model_images/{name}"
        for name in plot_names
    }

    return out
