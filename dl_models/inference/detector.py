"""
Deep learning inference for Parkinson's disease detection.

Provides ``DLDetector`` — loads the trained SE-ResNet multimodal model, runs a
forward pass on patient feature vectors, and returns a structured result
(API fields still use ``prediction`` / ``prediction_label`` for client compatibility)
plus explainability (attention, SE weights, scaled-input magnitude bars).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import torch

from dl_models.data.dataset import (
    GAIT_FEATURE_NAMES,
    HANDWRITING_FEATURE_NAMES,
    SPEECH_FEATURE_NAMES,
)
from dl_models.algorithm.networks import MultimodalPDNet


def _feature_importance_from_scaled_inputs(
    speech_t: torch.Tensor,
    hw_t: torch.Tensor,
    gait_t: torch.Tensor,
) -> dict[str, list[float]]:
    """Per-feature scores from |x| after scaling (UI bars; not gradient attribution)."""
    out: dict[str, list[float]] = {}
    for key, t in (
        ("speech", speech_t),
        ("handwriting", hw_t),
        ("gait", gait_t),
    ):
        x = t.detach().squeeze(0).abs().cpu().numpy()
        total = float(x.sum())
        n = len(x)
        if total <= 0.0:
            out[key] = [1.0 / n] * n
        else:
            out[key] = (x / total).tolist()
    return out


class DLDetector:
    """High-level deep learning detector.

    Loads the saved ``.pt`` model and scalers, runs inference,
    and returns structured results including explainability data.

    Args:
        model_dir: Directory containing model files (default ``models/``).
        model_file: Name of the ``.pt`` weights file.
        scalers_file: Name of the scalers joblib file.
        metrics_file: Name of the metrics JSON file.
        device: Compute device (auto-detected if ``None``).
    """

    def __init__(
        self,
        model_dir: str | Path = "models",
        model_file: str = "multimodal_pdnet.pt",
        scalers_file: str = "dl_scalers.joblib",
        metrics_file: str = "dl_model_metrics.json",
        device: Optional[str] = None,
    ) -> None:
        self.model_dir = Path(model_dir)
        self.device = device or self._auto_device()

        self.model: Optional[MultimodalPDNet] = None
        self.scalers: Optional[dict] = None
        self.metrics: Optional[dict] = None
        self._model_file = model_file
        self._scalers_file = scalers_file
        self._metrics_file = metrics_file

    # -- helpers ------------------------------------------------------ #

    @staticmethod
    def _auto_device() -> str:
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    @staticmethod
    def is_available(model_dir: str | Path = "models") -> bool:
        """Return ``True`` if a trained DL model exists."""
        p = Path(model_dir) / "multimodal_pdnet.pt"
        return p.is_file()

    # -- loading ------------------------------------------------------ #

    def load(self) -> None:
        """Load model weights, scalers, and metrics from disk."""
        model_path = self.model_dir / self._model_file
        scalers_path = self.model_dir / self._scalers_file
        metrics_path = self.model_dir / self._metrics_file

        if not model_path.is_file():
            raise FileNotFoundError(
                f"DL model not found at {model_path}. "
                "Run train_dl.py first.",
            )

        # Load model
        self.model = MultimodalPDNet()
        state = torch.load(model_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(state)
        self.model.to(self.device)
        self.model.eval()

        # Load scalers
        if scalers_path.is_file():
            self.scalers = joblib.load(scalers_path)

        # Load metrics
        if metrics_path.is_file():
            with open(metrics_path) as f:
                self.metrics = json.load(f)

    # -- detection --------------------------------------------------- #

    def detect(
        self,
        speech_features: Optional[np.ndarray] = None,
        handwriting_features: Optional[np.ndarray] = None,
        gait_features: Optional[np.ndarray] = None,
    ) -> dict[str, Any]:
        """Run detection on a single patient sample.

        Missing modalities are zero-filled.

        Args:
            speech_features: [22] numpy array (or None).
            handwriting_features: [10] numpy array (or None).
            gait_features: [10] numpy array (or None).

        Returns:
            Dict with:
                success: bool
                prediction: 0 or 1 (legacy API key)
                prediction_label: 'healthy' or 'parkinsons'
                confidence: float 0-1
                probabilities: {healthy, parkinsons}
                model_type: 'deep_learning'
                attention_weights: {speech, handwriting, gait}
                feature_importance: {speech: [...], ...} (|scaled input|, sum=1 per modality)
                se_weights: {speech: [...], handwriting: [...], gait: [...]}
                feature_names: {speech: [...], ...}
                modalities_used: [str, ...]
        """
        if self.model is None:
            raise RuntimeError("Model not loaded. Call .load() first.")

        modalities_used: list[str] = []

        # Prepare features -- zero-fill missing modalities
        if speech_features is not None:
            modalities_used.append("speech")
            speech = np.asarray(speech_features, dtype=np.float64).reshape(1, -1)
        else:
            speech = np.zeros((1, 22), dtype=np.float64)

        if handwriting_features is not None:
            modalities_used.append("handwriting")
            hw = np.asarray(handwriting_features, dtype=np.float64).reshape(1, -1)
        else:
            hw = np.zeros((1, 10), dtype=np.float64)

        if gait_features is not None:
            modalities_used.append("gait")
            gait = np.asarray(gait_features, dtype=np.float64).reshape(1, -1)
        else:
            gait = np.zeros((1, 10), dtype=np.float64)

        # Scale features
        if self.scalers:
            speech = self.scalers["speech"].transform(speech)
            hw = self.scalers["handwriting"].transform(hw)
            gait = self.scalers["gait"].transform(gait)

        # Convert to tensors
        speech_t = torch.tensor(speech, dtype=torch.float32).to(self.device)
        hw_t = torch.tensor(hw, dtype=torch.float32).to(self.device)
        gait_t = torch.tensor(gait, dtype=torch.float32).to(self.device)

        # Forward pass
        with torch.no_grad():
            out = self.model(speech_t, hw_t, gait_t)

        prob = out["probability"].item()
        prediction = 1 if prob >= 0.5 else 0
        confidence = prob if prediction == 1 else 1.0 - prob

        # Attention weights
        attn = out["attention_weights"].squeeze(0).cpu().numpy()
        attention_weights = {
            "speech": float(attn[0]),
            "handwriting": float(attn[1]),
            "gait": float(attn[2]),
        }

        # SE weights (from block2 of each encoder)
        se_weights = {}
        for name, info_key in [
            ("speech", "speech_info"),
            ("handwriting", "handwriting_info"),
            ("gait", "gait_info"),
        ]:
            se_w = out[info_key]["se_weights_2"].squeeze(0).cpu().numpy()
            se_weights[name] = se_w.tolist()

        feature_importance = _feature_importance_from_scaled_inputs(
            speech_t, hw_t, gait_t,
        )

        return {
            "success": True,
            "prediction": prediction,
            "prediction_label": "parkinsons" if prediction == 1 else "healthy",
            "confidence": round(confidence, 4),
            "probabilities": {
                "healthy": round(1.0 - prob, 4),
                "parkinsons": round(prob, 4),
            },
            "model_type": "deep_learning",
            "ensemble_method": "SE-ResNet + Attention Fusion",
            "modalities_used": modalities_used if modalities_used else ["speech", "handwriting", "gait"],
            "attention_weights": attention_weights,
            "feature_importance": feature_importance,
            "se_weights": se_weights,
            "feature_names": {
                "speech": SPEECH_FEATURE_NAMES,
                "handwriting": HANDWRITING_FEATURE_NAMES,
                "gait": GAIT_FEATURE_NAMES,
            },
        }

    def get_model_info(self) -> dict[str, Any]:
        """Return model metadata for the /model_info endpoint."""
        info = {
            "model_type": "SE-ResNet1D + Attention Fusion",
            "framework": "PyTorch",
            "available": self.model is not None,
        }
        if self.metrics:
            info.update({
                "test_accuracy": self.metrics.get("test_accuracy"),
                "test_f1": self.metrics.get("test_f1"),
                "test_roc_auc": self.metrics.get("test_roc_auc"),
                "total_params": self.metrics.get("total_params"),
                "epochs_trained": self.metrics.get("epochs_trained"),
            })
        return info
