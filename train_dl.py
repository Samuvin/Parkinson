#!/usr/bin/env python3
"""
Train the Multimodal SE-ResNet1D model for Parkinson's Disease prediction.

Usage::

    python train_dl.py                     # uses defaults from config.yaml
    python train_dl.py --epochs 200        # override epochs
    python train_dl.py --device cuda       # force GPU

The script will:
    1. Load speech, handwriting, and gait CSVs
    2. Apply SMOTE for class balancing
    3. Split into train / val / test (70 / 15 / 15)
    4. Train SE-ResNet with augmentation, early stopping, LR scheduling
    5. Save best model (.pt), metrics (.json), and plots to models/
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import yaml
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch

from dl_models.dataset import (
    MultimodalPDDataset,
    load_all_modalities,
)
from dl_models.networks import MultimodalPDNet
from dl_models.trainer import Trainer

# ------------------------------------------------------------------ #
#  Logging                                                            #
# ------------------------------------------------------------------ #

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger("train_dl")


# ------------------------------------------------------------------ #
#  Helpers                                                            #
# ------------------------------------------------------------------ #

def load_config(path: str = "config.yaml") -> dict:
    """Load YAML configuration."""
    with open(path) as f:
        return yaml.safe_load(f)


def apply_smote(
    speech: np.ndarray,
    handwriting: np.ndarray,
    gait: np.ndarray,
    labels: np.ndarray,
    random_state: int = 42,
    k_neighbors: int = 5,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Apply SMOTE on concatenated features, then split back.

    SMOTE requires a single feature matrix, so we concatenate all
    modalities, oversample, then split the columns back.
    """
    n_speech = speech.shape[1]
    n_hw = handwriting.shape[1]

    combined = np.hstack([speech, handwriting, gait])
    sm = SMOTE(random_state=random_state, k_neighbors=k_neighbors)
    combined_res, labels_res = sm.fit_resample(combined, labels)

    speech_res = combined_res[:, :n_speech]
    hw_res = combined_res[:, n_speech : n_speech + n_hw]
    gait_res = combined_res[:, n_speech + n_hw :]

    logger.info(
        "SMOTE: %d -> %d samples (PD=%d, Healthy=%d)",
        len(labels), len(labels_res),
        int(labels_res.sum()), int((labels_res == 0).sum()),
    )
    return speech_res, hw_res, gait_res, labels_res


# ------------------------------------------------------------------ #
#  Main                                                               #
# ------------------------------------------------------------------ #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Multimodal SE-ResNet1D for PD prediction",
    )
    parser.add_argument(
        "--config", default="config.yaml", help="Path to config.yaml",
    )
    parser.add_argument(
        "--epochs", type=int, default=None,
        help="Override max epochs from config",
    )
    parser.add_argument(
        "--batch-size", type=int, default=None,
        help="Override batch size from config",
    )
    parser.add_argument(
        "--device", default=None,
        help="Force device (cpu/cuda/mps)",
    )
    parser.add_argument(
        "--no-smote", action="store_true",
        help="Disable SMOTE oversampling",
    )
    args = parser.parse_args()

    # ---- config ----------------------------------------------------- #
    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})
    dl_cfg = cfg.get("deep_learning", {})

    raw_dir = data_cfg.get("raw_dir", "data/raw")
    random_state = data_cfg.get("random_state", 42)
    train_size = data_cfg.get("train_size", 0.70)
    val_size = data_cfg.get("val_size", 0.15)

    epochs = args.epochs or dl_cfg.get("epochs", 100)
    batch_size = args.batch_size or dl_cfg.get("batch_size", 32)
    lr = dl_cfg.get("learning_rate", 1e-3)
    weight_decay = dl_cfg.get("weight_decay", 1e-4)
    patience = dl_cfg.get("early_stopping_patience", 15)
    dropout = dl_cfg.get("dropout", 0.3)
    embed_dim = dl_cfg.get("embed_dim", 64)
    se_reduction = dl_cfg.get("se_reduction", 4)
    noise_std = dl_cfg.get("noise_std", 0.05)
    feature_dropout = dl_cfg.get("feature_dropout", 0.1)
    use_smote = (not args.no_smote) and dl_cfg.get("use_smote", True)

    save_dir = Path(dl_cfg.get("save_dir", "models"))
    save_dir.mkdir(parents=True, exist_ok=True)

    # ---- device ----------------------------------------------------- #
    if args.device:
        device = args.device
    elif torch.cuda.is_available():
        device = "cuda"
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = "mps"
    else:
        device = "cpu"
    logger.info("Using device: %s", device)

    # ---- load data -------------------------------------------------- #
    mm_spec = data_cfg.get("multimodal_features_config")
    logger.info("Loading data from %s ...", raw_dir)
    if mm_spec:
        logger.info("Using multimodal feature spec: %s", mm_spec)
    speech, handwriting, gait, labels = load_all_modalities(
        raw_dir, feature_spec_path=mm_spec,
    )
    logger.info(
        "Loaded %d samples: speech=%s, handwriting=%s, gait=%s",
        len(labels), speech.shape, handwriting.shape, gait.shape,
    )

    # ---- SMOTE ------------------------------------------------------ #
    if use_smote:
        speech, handwriting, gait, labels = apply_smote(
            speech, handwriting, gait, labels,
            random_state=random_state,
        )

    # ---- train / val / test split ----------------------------------- #
    # First split: train+val vs test
    test_ratio = 1.0 - train_size - val_size
    idx = np.arange(len(labels))
    idx_trainval, idx_test = train_test_split(
        idx, test_size=test_ratio, random_state=random_state,
        stratify=labels,
    )
    # Second split: train vs val
    val_frac = val_size / (train_size + val_size)
    idx_train, idx_val = train_test_split(
        idx_trainval, test_size=val_frac, random_state=random_state,
        stratify=labels[idx_trainval],
    )

    logger.info(
        "Split: train=%d, val=%d, test=%d",
        len(idx_train), len(idx_val), len(idx_test),
    )

    # ---- standardise ------------------------------------------------ #
    # Fit scalers on training data only
    speech_scaler = StandardScaler().fit(speech[idx_train])
    hw_scaler = StandardScaler().fit(handwriting[idx_train])
    gait_scaler = StandardScaler().fit(gait[idx_train])

    def scale(indices: np.ndarray) -> tuple[np.ndarray, ...]:
        return (
            speech_scaler.transform(speech[indices]),
            hw_scaler.transform(handwriting[indices]),
            gait_scaler.transform(gait[indices]),
            labels[indices],
        )

    train_data = scale(idx_train)
    val_data = scale(idx_val)
    test_data = scale(idx_test)

    train_ds = MultimodalPDDataset(*train_data)
    val_ds = MultimodalPDDataset(*val_data)
    test_ds = MultimodalPDDataset(*test_data)

    # ---- save scalers for inference --------------------------------- #
    import joblib
    scaler_path = save_dir / "dl_scalers.joblib"
    joblib.dump(
        {
            "speech": speech_scaler,
            "handwriting": hw_scaler,
            "gait": gait_scaler,
        },
        scaler_path,
    )
    logger.info("Scalers saved to %s", scaler_path)

    # ---- model ------------------------------------------------------ #
    model = MultimodalPDNet(
        speech_features=speech.shape[1],
        handwriting_features=handwriting.shape[1],
        gait_features=gait.shape[1],
        embed_dim=embed_dim,
        reduction=se_reduction,
        dropout=dropout,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(
        "Model: %d total params, %d trainable", total_params, trainable,
    )

    # ---- train ------------------------------------------------------ #
    trainer = Trainer(
        model, device=device, lr=lr, weight_decay=weight_decay,
        patience=patience, noise_std=noise_std,
        feature_dropout=feature_dropout,
    )

    history = trainer.fit(
        train_ds, val_ds, epochs=epochs, batch_size=batch_size,
    )

    # ---- evaluate --------------------------------------------------- #
    test_metrics = trainer.evaluate(test_ds, batch_size=batch_size)

    logger.info("=" * 60)
    logger.info("TEST RESULTS")
    logger.info("=" * 60)
    for key in ("accuracy", "precision", "recall", "f1", "roc_auc"):
        logger.info("  %-12s: %.4f", key, test_metrics[key])
    logger.info("=" * 60)

    # ---- save ------------------------------------------------------- #
    trainer.save_model(save_dir / "multimodal_pdnet.pt")

    # Combine history + test metrics for JSON
    all_metrics = {
        "model_type": "SE-ResNet1D + Attention Fusion",
        "input_dims": {
            "speech": int(speech.shape[1]),
            "handwriting": int(handwriting.shape[1]),
            "gait": int(gait.shape[1]),
        },
        "total_params": total_params,
        "trainable_params": trainable,
        "device": device,
        "epochs_trained": history["total_epochs"],
        "best_val_loss": history["best_val_loss"],
        "elapsed_seconds": history["elapsed_seconds"],
        "train_loss_history": [float(x) for x in history["train_losses"]],
        "val_loss_history": [float(x) for x in history["val_losses"]],
        "test_accuracy": test_metrics["accuracy"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"],
        "test_roc_auc": test_metrics["roc_auc"],
        "hyperparameters": {
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "weight_decay": weight_decay,
            "patience": patience,
            "dropout": dropout,
            "embed_dim": embed_dim,
            "se_reduction": se_reduction,
            "noise_std": noise_std,
            "feature_dropout": feature_dropout,
            "use_smote": use_smote,
            "multimodal_features_config": mm_spec,
        },
    }
    trainer.save_metrics(all_metrics, save_dir / "dl_model_metrics.json")

    # ---- plots ------------------------------------------------------ #
    trainer.plot_training_curves(save_dir / "dl_training_curves.png")
    Trainer.plot_roc_curve(
        test_metrics["y_true"],
        test_metrics["y_prob"],
        save_dir / "dl_roc_curve.png",
    )
    Trainer.plot_confusion_matrix(
        test_metrics["y_true"],
        test_metrics["y_pred"],
        save_dir / "dl_confusion_matrix.png",
    )

    logger.info("All artifacts saved to %s/", save_dir)
    logger.info("Done.")


if __name__ == "__main__":
    main()
