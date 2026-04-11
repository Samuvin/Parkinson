#!/usr/bin/env python3
"""
Train the Multimodal SE-ResNet1D model for Parkinson's Disease prediction.

Usage::

    python train_dl.py                     # uses defaults from config/project.yaml
    python train_dl.py --epochs 200        # override epochs
    python train_dl.py --device cuda       # force GPU
    python train_dl.py --skip-pipelines    # skip feature extraction (use existing CSVs)

The script will:
    1. Run feature-extraction pipelines (speech / gait / handwriting) if
       processed CSVs do not yet exist under ``data/processed/``
    2. Load the processed speech, handwriting, and gait CSVs
    3. Optionally apply SMOTE (with a safe ``k_neighbors`` for small cohorts)
    4. Split into train / val / test (70 / 15 / 15), stratified
    5. Train with AdamW, optional AMP (CUDA), cosine / OneCycle / plateau LR,
       class-weighted BCE when SMOTE is off, label smoothing, early stopping
       on val loss or val ROC-AUC
    6. Save best model (.pt), metrics (.json), and plots to models/
"""

from __future__ import annotations

import argparse
import random
from pathlib import Path
from typing import Optional

import numpy as np
import yaml
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import torch

from dl_models.algorithm import MultimodalPDNet
from dl_models.data import (
    ModalityMatchedDataset,
    MultimodalPDDataset,
    load_modalities_separate,
)
from dl_models.training import Trainer
from pipelines import run_all as run_pipelines


# ------------------------------------------------------------------ #
#  Helpers                                                            #
# ------------------------------------------------------------------ #

def set_seed(seed: int) -> None:
    """Fix RNG seeds for reproducibility (best-effort across libraries)."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_config(path: str = "config/project.yaml") -> dict:
    """Load YAML configuration."""
    with open(path) as f:
        return yaml.safe_load(f)


def _smote_single(
    feats: np.ndarray,
    labels: np.ndarray,
    k_neighbors: int = 5,
    random_state: int = 42,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Apply SMOTE to a single modality. Returns (feats, feats, labels)."""
    n_pos = int(labels.sum())
    n_neg = int(len(labels) - n_pos)
    k_eff = min(k_neighbors, min(n_pos, n_neg) - 1)
    if k_eff < 1:
        return feats, feats, labels
    sm = SMOTE(random_state=random_state, k_neighbors=k_eff)
    feats_res, labels_res = sm.fit_resample(feats, labels)
    return feats_res, feats_res, labels_res


# ------------------------------------------------------------------ #
#  Main                                                               #
# ------------------------------------------------------------------ #

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train Multimodal SE-ResNet1D for PD prediction",
    )
    parser.add_argument(
        "--config", default="config/project.yaml", help="Path to project YAML (config/project.yaml)",
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
    parser.add_argument(
        "--skip-pipelines", action="store_true",
        help="Skip feature-extraction pipelines (use existing processed CSVs)",
    )
    parser.add_argument(
        "--force-pipelines", action="store_true",
        help="Re-run all pipelines even if processed CSVs already exist",
    )
    args = parser.parse_args()

    # ---- config ----------------------------------------------------- #
    cfg = load_config(args.config)
    data_cfg = cfg.get("data", {})
    dl_cfg = cfg.get("deep_learning", {})

    raw_dir       = data_cfg.get("raw_dir",       "data/raw")
    processed_dir = data_cfg.get("processed_dir", "data/processed")
    random_state  = data_cfg.get("random_state", 42)
    train_size    = data_cfg.get("train_size", 0.70)
    val_size      = data_cfg.get("val_size", 0.15)

    # ---- feature-extraction pipelines -------------------------------- #
    if not args.skip_pipelines:
        print("=" * 55)
        print("  Step 1/2 — Feature extraction pipelines")
        print("=" * 55)
        run_pipelines(
            raw_dir=raw_dir,
            processed_dir=processed_dir,
            force=getattr(args, "force_pipelines", False),
        )
        print("=" * 55)
        print("  Step 2/2 — Model training")
        print("=" * 55)
    else:
        print("  Skipping pipelines — using existing processed CSVs")

    epochs = args.epochs or dl_cfg.get("epochs", 300)
    batch_size = args.batch_size or dl_cfg.get("batch_size", 32)
    lr = dl_cfg.get("learning_rate", 3e-4)
    weight_decay = dl_cfg.get("weight_decay", 1e-3)
    patience = dl_cfg.get("early_stopping_patience", 30)
    dropout = dl_cfg.get("dropout", 0.3)
    embed_dim = dl_cfg.get("embed_dim", 64)
    se_reduction = dl_cfg.get("se_reduction", 4)
    noise_std = dl_cfg.get("noise_std", 0.05)
    feature_dropout = dl_cfg.get("feature_dropout", 0.1)
    mixup_alpha = float(dl_cfg.get("mixup_alpha", 0.2))
    use_smote = (not args.no_smote) and dl_cfg.get("use_smote", False)
    seed = int(dl_cfg.get("seed", data_cfg.get("random_state", 42)))
    set_seed(seed)

    grad_clip_norm = float(dl_cfg.get("grad_clip_norm", 1.0))
    label_smoothing = float(dl_cfg.get("label_smoothing", 0.05))
    use_amp = bool(dl_cfg.get("use_amp", True))
    num_workers = int(dl_cfg.get("dataloader_num_workers", 0))
    lr_scheduler = str(dl_cfg.get("lr_scheduler", "cosine_warmup"))
    warmup_epochs = int(dl_cfg.get("warmup_epochs", 10))
    onecycle_pct_start = float(dl_cfg.get("onecycle_pct_start", 0.1))
    early_stop_monitor = str(dl_cfg.get("early_stop_monitor", "val_roc_auc"))
    smote_k_neighbors = int(dl_cfg.get("smote_k_neighbors", 5))
    use_focal_loss = bool(dl_cfg.get("focal_loss", True))
    focal_gamma = float(dl_cfg.get("focal_gamma", 2.0))
    accumulate_grad_batches = int(dl_cfg.get("accumulate_grad_batches", 1))

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

    # ---- load data (per-modality, no truncation) ------------------------ #
    mm_spec = data_cfg.get("multimodal_features_config")
    speech_feats, speech_labels, hw_feats, hw_labels, gait_feats, gait_labels = \
        load_modalities_separate(processed_dir, feature_spec_path=mm_spec)

    # ---- SMOTE (optional, per-modality) --------------------------------- #
    # ModalityMatchedDataset handles class-balance naturally via label-matched
    # sampling, so SMOTE is off by default. Enable with --no-smote=False in config.
    if use_smote:
        speech_feats, _, speech_labels = _smote_single(
            speech_feats, speech_labels, smote_k_neighbors, random_state,
        )
        hw_feats, _, hw_labels = _smote_single(
            hw_feats, hw_labels, smote_k_neighbors, random_state,
        )
        gait_feats, _, gait_labels = _smote_single(
            gait_feats, gait_labels, smote_k_neighbors, random_state,
        )

    # ---- train / val / test split (per modality, stratified) ------------ #
    test_ratio = 1.0 - train_size - val_size
    val_frac = val_size / (train_size + val_size)

    def _split(feats: np.ndarray, labels: np.ndarray):
        idx = np.arange(len(labels))
        idx_tv, idx_te = train_test_split(
            idx, test_size=test_ratio, random_state=random_state, stratify=labels,
        )
        idx_tr, idx_va = train_test_split(
            idx_tv, test_size=val_frac, random_state=random_state, stratify=labels[idx_tv],
        )
        return idx_tr, idx_va, idx_te

    s_tr, s_va, s_te = _split(speech_feats, speech_labels)
    h_tr, h_va, h_te = _split(hw_feats, hw_labels)
    g_tr, g_va, g_te = _split(gait_feats, gait_labels)

    # ---- standardise (fit on train split of each modality) -------------- #
    speech_scaler = StandardScaler().fit(speech_feats[s_tr])
    hw_scaler = StandardScaler().fit(hw_feats[h_tr])
    gait_scaler = StandardScaler().fit(gait_feats[g_tr])

    speech_sc = speech_scaler.transform(speech_feats)
    hw_sc = hw_scaler.transform(hw_feats)
    gait_sc = gait_scaler.transform(gait_feats)

    # ---- datasets ------------------------------------------------------- #
    # Training: dynamic label-matched sampling across all modalities.
    train_ds = ModalityMatchedDataset(
        speech_sc[s_tr], speech_labels[s_tr],
        hw_sc[h_tr],     hw_labels[h_tr],
        gait_sc[g_tr],   gait_labels[g_tr],
        dynamic=True, seed=seed,
    )
    # Val / test: pre-assigned pairs for reproducible evaluation.
    val_ds = ModalityMatchedDataset(
        speech_sc[s_va], speech_labels[s_va],
        hw_sc[h_va],     hw_labels[h_va],
        gait_sc[g_va],   gait_labels[g_va],
        dynamic=False, seed=seed,
    )
    test_ds = ModalityMatchedDataset(
        speech_sc[s_te], speech_labels[s_te],
        hw_sc[h_te],     hw_labels[h_te],
        gait_sc[g_te],   gait_labels[g_te],
        dynamic=False, seed=seed,
    )

    # pos_weight from gait train labels (anchor modality)
    n_pos = float(np.sum(gait_labels[g_tr] == 1))
    n_neg = float(np.sum(gait_labels[g_tr] == 0))
    pos_weight: Optional[float] = (n_neg / n_pos) if (n_pos > 0 and not use_focal_loss) else None

    idx_train, idx_val, idx_test = g_tr, g_va, g_te

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

    # MultimodalPDNet = 3× SE-ResNet1D encoder + AttentionFusion + DenseClassifier
    model = MultimodalPDNet(
        speech_features=speech_feats.shape[1],
        handwriting_features=hw_feats.shape[1],
        gait_features=gait_feats.shape[1],
        embed_dim=embed_dim,
        reduction=se_reduction,
        dropout=dropout,
    )

    total_params = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"\n  Model   : MultimodalPDNet  ({trainable:,} trainable params)")
    print(f"  Inputs  : speech={speech_feats.shape[1]}  "
          f"handwriting={hw_feats.shape[1]}  gait={gait_feats.shape[1]}")
    print(f"  Dataset : speech={len(s_tr)} train | "
          f"hw={len(h_tr)} train | gait={len(g_tr)} train  "
          f"(val={len(g_va)}  test={len(g_te)})")

    # ---- train ------------------------------------------------------ #
    trainer = Trainer(
        model,
        device=device,
        lr=lr,
        weight_decay=weight_decay,
        patience=patience,
        noise_std=noise_std,
        feature_dropout=feature_dropout,
        mixup_alpha=mixup_alpha,
        grad_clip_norm=grad_clip_norm,
        label_smoothing=label_smoothing,
        pos_weight=pos_weight,
        use_focal_loss=use_focal_loss,
        focal_gamma=focal_gamma,
        use_amp=use_amp,
        num_workers=num_workers,
        accumulate_grad_batches=accumulate_grad_batches,
        lr_scheduler=lr_scheduler,
        warmup_epochs=warmup_epochs,
        onecycle_pct_start=onecycle_pct_start,
        early_stop_monitor=early_stop_monitor,
    )

    history = trainer.fit(
        train_ds,
        val_ds,
        epochs=epochs,
        batch_size=batch_size,
    )

    # ---- evaluate --------------------------------------------------- #
    test_metrics = trainer.evaluate(test_ds, batch_size=batch_size)

    print(f"\n  ── Test Results ──────────────────────────")
    print(f"  Accuracy  : {test_metrics['accuracy']:.4f}")
    print(f"  Precision : {test_metrics['precision']:.4f}")
    print(f"  Recall    : {test_metrics['recall']:.4f}")
    print(f"  F1        : {test_metrics['f1']:.4f}")
    print(f"  ROC-AUC   : {test_metrics['roc_auc']:.4f}")
    print(f"  Threshold : {test_metrics['threshold_used']:.3f}  (F1-calibrated on val)")
    print(f"  ─────────────────────────────────────────\n")

    # ---- save ------------------------------------------------------- #
    trainer.save_model(save_dir / "multimodal_pdnet.pt")

    # Combine history + test metrics for JSON
    all_metrics = {
        "model_type": "SE-ResNet1D + Attention Fusion",
        "input_dims": {
            "speech": int(speech_feats.shape[1]),
            "handwriting": int(hw_feats.shape[1]),
            "gait": int(gait_feats.shape[1]),
        },
        "total_params": total_params,
        "trainable_params": trainable,
        "device": device,
        "epochs_trained": history["total_epochs"],
        "best_val_loss": history["best_val_loss"],
        "best_val_roc_auc": history["best_val_roc_auc"],
        "optimal_threshold": history["optimal_threshold"],
        "elapsed_seconds": history["elapsed_seconds"],
        "train_loss_history": [float(x) for x in history["train_losses"]],
        "val_loss_history": [float(x) for x in history["val_losses"]],
        "val_roc_auc_history": [float(x) for x in history["val_roc_aucs"]],
        "test_accuracy": test_metrics["accuracy"],
        "test_precision": test_metrics["precision"],
        "test_recall": test_metrics["recall"],
        "test_f1": test_metrics["f1"],
        "test_roc_auc": test_metrics["roc_auc"],
        "test_threshold": test_metrics["threshold_used"],
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
            "mixup_alpha": mixup_alpha,
            "use_focal_loss": use_focal_loss,
            "focal_gamma": focal_gamma,
            "use_smote": use_smote,
            "seed": seed,
            "grad_clip_norm": grad_clip_norm,
            "label_smoothing": label_smoothing,
            "use_amp": use_amp,
            "dataloader_num_workers": num_workers,
            "accumulate_grad_batches": accumulate_grad_batches,
            "lr_scheduler": lr_scheduler,
            "warmup_epochs": warmup_epochs,
            "onecycle_pct_start": onecycle_pct_start,
            "early_stop_monitor": early_stop_monitor,
            "smote_k_neighbors": smote_k_neighbors,
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

    elapsed = history.get("elapsed_seconds", 0)
    print(f"  Saved  → {save_dir}/")
    print(f"  Done!  Training took {elapsed:.1f}s  "
          f"({history['total_epochs']} epochs)  "
          f"Best val AUC={history['best_val_roc_auc']:.4f}")

if __name__ == "__main__":
    main()
