"""
Training utilities for the MultimodalPDNet.

Provides a ``Trainer`` class that handles:
    - Online data augmentation (Gaussian noise, feature dropout)
    - Train / validation loop with early stopping (val loss or val ROC-AUC)
    - Learning rate scheduling (ReduceLROnPlateau, cosine annealing, or OneCycle)
    - Optional mixed precision (CUDA), AdamW, class-weighted BCE, label smoothing
    - Best model checkpointing and metric plots
"""

from __future__ import annotations

import json
import logging
import time
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    ConfusionMatrixDisplay,
)
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader

from dl_models.data.dataset import MultimodalPDDataset
from dl_models.algorithm.networks import MultimodalPDNet

logger = logging.getLogger(__name__)


# ------------------------------------------------------------------ #
#  Data Augmentation                                                  #
# ------------------------------------------------------------------ #

def augment_batch(
    batch: dict[str, torch.Tensor],
    noise_std: float = 0.05,
    feature_dropout: float = 0.1,
) -> dict[str, torch.Tensor]:
    """Apply in-place Gaussian noise and random feature dropout.

    Args:
        batch: Dict with keys ``speech``, ``handwriting``, ``gait``,
               ``label``.  Tensors are modified in place.
        noise_std: Std-dev of additive Gaussian noise (features are z-scored).
        feature_dropout: Probability of zeroing a feature.

    Returns:
        The same batch dict (modified).
    """
    for key in ("speech", "handwriting", "gait"):
        x = batch[key]
        noise = torch.randn_like(x) * noise_std
        x = x + noise
        mask = torch.rand_like(x) > feature_dropout
        x = x * mask
        batch[key] = x
    return batch


# ------------------------------------------------------------------ #
#  Trainer                                                            #
# ------------------------------------------------------------------ #

class Trainer:
    """Training loop for ``MultimodalPDNet``.

    Args:
        model: An instance of ``MultimodalPDNet``.
        device: ``'cpu'``, ``'cuda'``, or ``'mps'``.
        lr: Initial / max learning rate (default 0.001).
        weight_decay: AdamW weight decay (default 1e-4).
        patience: Early stopping patience in epochs (default 15).
        noise_std: Augmentation noise level (default 0.05).
        feature_dropout: Augmentation feature dropout (default 0.1).
        grad_clip_norm: Max gradient norm (0 disables clipping).
        label_smoothing: Epsilon for soft binary targets in ``[0, 1]``.
        pos_weight: Positive-class weight for ``BCEWithLogitsLoss`` (imbalanced data).
        use_amp: Use automatic mixed precision on CUDA only.
        num_workers: ``DataLoader`` worker processes.
        lr_scheduler: ``plateau``, ``cosine``, or ``onecycle``.
        plateau_factor: LR multiply factor when plateau fires.
        plateau_patience: Epochs with no val-loss improvement before LR drop.
        onecycle_pct_start: Fraction of steps in the increasing-LR phase.
        early_stop_monitor: ``val_loss`` (minimize) or ``val_roc_auc`` (maximize).
    """

    def __init__(
        self,
        model: MultimodalPDNet,
        device: str = "cpu",
        lr: float = 1e-3,
        weight_decay: float = 1e-4,
        patience: int = 15,
        noise_std: float = 0.05,
        feature_dropout: float = 0.1,
        *,
        grad_clip_norm: float = 1.0,
        label_smoothing: float = 0.0,
        pos_weight: Optional[float] = None,
        use_amp: bool = False,
        num_workers: int = 0,
        lr_scheduler: str = "plateau",
        plateau_factor: float = 0.5,
        plateau_patience: int = 5,
        onecycle_pct_start: float = 0.1,
        early_stop_monitor: str = "val_loss",
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.base_lr = lr
        self.patience = patience
        self.noise_std = noise_std
        self.feature_dropout = feature_dropout
        self.grad_clip_norm = grad_clip_norm
        self.label_smoothing = label_smoothing
        self.use_amp = bool(
            use_amp and device == "cuda" and torch.cuda.is_available()
        )
        self.num_workers = max(0, int(num_workers))
        self.lr_scheduler_kind = lr_scheduler.strip().lower()
        self.plateau_factor = plateau_factor
        self.plateau_patience = plateau_patience
        self.onecycle_pct_start = onecycle_pct_start
        esm = early_stop_monitor.strip().lower()
        if esm not in ("val_loss", "val_roc_auc"):
            raise ValueError(
                "early_stop_monitor must be 'val_loss' or 'val_roc_auc'",
            )
        self.early_stop_monitor = esm

        if self.lr_scheduler_kind not in ("plateau", "cosine", "onecycle"):
            raise ValueError(
                "lr_scheduler must be 'plateau', 'cosine', or 'onecycle'",
            )

        pw_tensor: Optional[torch.Tensor] = None
        if pos_weight is not None and pos_weight > 0:
            pw_tensor = torch.tensor(
                [float(pos_weight)], device=device, dtype=torch.float32,
            )
        self.criterion = nn.BCEWithLogitsLoss(pos_weight=pw_tensor)

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay,
        )

        self._epoch_scheduler: Any = None
        self._onecycle: Any = None
        self._scaler: Optional[GradScaler] = None
        if self.use_amp:
            self._scaler = GradScaler()

        # History
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.val_roc_aucs: list[float] = []
        self.best_val_loss = float("inf")
        self.best_val_roc_auc = float("-inf")
        self._best_monitor_score = float("inf")
        self.epochs_no_improve = 0
        self.best_state: Optional[dict] = None

    def _smooth_labels(self, labels: torch.Tensor) -> torch.Tensor:
        if self.label_smoothing <= 0.0:
            return labels
        eps = self.label_smoothing
        return labels * (1.0 - eps) + (1.0 - labels) * eps

    # -- single epoch ------------------------------------------------- #

    def _train_epoch(
        self,
        loader: DataLoader,
        augment: bool = True,
    ) -> float:
        """Run one training epoch. Returns average loss."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0

        for batch in loader:
            if augment:
                batch = augment_batch(
                    batch, self.noise_std, self.feature_dropout,
                )

            speech = batch["speech"].to(self.device)
            handwriting = batch["handwriting"].to(self.device)
            gait = batch["gait"].to(self.device)
            labels = self._smooth_labels(batch["label"].to(self.device))

            self.optimizer.zero_grad(set_to_none=True)

            if self.use_amp and self._scaler is not None:
                with autocast():
                    out = self.model(speech, handwriting, gait)
                    loss = self.criterion(
                        out["logit"].squeeze(-1), labels,
                    )
                self._scaler.scale(loss).backward()
                if self.grad_clip_norm > 0:
                    self._scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=self.grad_clip_norm,
                    )
                self._scaler.step(self.optimizer)
                self._scaler.update()
            else:
                out = self.model(speech, handwriting, gait)
                loss = self.criterion(out["logit"].squeeze(-1), labels)
                loss.backward()
                if self.grad_clip_norm > 0:
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=self.grad_clip_norm,
                    )
                self.optimizer.step()

            if self._onecycle is not None:
                self._onecycle.step()

            total_loss += loss.item()
            n_batches += 1

        return total_loss / max(n_batches, 1)

    @torch.no_grad()
    def _eval_epoch(self, loader: DataLoader) -> tuple[float, dict[str, Any]]:
        """Evaluate on a data loader. Returns (loss, metrics_dict)."""
        self.model.eval()
        total_loss = 0.0
        n_batches = 0
        all_probs: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []

        for batch in loader:
            speech = batch["speech"].to(self.device)
            handwriting = batch["handwriting"].to(self.device)
            gait = batch["gait"].to(self.device)
            labels = batch["label"].to(self.device)

            if self.use_amp:
                with autocast():
                    out = self.model(speech, handwriting, gait)
                    loss = self.criterion(out["logit"].squeeze(-1), labels)
            else:
                out = self.model(speech, handwriting, gait)
                loss = self.criterion(out["logit"].squeeze(-1), labels)
            total_loss += loss.item()
            n_batches += 1

            probs = out["probability"].squeeze(-1).cpu().numpy()
            all_probs.append(probs)
            all_labels.append(labels.cpu().numpy())

        avg_loss = total_loss / max(n_batches, 1)

        y_true = np.concatenate(all_labels)
        y_prob = np.concatenate(all_probs)
        y_pred = (y_prob >= 0.5).astype(int)

        metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": float(roc_auc_score(y_true, y_prob))
            if len(np.unique(y_true)) > 1
            else 0.0,
        }

        return avg_loss, metrics

    def _setup_schedulers(self, epochs: int, steps_per_epoch: int) -> None:
        """Attach per-epoch or OneCycle LR schedulers for this run."""
        self._epoch_scheduler = None
        self._onecycle = None
        kind = self.lr_scheduler_kind
        epochs_i = max(int(epochs), 1)
        steps_i = max(int(steps_per_epoch), 1)

        if kind == "plateau":
            self._epoch_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode="min",
                factor=self.plateau_factor,
                patience=self.plateau_patience,
            )
        elif kind == "cosine":
            eta_min = max(self.base_lr * 0.01, 1e-6)
            self._epoch_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=epochs_i,
                eta_min=eta_min,
            )
        elif kind == "onecycle":
            self._onecycle = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=self.base_lr,
                epochs=epochs_i,
                steps_per_epoch=steps_i,
                pct_start=self.onecycle_pct_start,
            )

    def _dataloader_kwargs(self) -> dict[str, Any]:
        return {
            "num_workers": self.num_workers,
            "pin_memory": self.device == "cuda",
        }

    def _early_stop_improved(
        self, val_loss: float, val_roc_auc: float,
    ) -> bool:
        if self.early_stop_monitor == "val_roc_auc":
            return val_roc_auc > self._best_monitor_score + 1e-6
        return val_loss < self._best_monitor_score - 1e-6

    def _early_stop_update_best(self, val_loss: float, val_roc_auc: float) -> None:
        if self.early_stop_monitor == "val_roc_auc":
            self._best_monitor_score = val_roc_auc
        else:
            self._best_monitor_score = val_loss

    # -- full training loop ------------------------------------------- #

    def fit(
        self,
        train_dataset: MultimodalPDDataset,
        val_dataset: MultimodalPDDataset,
        epochs: int = 100,
        batch_size: int = 32,
        augment: bool = True,
    ) -> dict[str, Any]:
        """Train the model with early stopping.

        Args:
            train_dataset: Training ``MultimodalPDDataset``.
            val_dataset: Validation ``MultimodalPDDataset``.
            epochs: Maximum number of epochs.
            batch_size: Batch size.
            augment: Whether to apply online augmentation.

        Returns:
            Dict with final training metrics and history.
        """
        dl_kw = self._dataloader_kwargs()
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            drop_last=False,
            **dl_kw,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            **dl_kw,
        )

        steps_per_epoch = len(train_loader)
        self._setup_schedulers(epochs, steps_per_epoch)

        self.train_losses.clear()
        self.val_losses.clear()
        self.val_roc_aucs.clear()
        self.best_val_loss = float("inf")
        self.best_val_roc_auc = float("-inf")
        if self.early_stop_monitor == "val_roc_auc":
            self._best_monitor_score = float("-inf")
        else:
            self._best_monitor_score = float("inf")
        self.epochs_no_improve = 0
        self.best_state = None

        start_time = time.time()
        logger.info(
            "Starting training: %d epochs, batch_size=%d, device=%s, "
            "scheduler=%s, early_stop=%s, amp=%s, workers=%d",
            epochs,
            batch_size,
            self.device,
            self.lr_scheduler_kind,
            self.early_stop_monitor,
            self.use_amp,
            self.num_workers,
        )

        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch(train_loader, augment=augment)
            val_loss, val_metrics = self._eval_epoch(val_loader)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            self.val_roc_aucs.append(val_metrics["roc_auc"])

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
            auc = val_metrics["roc_auc"]
            if auc > self.best_val_roc_auc:
                self.best_val_roc_auc = auc

            if self._epoch_scheduler is not None:
                if self.lr_scheduler_kind == "plateau":
                    self._epoch_scheduler.step(val_loss)
                elif self.lr_scheduler_kind == "cosine":
                    self._epoch_scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]

            if epoch % 5 == 0 or epoch == 1:
                logger.info(
                    "Epoch %3d/%d | Train Loss: %.4f | Val Loss: %.4f | "
                    "Val Acc: %.3f | Val AUC: %.3f | LR: %.6f",
                    epoch,
                    epochs,
                    train_loss,
                    val_loss,
                    val_metrics["accuracy"],
                    val_metrics["roc_auc"],
                    current_lr,
                )

            if self._early_stop_improved(val_loss, val_metrics["roc_auc"]):
                self._early_stop_update_best(val_loss, val_metrics["roc_auc"])
                self.epochs_no_improve = 0
                self.best_state = {
                    k: v.cpu().clone()
                    for k, v in self.model.state_dict().items()
                }
            else:
                self.epochs_no_improve += 1

            if self.epochs_no_improve >= self.patience:
                logger.info(
                    "Early stopping at epoch %d (patience=%d, monitor=%s).",
                    epoch,
                    self.patience,
                    self.early_stop_monitor,
                )
                break

        elapsed = time.time() - start_time
        logger.info("Training completed in %.1f seconds.", elapsed)

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)
            logger.info(
                "Restored best checkpoint (monitor=%s, best_val_loss=%.4f, "
                "best_val_auc=%.4f).",
                self.early_stop_monitor,
                self.best_val_loss,
                self.best_val_roc_auc,
            )

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_roc_aucs": self.val_roc_aucs,
            "best_val_loss": self.best_val_loss,
            "best_val_roc_auc": self.best_val_roc_auc,
            "total_epochs": len(self.train_losses),
            "elapsed_seconds": elapsed,
        }

    # -- evaluation --------------------------------------------------- #

    def evaluate(
        self, test_dataset: MultimodalPDDataset, batch_size: int = 32,
    ) -> dict[str, Any]:
        """Evaluate on the test set.

        Returns:
            Dict with metrics, predictions, probabilities, and labels.
        """
        loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            **self._dataloader_kwargs(),
        )
        loss, metrics = self._eval_epoch(loader)
        metrics["test_loss"] = loss

        # Collect predictions for plots
        all_probs = []
        all_labels = []
        with torch.no_grad():
            for batch in loader:
                out = self.model(
                    batch["speech"].to(self.device),
                    batch["handwriting"].to(self.device),
                    batch["gait"].to(self.device),
                )
                all_probs.append(
                    out["probability"].squeeze(-1).cpu().numpy(),
                )
                all_labels.append(batch["label"].numpy())

        metrics["y_true"] = np.concatenate(all_labels)
        metrics["y_prob"] = np.concatenate(all_probs)
        metrics["y_pred"] = (metrics["y_prob"] >= 0.5).astype(int)

        return metrics

    # -- saving ------------------------------------------------------- #

    def save_model(self, path: str | Path) -> None:
        """Save model state dict to *path*."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)
        logger.info("Model saved to %s", path)

    def save_metrics(
        self, metrics: dict[str, Any], path: str | Path,
    ) -> None:
        """Save JSON-serialisable metrics to *path*."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Strip numpy arrays for JSON
        clean: dict[str, Any] = {}
        for k, v in metrics.items():
            if isinstance(v, np.ndarray):
                continue
            if isinstance(v, (np.floating, np.integer)):
                clean[k] = float(v)
            else:
                clean[k] = v

        with open(path, "w") as f:
            json.dump(clean, f, indent=2)
        logger.info("Metrics saved to %s", path)

    # -- plotting ----------------------------------------------------- #

    def plot_training_curves(self, save_path: str | Path) -> None:
        """Save train/val loss curves and validation ROC-AUC."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(self.train_losses, label="Train Loss", color="tab:blue")
        ax1.plot(self.val_losses, label="Val Loss", color="tab:orange")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("BCE Loss")
        ax1.grid(True, alpha=0.3)

        if self.val_roc_aucs:
            ax2 = ax1.twinx()
            ax2.plot(
                self.val_roc_aucs,
                label="Val ROC-AUC",
                color="tab:green",
                alpha=0.85,
            )
            ax2.set_ylabel("ROC-AUC")
            ax2.set_ylim(0.0, 1.02)

        ax1.set_title("Training curves — loss and validation AUC")
        lines1, lab1 = ax1.get_legend_handles_labels()
        if self.val_roc_aucs:
            lines2, lab2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, lab1 + lab2, loc="center right")
        else:
            ax1.legend(loc="best")
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        logger.info("Training curves saved to %s", save_path)

    @staticmethod
    def plot_roc_curve(
        y_true: np.ndarray,
        y_prob: np.ndarray,
        save_path: str | Path,
    ) -> None:
        """Save ROC curve plot."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        fpr, tpr, _ = roc_curve(y_true, y_prob)
        auc_val = roc_auc_score(y_true, y_prob)

        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(fpr, tpr, label=f"SE-ResNet (AUC = {auc_val:.3f})")
        ax.plot([0, 1], [0, 1], "k--", alpha=0.5)
        ax.set_xlabel("False Positive Rate")
        ax.set_ylabel("True Positive Rate")
        ax.set_title("ROC Curve — Multimodal SE-ResNet")
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        logger.info("ROC curve saved to %s", save_path)

    @staticmethod
    def plot_confusion_matrix(
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save_path: str | Path,
    ) -> None:
        """Save confusion matrix plot."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        cm = confusion_matrix(y_true, y_pred)
        fig, ax = plt.subplots(figsize=(6, 5))
        disp = ConfusionMatrixDisplay(
            cm, display_labels=["Healthy", "PD"],
        )
        disp.plot(ax=ax, cmap="Blues")
        ax.set_title("Confusion Matrix — SE-ResNet")
        fig.tight_layout()
        fig.savefig(save_path, dpi=150)
        plt.close(fig)
        logger.info("Confusion matrix saved to %s", save_path)
