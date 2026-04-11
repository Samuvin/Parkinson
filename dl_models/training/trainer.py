"""
Training utilities for the MultimodalPDNet.

Provides a ``Trainer`` class that handles:
    - Online data augmentation: Gaussian noise, feature dropout, MixUp
    - Focal Loss for class-imbalance robustness
    - Train / validation loop with early stopping (val loss or val ROC-AUC)
    - LR scheduling: ReduceLROnPlateau, CosineAnnealing, OneCycleLR,
      or cosine_warmup (linear warm-up → cosine decay — recommended)
    - Gradient accumulation for stable updates with small batches
    - Optional mixed precision (CUDA)
    - Validation-set threshold calibration (F1-optimal threshold)
    - Best model checkpointing and metric plots
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Optional

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
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
from torch.utils.data import DataLoader, Dataset

from dl_models.algorithm.networks import MultimodalPDNet


# --------------------------------------------------------------------------- #
#  Focal Loss                                                                   #
# --------------------------------------------------------------------------- #

class FocalBCELoss(nn.Module):
    """Focal loss for binary classification (Lin et al., RetinaNet 2017).

    Reduces the contribution from easy, well-classified examples by scaling
    the standard BCE loss by ``(1 - p_t) ** gamma``, focusing training on
    hard, misclassified examples.

    Args:
        gamma: Focusing parameter (0 = standard BCE, 2 = recommended default).
        pos_weight: Optional scalar weight for the positive class (same
                    semantics as ``nn.BCEWithLogitsLoss(pos_weight=…)``).
    """

    def __init__(
        self,
        gamma: float = 2.0,
        pos_weight: Optional[torch.Tensor] = None,
    ) -> None:
        super().__init__()
        self.gamma = gamma
        if pos_weight is not None:
            self.register_buffer("pos_weight", pos_weight)
        else:
            self.pos_weight: Optional[torch.Tensor] = None

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        pw = getattr(self, "pos_weight", None)
        bce = F.binary_cross_entropy_with_logits(
            logits, targets, pos_weight=pw, reduction="none",
        )
        prob = torch.sigmoid(logits)
        p_t = targets * prob + (1.0 - targets) * (1.0 - prob)
        focal_weight = (1.0 - p_t).pow(self.gamma)
        return (focal_weight * bce).mean()


# --------------------------------------------------------------------------- #
#  Data Augmentation                                                            #
# --------------------------------------------------------------------------- #

def augment_batch(
    batch: dict[str, torch.Tensor],
    noise_std: float = 0.05,
    feature_dropout: float = 0.1,
    mixup_alpha: float = 0.0,
) -> dict[str, torch.Tensor]:
    """Apply Gaussian noise, feature dropout, and optionally MixUp.

    MixUp (Zhang et al., 2018) linearly interpolates two random training
    samples (and their labels) which acts as a strong regulariser for
    small tabular datasets.

    Args:
        batch: Dict with keys ``speech``, ``handwriting``, ``gait``, ``label``.
        noise_std: Std-dev of additive Gaussian noise.
        feature_dropout: Probability of zeroing each feature independently.
        mixup_alpha: Beta-distribution alpha for MixUp (0 disables MixUp).

    Returns:
        The same batch dict with augmented tensors.
    """
    for key in ("speech", "handwriting", "gait"):
        x = batch[key]
        x = x + torch.randn_like(x) * noise_std
        mask = torch.rand_like(x) > feature_dropout
        batch[key] = x * mask

    if mixup_alpha > 0.0:
        B = batch["speech"].shape[0]
        if B > 1:
            lam = float(np.random.beta(mixup_alpha, mixup_alpha))
            perm = torch.randperm(B, device=batch["speech"].device)
            for key in ("speech", "handwriting", "gait", "label"):
                batch[key] = lam * batch[key] + (1.0 - lam) * batch[key][perm]

    return batch


# --------------------------------------------------------------------------- #
#  Trainer                                                                      #
# --------------------------------------------------------------------------- #

class Trainer:
    """Training loop for ``MultimodalPDNet``.

    Args:
        model: An instance of ``MultimodalPDNet``.
        device: ``'cpu'``, ``'cuda'``, or ``'mps'``.
        lr: Initial / max learning rate (default 3e-4).
        weight_decay: AdamW weight decay (default 1e-3).
        patience: Early stopping patience in epochs (default 30).
        noise_std: Augmentation noise std-dev (default 0.05).
        feature_dropout: Augmentation feature dropout probability (default 0.1).
        mixup_alpha: MixUp Beta-distribution alpha; 0 disables MixUp (default 0.2).
        grad_clip_norm: Max gradient norm; 0 disables clipping (default 1.0).
        label_smoothing: Soft-label epsilon in ``[0, 1]`` (default 0.05).
        pos_weight: Positive-class weight for the loss (imbalanced data).
        use_focal_loss: Use Focal BCE instead of standard BCE (default True).
        focal_gamma: Focusing exponent for Focal Loss (default 2.0).
        use_amp: Enable AMP on CUDA (default True).
        num_workers: DataLoader worker count (default 0).
        accumulate_grad_batches: Gradient accumulation steps (default 1).
        lr_scheduler: ``plateau`` | ``cosine`` | ``onecycle`` | ``cosine_warmup``.
        warmup_epochs: Linear warm-up epochs for ``cosine_warmup`` (default 10).
        plateau_factor: LR reduction factor for plateau scheduler (default 0.5).
        plateau_patience: Plateau patience in epochs (default 5).
        onecycle_pct_start: Fraction of steps in OneCycle rising phase (default 0.1).
        early_stop_monitor: ``val_loss`` (minimise) or ``val_roc_auc`` (maximise).
    """

    def __init__(
        self,
        model: MultimodalPDNet,
        device: str = "cpu",
        lr: float = 3e-4,
        weight_decay: float = 1e-3,
        patience: int = 30,
        noise_std: float = 0.05,
        feature_dropout: float = 0.1,
        mixup_alpha: float = 0.2,
        *,
        grad_clip_norm: float = 1.0,
        label_smoothing: float = 0.05,
        pos_weight: Optional[float] = None,
        use_focal_loss: bool = True,
        focal_gamma: float = 2.0,
        use_amp: bool = False,
        num_workers: int = 0,
        accumulate_grad_batches: int = 1,
        lr_scheduler: str = "cosine_warmup",
        warmup_epochs: int = 10,
        plateau_factor: float = 0.5,
        plateau_patience: int = 5,
        onecycle_pct_start: float = 0.1,
        early_stop_monitor: str = "val_roc_auc",
    ) -> None:
        self.model = model.to(device)
        self.device = device
        self.base_lr = lr
        self.patience = patience
        self.noise_std = noise_std
        self.feature_dropout = feature_dropout
        self.mixup_alpha = mixup_alpha
        self.grad_clip_norm = grad_clip_norm
        self.label_smoothing = label_smoothing
        self.accumulate_grad_batches = max(1, int(accumulate_grad_batches))
        self.use_amp = bool(
            use_amp and device == "cuda" and torch.cuda.is_available()
        )
        self.num_workers = max(0, int(num_workers))
        self.lr_scheduler_kind = lr_scheduler.strip().lower()
        self.warmup_epochs = max(1, int(warmup_epochs))
        self.plateau_factor = plateau_factor
        self.plateau_patience = plateau_patience
        self.onecycle_pct_start = onecycle_pct_start

        _valid_schedulers = ("plateau", "cosine", "onecycle", "cosine_warmup")
        if self.lr_scheduler_kind not in _valid_schedulers:
            raise ValueError(
                f"lr_scheduler must be one of {_valid_schedulers}",
            )

        esm = early_stop_monitor.strip().lower()
        if esm not in ("val_loss", "val_roc_auc"):
            raise ValueError(
                "early_stop_monitor must be 'val_loss' or 'val_roc_auc'",
            )
        self.early_stop_monitor = esm

        pw_tensor: Optional[torch.Tensor] = None
        if pos_weight is not None and pos_weight > 0:
            pw_tensor = torch.tensor(
                [float(pos_weight)], device=device, dtype=torch.float32,
            )

        if use_focal_loss:
            self.criterion: nn.Module = FocalBCELoss(
                gamma=focal_gamma, pos_weight=pw_tensor,
            )
        else:
            self.criterion = nn.BCEWithLogitsLoss(pos_weight=pw_tensor)

        self.optimizer = torch.optim.AdamW(
            model.parameters(), lr=lr, weight_decay=weight_decay,
        )

        self._epoch_scheduler: Any = None
        self._onecycle: Any = None
        self._scaler: Optional[GradScaler] = None
        if self.use_amp:
            self._scaler = GradScaler()

        self.threshold: float = 0.5

        # History
        self.train_losses: list[float] = []
        self.val_losses: list[float] = []
        self.val_roc_aucs: list[float] = []
        self.best_val_loss = float("inf")
        self.best_val_roc_auc = float("-inf")
        self._best_monitor_score = float("inf")
        self.epochs_no_improve = 0
        self.best_state: Optional[dict] = None

    # --------------------------------------------------------------------- #
    #  Label smoothing                                                        #
    # --------------------------------------------------------------------- #

    def _smooth_labels(self, labels: torch.Tensor) -> torch.Tensor:
        if self.label_smoothing <= 0.0:
            return labels
        eps = self.label_smoothing
        return labels * (1.0 - eps) + (1.0 - labels) * eps

    # --------------------------------------------------------------------- #
    #  Single epoch                                                           #
    # --------------------------------------------------------------------- #

    def _train_epoch(
        self,
        loader: DataLoader,
        augment: bool = True,
    ) -> float:
        """Run one training epoch with gradient accumulation. Returns avg loss."""
        self.model.train()
        total_loss = 0.0
        n_batches = 0
        accum = self.accumulate_grad_batches
        self.optimizer.zero_grad(set_to_none=True)

        for step, batch in enumerate(loader):
            if augment:
                batch = augment_batch(
                    batch, self.noise_std, self.feature_dropout, self.mixup_alpha,
                )

            speech = batch["speech"].to(self.device)
            handwriting = batch["handwriting"].to(self.device)
            gait = batch["gait"].to(self.device)
            labels = self._smooth_labels(batch["label"].to(self.device))

            if self.use_amp and self._scaler is not None:
                with autocast():
                    out = self.model(speech, handwriting, gait)
                    loss = self.criterion(out["logit"].squeeze(-1), labels) / accum
                self._scaler.scale(loss).backward()
            else:
                out = self.model(speech, handwriting, gait)
                loss = self.criterion(out["logit"].squeeze(-1), labels) / accum
                loss.backward()

            is_last_batch = (step + 1) == len(loader)
            if (step + 1) % accum == 0 or is_last_batch:
                if self.grad_clip_norm > 0:
                    if self._scaler is not None:
                        self._scaler.unscale_(self.optimizer)
                    nn.utils.clip_grad_norm_(
                        self.model.parameters(), max_norm=self.grad_clip_norm,
                    )
                if self._scaler is not None:
                    self._scaler.step(self.optimizer)
                    self._scaler.update()
                else:
                    self.optimizer.step()

                self.optimizer.zero_grad(set_to_none=True)

                if self._onecycle is not None:
                    self._onecycle.step()

            total_loss += loss.item() * accum
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
        y_pred = (y_prob >= self.threshold).astype(int)

        metrics: dict[str, Any] = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "f1": float(f1_score(y_true, y_pred, zero_division=0)),
            "roc_auc": (
                float(roc_auc_score(y_true, y_prob))
                if len(np.unique(y_true)) > 1
                else 0.0
            ),
            "y_true": y_true,
            "y_prob": y_prob,
        }
        return avg_loss, metrics

    # --------------------------------------------------------------------- #
    #  Schedulers                                                             #
    # --------------------------------------------------------------------- #

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
        elif kind == "cosine_warmup":
            # Linear warm-up for `warmup_epochs`, then cosine decay to 1 % of base_lr.
            # SequentialLR chains two schedulers at a milestone epoch.
            w = min(self.warmup_epochs, epochs_i - 1)
            warmup_sched = torch.optim.lr_scheduler.LinearLR(
                self.optimizer,
                start_factor=1e-3,
                end_factor=1.0,
                total_iters=w,
            )
            cosine_sched = torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=max(epochs_i - w, 1),
                eta_min=max(self.base_lr * 0.01, 1e-7),
            )
            self._epoch_scheduler = torch.optim.lr_scheduler.SequentialLR(
                self.optimizer,
                schedulers=[warmup_sched, cosine_sched],
                milestones=[w],
            )

    # --------------------------------------------------------------------- #
    #  Early stopping helpers                                                 #
    # --------------------------------------------------------------------- #

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

    def _dataloader_kwargs(self) -> dict[str, Any]:
        return {
            "num_workers": self.num_workers,
            "pin_memory": self.device == "cuda",
        }

    # --------------------------------------------------------------------- #
    #  Threshold calibration                                                  #
    # --------------------------------------------------------------------- #

    def calibrate_threshold(
        self, val_dataset: Dataset, batch_size: int = 32,
    ) -> float:
        """Find the F1-maximising classification threshold on the validation set.

        Scans 81 candidate thresholds in [0.10, 0.90] and picks the one that
        yields the highest macro-F1.  Updates ``self.threshold`` in place.

        Returns:
            The optimal threshold (also stored as ``self.threshold``).
        """
        loader = DataLoader(
            val_dataset, batch_size=batch_size, shuffle=False,
            **self._dataloader_kwargs(),
        )
        _, metrics = self._eval_epoch(loader)
        y_true = metrics["y_true"]
        y_prob = metrics["y_prob"]

        best_t, best_f1 = 0.5, 0.0
        for t in np.linspace(0.10, 0.90, 81):
            preds = (y_prob >= t).astype(int)
            f1 = f1_score(y_true, preds, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, float(t)

        self.threshold = best_t
        return best_t

    # --------------------------------------------------------------------- #
    #  Full training loop                                                     #
    # --------------------------------------------------------------------- #

    def fit(
        self,
        train_dataset: Dataset,
        val_dataset: Dataset,
        epochs: int = 300,
        batch_size: int = 32,
        augment: bool = True,
    ) -> dict[str, Any]:
        """Train the model with early stopping and threshold calibration.

        Args:
            train_dataset: Training dataset (``MultimodalPDDataset`` or
                           ``ModalityMatchedDataset``).
            val_dataset: Validation dataset (same interface).
            epochs: Maximum number of epochs.
            batch_size: Mini-batch size.
            augment: Enable online data augmentation.

        Returns:
            Dict with training history and best validation metrics.
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
        self._best_monitor_score = (
            float("-inf") if self.early_stop_monitor == "val_roc_auc" else float("inf")
        )
        self.epochs_no_improve = 0
        self.best_state = None

        start_time = time.time()

        print(f"\n  Device    : {self.device}")
        print(f"  Train     : {len(train_dataset)} samples  |  Val: {len(val_dataset)} samples")
        print(f"  Epochs    : up to {epochs}  |  Patience: {self.patience}")
        print(f"  Scheduler : {self.lr_scheduler_kind}"
              + (f"  (warmup={self.warmup_epochs})" if self.lr_scheduler_kind == "cosine_warmup" else ""))
        print(f"  Loss      : {'Focal BCE' if isinstance(self.criterion, FocalBCELoss) else 'BCE'}"
              + f"  |  MixUp α={self.mixup_alpha}  |  GradAccum={self.accumulate_grad_batches}")
        print(f"\n  {'Epoch':>6}  {'Train Loss':>10}  {'Val Loss':>9}  "
              f"{'Val AUC':>8}  {'LR':>9}  Status")
        print(f"  {'-'*6}  {'-'*10}  {'-'*9}  {'-'*8}  {'-'*9}  {'-'*18}")

        for epoch in range(1, epochs + 1):
            train_loss = self._train_epoch(train_loader, augment=augment)
            val_loss, val_metrics = self._eval_epoch(val_loader)

            self.train_losses.append(train_loss)
            self.val_losses.append(val_loss)
            auc = val_metrics["roc_auc"]
            self.val_roc_aucs.append(auc)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
            if auc > self.best_val_roc_auc:
                self.best_val_roc_auc = auc

            if self._epoch_scheduler is not None:
                if self.lr_scheduler_kind == "plateau":
                    self._epoch_scheduler.step(val_loss)
                else:
                    self._epoch_scheduler.step()

            current_lr = self.optimizer.param_groups[0]["lr"]

            improved = self._early_stop_improved(val_loss, auc)
            if improved:
                self._early_stop_update_best(val_loss, auc)
                self.epochs_no_improve = 0
                self.best_state = {
                    k: v.cpu().clone()
                    for k, v in self.model.state_dict().items()
                }
                status = "✓ best"
            else:
                self.epochs_no_improve += 1
                status = f"no-improve {self.epochs_no_improve}/{self.patience}"

            print(
                f"  {epoch:>6}  {train_loss:>10.4f}  {val_loss:>9.4f}  "
                f"{auc:>8.4f}  {current_lr:>9.2e}  {status}",
                flush=True,
            )

            if self.epochs_no_improve >= self.patience:
                print(f"\n  Early stopping at epoch {epoch} "
                      f"(no improvement for {self.patience} epochs)")
                break

        elapsed = time.time() - start_time

        if self.best_state is not None:
            self.model.load_state_dict(self.best_state)

        # ---- threshold calibration on val set ----------------------------- #
        best_t = self.calibrate_threshold(val_dataset, batch_size=batch_size)
        print(f"\n  Threshold calibration (F1-optimal on val): {best_t:.3f}")

        return {
            "train_losses": self.train_losses,
            "val_losses": self.val_losses,
            "val_roc_aucs": self.val_roc_aucs,
            "best_val_loss": self.best_val_loss,
            "best_val_roc_auc": self.best_val_roc_auc,
            "optimal_threshold": best_t,
            "total_epochs": len(self.train_losses),
            "elapsed_seconds": elapsed,
        }

    # --------------------------------------------------------------------- #
    #  Evaluation                                                             #
    # --------------------------------------------------------------------- #

    def evaluate(
        self,
        test_dataset: Dataset,
        batch_size: int = 32,
        threshold: Optional[float] = None,
    ) -> dict[str, Any]:
        """Evaluate on the test set using the calibrated threshold.

        Args:
            test_dataset: Test dataset.
            batch_size: Batch size for inference.
            threshold: Override threshold (default: ``self.threshold``).

        Returns:
            Dict with metrics, predictions, probabilities, and labels.
        """
        saved_t = self.threshold
        if threshold is not None:
            self.threshold = threshold

        loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            **self._dataloader_kwargs(),
        )
        loss, metrics = self._eval_epoch(loader)
        metrics["test_loss"] = loss
        metrics["threshold_used"] = self.threshold

        # Re-compute preds using calibrated threshold (already done in _eval_epoch)
        metrics["y_pred"] = (metrics["y_prob"] >= self.threshold).astype(int)

        self.threshold = saved_t
        return metrics

    # --------------------------------------------------------------------- #
    #  Saving                                                                 #
    # --------------------------------------------------------------------- #

    def save_model(self, path: str | Path) -> None:
        """Save model state dict to *path*."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(self.model.state_dict(), path)

    def save_metrics(
        self, metrics: dict[str, Any], path: str | Path,
    ) -> None:
        """Save JSON-serialisable metrics to *path*."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

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

    # --------------------------------------------------------------------- #
    #  Plotting                                                               #
    # --------------------------------------------------------------------- #

    def plot_training_curves(self, save_path: str | Path) -> None:
        """Save train/val loss curves and validation ROC-AUC."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        fig, ax1 = plt.subplots(figsize=(8, 5))
        ax1.plot(self.train_losses, label="Train Loss", color="tab:blue")
        ax1.plot(self.val_losses, label="Val Loss", color="tab:orange")
        ax1.set_xlabel("Epoch")
        ax1.set_ylabel("Loss")
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
