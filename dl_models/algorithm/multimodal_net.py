"""Attention fusion across modality embeddings and dense classification head."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from dl_models.algorithm.se_resnet_1d import ModalitySEResNet1D


class AttentionFusion(nn.Module):
    """Modality-level attention fusion.

    Learns a scalar importance score for each modality embedding,
    applies softmax across modalities, and returns the weighted sum.

    Args:
        embed_dim: Dimensionality of each modality embedding.
        num_modalities: Number of modalities (default 3).
    """

    def __init__(self, embed_dim: int = 64, num_modalities: int = 3) -> None:
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.Tanh(),
            nn.Linear(embed_dim // 2, 1),
        )
        self.num_modalities = num_modalities

    def forward(
        self, embeddings: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Fuse modality embeddings.

        Args:
            embeddings: [B, num_modalities, embed_dim].

        Returns:
            fused: [B, embed_dim] weighted combination.
            weights: [B, num_modalities] attention weights (sum to 1).
        """
        scores = self.attention(embeddings)
        weights = F.softmax(scores, dim=1)
        fused = (weights * embeddings).sum(dim=1)
        return fused, weights.squeeze(-1)


class DenseClassifier(nn.Module):
    """Two-layer dense head with dropout.

    Args:
        embed_dim: Input dimensionality (default 64).
        hidden: Hidden layer size (default 32).
        dropout: Dropout probability (default 0.3).
    """

    def __init__(
        self, embed_dim: int = 64, hidden: int = 32, dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return raw logit [B, 1]."""
        return self.net(x)


class MultimodalPDNet(nn.Module):
    """Full graph: three ``ModalitySEResNet1D`` encoders → ``fusion`` → ``classifier``.

    Args:
        speech_features: Number of speech input features (default 22).
        handwriting_features: Number of handwriting features (default 10).
        gait_features: Number of gait features (default 10).
        embed_dim: Shared embedding dimensionality (default 64).
        reduction: SE reduction ratio (default 4).
        dropout: Global dropout probability (default 0.3).
    """

    def __init__(
        self,
        speech_features: int = 22,
        handwriting_features: int = 10,
        gait_features: int = 10,
        embed_dim: int = 64,
        reduction: int = 4,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.speech_encoder = ModalitySEResNet1D(
            speech_features, embed_dim, reduction, dropout,
        )
        self.handwriting_encoder = ModalitySEResNet1D(
            handwriting_features, embed_dim, reduction, dropout,
        )
        self.gait_encoder = ModalitySEResNet1D(
            gait_features, embed_dim, reduction, dropout,
        )

        self.fusion = AttentionFusion(embed_dim, num_modalities=3)
        self.classifier = DenseClassifier(embed_dim, embed_dim // 2, dropout)

    def forward(
        self,
        speech: torch.Tensor,
        handwriting: torch.Tensor,
        gait: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Full forward pass.

        Args:
            speech: [B, 22] speech feature vectors.
            handwriting: [B, 10] handwriting feature vectors.
            gait: [B, 10] gait feature vectors.

        Returns:
            Dictionary with keys:
                logit: [B, 1] raw logit.
                probability: [B, 1] sigmoid probability.
                attention_weights: [B, 3] modality attention weights.
                speech_info: SE weights and last conv from speech encoder.
                handwriting_info: Same for handwriting.
                gait_info: Same for gait.
        """
        s_emb, s_info = self.speech_encoder(speech)
        h_emb, h_info = self.handwriting_encoder(handwriting)
        g_emb, g_info = self.gait_encoder(gait)

        stacked = torch.stack([s_emb, h_emb, g_emb], dim=1)

        fused, attn_weights = self.fusion(stacked)
        logit = self.classifier(fused)
        prob = torch.sigmoid(logit)

        return {
            "logit": logit,
            "probability": prob,
            "attention_weights": attn_weights,
            "speech_info": s_info,
            "handwriting_info": h_info,
            "gait_info": g_info,
        }
