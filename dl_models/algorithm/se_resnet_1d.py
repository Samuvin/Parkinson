"""
1D CNN stem + SE-ResNet encoder for tabular features treated as a length-L sequence.

Per modality (e.g. speech 22-D, handwriting 10-D, gait 10-D)::

    ModalitySEResNet1D
        1) **1D CNN stem:** Conv1d(1→32, k=3) + BatchNorm + ReLU
        2) **ResidualSEBlock1D:** two Conv1d-BN stacks + squeeze–excitation (ResNet-style)
        3) **ResidualSEBlock1D:** channels 32→64 + SE
        4) AdaptiveAvgPool1d(1) + Dropout + Linear → embedding (default 64-D)

``ResidualSEBlock1D`` follows He et al. ResNet (CVPR 2016); ``SEBlock1D`` follows
Hu et al. Squeeze-and-Excitation (CVPR 2018), adapted to 1D.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class SEBlock1D(nn.Module):
    """Squeeze-and-Excitation channel attention for 1-D feature maps.

    Squeezes global spatial information into a channel descriptor via
    adaptive average pooling, then excites (re-weights) channels through
    a two-layer bottleneck with sigmoid gating.

    Args:
        channels: Number of input channels.
        reduction: Bottleneck reduction ratio (default 4).
    """

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        mid = max(channels // reduction, 1)
        self.squeeze = nn.AdaptiveAvgPool1d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, mid),
            nn.ReLU(inplace=True),
            nn.Linear(mid, channels),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (re-weighted x, channel weights [B, C])."""
        b, c, _ = x.size()
        se = self.squeeze(x).view(b, c)
        weights = self.excitation(se)
        out = x * weights.unsqueeze(-1)
        return out, weights


class ResidualSEBlock1D(nn.Module):
    """ResNet residual block with SE channel attention for 1-D data.

    Conv1d -> BN -> ReLU -> Dropout -> Conv1d -> BN -> SE -> + skip -> ReLU

    A 1x1 convolution is used on the skip path when in_channels
    differs from out_channels.

    Args:
        in_channels: Input channel count.
        out_channels: Output channel count.
        reduction: SE bottleneck ratio (default 4).
        dropout: Dropout probability (default 0.3).
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        reduction: int = 4,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()

        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size=3, padding=1, bias=False,
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size=3, padding=1, bias=False,
        )
        self.bn2 = nn.BatchNorm1d(out_channels)

        self.se = SEBlock1D(out_channels, reduction)

        if in_channels != out_channels:
            self.skip = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=False),
                nn.BatchNorm1d(out_channels),
            )
        else:
            self.skip = nn.Identity()

    def forward(
        self, x: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (block output, SE channel weights)."""
        identity = self.skip(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = F.relu(out, inplace=True)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out, se_weights = self.se(out)
        out = out + identity
        out = F.relu(out, inplace=True)
        return out, se_weights


class ModalitySEResNet1D(nn.Module):
    """SE-ResNet1D encoder for a single data modality.

    Architecture::

        Conv1d(1->32) + BN + ReLU
        ResidualSEBlock1D(32->32)
        ResidualSEBlock1D(32->64)
        AdaptiveAvgPool1d(1)
        Dropout -> Linear(64->embed_dim)

    Args:
        num_features: Length of the input feature vector (e.g. 22 for speech).
        embed_dim: Embedding dimensionality (default 64).
        reduction: SE reduction ratio (default 4).
        dropout: Dropout probability (default 0.3).
    """

    def __init__(
        self,
        num_features: int,
        embed_dim: int = 64,
        reduction: int = 4,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        self.num_features = num_features

        self.initial = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm1d(32),
            nn.ReLU(inplace=True),
        )

        self.block1 = ResidualSEBlock1D(32, 32, reduction, dropout)
        self.block2 = ResidualSEBlock1D(32, 64, reduction, dropout)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(64, embed_dim)

    def forward(
        self, x: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Forward pass.

        Args:
            x: Feature vector [B, num_features] or [B, 1, num_features].

        Returns:
            embedding: [B, embed_dim] embedding vector.
            info: Dict with ``se_weights_1``, ``se_weights_2``, ``last_conv_output``.
        """
        if x.dim() == 2:
            x = x.unsqueeze(1)

        out = self.initial(x)

        out, se_w1 = self.block1(out)
        out, se_w2 = self.block2(out)

        last_conv = out

        out = self.gap(out).squeeze(-1)
        out = self.dropout(out)
        embedding = self.fc(out)

        info = {
            "se_weights_1": se_w1,
            "se_weights_2": se_w2,
            "last_conv_output": last_conv,
        }
        return embedding, info
