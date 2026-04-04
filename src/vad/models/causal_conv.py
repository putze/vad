from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CausalConv1d(nn.Module):
    """
    1D causal convolution layer.

    Ensures that output at time t depends only on inputs at time <= t.
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        bias: bool = True,
    ) -> None:
        super().__init__()

        self.left_padding = (kernel_size - 1) * dilation

        self.conv: nn.Conv1d = nn.Conv1d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            dilation=dilation,
            padding=0,
            bias=bias,
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor [B, C, T]

        Returns:
            Tensor [B, C_out, T]
        """
        x = F.pad(x, (self.left_padding, 0))
        return cast(Tensor, self.conv(x))


class CausalVAD(nn.Module):
    """
    Fully convolutional causal Voice Activity Detection (VAD) model.

    This model performs per-frame binary classification (speech vs non-speech)
    using only past and current context (causal).

    Input:
        x: Tensor [B, n_mels, T]

    Output:
        logits: Tensor [B, num_classes, T]
    """

    def __init__(
        self,
        n_mels: int = 40,
        hidden_channels: int = 128,
        num_classes: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()

        self.net: nn.Sequential = nn.Sequential(
            CausalConv1d(n_mels, hidden_channels, kernel_size=5),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            CausalConv1d(hidden_channels, hidden_channels, kernel_size=5),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            CausalConv1d(hidden_channels, hidden_channels, kernel_size=5),
            nn.BatchNorm1d(hidden_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv1d(hidden_channels, num_classes, kernel_size=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Args:
            x: Tensor [B, n_mels, T]

        Returns:
            logits: Tensor [B, num_classes, T]
        """
        return cast(Tensor, self.net(x))

    @torch.no_grad()
    def predict(self, x: Tensor) -> Tensor:
        """
        Predict class labels (argmax).

        Args:
            x: Tensor [B, n_mels, T]

        Returns:
            Tensor [B, T] with class indices
        """
        logits = self.forward(x)
        return logits.argmax(dim=1)
