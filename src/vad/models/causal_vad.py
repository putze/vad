from __future__ import annotations

from typing import cast

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor


class CausalConv1d(nn.Module):
    """
    One-dimensional causal convolution.

    This layer applies left padding so that the output at time step ``t``
    depends only on input time steps ``<= t``. No future context is used.

    Input shape:
        ``[batch_size, in_channels, num_frames]``

    Output shape:
        ``[batch_size, out_channels, num_frames]``
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        bias: bool = True,
    ) -> None:
        """
        Initialize the causal convolution layer.

        Args:
            in_channels: Number of input channels.
            out_channels: Number of output channels.
            kernel_size: Convolution kernel size.
            dilation: Dilation factor.
            bias: Whether to include a learnable bias term.

        Raises:
            ValueError: If any integer hyperparameter is not positive.
        """
        super().__init__()

        if in_channels <= 0:
            raise ValueError(f"`in_channels` must be positive, got {in_channels}")
        if out_channels <= 0:
            raise ValueError(f"`out_channels` must be positive, got {out_channels}")
        if kernel_size <= 0:
            raise ValueError(f"`kernel_size` must be positive, got {kernel_size}")
        if dilation <= 0:
            raise ValueError(f"`dilation` must be positive, got {dilation}")

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
        Apply causal convolution.

        Args:
            x: Input tensor of shape ``[batch_size, in_channels, num_frames]``.

        Returns:
            Output tensor of shape ``[batch_size, out_channels, num_frames]``.
        """
        if x.ndim != 3:
            raise ValueError(f"Expected input with shape [B, C, T], got {tuple(x.shape)}")

        x = F.pad(x, (self.left_padding, 0))
        return cast(Tensor, self.conv(x))


class CausalVAD(nn.Module):
    """
    Fully convolutional causal voice activity detection model.

    The model takes frame-level acoustic features of shape
    ``[batch_size, n_mels, num_frames]`` and produces one logit per frame.

    It uses stacked causal convolutions so each prediction can depend only on
    the current frame and a finite amount of past context.

    Forward output:
        ``[batch_size, 1, num_frames]``
    """

    def __init__(
        self,
        n_mels: int = 40,
        hidden_channels: int = 128,
        dropout: float = 0.1,
    ) -> None:
        """
        Initialize the causal VAD model.

        Args:
            n_mels: Number of input Mel-frequency channels.
            hidden_channels: Number of channels in the hidden convolution layers.
            dropout: Dropout probability applied after each hidden block.

        Raises:
            ValueError: If model hyperparameters are invalid.
        """
        super().__init__()

        if n_mels <= 0:
            raise ValueError(f"`n_mels` must be positive, got {n_mels}")
        if hidden_channels <= 0:
            raise ValueError(f"`hidden_channels` must be positive, got {hidden_channels}")
        if not 0.0 <= dropout < 1.0:
            raise ValueError(f"`dropout` must be in [0, 1), got {dropout}")

        self.net: nn.Sequential = nn.Sequential(
            self._block(n_mels, hidden_channels, dropout),
            self._block(hidden_channels, hidden_channels, dropout),
            self._block(hidden_channels, hidden_channels, dropout),
            nn.Conv1d(hidden_channels, 1, kernel_size=1),
        )

    @staticmethod
    def _block(in_ch: int, out_ch: int, dropout: float) -> nn.Sequential:
        """
        Build one hidden causal convolution block.

        Args:
            in_ch: Number of input channels.
            out_ch: Number of output channels.
            dropout: Dropout probability.

        Returns:
            Sequential block consisting of causal convolution, group
            normalization, ReLU, and dropout.
        """
        return nn.Sequential(
            CausalConv1d(in_ch, out_ch, kernel_size=5),
            nn.GroupNorm(1, out_ch),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        """
        Compute frame-level logits.

        Args:
            x: Input feature tensor of shape
                ``[batch_size, n_mels, num_frames]``.

        Returns:
            Logit tensor of shape ``[batch_size, 1, num_frames]``.
        """
        if x.ndim != 3:
            raise ValueError(f"Expected input with shape [B, n_mels, T], got {tuple(x.shape)}")

        return cast(Tensor, self.net(x))

    @torch.no_grad()
    def predict_proba(self, x: Tensor) -> Tensor:
        """
        Predict speech probabilities for each frame.

        Args:
            x: Input feature tensor of shape
                ``[batch_size, n_mels, num_frames]``.

        Returns:
            Probability tensor of shape ``[batch_size, num_frames]`` with values
            in ``[0, 1]``.
        """
        logits = self.forward(x)[:, 0, :]
        return torch.sigmoid(logits)

    @torch.no_grad()
    def predict(self, x: Tensor, threshold: float = 0.5) -> Tensor:
        """
        Predict binary speech decisions for each frame.

        Args:
            x: Input feature tensor of shape
                ``[batch_size, n_mels, num_frames]``.
            threshold: Probability threshold used to convert probabilities into
                binary predictions.

        Returns:
            Tensor of shape ``[batch_size, num_frames]`` containing binary
            predictions in ``{0, 1}``.

        Raises:
            ValueError: If ``threshold`` is outside ``[0, 1]``.
        """
        if not 0.0 <= threshold <= 1.0:
            raise ValueError(f"`threshold` must be in [0, 1], got {threshold}")

        probs = self.predict_proba(x)
        return (probs >= threshold).long()
