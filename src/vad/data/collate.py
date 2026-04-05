from __future__ import annotations

from typing import Sequence

import torch
from torch import Tensor


def pad_collate_fn(
    batch: Sequence[tuple[Tensor, Tensor]],
) -> tuple[Tensor, Tensor, Tensor]:
    """
    Pad a batch of variable-length feature sequences and labels.

    Args:
        batch (Sequence[tuple[Tensor, Tensor]]):
            List of (features, labels) where:
                - features: [n_mels, T]
                - labels:   [T]

    Returns:
        tuple[Tensor, Tensor, Tensor]:
            - x_padded: [B, n_mels, T_max]
            - y_padded: [B, T_max] (padded with -100)
            - lengths:  [B] original sequence lengths
    """
    xs, ys = zip(*batch)

    B = len(xs)
    n_mels = xs[0].shape[0]
    lengths = torch.tensor([x.shape[1] for x in xs], dtype=torch.long)
    T_max = int(lengths.max().item())

    x_padded = torch.zeros(B, n_mels, T_max, dtype=xs[0].dtype)
    y_padded = torch.full((B, T_max), -100, dtype=torch.long)

    for i, (x, y) in enumerate(zip(xs, ys)):
        T = x.shape[1]
        x_padded[i, :, :T] = x
        y_padded[i, :T] = y.long()

    return x_padded, y_padded, lengths
