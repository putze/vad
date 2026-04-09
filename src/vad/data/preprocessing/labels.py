from __future__ import annotations

import torch.nn.functional as F
from torch import Tensor


class LabelAligner:
    """
    Align sample-level labels to the feature frame grid.

    Each output frame corresponds to a sliding window over the sample-level
    labels. A frame is labeled as speech if any sample inside that window is
    labeled as speech.

    This implements a max-pooling alignment rule:

    - frame label = 1 if any label in the frame window is 1,
    - frame label = 0 otherwise.

    When ``center=True``, symmetric zero-padding is applied before windowing to
    mirror centered STFT-style framing. The final output is trimmed or padded to
    match the requested ``num_frames`` exactly.
    """

    def __init__(
        self,
        hop_length: int,
        frame_length: int,
        center: bool = True,
    ) -> None:
        """
        Initialize the label aligner.

        Args:
            hop_length: Number of samples between consecutive frame starts.
            frame_length: Number of samples in each frame window.
            center: Whether to apply symmetric padding before frame extraction
                to mimic centered feature framing.

        Raises:
            ValueError: If ``hop_length`` or ``frame_length`` is not positive.
        """
        if hop_length <= 0:
            raise ValueError(f"`hop_length` must be positive, got {hop_length}")
        if frame_length <= 0:
            raise ValueError(f"`frame_length` must be positive, got {frame_length}")

        self.hop_length = hop_length
        self.frame_length = frame_length
        self.center = center

    def __call__(self, labels: Tensor, num_frames: int) -> Tensor:
        """
        Convert sample-level labels to frame-level labels.

        Args:
            labels: One-dimensional tensor of shape ``[num_samples]`` containing
                binary sample-level labels.
            num_frames: Expected number of output frames, typically taken from
                the feature extractor output.

        Returns:
            Frame-level label tensor of shape ``[num_frames]``.

        Raises:
            ValueError: If ``labels`` is not one-dimensional or if
                ``num_frames`` is not positive.
        """
        if labels.ndim != 1:
            raise ValueError(f"Expected 1D labels [num_samples], got shape {tuple(labels.shape)}")

        if num_frames <= 0:
            raise ValueError(f"`num_frames` must be positive, got {num_frames}")

        labels = labels.float()

        if self.center:
            pad = self.frame_length // 2
            labels = F.pad(labels, (pad, pad), mode="constant", value=0.0)

        if labels.shape[0] < self.frame_length:
            extra = self.frame_length - labels.shape[0]
            labels = F.pad(labels, (0, extra), mode="constant", value=0.0)

        windows = labels.unfold(
            dimension=0,
            size=self.frame_length,
            step=self.hop_length,
        )

        frame_labels = windows.max(dim=1).values
        current_num_frames = frame_labels.shape[0]

        if current_num_frames > num_frames:
            frame_labels = frame_labels[:num_frames]
        elif current_num_frames < num_frames:
            pad_amount = num_frames - current_num_frames
            frame_labels = F.pad(
                frame_labels,
                (0, pad_amount),
                mode="constant",
                value=0.0,
            )

        return frame_labels
