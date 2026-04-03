import torch
import torch.nn.functional as F
from torch import Tensor


class LabelAligner:
    """
    Align sample-level labels to frame-level labels using max-pooling logic.

    A frame is labeled as speech (1) if any sample within its window is speech,
    otherwise non-speech (0).
    """

    def __init__(
        self,
        hop_length: int,
        frame_length: int,
        center: bool = True,
    ) -> None:
        """
        Args:
            hop_length (int): Number of samples between consecutive frames.
            frame_length (int): Number of samples per frame window.
            center (bool): Whether frames are centered (applies symmetric padding).
        """
        self.hop_length = hop_length
        self.frame_length = frame_length
        self.center = center

    def __call__(self, labels: Tensor, num_frames: int) -> Tensor:
        """
        Convert sample-level labels to frame-level labels.

        Args:
            labels (Tensor): 1D tensor [T] with binary labels (0 or 1).
            num_frames (int): Number of output frames.

        Returns:
            Tensor: Frame-level labels [num_frames].

        Raises:
            ValueError: If input labels are not 1D.
        """
        if labels.ndim != 1:
            raise ValueError(f"Expected 1D labels [T], got shape {tuple(labels.shape)}")

        labels = labels.float()

        if self.center:
            pad = self.frame_length // 2
            labels = F.pad(labels, (pad, pad), mode="constant", value=0.0)

        frame_labels = torch.zeros(num_frames, dtype=torch.float32)

        for frame_idx in range(num_frames):
            start = frame_idx * self.hop_length
            end = start + self.frame_length

            chunk = labels[start:end]
            if chunk.numel() == 0:
                continue

            frame_labels[frame_idx] = 1.0 if chunk.max() > 0 else 0.0

        return frame_labels
