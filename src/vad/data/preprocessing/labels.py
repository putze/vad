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

        # Create sliding windows: shape → [num_windows, frame_length]
        windows = labels.unfold(
            dimension=0,
            size=self.frame_length,
            step=self.hop_length,
        )

        # Handle case where unfold gives more frames than expected
        windows = windows[:num_frames]

        # Max over each frame window → [num_frames]
        frame_labels = windows.max(dim=1).values

        return frame_labels
