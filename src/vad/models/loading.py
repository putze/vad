from __future__ import annotations

from pathlib import Path

import torch

from src.vad.models import CausalVAD


def load_model(
    checkpoint_path: str | Path,
    device: torch.device,
    n_mels: int,
) -> CausalVAD:
    """
    Load a trained :class:`CausalVAD` checkpoint.

    Supports a raw state dict or a checkpoint containing
    ``model_state_dict``.

    Args:
        checkpoint_path: Path to the checkpoint file.
        device: Torch device used for inference.
        n_mels: Number of mel bins expected by the model.

    Returns:
        Loaded model in evaluation mode.
    """
    model = CausalVAD(n_mels=n_mels)

    checkpoint = torch.load(checkpoint_path, map_location=device)
    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    return model
