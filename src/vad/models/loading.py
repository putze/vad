from __future__ import annotations

from pathlib import Path

import torch

from vad.models import CausalVAD


def load_model(
    checkpoint_path: str | Path,
    device: torch.device | str = "cpu",
    n_mels: int = 40,
) -> CausalVAD:
    """
    Load a trained ``CausalVAD`` model from a checkpoint.

    The checkpoint may be either:
    - a raw model state dict, or
    - a checkpoint dictionary containing a ``"model_state_dict"`` entry.

    The model is instantiated, its weights are loaded, moved to the requested
    device, and switched to evaluation mode.

    Args:
        checkpoint_path: Path to the checkpoint file.
        device: Torch device or device string used to place the model.
        n_mels: Number of Mel bins expected by the model input.

    Returns:
        Loaded ``CausalVAD`` model in evaluation mode.

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
        ValueError: If ``n_mels`` is not positive.
        RuntimeError: If the checkpoint cannot be loaded or is incompatible
            with the model architecture.
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    if n_mels <= 0:
        raise ValueError(f"`n_mels` must be positive, got {n_mels}")

    device = torch.device(device)
    model = CausalVAD(n_mels=n_mels)

    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    state_dict = (
        checkpoint["model_state_dict"]
        if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint
        else checkpoint
    )

    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model
