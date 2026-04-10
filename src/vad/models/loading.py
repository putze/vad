from __future__ import annotations

from pathlib import Path

import torch

from vad.config import AudioConfig, TrainingConfig
from vad.models import CausalVAD


def load_model(
    checkpoint_path: str | Path,
    device: torch.device | str = "cpu",
) -> tuple[CausalVAD, AudioConfig, TrainingConfig]:
    """
    Load a trained ``CausalVAD`` model and its saved configs from a checkpoint.

    The checkpoint may be either:
    - a raw model state dict, or
    - a checkpoint dictionary containing a ``"model_state_dict"`` entry.

    If configuration dictionaries are present in ``extra_state``, they are
    reconstructed and returned. Otherwise default configs are used.

    Args:
        checkpoint_path: Path to the checkpoint file.
        device: Torch device or device string used to place the model.

    Returns:
        Tuple of:
            - loaded ``CausalVAD`` model in evaluation mode
            - ``AudioConfig``
            - ``TrainingConfig``

    Raises:
        FileNotFoundError: If the checkpoint file does not exist.
        RuntimeError: If the checkpoint cannot be loaded or is incompatible
            with the model architecture.
    """
    checkpoint_path = Path(checkpoint_path)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint file not found: {checkpoint_path}")

    device = torch.device(device)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if isinstance(checkpoint, dict):
        extra_state = checkpoint.get("extra_state", {})
        audio_config_dict = extra_state.get("audio_config")
        training_config_dict = extra_state.get("training_config")

        audio_config = (
            AudioConfig(**audio_config_dict) if audio_config_dict is not None else AudioConfig()
        )
        training_config = (
            TrainingConfig(**training_config_dict)
            if training_config_dict is not None
            else TrainingConfig()
        )

        state_dict = checkpoint.get("model_state_dict", checkpoint)
    else:
        audio_config = AudioConfig()
        training_config = TrainingConfig()
        state_dict = checkpoint

    model = CausalVAD(n_mels=audio_config.n_mels)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    return model, audio_config, training_config
