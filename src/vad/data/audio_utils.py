from torch import Tensor


def ensure_mono_waveform(waveform: Tensor) -> Tensor:
    """
    Ensure a waveform is mono with shape ``[num_samples]``.

    If the input has multiple channels, they are averaged.

    Args:
        waveform: Tensor of shape ``[num_samples]`` or ``[channels, num_samples]``.

    Returns:
        Mono waveform of shape ``[num_samples]``.

    Raises:
        ValueError: If the input shape is unsupported.
    """
    if waveform.ndim == 2:
        # [C, N] → average channels
        waveform = waveform.mean(dim=0)

    if waveform.ndim != 1:
        raise ValueError(
            f"Expected waveform with shape [N] or [C, N], got {tuple(waveform.shape)}."
        )

    return waveform
