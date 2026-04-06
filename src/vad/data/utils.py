from __future__ import annotations

from torch import Tensor


def ensure_mono_waveform(waveform: Tensor) -> Tensor:
    """
    Convert a waveform to mono shape ``[N]``.

    Args:
        waveform: Waveform of shape ``[N]`` or ``[C, N]``.

    Returns:
        Mono waveform of shape ``[N]``.

    Raises:
        ValueError: If the waveform shape is unsupported.
    """
    if waveform.ndim == 2:
        waveform = waveform.squeeze(0) if waveform.shape[0] == 1 else waveform.mean(dim=0)

    if waveform.ndim != 1:
        raise ValueError(
            f"Expected waveform with shape [N] or [C, N], got {tuple(waveform.shape)}."
        )

    return waveform
