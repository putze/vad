from __future__ import annotations

import tempfile
from pathlib import Path

import torch
import torchaudio


def load_audio_from_upload(uploaded_file) -> tuple[torch.Tensor, int]:
    """Read an uploaded audio file with torchaudio."""
    suffix = Path(uploaded_file.name).suffix or ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    waveform, sample_rate = torchaudio.load(tmp_path)
    return waveform, sample_rate
