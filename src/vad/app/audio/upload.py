from __future__ import annotations

import tempfile
from pathlib import Path

import torch

from vad.data.file_utils import load_audio
from vad.data.preprocessing.audio import AudioPreprocessor


def load_audio_from_upload(uploaded_file, target_sample_rate=None) -> tuple[torch.Tensor, int]:
    """Read an uploaded audio file with torchaudio."""
    suffix = Path(uploaded_file.name).suffix or ".wav"

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name

    waveform, sample_rate = load_audio(tmp_path)
    preprocessor = AudioPreprocessor(target_sample_rate)
    waveform, sample_rate = preprocessor.process_waveform(waveform, target_sample_rate)
    return waveform, sample_rate
