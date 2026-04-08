from __future__ import annotations

from dataclasses import dataclass, field

import torch
import webrtcvad
from torch import Tensor
from vad.data.preprocessing.audio import AudioPreprocessor
from vad.data.utils import ensure_mono_waveform


@dataclass(slots=True)
class BaselinePrediction:
    frame_times: Tensor  # [T]
    probabilities: Tensor  # [T], can be hard 0/1 if baseline has no probas
    predictions: Tensor  # [T], binary
    sample_rate: int


@dataclass(slots=True)
class WebRTCVADBaseline:
    aggressiveness: int = 2
    frame_duration_ms: int = 10
    target_sample_rate: int = 16000
    name: str = "webrtcvad"
    vad: webrtcvad.Vad = field(init=False, repr=False)
    audio_preprocessor: AudioPreprocessor = field(init=False, repr=False)

    def __post_init__(self) -> None:
        if self.frame_duration_ms not in (10, 20, 30):
            raise ValueError("WebRTC VAD only supports 10, 20, or 30 ms frames.")
        if self.aggressiveness not in (0, 1, 2, 3):
            raise ValueError("Aggressiveness must be 0, 1, 2, or 3.")
        self.vad = webrtcvad.Vad(self.aggressiveness)
        if self.target_sample_rate < 0:
            raise ValueError("Target sample rate must be positive.")
        self.audio_preprocessor = AudioPreprocessor(self.target_sample_rate)

    def _prepare_waveform(self, waveform: Tensor, sample_rate: int) -> Tensor:
        waveform = ensure_mono_waveform(waveform).float()
        waveform, sample_rate = self.audio_preprocessor.process_waveform(
            waveform,
            sample_rate,
        )
        # WebRTC expects 16-bit signed PCM [range -32768, 32767]
        pcm = (waveform * 32767.0).to(torch.int16)
        return pcm

    def predict_waveform(self, waveform: Tensor, sample_rate: int) -> BaselinePrediction:
        waveform = self._prepare_waveform(waveform, sample_rate)
        sr = self.target_sample_rate

        frame_len = int(sr * self.frame_duration_ms / 1000)
        total_frames = waveform.numel() // frame_len
        waveform = waveform[: total_frames * frame_len]

        if total_frames == 0:
            empty = torch.empty(0, dtype=torch.float32)
            return BaselinePrediction(
                frame_times=empty,
                probabilities=empty,
                predictions=empty.to(torch.int64),
                sample_rate=sr,
            )

        chunks = waveform.view(total_frames, frame_len)

        preds = []
        for chunk in chunks:
            is_speech = self.vad.is_speech(chunk.numpy().tobytes(), sr)
            preds.append(1 if is_speech else 0)

        predictions = torch.tensor(preds, dtype=torch.int64)
        probabilities = predictions.float()  # WebRTC is hard decision only
        frame_times = torch.arange(total_frames, dtype=torch.float32) * (
            self.frame_duration_ms / 1000.0
        )

        return BaselinePrediction(
            frame_times=frame_times,
            probabilities=probabilities,
            predictions=predictions,
            sample_rate=sr,
        )
