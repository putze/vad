from __future__ import annotations

from pathlib import Path

import streamlit as st
import torch

from vad.app.components.offline import render_offline_tab
from vad.app.components.online import render_online_tab
from vad.app.sidebar import sidebar
from vad.config import AudioConfig
from vad.inference.offline import OfflineVADInferencer
from vad.inference.streaming import StreamingVADInferencer

APP_TITLE = "Voice Activity Detection Demo"


@st.cache_resource(show_spinner=False)
def create_inferencer(
    checkpoint_path: str,
    device_str: str = "cpu",
) -> OfflineVADInferencer:
    """Create and cache a VAD inferencer."""

    device = torch.device(device_str)
    audio_config = AudioConfig()

    return OfflineVADInferencer(
        checkpoint_path=checkpoint_path,
        device=device,
        target_sample_rate=audio_config.sample_rate,
        n_mels=audio_config.n_mels,
        frame_length_ms=audio_config.frame_length_ms,
        frame_shift_ms=audio_config.frame_shift_ms,
    )


@st.cache_resource(show_spinner=False)
def create_streaming_inferencer(
    checkpoint_path: str,
    device_str: str = "cpu",
) -> StreamingVADInferencer:
    """Create and cache the inferencer used by the online tab.

    The current online tab simulates streaming by sending successive chunks
    through the same inferencer.
    """
    audio_config = AudioConfig()

    return StreamingVADInferencer(
        checkpoint_path=checkpoint_path,
        device=torch.device(device_str),
        target_sample_rate=audio_config.sample_rate,
        n_mels=audio_config.n_mels,
        frame_length_ms=audio_config.frame_length_ms,
        frame_shift_ms=audio_config.frame_shift_ms,
    )


def main() -> None:
    """Entry point for the Streamlit VAD demo."""
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.write("A simple demo with offline inference and simulated online inference.")

    mode, checkpoint_path, threshold, chunk_seconds, window_seconds = sidebar()

    if not Path(checkpoint_path).exists():
        st.error(f"Checkpoint not found: {checkpoint_path}")
        st.stop()

    if mode == "Offline":
        inferencer = create_inferencer(checkpoint_path)
        render_offline_tab(inferencer, threshold)
    else:
        inferencer = create_streaming_inferencer(checkpoint_path)
        render_online_tab(inferencer, threshold, chunk_seconds, window_seconds)


if __name__ == "__main__":
    main()
