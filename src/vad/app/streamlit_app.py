from __future__ import annotations

from pathlib import Path

import streamlit as st
import torch

from vad.app.components.offline import render_offline_tab
from vad.app.components.online import render_online_tab
from vad.app.sidebar import sidebar
from vad.config import InferenceConfig, StreamingConfig
from vad.inference.offline import OfflineVADInferencer
from vad.inference.streaming import StreamingVADInferencer

APP_TITLE = "Voice Activity Detection Demo"


@st.cache_resource(show_spinner=False)
def create_inferencer(
    checkpoint_path: str,
    inference_config: InferenceConfig,
    device_str: str = "cpu",
) -> OfflineVADInferencer:
    """Create and cache an offline VAD inferencer."""
    return OfflineVADInferencer(
        checkpoint_path=checkpoint_path,
        device=torch.device(device_str),
        inference_config=inference_config,
    )


@st.cache_resource(show_spinner=False)
def create_streaming_inferencer(
    checkpoint_path: str,
    inference_config: InferenceConfig,
    streaming_config: StreamingConfig,
    device_str: str = "cpu",
) -> StreamingVADInferencer:
    """Create and cache the inferencer used by the online tab."""
    return StreamingVADInferencer.from_checkpoint(
        checkpoint_path=checkpoint_path,
        device=torch.device(device_str),
        inference_config=inference_config,
        streaming_config=streaming_config,
    )


def main() -> None:
    """Entry point for the Streamlit VAD demo."""
    st.set_page_config(page_title=APP_TITLE, layout="wide")
    st.title(APP_TITLE)
    st.write("A simple demo with offline inference and simulated online inference.")

    mode, checkpoint_path, inference_config, streaming_config = sidebar()

    if not Path(checkpoint_path).exists():
        st.error(f"Checkpoint not found: {checkpoint_path}")
        st.stop()

    if mode == "Offline":
        inferencer = create_inferencer(
            checkpoint_path=checkpoint_path,
            inference_config=inference_config,
        )
        render_offline_tab(inferencer, inference_config)
    else:
        inferencer = create_streaming_inferencer(
            checkpoint_path=checkpoint_path,
            inference_config=inference_config,
            streaming_config=streaming_config,
        )
        render_online_tab(
            inferencer,
            inference_config.threshold,
            streaming_config.chunk_seconds,
            streaming_config.window_seconds,
        )


if __name__ == "__main__":
    main()
