from __future__ import annotations

import streamlit as st

from vad.config import InferenceConfig, StreamingConfig


def sidebar() -> tuple[str, str, float, float, float]:
    """Render sidebar controls."""
    st.sidebar.header("Controls")
    inference_config = InferenceConfig()
    streaming_config = StreamingConfig()

    mode = st.sidebar.radio("Mode", ["Offline", "Online"])
    checkpoint_path = st.sidebar.text_input("Checkpoint", "checkpoints/best_causal_vad.pt")
    threshold = st.sidebar.slider(
        "Speech threshold",
        min_value=0.05,
        max_value=0.95,
        value=inference_config.threshold,
        step=0.05,
    )
    chunk_seconds = st.sidebar.slider(
        "Chunk duration (online)",
        min_value=0.25,
        max_value=2.0,
        value=streaming_config.chunk_seconds,
        step=0.25,
    )
    window_seconds = st.sidebar.slider(
        "Rolling window (online)",
        min_value=3.0,
        max_value=30.0,
        value=streaming_config.rolling_window_seconds,
        step=1.0,
    )
    return mode, checkpoint_path, threshold, chunk_seconds, window_seconds
