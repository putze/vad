from __future__ import annotations

import streamlit as st

from vad.config import InferenceConfig, StreamingConfig


def sidebar() -> tuple[str, str, InferenceConfig, StreamingConfig]:
    """Render sidebar controls."""
    st.sidebar.header("⚙️ VAD Controls")

    inference_config = InferenceConfig()
    streaming_config = StreamingConfig()

    # --- Mode ---
    mode = st.sidebar.radio(
        "Inference mode",
        ["Offline", "Streaming"],
        help="Offline processes a full file at once. Streaming simulates real-time audio.",
    )

    st.sidebar.divider()

    # --- Model ---
    st.sidebar.subheader("Model")
    checkpoint_path = st.sidebar.text_input(
        "Checkpoint path",
        "checkpoints/causal_vad.pt",
        help="Path to the trained VAD model checkpoint.",
    )

    inference_config.threshold = st.sidebar.slider(
        "Speech detection threshold",
        min_value=0.05,
        max_value=0.95,
        value=float(inference_config.threshold),
        step=0.05,
        help=(
            "Probability threshold for speech detection.\n"
            "Lower = more sensitive (more false positives).\n"
            "Higher = stricter (more false negatives)."
        ),
    )

    # --- Streaming settings ---
    if mode == "Streaming":
        st.sidebar.divider()
        st.sidebar.subheader("Streaming settings")

        streaming_config.chunk_seconds = st.sidebar.slider(
            "Chunk size (s)",
            min_value=0.1,
            max_value=2.0,
            value=float(streaming_config.chunk_seconds),
            step=0.1,
            help=(
                "Size of incoming audio chunks.\n"
                "Smaller = faster updates, higher overhead.\n"
                "Larger = smoother but slower updates."
            ),
        )

        streaming_config.min_buffer_seconds = st.sidebar.slider(
            "Minimum buffer (s)",
            min_value=1.0,
            max_value=10.0,
            value=float(streaming_config.min_buffer_seconds),
            step=0.5,
            help=(
                "Minimum audio required before inference runs.\n"
                "Lower = lower latency.\n"
                "Higher = more stable predictions."
            ),
        )

    return mode, checkpoint_path, inference_config, streaming_config
