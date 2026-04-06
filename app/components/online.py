from __future__ import annotations

import time

import streamlit as st
import torch

from app.audio.io import load_audio_from_upload
from app.components.utils import append_chunk_to_state, chunk_waveform, maybe_resample
from app.plots import (
    plot_streaming_state,
)
from app.state import StreamingState
from src.vad.config import AudioConfig
from src.vad.inference import StreamingVADInferencer
from src.vad.inference.streaming import StreamingPrediction
from src.vad.inference.utils import ensure_mono_waveform


def init_streaming_state() -> StreamingState:
    """Initialize empty simulated streaming state."""
    return StreamingState(
        times=[], probabilities=[], predictions=[], waveform_times=[], waveform_values=[]
    )


def run_online_inference(
    inferencer: StreamingVADInferencer,
    waveform: torch.Tensor,
    sample_rate: int,
    threshold: float,
) -> StreamingPrediction:
    """Run offline inference and apply selected threshold if needed."""
    prediction = inferencer.predict_waveform(waveform, sample_rate)

    # If your current inferencer already thresholds internally, remove this block.
    prediction.predictions = (prediction.probabilities >= threshold).to(torch.int64)
    return prediction


def render_online_tab(
    inferencer: StreamingVADInferencer,
    threshold: float,
    chunk_seconds: float,
    window_seconds: float,
) -> None:
    """Render simulated streaming workflow.

    This version streams an uploaded file chunk by chunk. You can later replace
    the input source with a microphone component.
    """
    st.subheader("Online mode")
    st.write("Simulated streaming: the uploaded file is processed chunk by chunk.")

    uploaded_file = st.file_uploader(
        "Upload audio for simulated streaming", type=["wav", "flac", "mp3"], key="online_uploader"
    )
    if uploaded_file is None:
        st.info("Later you can swap this for a microphone recorder component.")
        return

    audio_config = AudioConfig()
    waveform, sample_rate = load_audio_from_upload(uploaded_file)
    waveform = ensure_mono_waveform(waveform)
    waveform, sample_rate = maybe_resample(waveform, sample_rate, audio_config.sample_rate)
    chunks = chunk_waveform(waveform, sample_rate, chunk_seconds)

    if "streaming_state" not in st.session_state:
        st.session_state.streaming_state = init_streaming_state()

    col1, col2 = st.columns(2)
    start_clicked = col1.button("Start online simulation")
    reset_clicked = col2.button("Reset online state")

    if reset_clicked:
        st.session_state.streaming_state = init_streaming_state()
        st.rerun()

    placeholder = st.empty()
    status_placeholder = st.empty()

    if start_clicked:
        st.session_state.streaming_state = init_streaming_state()

        for chunk in chunks:
            chunk_prediction = run_online_inference(inferencer, chunk, sample_rate, threshold)
            append_chunk_to_state(
                st.session_state.streaming_state,
                chunk,
                sample_rate,
                chunk_prediction,
                max_window_seconds=window_seconds,
            )

            current_pred = (
                st.session_state.streaming_state.predictions[-1]
                if st.session_state.streaming_state.predictions
                else 0
            )
            current_prob = (
                st.session_state.streaming_state.probabilities[-1]
                if st.session_state.streaming_state.probabilities
                else 0.0
            )

            status_placeholder.metric(
                "Current decision",
                "Speech" if current_pred == 1 else "Non-speech",
                delta=f"p={current_prob:.2f}",
            )
            placeholder.plotly_chart(
                plot_streaming_state(st.session_state.streaming_state), width="stretch"
            )
            time.sleep(chunk_seconds)

    elif st.session_state.streaming_state.times:
        current_pred = st.session_state.streaming_state.predictions[-1]
        current_prob = st.session_state.streaming_state.probabilities[-1]
        status_placeholder.metric(
            "Current decision",
            "Speech" if current_pred == 1 else "Non-speech",
            delta=f"p={current_prob:.2f}",
        )
        placeholder.plotly_chart(
            plot_streaming_state(st.session_state.streaming_state), width="stretch"
        )

    st.caption(
        "For a real microphone path, replace the uploaded-file source with a "
        "Streamlit microphone component and feed audio chunks into the same state-update logic."
    )
