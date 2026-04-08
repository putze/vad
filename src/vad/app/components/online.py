from __future__ import annotations

import time

import streamlit as st
import torch

from vad.app.audio.upload import load_audio_from_upload
from vad.app.components.utils import append_chunk_to_state, chunk_waveform
from vad.app.plots import (
    plot_streaming_state,
)
from vad.app.state import StreamingState
from vad.config import AudioConfig
from vad.inference import StreamingVADInferencer
from vad.inference.streaming import StreamingPrediction


def init_streaming_state() -> StreamingState:
    """Initialize empty simulated streaming state."""
    return StreamingState(
        times=[], probabilities=[], predictions=[], waveform_times=[], waveform_values=[]
    )


def run_online_inference(
    inferencer: StreamingVADInferencer,
    chunk: torch.Tensor,
    threshold: float,
) -> StreamingPrediction | None:
    """Run online inference and apply selected threshold if a prediction is available."""
    prediction = inferencer.process_chunk(chunk)
    if prediction is None:
        return None

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

    audio_config = AudioConfig()
    waveform, sample_rate = load_audio_from_upload(uploaded_file, audio_config.sample_rate)
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
            for chunk in chunks:
                chunk_prediction = run_online_inference(inferencer, chunk, threshold)
            if chunk_prediction is None:
                continue

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
