Design and Approach
===================

Problem
-------

The goal is to build a **frame-level Voice Activity Detection (VAD)** system
that classifies speech vs. non-speech.

Constraints:

- Frame resolution: 10 ms
- Binary classification
- Streaming-compatible inference
- Low latency
- CPU-friendly


System Overview
---------------

The system follows a standard pipeline:

.. code-block:: text

   Audio → Preprocessing → Features → Model → Predictions

- Input: raw waveform
- Output: frame-level speech probabilities and binary decisions


Project Structure
-----------------

.. code-block:: text

   vad/
   ├── data/        # Dataset loading and preprocessing
   ├── models/      # Model definitions
   ├── training/    # Training loop and metrics
   ├── inference/   # Offline and streaming inference
   ├── app/         # Streamlit demo
   ├── cli/         # Command-line interfaces

Each module is designed to be independent:

- ``data``: prepares inputs and labels
- ``models``: defines the neural architecture
- ``training``: handles optimization and logging
- ``inference``: runs trained models
- ``app``: provides visualization


Design Choices
--------------

**Feature-based approach**

- Log-mel spectrograms are used as model input
- Reduces input dimensionality
- Avoids raw waveform modeling complexity

**Causal modeling**

- Model uses only past context
- Enables streaming inference
- No look-ahead latency

**Frame-level prediction**

- Direct mapping from features to speech probabilities
- Keeps the system simple and interpretable


Streaming Inference
-------------------

The system supports **streaming (online) inference**, where audio is processed
incrementally.


Processing Strategy
~~~~~~~~~~~~~~~~~~~

- Audio is split into fixed-size chunks
- Each chunk is converted to features
- Features are appended to a buffer
- The model is applied once enough context is available

A minimum buffer is required to ensure valid feature extraction.


Causality Requirement
~~~~~~~~~~~~~~~~~~~~~

Streaming is enabled by the causal model:

- Predictions depend only on past frames
- No future context is required
- Compatible with real-time deployment


Latency Considerations
~~~~~~~~~~~~~~~~~~~~~~

Latency is introduced by:

- Feature extraction window (e.g. 25 ms)
- Chunk size (e.g. 100 ms)
- Buffering before inference

Trade-offs:

- Smaller chunks → lower latency but noisier predictions
- Larger chunks → more stable predictions but higher latency


Offline vs Streaming
~~~~~~~~~~~~~~~~~~~~

- Offline: full sequence available
- Streaming: partial context only

This may result in:

- Slightly lower accuracy
- Increased variability near chunk boundaries


What works well
---------------

- Simple and modular pipeline
- Fully streaming-compatible
- Efficient on CPU
- Easy to train and debug


Limitations
-----------

- Frame-level predictions can be noisy
- No temporal smoothing
- Limited temporal context
- Streaming inference recomputes over buffered context (stateless)


Possible Improvements
---------------------

- Add temporal smoothing (median filtering, hysteresis)
- Increase temporal context (dilated CNN, RNN, Transformer)
- Implement stateful streaming inference
- Improve robustness to noise
- Calibrate output probabilities
