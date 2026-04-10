Demo
====

Overview
--------

An interactive demo is provided to visualize Voice Activity Detection (VAD)
predictions on audio signals.

The demo allows:

- Uploading an audio file
- Running the trained model
- Visualizing waveform, probabilities, and predictions


Running the demo
----------------

Launch the Streamlit app:

.. code-block:: bash

   vad-demo

This will open a local web interface in your browser.


Features
--------

**Audio input**

- Upload a ``.wav`` file
- Audio is automatically preprocessed

**Model inference**

- Runs the trained VAD model
- Outputs frame-level speech probabilities

**Visualization**

The demo displays:

- Waveform
- Speech probabilities over time
- Binary speech / non-speech decisions

This makes it easy to understand model behavior.


What to look for
----------------

The demo helps identify:

- Missed speech segments (false negatives)
- False detections in silence or noise
- Temporal jitter in predictions
- Effect of thresholding


Connection to evaluation
------------------------

The demo complements quantitative evaluation by providing
a qualitative view of model performance.

It is particularly useful for:

- Debugging model behavior
- Understanding failure cases
- Comparing predictions visually


Limitations
-----------

- Offline processing only (no real microphone streaming)
- No direct comparison with baseline models in the UI


Possible Improvements
---------------------

- Add real-time microphone input
- Overlay ground-truth labels
- Compare multiple models side-by-side
- Interactive threshold tuning
