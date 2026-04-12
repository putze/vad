Quickstart
==========

This section shows how to:

- Train a model
- Run inference (offline and streaming)
- Compare against baselines with ROC/AUC
- Launch an interactive demo


Train a model
-------------

Train the VAD model on LibriVAD-style data:

.. code-block:: bash

   vad-train \
       --results-root path/to/audio \
       --labels-root path/to/labels

Example:

.. code-block:: bash

   vad-train \
       --results-root data/Results \
       --labels-root data/Labels \
       --checkpoint-dir checkpoints \
       --log-dir runs \
       --num-epochs 10 \
       --device cpu


Run offline inference
---------------------

Run VAD on a single audio file:

.. code-block:: bash

   vad-infer-offline \
       --audio path/to/audio.wav

Example:

.. code-block:: bash

   vad-infer-offline \
       --audio example.wav \
       --checkpoint checkpoints/best_causal_vad.pt \
       --output-dir outputs \
       --show-plot

This command produces:

- Frame-level predictions
- Speech probabilities
- Optional visualization (waveform + predictions)


Run streaming inference
-----------------------

Simulate real-time VAD on an audio file:

.. code-block:: bash

   vad-stream-file \
       --audio path/to/audio.wav \
       --checkpoint checkpoints/best_causal_vad.pt

Optional parameters:

- ``--chunk-ms``: chunk size (default: 100 ms)
- ``--min-buffer-ms``: minimum buffer before inference
- ``--threshold``: decision threshold

This mode mimics **online/low-latency inference**.


Compare with baselines
----------------------

Evaluate your model against baselines (see :ref:`baselines`):

.. code-block:: bash

   vad-compare-models \
       --checkpoint checkpoints/best_causal_vad.pt

Example:

.. code-block:: bash

   vad-compare-models \
       --checkpoint checkpoints/best_causal_vad.pt \
       --results-root data/Results \
       --labels-root data/Labels \
       --split dev-clean \
       --webrtc-mode 2

This prints:

- Accuracy, precision, recall, F1
- False positive / negative rates


ROC curve and AUC
-----------------

To compute and visualize the ROC curve:

.. code-block:: bash

   vad-compare-models \
       --checkpoint checkpoints/best_causal_vad.pt \
       --split dev-clean \
       --all-webrtc \
       --roc-output outputs/roc.png

This will:

- Compute the **ROC curve** and **AUC** for the trained model
- Plot **WebRTC operating points** (aggressiveness 0–3)
- Save the figure to ``outputs/roc.png``

You can display it interactively with:

.. code-block:: bash

   vad-compare-models \
       --checkpoint checkpoints/best_causal_vad.pt \
       --all-webrtc \
       --show-roc


Run the demo
------------

Launch the interactive Streamlit demo:

.. code-block:: bash

   vad-demo

The demo allows:

- Uploading audio files
- Visualizing predictions


Next steps
----------

- See :doc:`evaluation` for details on metrics and ROC analysis
- See :ref:`baselines` for baseline descriptions
- Explore qualitative outputs to understand model behavior
