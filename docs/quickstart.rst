Quickstart
==========

This section shows how to train a model, run inference, and compare it to a baseline.


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


Compare with WebRTC VAD
------------------------

Evaluate your model against the WebRTC baseline:

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


Run the demo
------------

Launch the interactive Streamlit demo:

.. code-block:: bash

   vad-demo
