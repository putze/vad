Baselines
=========

Overview
--------

To evaluate the proposed model, we compare it against a strong and widely used baseline:

- **WebRTC VAD**

This provides a reference point for performance and helps assess whether the learned model brings improvements.


WebRTC VAD
----------

WebRTC VAD is a rule-based voice activity detector commonly used in real-time systems.

Key properties:

- No training required
- Designed for low-latency streaming
- Lightweight and CPU-efficient
- Widely deployed in production systems


Aggressiveness Modes
--------------------

WebRTC VAD exposes a parameter controlling detection strictness:

- ``0``: least aggressive (more false positives)
- ``3``: most aggressive (more false negatives)

This allows trading off precision vs. recall.


Why this baseline?
------------------

WebRTC VAD is a strong baseline because:

- It is widely used in practice
- It performs reasonably well across conditions
- It requires no training data

Comparing against it provides a meaningful real-world reference.


Comparison Protocol
-------------------

Both models are evaluated under the same conditions:

- Same dataset and splits
- Same frame resolution (10 ms)
- Same label alignment procedure

The learned model uses probability thresholding,
while WebRTC produces binary decisions directly.


Running the comparison
----------------------

Example:

.. code-block:: bash

   vad-compare-models \
       --checkpoint checkpoints/best_causal_vad.pt \
       --results-root data/Results \
       --labels-root data/Labels \
       --split dev-clean \
       --webrtc-mode 2


What to analyze
---------------

When comparing models, focus on:

- **F1-score**: overall performance
- **Precision vs. recall** trade-offs
- **Robustness to noise**
- **Consistency over time** (jitter)


Limitations of the baseline
--------------------------

- Rule-based, not learned from data
- Limited adaptability to new domains
- Performance depends on aggressiveness tuning


Possible Extensions
-------------------

Additional baselines could be added:

- **Silero VAD** (neural, pretrained)
- Energy-based VAD (simple heuristic)
- Transformer-based VAD models

This would provide a broader comparison spectrum.
