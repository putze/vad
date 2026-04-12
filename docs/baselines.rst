.. _baselines:

Baselines
=========

Overview
--------

To evaluate the proposed model, we compare it against a strong and widely used baseline:

- **WebRTC VAD**

This provides a practical reference point and helps assess whether the learned model
brings improvements over a production-ready system.


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

- ``0``: least aggressive (higher recall, more false positives)
- ``1``: moderate
- ``2``: balanced (default)
- ``3``: most aggressive (higher precision, more false negatives)

This parameter allows controlling the trade-off between false positives and missed speech.


Why this baseline?
------------------

WebRTC VAD is a strong baseline because:

- It is widely used in real-world applications
- It performs reasonably well across a range of acoustic conditions
- It requires no training data or model tuning

Comparing against it provides a meaningful real-world reference for performance.


Comparison Protocol
-------------------

Both models are evaluated under identical conditions:

- Same dataset and splits
- Same frame resolution
- Same label alignment procedure

The learned model produces **probabilities** that are thresholded,
while WebRTC produces **binary decisions directly**.


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

- **F1-score**: overall performance at a fixed threshold
- **Precision vs. recall** trade-offs
- **Robustness to noise conditions**
- **Temporal consistency** (e.g., prediction jitter)


Limitations of the baseline
---------------------------

- Rule-based, not learned from data
- Limited adaptability to new domains
- Performance depends on aggressiveness tuning


Possible Extensions
-------------------

Additional baselines could be included:

- **Silero VAD** (neural, pretrained)
- Transformer-based VAD models

This would provide a broader comparison spectrum.
