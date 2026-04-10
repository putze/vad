Evaluation
==========

Evaluation Protocol
-------------------

The model is evaluated as a **frame-level binary classifier**.

For each audio sample:

1. Run inference to obtain frame-level predictions
2. Align ground-truth labels to frame resolution
3. Compare predictions and targets

Small alignment mismatches (±1–2 frames) are trimmed.


Metrics
-------

The following metrics are computed:

- **Precision**: proportion of predicted speech frames that are correct
- **Recall**: proportion of true speech frames detected
- **F1-score**: harmonic mean of precision and recall
- **Accuracy**: overall frame-level correctness

These metrics are averaged over the dataset.


Decision Threshold
------------------

The model outputs probabilities.

- A threshold (default: ``0.5``) is applied to obtain binary predictions
- The threshold can be tuned to trade off precision vs. recall


Baseline Comparison
-------------------

The model is compared against **WebRTC VAD**, a widely used rule-based baseline.

WebRTC characteristics:

- No training required
- Designed for real-time applications
- Tunable aggressiveness mode (0–3)

Comparison is performed on the same dataset and splits.


Example command:

.. code-block:: bash

   vad-compare-models \
       --checkpoint checkpoints/best_causal_vad.pt \
       --results-root data/Results \
       --labels-root data/Labels \
       --split dev-clean


What to look at
---------------

- F1-score comparison between models
- Precision/recall trade-offs
- Sensitivity to noise conditions
- Effect of WebRTC aggressiveness mode


Limitations
-----------

- Evaluation is frame-level only (no segment-level metrics)
- No ROC or PR curves currently computed
- No statistical significance analysis


Possible Improvements
---------------------

- Plot ROC and Precision–Recall curves
- Evaluate segment-level metrics (e.g. speech onset/offset)
- Analyze performance across noise conditions
- Tune threshold on validation set

Qualitative Evaluation
----------------------

Predictions can be visualized alongside the waveform and ground truth.

This helps identify:

- Missed speech segments
- False positives in noise
- Temporal jitter

Visualization is available via:

.. code-block:: bash

   vad-infer-offline --audio example.wav --show-plot
