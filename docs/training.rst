Training
========

Objective
---------

The model is trained as a binary classifier.

- Loss: Binary Cross-Entropy
- Output: speech probability per frame


Optimization
------------

- Optimizer: Adam
- Mini-batch training


Handling Variable Length
-------------------------

Audio sequences have different lengths.

- Sequences are padded within a batch
- A collate function ensures proper batching


Evaluation During Training
--------------------------

Metrics:

- Precision
- Recall
- F1-score

Computed at the frame level.


Limitations
-----------

- No class imbalance handling
- No advanced scheduling or regularization


Possible Improvements
---------------------

- Weighted loss for imbalance
- Learning rate scheduling
- Data augmentation
