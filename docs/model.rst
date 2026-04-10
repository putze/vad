Model
=====

Architecture
------------

The model is a **causal 1D convolutional network** operating on time frames.

- Input: log-mel features (time × frequency)
- Convolutions applied over time
- Output: frame-level speech probabilities


Key Properties
--------------

**Causality**

- Output at time ``t`` depends only on past inputs
- Achieved via left-padding in convolution layers

**Efficiency**

- Lightweight architecture
- Suitable for CPU inference


Why convolutional?
------------------

- Simple and fast
- Captures local temporal patterns
- Easier to train than sequence models


Alternatives
------------

- RNNs (LSTM/GRU): better temporal modeling but slower
- Transformers: strong performance but higher compute cost


Limitations
-----------

- Limited receptive field
- No explicit long-range dependencies


Possible Improvements
---------------------

- Dilated convolutions to increase context
- Hybrid CNN + RNN architecture
- Transformer-based VAD
