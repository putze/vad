Data and Features
=================

Input Representation
--------------------

Audio is processed as mono waveforms.

- Sample rate: 16 kHz
- Variable-length recordings


Feature Extraction
------------------

Log-mel spectrograms are used as input features.

Parameters:

- Window size: 25 ms
- Hop size: 10 ms
- Number of mel bins: configurable (default: 40)

Rationale:

- Standard representation for speech tasks
- Compact and efficient
- More robust to noise than raw waveform input


Label Alignment
---------------

Labels are provided at the sample level and converted to frame-level targets.

- A frame is labeled as speech if any sample in the frame is speech
- Implemented via max-pooling over the frame window

This ensures alignment with feature frames.


Limitations
-----------

- Hard alignment may introduce boundary noise
- No soft labeling or uncertainty modeling


Possible Improvements
---------------------

- Soft labels based on speech proportion
- Data augmentation (noise, reverberation)
- Multi-condition training for robustness
