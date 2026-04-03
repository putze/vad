# Voice Activity Detection

## Problem definition

- output type: frame level (every 10 ms)
- real time: streaming
- Classes: speech / nonspeech
- latency target: < 200ms
- environment: CPU

## Architecture

Audio -> preprocessing -> features -> model -> post-processing -> output

## Data Layer

### Dataset interface

Base Dataset class

required:
- unify multiple datasets (currently only LibriVAD...)
- handle file paths (metadata?)

### Dataset indexing

Sample should contain:
- audio_path (string)
- start (float)?
- end (float)?
- label (int)

### Data Validation

- Sample rate consistency
- label alignement
- duration mismatches
- class imbalance

## Preprocessing

### Feature extraction

- log-mel spectrogram
- window 25 ms
- hop 10 ms

### Data augmentation

- noise
- background music
- random silence

## split strategy

- split by speaker or recording
- keep domains separate

## Baseline

- todo

## Model Layer

- CNN
- CRNN ?
- ViT

## Loss, Metrics

- Loss: Binary Cross Entropy

Metrics:
- frame accuracy
- precision / recall
- F1 score

## Training loop

- batch (padding?)
- logging

## Monitoring

- loss curves
- F1 score
- FP, FN

Tensorboard

## Post processing
- threshodling
- minimum speech duration
- merge close segments

## Inference pipeline

audio -> buffer -> features -> model -> smoothing -> output

## Deployment

export model

## Experiment tracking

MLFlow?

## Testing

## UI

- waveform / mel spectrogram + VAD overlay
- real time mic demo
- debugging tool?
