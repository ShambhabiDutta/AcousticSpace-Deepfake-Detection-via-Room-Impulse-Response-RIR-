
# AcousticSpace — Audio ML Pipeline

## Purpose
This module prepares audio data for deepfake detection using
Librosa-based preprocessing and acoustic feature extraction.

## Project Structure

- `config.py` — Audio and feature configuration
- `audio_processor.py` — Audio loading and validation
- `feature_extractor.py` — Log-mel spectrogram extraction
- `dataset.py` — Label mapping and feature preparation
- `data/raw/` — Original audio samples
- `data/processed/` — Processed audio data
- `features/` — Extracted feature artifacts
- `models/` — Saved model weights
- `tests/` — Automated tests

## Audio Preprocessing

Audio is loaded as mono and resampled to 16,000 Hz.

The log-mel spectrogram uses:
- FFT size: 1024
- Hop length: 256
- Mel bands: 64

## Labels

- `0` — Real audio (`bonafide` or `real`)
- `1` — Spoofed audio (`spoof` or `fake`)

Confirm the dataset's official labels before processing the full dataset.

## Feature Format

The feature extractor returns a NumPy float32 array with shape:

`(64, time_frames)`

The time dimension depends on the input audio duration.

For CNN training, features may need padding or truncation to a
consistent time dimension. This must be implemented before batching
variable-length audio samples.

## Usage

From the repository root:

```python
from audio_ml.feature_extractor import extract_log_mel_spectrogram

features = extract_log_mel_spectrogram("path/to/audio.wav")
print(features.shape)
```

## Current Limitations

- The pipeline has not yet demonstrated reliable RIR isolation.
- Baseline model training and evaluation are separate tasks.
- Feature extraction must be tested with real audio files.
- Performance metrics must come from held-out evaluation data.
