
# AcousticSpace — Audio ML

This module handles dataset preparation, audio preprocessing,
acoustic feature extraction, and evaluation for deepfake audio detection.

## Planned components

- `data/raw/`: Original audio samples
- `data/processed/`: Processed audio data
- `features/`: Extracted acoustic features
- `models/`: Saved model artifacts
- `tests/`: Pipeline tests

## Dependencies

Install the audio-processing dependencies from the project root:

```bash
python -m pip install -r audio_ml/requirements.txt
```

## Week 2 objectives

- Validate audio loading and preprocessing with Librosa.
- Extract spectrograms and basic acoustic features.
- Prepare consistent features and labels for baseline classification.
- Evaluate the baseline model using held-out test data.
