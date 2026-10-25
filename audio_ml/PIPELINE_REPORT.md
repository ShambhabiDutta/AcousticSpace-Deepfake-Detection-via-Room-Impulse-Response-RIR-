# AcousticSpace — Audio ML Pipeline Report

## 1. Purpose
This module supports dataset preparation, audio feature extraction, reverberation-related feature investigation, and evaluation of real-versus-spoof audio classification.

## 2. Dataset Preparation
- Parse a six-field, ASVspoof-style protocol file.
- Map bona fide/real samples to label `0` and spoof/fake samples to label `1`.
- Resolve audio files by ID and validate audio availability.
- Summarize class counts and missing audio files.

## 3. Audio Preprocessing
- Load audio as mono using Librosa.
- Resample audio to the configured sample rate.
- Extract log-mel spectrograms.
- Convert spectrograms to fixed-length arrays through truncation or padding.
- Export model-ready arrays as `X.npy`, `y.npy`, and `sample_ids.npy`.

## 4. Reverberation Investigation
The current implementation calculates duration, mean RMS energy, final-segment energy ratio, and zero-crossing rate.

These are exploratory acoustic descriptors, not direct measurements of room impulse responses or reverberation time. Their ability to distinguish recording environments or synthetic audio requires experimental validation.

## 5. Evaluation
The evaluation module calculates:
- Accuracy
- Precision
- Recall
- F1-score
- Confusion matrix
- Classification report

A confusion matrix plot can also be saved for analysis.

## 6. Testing
Automated tests cover audio feature extraction, fixed-length preprocessing, label handling, protocol parsing, dataset export, missing audio detection, reverberation feature extraction, and evaluation metrics.

Run the test suite from the repository root:

`python -m pytest audio_ml/tests -v`

## 7. Limitations
- No real dataset performance is established by synthetic test audio.
- Actual evaluation requires a labeled dataset and predictions from the trained baseline model.
- The dataset must be split appropriately into training and held-out evaluation sets.
- Padding and truncation may discard useful information or introduce artifacts.
- The current acoustic descriptors do not constitute a validated RIR estimation method.
- Model accuracy, generalization, and robustness remain unverified until experiments are run on real labeled data.

## 8. Next Steps
1. Obtain a suitable labeled real/spoof dataset.
2. Run the pipeline on real audio files and inspect the exported arrays.
3. Coordinate with the model developer to obtain held-out predictions.
4. Record measured metrics and analyze the confusion matrix.
5. Compare results with a simple baseline and document experimental limitations.
