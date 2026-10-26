# AcousticSpace: Deepfake Detection via Room Impulse Response (RIR)

## Overview

AcousticSpace explores audio deepfake detection using Room Impulse Response (RIR) and acoustic features. The project investigates whether reverberation and other acoustic characteristics can contribute to identifying synthetic or manipulated audio.

## Technology Stack

- Python
- FastAPI and Uvicorn
- Librosa, NumPy, Pandas, Matplotlib, SciPy, and PyTorch
- pytest and HTTPX

## Backend API

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | API status |
| GET | `/health` | Health check |
| POST | `/upload` | Upload and validate audio |
| POST | `/predict` | Prediction endpoint placeholder |

Supported audio extensions: WAV, MP3, FLAC, OGG. Maximum upload size: 20 MB.

The `/predict` endpoint returns HTTP 503 until a trained model is available.

## Audio Processing

The `audio_ml/` module includes:

- Audio preprocessing and mono conversion
- Resampling to 16 kHz
- Log-mel spectrogram extraction
- Dataset utilities and label validation
- Reverberation-related feature extraction
- Evaluation utilities and confusion matrix plotting
- Automated tests

Configuration for log-mel spectrogram extraction:

- FFT size: 1,024
- Hop length: 256
- Mel bands: 64

## RIR Acoustic Analysis

The script `audio_ml/run_rir_analysis.py` regenerates acoustic-condition summaries, paired comparisons, statistical tests, correlations, and plots from the existing feature dataset.

### Input

The script reads:

`audio_ml/features/sample_features_with_rir_metadata.csv`

The current sample contains 100 feature rows across five conditions:

- Clean
- Reverberation
- Noise at 10 dB SNR
- Noise at 5 dB SNR
- Noise at 0 dB SNR

The paired reverberation analysis compares 20 matched sample IDs under clean and reverberant conditions.

### Features analyzed

- RMS amplitude
- Peak amplitude
- Zero-crossing rate (ZCR)
- Spectral centroid
- RIR metadata, including T60, DRR, C50, EDT, and available SNR values

### Analysis methods

- Summary statistics by acoustic condition
- Paired reverb-minus-clean feature differences
- Wilcoxon signed-rank tests with Holm correction
- Spearman correlations between T60 and paired feature changes
- Plots comparing acoustic conditions and paired feature changes

### Run the analysis

From the project root:

```powershell
python audio_ml\run_rir_analysis.py
```

The script writes its output to `audio_ml/features/`.

Main output files:

| File | Purpose |
|---|---|
| `condition_feature_summary.csv` | Feature statistics by condition |
| `sample_feature_comparison.png` | Acoustic feature comparison plots |
| `feature_comparison_100_samples.png` | Feature comparison figure |
| `paired_reverb_changes.csv` | Per-sample reverb-minus-clean differences |
| `paired_reverb_statistical_tests.csv` | Paired tests and Holm-adjusted p-values |
| `paired_reverb_changes.png` | Distribution of paired changes |
| `t60_paired_change_correlations.csv` | T60 correlation statistics |
| `t60_paired_change_scatter.png` | T60 versus paired feature changes |
| `t60_vs_spectral_centroid_reverb.png` | T60 versus spectral centroid for reverb samples |

### Interpretation and limitations

These analyses describe how acoustic features change under the dataset's simulated reverberation and noise conditions. The current paired analysis uses 20 matched samples, so its findings are preliminary.

T60, DRR, C50, and EDT are dataset-provided acoustic descriptors. They should not be presented as independently measured physical room properties.

**Important:** The current analysis does not establish genuine-versus-deepfake classification performance. Acoustic-condition labels such as `clean`, `reverb`, and `noisy_snr10` are not real/spoof labels. A validated labelled dataset and trained, evaluated detection model are still required.

## Setup

Create and activate a virtual environment if needed, then install the backend dependencies:

```powershell
python -m venv backend\venv
.\backend\venv\Scripts\Activate.ps1
python -m pip install -r backend\requirements.txt

```
