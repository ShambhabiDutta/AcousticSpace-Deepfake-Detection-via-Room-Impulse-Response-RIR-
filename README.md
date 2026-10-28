
# AcousticSpace: Deepfake Detection via Room Impulse Response (RIR)

## Overview

AcousticSpace is a research project exploring audio deepfake detection using Room Impulse Response (RIR) and acoustic features. It investigates whether acoustic characteristics can help distinguish genuine speech from synthetic or manipulated audio.

The project includes a Python audio-processing pipeline, a baseline machine-learning model, and a FastAPI backend. Frontend integration is the next development stage.

## Technology Stack

- Python
- FastAPI
- Uvicorn
- NumPy
- Librosa
- Scikit-learn
- Joblib
- pytest
- HTTPX
- React and TypeScript (frontend integration)

## Project Structure

```text
AcousticSpace/
├── audio_ml/
│   ├── dataset.py
│   ├── feature_extractor.py
│   ├── evaluate_thresholds.py
│   ├── train_baseline.py
│   ├── models/
│   │   └── baseline_logreg.joblib
│   └── tests/
├── backend/
│   ├── main.py
│   ├── test_main.py
│   ├── test_cnn_model.py
│   ├── requirements.txt
│   └── uploads/
├── frontend/                 # Frontend integration, if present on this branch
├── .gitignore
└── README.md
```

The `venv/` directory, uploaded audio, and generated datasets should remain excluded from version control where appropriate. The trained model is currently stored at `audio_ml/models/baseline_logreg.joblib`.

## Backend API

Start the backend from the project root:

```powershell
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

If your local import configuration requires starting from the `backend` directory, use the command that works in your environment.

### Available Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Confirms that the API is running |
| GET | `/health` | Checks backend health |
| POST | `/upload` | Validates and saves an audio file |
| POST | `/predict` | Runs baseline model inference on an audio file |

### Audio Upload

The `/upload` endpoint accepts a multipart form field named `file`.

Supported extensions:

- `.wav`
- `.mp3`
- `.flac`
- `.ogg`

The configured maximum file size is 20 MB. Empty files and unsupported formats are rejected. Files are saved with unique generated filenames to avoid overwriting existing uploads.

### Deepfake Prediction

The `/predict` endpoint accepts an audio file through the multipart field `file`.

The backend processes the audio, extracts a fixed-length log-mel spectrogram, generates summary features, and passes them to the trained baseline classifier.

A successful response includes:

- `filename`
- `prediction` (`real` or `spoof`)
- `spoof_probability`
- `threshold`
- `status`

The current configured decision threshold is `0.70`. The returned probability is a model score, not proof that a recording is genuine or fake.

## Frontend Integration

The frontend will connect to the FastAPI backend at:

```text
http://127.0.0.1:8000
```

The React upload form should send audio to `POST /predict` using `multipart/form-data` with the field name `file`.

The dashboard should display the selected filename, upload or processing status, predicted label, and spoof probability.

The frontend should handle API errors, invalid files, and network failures gracefully. CORS must permit the frontend's actual local development origin.

**Status:** Frontend integration is pending until the frontend code is connected and tested end to end.

## Audio Processing

The audio pipeline includes:

1. Loading audio and converting it to mono.
2. Resampling to 16 kHz.
3. Extracting log-mel spectrograms.
4. Padding or truncating spectrograms to a consistent shape.
5. Summarizing the features for baseline model inference.

The feature extractor uses 64 mel bands and a fixed-length spectrogram representation of 400 frames.

## Baseline Model

The baseline training script is:

```powershell
python -m audio_ml.train_baseline
```

The threshold evaluation script is:

```powershell
python -m audio_ml.evaluate_thresholds
```

The trained model is stored at:

```text
audio_ml/models/baseline_logreg.joblib
```

The baseline model is an initial research implementation. Performance on development data does not guarantee accuracy on unseen real-world recordings.

## Run Tests

From the project root, with the project's virtual environment activated:

```powershell
python -m pytest -v
```

The current test suite contains 23 tests covering audio processing, dataset summaries, evaluation metrics, RIR features, CNN output checks, and backend endpoints.

## Current Project Status

### Completed

- FastAPI backend setup.
- Health-check and root endpoints.
- Audio-upload validation.
- Baseline model training and storage.
- Audio feature extraction.
- Prediction endpoint integration.
- Threshold evaluation script.
- Automated tests: 23 passed.
- Backend changes committed and pushed to the `KarolineUrmitha` branch.

### In Progress

- React frontend integration.
- Frontend-to-backend upload and prediction workflow.
- Displaying prediction results in the dashboard.
- End-to-end testing.

## Limitations

The current classifier is a baseline model and has not been established as a reliable production deepfake detector. Results may vary across datasets, recording environments, codecs, and synthesis methods.

RIR-based feature exploration, further model evaluation, and end-to-end frontend testing remain important future tasks.


