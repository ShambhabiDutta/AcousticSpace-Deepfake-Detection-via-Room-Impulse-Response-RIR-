# AcousticSpace API Guide

## 1. Overview

AcousticSpace provides a FastAPI backend for uploading audio recordings and classifying them as **bonafide (genuine)** or **spoof (fake or manipulated)**.

## 2. Starting the API

From the project root, run:

```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

Base URL: `http://127.0.0.1:8000`

Interactive API documentation: `http://127.0.0.1:8000/docs`

## 3. Available Endpoints

### GET /

Checks whether the API root endpoint is accessible.

### GET /health

Checks backend health.

### POST /upload

Accepts an audio file and validates its format and size.

Supported formats:
- WAV
- MP3
- FLAC
- OGG

Maximum file size: 20 MB.

### POST /predict

Classifies an uploaded audio recording using the currently configured Logistic Regression model.

**Model file:** `audio_ml/models/baseline_logreg.joblib`

**Prediction threshold:** 0.70

Classification:
- `real`: spoof probability is below 0.70.
- `spoof`: spoof probability is greater than or equal to 0.70.

Example successful response:

```json
{
  "filename": "sample.wav",
  "prediction": "spoof",
  "spoof_probability": 0.8,
  "threshold": 0.7,
  "status": "success"
}# AcousticSpace API Guide

## 1. Overview

AcousticSpace provides a FastAPI backend for uploading audio recordings and classifying them as **bonafide (genuine)** or **spoof (fake or manipulated)**.

The backend validates uploaded audio files, extracts audio features and uses a machine-learning model to estimate whether a recording is genuine or spoofed.

## 2. Starting the API

From the project root, activate the virtual environment and run:

```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

**Base URL:** `http://127.0.0.1:8000`

**Interactive API documentation:** `http://127.0.0.1:8000/docs`

The interactive documentation allows developers to inspect and test the available endpoints.

## 3. Available Endpoints

### GET /

Checks whether the API root endpoint is accessible.

### GET /health

Checks backend health and confirms that the API is responding.

### POST /upload

Accepts an audio file and validates its format and size.

**Supported formats:**
- WAV
- MP3
- FLAC
- OGG

**Maximum file size:** 20 MB.

### POST /predict

Classifies an uploaded audio recording using the currently configured Logistic Regression model.

**Model file:** `audio_ml/models/baseline_logreg.joblib`

**Prediction threshold:** `0.70`

**Classification logic:**
- `real`: spoof probability is below 0.70.
- `spoof`: spoof probability is greater than or equal to 0.70.

**Example successful response:**

```json
{
  "filename": "sample.wav",
  "prediction": "spoof",
  "spoof_probability": 0.8,
  "threshold": 0.7,
  "status": "success"
}
```

The spoof probability is a model score, not a guarantee that an audio recording is genuine or fake.

## 4. Prediction Workflow

The prediction endpoint follows these steps:

1. Receive the uploaded audio file.
2. Validate the file format and size.
3. Save the uploaded audio temporarily for processing.
4. Extract fixed-length log-mel spectrogram features.
5. Summarize the extracted features into a model input vector.
6. Pass the features to the Logistic Regression model.
7. Calculate the spoof probability.
8. Compare the probability with the configured threshold.
9. Return the prediction result.
10. Remove the temporary audio file after processing.

## 5. Model and Feature Processing

The currently configured API prediction model is Logistic Regression.

The feature-processing pipeline:
- Extracts log-mel spectrogram features.
- Uses a fixed-size spectrogram representation.
- Summarizes the features into a 128-dimensional input vector.
- Generates a spoof probability using the trained model.

The AST (Audio Spectrogram Transformer) model is evaluated separately. It should not be considered the active API prediction model until its evaluation and integration are completed.

## 6. Error Handling and Validation

The API validates incoming audio uploads before processing.

Possible validation cases include:
- Empty audio uploads.
- Unsupported file formats.
- Files exceeding the 20 MB limit.
- Invalid or unreadable audio files.
- Prediction-processing errors.

The API returns an appropriate error response when an upload cannot be processed.

## 7. Current Limitations

- The current API performs whole-recording classification.
- It does not identify suspicious timestamps or individual audio segments.
- The active prediction API uses Logistic Regression.
- AST evaluation is a separate model-evaluation workflow.
- The prediction score should not be treated as a guarantee.
- Model performance should be assessed using independent evaluation data and appropriate class-sensitive metrics.

## 8. Running Tests

From the project root, run:

```powershell
python -m pytest backend -v
```

This runs the backend test suite, including endpoint, upload validation and prediction tests.

## 9. Summary

The AcousticSpace API provides audio upload validation, backend health checks and whole-recording spoof classification.

Its current prediction workflow uses log-mel feature extraction and a Logistic Regression model. AST evaluation is maintained separately until its results support a model-integration decision.
