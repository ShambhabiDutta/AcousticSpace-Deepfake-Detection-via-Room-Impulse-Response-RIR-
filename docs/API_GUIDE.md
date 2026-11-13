# AcousticSpace API Guide

## 1. Overview

AcousticSpace provides a FastAPI backend for detecting bonafide (genuine) and spoofed (fake or manipulated) audio recordings.

The API supports audio upload validation and prediction using three configured models:

- Logistic Regression
- Audio Spectrogram Transformer (AST)
- Convolutional Neural Network (CNN)

## 2. Starting the API Locally

From the project root, activate the virtual environment and run:

```powershell
python -m uvicorn backend.main:app --host 127.0.0.1 --port 8000
```

**Base URL:** `http://127.0.0.1:8000`

**Interactive API documentation:** `http://127.0.0.1:8000/docs`

## 3. Available Endpoints

| Method | Endpoint | Description |
|---|---|---|
| GET | `/` | API root |
| GET | `/health` | Backend health check |
| POST | `/upload` | Upload and validate audio |
| POST | `/predict` | Logistic Regression prediction |
| POST | `/predict/ast` | AST prediction |
| POST | `/predict/cnn` | CNN prediction |

### 3.1 GET /

Checks whether the API root endpoint is accessible.

### 3.2 GET /health

Checks backend health and confirms that the API is responding.

### 3.3 POST /upload

Accepts an audio file and validates its format and size.

Supported formats:
- WAV
- MP3
- FLAC
- OGG

Maximum file size: 20 MB.

### 3.4 POST /predict

Performs audio spoof classification using the Logistic Regression model.

- Model: Logistic Regression
- Model file: `audio_ml/models/baseline_logreg.joblib`
- Prediction threshold: `0.70`

Classification:
- Real: spoof probability is below 0.70.
- Spoof: spoof probability is greater than or equal to 0.70.

### 3.5 POST /predict/ast

Performs audio spoof classification using the Audio Spectrogram Transformer.

- Model: AST
- Model directory: `audio_ml/models/ast_binary`
- Configured prediction threshold: `0.50`

Classification:
- Real: spoof probability is below 0.50.
- Spoof: spoof probability is greater than or equal to 0.50.

### 3.6 POST /predict/cnn

Performs audio spoof classification using the Convolutional Neural Network.

- Model: CNN
- Configured prediction threshold: `0.50`
- The threshold is provisional and requires validation against evaluation results.

Classification:
- Real: spoof probability is below 0.50.
- Spoof: spoof probability is greater than or equal to 0.50.

## 4. Prediction Request Format

All prediction endpoints accept an uploaded audio file using multipart form data.

The required field name is:

`file`

Example using PowerShell:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/predict/ast" -F "file=@sample.flac"
```

For the other models, use:

```powershell
curl.exe -X POST "http://127.0.0.1:8000/predict" -F "file=@sample.flac"

curl.exe -X POST "http://127.0.0.1:8000/predict/cnn" -F "file=@sample.flac"
```

Replace `sample.flac` with the path to your audio file.

## 5. Prediction Response

A successful prediction returns information such as:

- Uploaded filename
- Model used
- Predicted class
- Real and spoof probabilities
- Spoof probability
- Confidence score
- Configured threshold
- Analysis status

Example AST response:

```json
{
  "filename": "sample.flac",
  "model": "AST",
  "prediction": "spoof",
  "predicted_class": "spoof",
  "class_probabilities": {
    "real": 0.067,
    "spoof": 0.933
  },
  "spoof_probability": 0.933,
  "confidence": 0.933,
  "analysis": {
    "threshold": 0.5,
    "classification_rule": "spoof_probability >= threshold means spoof; otherwise real"
  },
  "analysis_status": "success",
  "status": "success"
}
```

The confidence and probability values are model outputs, not guarantees that a recording is genuine or manipulated.

## 6. Prediction Workflow

The prediction workflow generally follows these steps:

1. Receive the uploaded audio file.
2. Validate the file format and size.
3. Save the audio temporarily for processing.
4. Decode and preprocess the audio.
5. Extract the required audio features.
6. Pass the processed input to the selected model.
7. Calculate the spoof probability.
8. Compare the probability with the configured threshold.
9. Return the prediction and analysis result.
10. Clean up temporary processing files.

The preprocessing and inference implementation differ between models.

## 7. Docker Deployment

The project includes a Dockerfile for deploying the FastAPI backend with its required system dependencies and model assets.

### Build the Docker image

From the project root:

```powershell
docker build -t acousticspace-api:week4 .
```

### Run the container

```powershell
docker run --name acousticspace-week4 -p 8000:8000 acousticspace-api:week4
```

### Verify the deployment

Open another PowerShell window and run:

```powershell
curl.exe http://127.0.0.1:8000/health
```

Expected healthy response:

```json
{
  "status": "healthy"
}
```

The interactive API documentation is available at:

`http://127.0.0.1:8000/docs`

## 8. Error Handling and Validation

The API validates uploaded audio before processing.

Validation cases include:

- Empty or missing audio uploads.
- Unsupported file formats.
- Files exceeding the maximum upload size.
- Invalid or unreadable audio files.
- Prediction-processing errors.

For example, a missing file can return HTTP 422, while an unsupported file format can return HTTP 400.

## 9. Running Tests

Run the backend tests locally:

```powershell
python -m pytest tests/test_prediction_api.py backend/test_main.py -q
```

The project also includes a GitHub Actions workflow:

`.github/workflows/backend-ci.yml`

The workflow runs on pushes and pull requests. It installs Python, audio system dependencies, CPU PyTorch and project requirements before executing the backend prediction tests.

## 10. Model Evaluation and Limitations

The project supports multiple prediction models, but their thresholds and performance must be evaluated independently.

Important considerations:

- The Logistic Regression threshold is currently 0.70.
- The AST threshold is currently 0.50 and should be assessed against the full evaluation results.
- The CNN threshold is provisional at 0.50.
- Model scores should not be interpreted as guarantees.
- Evaluation should consider accuracy, precision, recall, F1-score and ROC-AUC, with particular attention to spoof detection performance.
- Performance can vary depending on recording conditions, audio quality and the type of spoofing.

## 11. Summary

AcousticSpace provides a FastAPI-based audio classification backend with upload validation, health checks, and three prediction endpoints for Logistic Regression, AST and CNN.

The application supports local execution and Docker deployment, with automated backend testing through GitHub Actions.
