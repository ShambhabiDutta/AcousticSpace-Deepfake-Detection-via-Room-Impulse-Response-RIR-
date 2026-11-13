# AcousticSpace: Deepfake Detection via Room Impulse Response (RIR)

## 1. Project Overview

**AcousticSpace** is a research project exploring audio deepfake detection using Room Impulse Response (RIR) and acoustic characteristics. It investigates how audio features can help distinguish genuine speech from synthetic or manipulated recordings.

The project includes:

- Audio preprocessing and feature extraction.
- A Logistic Regression baseline classifier.
- A CNN-based audio classifier.
- An Audio Spectrogram Transformer (AST) inference pipeline.
- A FastAPI backend for audio upload and prediction.
- Model evaluation and automated testing.
- Docker-based backend containerization.
- GitHub Actions continuous integration.

The current implementation provides separate prediction endpoints for the baseline, CNN, and AST models.

## 2. Technology Stack

| Area | Technologies |
|---|---|
| Programming language | Python |
| Backend API | FastAPI, Uvicorn |
| Audio processing | Librosa, NumPy |
| Machine learning | Scikit-learn, PyTorch |
| Transformer model | Hugging Face Transformers |
| Model storage | Joblib, PyTorch checkpoints, Safetensors |
| Testing | Pytest, HTTPX |
| Containerization | Docker |
| CI | GitHub Actions |
| Version control | Git, GitHub |

## 3. Project Structure

```text
AcousticSpace/
│
├── audio_ml/
│   ├── dataset.py
│   ├── feature_extractor.py
│   ├── evaluate_thresholds.py
│   ├── train_baseline.py
│   ├── models/
│   │   ├── baseline_logreg.joblib
│   │   ├── cnn_baseline.pt
│   │   └── ast_binary/
│   │       ├── config.json
│   │       └── model.safetensors
│   └── evaluation/
│       └── evaluation reports
│
├── backend/
│   ├── main.py
│   ├── ast_inference.py
│   ├── cnn_inference.py
│   ├── requirements.txt
│   └── training/
│       ├── train_ast.py
│       ├── train_cnn.py
│       ├── evaluate_ast_full_dev.py
│       ├── evaluate_cnn_dev.py
│       └── evaluate_cnn_validation.py
│
├── tests/
│   └── test_prediction_api.py
│
├── data/
├── docs/
│   └── MODEL_EVALUATION.md
├── uploads/
├── .github/
│   └── workflows/
│       └── backend-ci.yml
├── .dockerignore
├── .gitattributes
├── .gitignore
├── Dockerfile
└── README.md
```

*Note: This tree shows the principal project files. Additional experimental, test, and evaluation files may be present. Large datasets, virtual environments, temporary uploads, and generated experimental artifacts should not be committed unnecessarily.*

## 4. Environment Setup

### Prerequisites

- Python 3.12 or a compatible Python environment.
- pip.
- Git.
- Docker Desktop (for containerized deployment).
- A compatible virtual environment.

### Create and activate a virtual environment

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

### Install backend dependencies

```powershell
pip install -r backend/requirements.txt
```

The CNN and AST workflows also require their respective machine-learning and audio-processing dependencies.

## 5. Audio Processing Pipeline

The audio-processing pipeline prepares recordings for machine-learning inference.

The main preprocessing stages are:

1. Load the input audio.
2. Convert audio to mono where required.
3. Resample the recording to 16 kHz.
4. Extract log-mel spectrogram features.
5. Normalize the extracted features.
6. Pad or truncate the representation to the required input dimensions.
7. Pass the processed features to the selected model.

The CNN baseline uses a fixed spectrogram representation with:

- 64 mel bands.
- 400 time frames.
- Input shape: `1 × 64 × 400`.

The AST inference pipeline uses its configured Hugging Face feature extractor and model input requirements.

## 6. Backend API

The backend is implemented using FastAPI.

### Start the API locally

From the project root:

```powershell
python -m uvicorn backend.main:app --reload --host 127.0.0.1 --port 8000
```

If the local import configuration requires running from the backend directory:

```powershell
cd backend
python -m uvicorn main:app --host 127.0.0.1 --port 8000
```

The API is available at:

- API: `http://127.0.0.1:8000`
- Interactive documentation: `http://127.0.0.1:8000/docs`

### Available Endpoints

| Method | Endpoint | Purpose |
|---|---|---|
| GET | `/` | Confirms that the API is running |
| GET | `/health` | Checks backend health |
| POST | `/upload` | Validates and saves an audio file |
| POST | `/predict` | Runs Logistic Regression baseline inference |
| POST | `/predict/cnn` | Runs CNN-based deepfake inference |
| POST | `/predict/ast` | Runs AST-based deepfake inference |

Prediction endpoints accept audio through a multipart form field named `file`.

### Audio Upload Validation

The upload API supports:

- `.wav`
- `.mp3`
- `.flac`
- `.ogg`

The configured maximum upload size is 20 MB.

The backend validates uploaded files, rejects empty files and unsupported formats, and saves accepted uploads using generated filenames to avoid overwriting existing files.

### Prediction Response

The CNN and AST prediction endpoints return information including:

- `filename`
- `model`
- `prediction` (`real` or `spoof`)
- `class_probabilities`
- `spoof_probability`
- `confidence`
- `analysis`
- `status`

The `analysis` section includes the decision threshold and classification rule.

**Classification rule:**

- If spoof probability is greater than or equal to the configured threshold, the prediction is `spoof`.
- Otherwise, the prediction is `real`.

Confidence and probability values are model outputs. They should not be interpreted as proof that a recording is genuine or manipulated.

## 7. Machine-Learning Models

### 7.1 Logistic Regression Baseline

The Logistic Regression classifier serves as the initial baseline model.

Model checkpoint:

```text
audio_ml/models/baseline_logreg.joblib
```

The baseline provides a reference point for comparing the CNN and AST approaches.

### 7.2 CNN Model

The CNN model uses log-mel spectrogram features to classify audio into two categories:

| Class | Label |
|---|---:|
| Real | 0 |
| Spoof | 1 |

Saved checkpoint:

```text
audio_ml/models/cnn_baseline.pt
```

The checkpoint stores:

- Model state dictionary.
- Number of output classes.
- Input shape.
- Class mapping.
- Training seed and configuration.
- Validation balanced accuracy.

The CNN inference module is:

```text
backend/cnn_inference.py
```

It loads the saved checkpoint, validates its metadata, preprocesses audio, performs inference, and returns the spoof probability.

### 7.3 Audio Spectrogram Transformer (AST)

The project also includes an AST inference pipeline using Hugging Face Transformers.

Model directory:

```text
audio_ml/models/ast_binary/
```

Inference module:

```text
backend/ast_inference.py
```

The model configuration uses:

- 16 kHz audio sampling rate.
- 128 mel bins.
- A configured maximum sequence length of 1024.
- Class mapping: bonafide = 0 and spoof = 1.

An attack-aware AST training script has also been prepared:

```text
backend/training/train_ast.py
```

The script supports selecting genuine samples and spoof samples from attack categories A01–A06. Its separate experiment output directory is intended to protect the existing AST checkpoint.

**Status:** AST inference and evaluation have been performed. The separate attack-aware fine-tuning script has been prepared, but a completed fine-tuned experiment should not be claimed until it has actually been run and evaluated.

## 8. CNN Training and Evaluation

### Train the CNN

```powershell
python -m backend.training.train_cnn
```

The saved model is:

```text
audio_ml/models/cnn_baseline.pt
```

### Evaluate on the validation set

```powershell
python -m backend.training.evaluate_cnn_validation
```

### Evaluate on the development set

```powershell
python -m backend.training.evaluate_cnn_dev
```

### CNN Evaluation Results

| Metric | CNN Validation | CNN Development |
|---|---:|---:|
| ROC-AUC | 0.9243 | 0.9434 |
| Best tested balanced accuracy | 87.44% | 91.19% |
| Threshold associated with best tested balanced accuracy | 0.70 | 0.70 |

### CNN Development Threshold Comparison

| Threshold | Accuracy | Balanced Accuracy | Real Recall | Spoof Recall |
|---:|---:|---:|---:|---:|
| 0.30 | 90.42% | 72.28% | 49.45% | 95.11% |
| 0.40 | 90.50% | 78.16% | 62.64% | 93.68% |
| 0.50 | 90.33% | 83.63% | 75.20% | 92.06% |
| 0.60 | 89.57% | 88.45% | 87.05% | 89.85% |
| 0.70 | 87.35% | 91.19% | 96.04% | 86.35% |

A higher threshold improves genuine-audio recall in these results, while a lower threshold catches more spoof recordings.

The API's CNN threshold remains provisionally configured at 0.50. The threshold comparison above is evaluation evidence, not a finalized production calibration.

## 9. AST Evaluation

The AST model was evaluated on the ASVspoof 2019 LA development data.

Recorded full development-set results:

| Metric | AST |
|---|---:|
| ROC-AUC | 0.9069 |
| Balanced accuracy at threshold 0.30 | 82.17% |
| Spoof recall at threshold 0.30 | 73.64% |
| Balanced accuracy at threshold 0.50 | 77.80% |
| Spoof recall at threshold 0.50 | 58.54% |

These results show that the decision threshold changes the balance between detecting spoof recordings and correctly recognizing genuine recordings.

Detailed evaluation reports are available in:

```text
audio_ml/evaluation/
```

Additional experiment details are documented in:

```text
docs/MODEL_EVALUATION.md
```

## 10. Model Selection and Limitations

The CNN achieved a higher recorded development ROC-AUC and balanced accuracy than the AST result listed above.

The CNN is therefore the current deployment candidate for the next development stage. The AST pipeline remains available for comparison and further experiments.

Important limitations:

- The development dataset is not an independent final test set.
- Threshold results depend on the evaluation dataset and selected operating point.
- The CNN checkpoint is a baseline experiment, not a production-certified detector.
- Real-world performance may vary with microphones, rooms, codecs, noise, speakers, and synthesis methods.
- RIR-based research remains an important part of the project's direction.
- The current prediction pipeline does not provide verified timestamp-level suspicious-segment localization.

## 11. Automated Testing

The project uses Pytest and HTTPX for backend and API testing.

Run the full test suite from the project root:

```powershell
python -m pytest -v
```

The latest verified targeted API/backend test command was:

```powershell
python -m pytest tests/test_prediction_api.py backend/test_main.py -q
```

Result:

```text
18 passed
```

The CNN API tests cover:

- Successful prediction responses.
- Invalid audio handling.
- Model inference failure handling.

A synthetic audio smoke test was also performed to verify audio decoding, feature extraction, CNN inference, and the prediction API response. This smoke test verifies the inference pipeline, not real-world deepfake accuracy.

## 12. Docker Deployment

The backend can be packaged and run as a Docker container.

The root-level `Dockerfile`:

- Uses Python 3.12 slim as its base image.
- Installs FFmpeg and `libsndfile1` for audio processing.
- Installs CPU-based PyTorch and backend requirements.
- Copies the project files into the container.
- Exposes port 8000.
- Starts the FastAPI application using Uvicorn.

The `.dockerignore` file excludes virtual environments, temporary uploads, datasets, test caches, and other unnecessary files from the Docker build context.

### Build the Docker image

From the project root:

```powershell
docker build -t acousticspace-backend .
```

### Run the container

```powershell
docker run --name acousticspace-week4 -p 8000:8000 acousticspace-backend
```

The containerized API is available at:

```text
http://localhost:8000
```

API documentation:

```text
http://localhost:8000/docs
```

### Check container health

```powershell
Invoke-RestMethod http://localhost:8000/health
```

Expected healthy response:

```json
{
  "status": "healthy"
}
```

### Test CNN inference

```powershell
curl.exe -X POST -F "file=@uploads/cnn_smoke_test.wav" http://localhost:8000/predict/cnn
```

### Test AST inference

```powershell
curl.exe --max-time 180 -X POST -F "file=@uploads/cnn_smoke_test.wav" http://localhost:8000/predict/ast
```

### Stop and remove the container

```powershell
docker stop acousticspace-week4
docker rm acousticspace-week4
```

### Local Docker verification

The following checks were completed locally:

- Docker image built successfully.
- Container started and exposed port 8000.
- Health endpoint returned a healthy response.
- CNN prediction endpoint returned HTTP 200.
- AST prediction endpoint returned HTTP 200.
- Invalid audio format was rejected with HTTP 400.
- Model inference ran inside the container.

The measured average CNN HTTP response time across five local requests was approximately 186 ms. This includes the HTTP request and response process, not just model execution time. The first request was slower than subsequent requests.

These results represent local development verification, not a production deployment or cloud-hosted performance benchmark.

## 13. Continuous Integration — GitHub Actions

A GitHub Actions workflow has been added at:

```text
.github/workflows/backend-ci.yml
```

The workflow is configured to run on pushes and pull requests.

Its steps include:

1. Checking out the repository.
2. Setting up Python 3.12.
3. Installing FFmpeg and `libsndfile1`.
4. Installing CPU-based PyTorch.
5. Installing backend dependencies.
6. Running the backend prediction API tests.

### CI verification status

The workflow file has been created and staged locally. Its execution on GitHub must be verified after the changes are pushed to the repository.

To verify the workflow:

1. Push the approved changes to the GitHub branch.
2. Open the repository on GitHub.
3. Select the **Actions** tab.
4. Open the **AcousticSpace Backend CI** workflow.
5. Confirm that the latest run completes successfully.

A successful local test run does not automatically mean the remote GitHub Actions workflow has passed.

## 14. Development Status Through Week 4

### Week 3 — Advanced AI, Model Inference and Evaluation

- CNN baseline training and checkpoint.
- CNN inference module.
- CNN prediction API endpoint.
- AST inference integration.
- AST evaluation experiments.
- CNN validation and development evaluation.
- Threshold comparison.
- Updated model evaluation documentation.
- Automated CNN API tests.
- Week 3 changes committed and pushed to the `KarolineUrmitha` GitHub branch.

### Week 4 — Backend, ML Deployment and API

- Finalized the current CNN deployment candidate while retaining AST for comparison.
- Organized model weights, configuration, and inference dependencies.
- Added the root-level Dockerfile.
- Added Docker build exclusions through `.dockerignore`.
- Containerized the FastAPI and ML inference service.
- Verified container health.
- Tested CNN and AST prediction endpoints inside Docker.
- Verified invalid audio handling.
- Measured local CNN API response time.
- Added the GitHub Actions backend CI workflow.
- Verified 18 targeted backend/API tests locally.
- Remote GitHub Actions execution remains pending verification.

## 15. Future Work

- Run and evaluate the separate attack-aware AST fine-tuning experiment.
- Perform additional model calibration on suitable validation data.
- Evaluate the selected model on an independent test set.
- Improve inference latency and resource usage.
- Investigate robust RIR-based feature integration.
- Strengthen generalization across recording environments and spoofing methods.
- Add verified timestamp-level suspicious-segment localization if supported by future model development.
- Verify the GitHub Actions workflow on the remote repository.
- Explore cloud deployment and production monitoring.
- Document further deployment and evaluation evidence.
