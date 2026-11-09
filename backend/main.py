
import sys
import tempfile
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# Add the project root so imports work when Uvicorn is started
# from either the project root or the backend directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audio_ml.feature_extractor import extract_fixed_length_spectrogram
from audio_ml.train_baseline import summarize_spectrograms
from backend.ast_inference import AudioDecodeError, predict_ast_file
from backend.cnn_inference import (
    CNNAudioError,
    CNNModelError,
    predict_cnn_file,
)


app = FastAPI(
    title="AcousticSpace API",
    description="Backend API for audio deepfake detection research",
    version="0.3.0",
)

# Allow the local React frontend to communicate with FastAPI.
app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:5173",
        "http://127.0.0.1:5173",
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths and model settings.
UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = PROJECT_ROOT / "audio_ml" / "models" / "baseline_logreg.joblib"

# Existing Logistic Regression threshold.
PREDICTION_THRESHOLD = 0.70

# Existing AST threshold.
AST_PREDICTION_THRESHOLD = 0.50

# CNN threshold is provisional and must be validated.
CNN_PREDICTION_THRESHOLD = 0.50

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg"}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB

if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")

# Load the Logistic Regression model once when the API starts.
# AST and CNN models are loaded separately by their inference modules.
model = joblib.load(MODEL_PATH)


def validate_filename(filename: str | None) -> tuple[str, str]:
    original_name = Path(filename or "").name
    extension = Path(original_name).suffix.lower()

    if not original_name or extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported audio format. Use WAV, MP3, FLAC, or OGG.",
        )

    return original_name, extension


async def read_audio_upload(file: UploadFile) -> tuple[str, str, bytes]:
    original_name, extension = validate_filename(file.filename)

    content = await file.read(MAX_FILE_SIZE + 1)

    if not content:
        raise HTTPException(
            status_code=400,
            detail="The uploaded file is empty.",
        )

    if len(content) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail="File is too large. Maximum size is 20 MB.",
        )

    return original_name, extension, content


def validate_probability(probability: float, model_name: str) -> None:
    """Ensure the model returned a valid spoof probability."""

    if not np.isfinite(probability) or not 0.0 <= probability <= 1.0:
        raise HTTPException(
            status_code=500,
            detail=f"{model_name} returned an invalid prediction probability.",
        )


def build_prediction_response(
    filename: str,
    model_name: str,
    spoof_probability: float,
    threshold: float,
) -> dict:
    """Build a consistent prediction response for all models."""

    prediction = (
        "spoof"
        if spoof_probability >= threshold
        else "real"
    )

    real_probability = 1.0 - spoof_probability

    confidence = (
        spoof_probability
        if prediction == "spoof"
        else real_probability
    )

    return {
        "filename": filename,
        "model": model_name,
        "prediction": prediction,
        "class_probabilities": {
            "real": round(real_probability, 4),
            "spoof": round(spoof_probability, 4),
        },
        "spoof_probability": round(spoof_probability, 4),
        "confidence": round(confidence, 4),
        "analysis": {
            "threshold": threshold,
            "classification_rule": (
                "spoof_probability >= threshold means spoof; "
                "otherwise real"
            ),
        },
        "status": "success",
    }


@app.get("/")
def root():
    return {"message": "AcousticSpace API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    try:
        original_name, extension, content = await read_audio_upload(file)

        saved_name = f"{uuid4().hex}{extension}"
        destination = UPLOAD_DIR / saved_name
        destination.write_bytes(content)

        return {
            "message": "Audio uploaded successfully",
            "filename": original_name,
            "saved_filename": saved_name,
            "size_bytes": len(content),
            "status": "uploaded",
        }

    finally:
        await file.close()


@app.post("/predict")
async def predict_audio(file: UploadFile = File(...)):
    """Logistic Regression audio deepfake prediction endpoint."""

    temp_path = None

    try:
        original_name, extension, content = await read_audio_upload(file)

        # Save temporarily for audio preprocessing.
        with tempfile.NamedTemporaryFile(
            suffix=extension,
            delete=False,
        ) as temp_file:
            temp_file.write(content)
            temp_path = Path(temp_file.name)

        # Produce the same 64 x 400 log-mel features used in training.
        try:
            spectrogram = extract_fixed_length_spectrogram(str(temp_path))

        except Exception as exc:
            raise HTTPException(
                status_code=400,
                detail="Could not decode this file as valid audio.",
            ) from exc

        # The trained model expects 128 features per recording:
        # 64 mel-band means + 64 mel-band standard deviations.
        try:
            batch = np.expand_dims(spectrogram, axis=0)
            X = summarize_spectrograms(batch)

            spoof_probability = float(model.predict_proba(X)[0, 1])

        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail="Prediction failed while processing the audio.",
            ) from exc

        validate_probability(spoof_probability, "Logistic Regression")

        return build_prediction_response(
            filename=original_name,
            model_name="Logistic Regression",
            spoof_probability=spoof_probability,
            threshold=PREDICTION_THRESHOLD,
        )

    finally:
        await file.close()

        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


@app.post("/predict/ast")
async def predict_audio_ast(file: UploadFile = File(...)):
    """AST-based audio deepfake prediction endpoint."""

    temp_path = None

    try:
        original_name, extension, content = await read_audio_upload(file)

        # Save the uploaded audio temporarily.
        with tempfile.NamedTemporaryFile(
            suffix=extension,
            delete=False,
        ) as temp_file:
            temp_file.write(content)
            temp_path = Path(temp_file.name)

        # Run inference using the separately saved AST model.
        try:
            spoof_probability = float(predict_ast_file(temp_path))

        except AudioDecodeError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail="AST prediction failed while processing the audio.",
            ) from exc

        validate_probability(spoof_probability, "AST")

        return build_prediction_response(
            filename=original_name,
            model_name="AST",
            spoof_probability=spoof_probability,
            threshold=AST_PREDICTION_THRESHOLD,
        )

    finally:
        await file.close()

        if temp_path is not None:
            temp_path.unlink(missing_ok=True)


@app.post("/predict/cnn")
async def predict_audio_cnn(file: UploadFile = File(...)):
    """CNN-based audio deepfake prediction endpoint."""

    temp_path = None

    try:
        original_name, extension, content = await read_audio_upload(file)

        # Save the uploaded audio temporarily.
        with tempfile.NamedTemporaryFile(
            suffix=extension,
            delete=False,
        ) as temp_file:
            temp_file.write(content)
            temp_path = Path(temp_file.name)

        # Run inference using the trained CNN model.
        try:
            spoof_probability = float(predict_cnn_file(temp_path))

        except CNNAudioError as exc:
            raise HTTPException(
                status_code=400,
                detail=str(exc),
            ) from exc

        except CNNModelError as exc:
            raise HTTPException(
                status_code=500,
                detail="CNN model is unavailable or could not complete prediction.",
            ) from exc

        except Exception as exc:
            raise HTTPException(
                status_code=500,
                detail="CNN prediction failed while processing the audio.",
            ) from exc

        validate_probability(spoof_probability, "CNN")

        return build_prediction_response(
            filename=original_name,
            model_name="CNN",
            spoof_probability=spoof_probability,
            threshold=CNN_PREDICTION_THRESHOLD,
        )

    finally:
        await file.close()

        if temp_path is not None:
            temp_path.unlink(missing_ok=True)