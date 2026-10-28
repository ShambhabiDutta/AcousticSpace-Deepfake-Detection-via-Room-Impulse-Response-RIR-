
import sys
import tempfile
from pathlib import Path
from uuid import uuid4

import joblib
import numpy as np
from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

# Add the project root so audio_ml imports work when Uvicorn
# is started from either the project root or backend directory.
PROJECT_ROOT = Path(__file__).resolve().parent.parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from audio_ml.feature_extractor import extract_fixed_length_spectrogram
from audio_ml.train_baseline import summarize_spectrograms


app = FastAPI(
    title="AcousticSpace API",
    description="Backend API for audio deepfake detection research",
    version="0.2.0",
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

# Paths and model settings
UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = PROJECT_ROOT / "audio_ml" / "models" / "baseline_logreg.joblib"
PREDICTION_THRESHOLD = 0.70

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg"}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB

if not MODEL_PATH.is_file():
    raise FileNotFoundError(f"Trained model not found: {MODEL_PATH}")

# Load the trained model once when the API starts.
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
    temp_path = None

    try:
        original_name, extension, content = await read_audio_upload(file)

        # Save temporarily so the existing audio preprocessing pipeline
        # can decode the file. The temporary file is deleted afterward.
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

        if not np.isfinite(spoof_probability):
            raise HTTPException(
                status_code=500,
                detail="The model returned an invalid prediction score.",
            )

        prediction = (
            "spoof"
            if spoof_probability >= PREDICTION_THRESHOLD
            else "real"
        )

        return {
            "filename": original_name,
            "prediction": prediction,
            "spoof_probability": round(spoof_probability, 4),
            "threshold": PREDICTION_THRESHOLD,
            "status": "success",
        }

    finally:
        await file.close()

        if temp_path is not None:
            temp_path.unlink(missing_ok=True)
