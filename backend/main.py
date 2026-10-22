
from pathlib import Path
from uuid import uuid4

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(
    title="AcousticSpace API",
    description="Backend API for audio deepfake detection research",
    version="0.1.0",
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

# Upload settings
UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg"}
MAX_FILE_SIZE = 20 * 1024 * 1024  # 20 MB


@app.get("/")
def root():
    return {"message": "AcousticSpace API is running"}


@app.get("/health")
def health():
    return {"status": "healthy"}


@app.post("/upload")
async def upload_audio(file: UploadFile = File(...)):
    # Validate the filename and extension.
    original_name = Path(file.filename or "").name
    extension = Path(original_name).suffix.lower()

    if not original_name or extension not in ALLOWED_EXTENSIONS:
        await file.close()
        raise HTTPException(
            status_code=400,
            detail="Unsupported audio format. Use WAV, MP3, FLAC, or OGG.",
        )

    try:
        # Read at most 20 MB plus one byte to detect oversized files.
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

        # Generate a unique filename to avoid overwriting uploads.
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
    await file.close()
    raise HTTPException(
        status_code=503,
        detail=(
            "Prediction is unavailable until the trained model "
            "and audio preprocessing are configured."
        ),
    )

