
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile

app = FastAPI(
    title="AcousticSpace API",
    description="Backend API for audio deepfake detection research",
    version="0.1.0",
)

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

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
    extension = Path(file.filename or "").suffix.lower()

    if extension not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail="Unsupported audio format. Use WAV, MP3, FLAC, or OGG.",
        )

    try:
        content = await file.read(MAX_FILE_SIZE + 1)

        if not content:
            raise HTTPException(status_code=400, detail="The uploaded file is empty.")

        if len(content) > MAX_FILE_SIZE:
            raise HTTPException(
                status_code=413,
                detail="File is too large. Maximum size is 20 MB.",
            )

        safe_name = Path(file.filename or "audio").name
        destination = UPLOAD_DIR / safe_name

        # Avoid overwriting an existing upload.
        if destination.exists():
            destination = UPLOAD_DIR / f"{destination.stem}_{__import__('uuid').uuid4().hex[:8]}{destination.suffix}"

        destination.write_bytes(content)

        return {
            "message": "Audio uploaded successfully",
            "filename": destination.name,
            "size_bytes": len(content),
            "status": "uploaded",
        }
    finally:
        await file.close()
