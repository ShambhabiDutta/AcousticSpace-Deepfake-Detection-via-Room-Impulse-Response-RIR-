
import io
import wave

import pytest
from fastapi.testclient import TestClient

from backend.main import app


client = TestClient(app)

MAX_FILE_SIZE = 20 * 1024 * 1024


def create_valid_wav():
    """Create a small valid WAV file in memory."""
    buffer = io.BytesIO()

    with wave.open(buffer, "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(16000)
        audio.writeframes(b"\x00\x00" * 16000)

    return buffer.getvalue()


def test_root():
    response = client.get("/")

    assert response.status_code == 200


def test_health():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json()["status"] == "healthy"


def test_valid_audio_upload():
    response = client.post(
        "/upload",
        files={
            "file": (
                "sample.wav",
                create_valid_wav(),
                "audio/wav",
            )
        },
    )

    assert response.status_code == 200


def test_unsupported_format():
    response = client.post(
        "/upload",
        files={
            "file": (
                "sample.txt",
                b"sample data",
                "text/plain",
            )
        },
    )

    assert response.status_code == 400


def test_empty_audio_upload():
    response = client.post(
        "/upload",
        files={
            "file": (
                "empty.wav",
                b"",
                "audio/wav",
            )
        },
    )

    assert response.status_code == 400


def test_file_too_large():
    response = client.post(
        "/upload",
        files={
            "file": (
                "large.wav",
                b"x" * (MAX_FILE_SIZE + 1),
                "audio/wav",
            )
        },
    )

    assert response.status_code == 413


def test_predict_rejects_invalid_audio():
    response = client.post(
        "/predict",
        files={
            "file": (
                "sample.wav",
                b"sample audio data",
                "audio/wav",
            )
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == (
        "Could not decode this file as valid audio."
    )
