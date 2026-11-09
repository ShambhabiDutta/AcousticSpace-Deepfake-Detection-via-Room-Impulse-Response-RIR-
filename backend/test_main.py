

import io
import wave

import numpy as np
import pytest
from fastapi.testclient import TestClient

import backend.main as main_module
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
    assert response.json()["message"] == "AcousticSpace API is running"


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

    result = response.json()

    assert result["message"] == "Audio uploaded successfully"
    assert result["filename"] == "sample.wav"
    assert result["status"] == "uploaded"
    assert result["size_bytes"] > 0


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
    assert response.json()["detail"] == "The uploaded file is empty."


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


def test_successful_prediction(monkeypatch):
    """Test a successful prediction using a mocked model."""

    class MockModel:
        def predict_proba(self, X):
            return np.array([[0.2, 0.8]])

    monkeypatch.setattr(
        main_module,
        "extract_fixed_length_spectrogram",
        lambda path: np.zeros((64, 400), dtype=np.float32),
    )

    monkeypatch.setattr(
        main_module,
        "summarize_spectrograms",
        lambda batch: np.zeros((1, 128), dtype=np.float32),
    )

    monkeypatch.setattr(
        main_module,
        "model",
        MockModel(),
    )

    response = client.post(
        "/predict",
        files={
            "file": (
                "sample.wav",
                create_valid_wav(),
                "audio/wav",
            )
        },
    )

    assert response.status_code == 200

    result = response.json()

    assert result["filename"] == "sample.wav"
    assert result["prediction"] == "spoof"
    assert result["spoof_probability"] == 0.8
    assert result["analysis"]["threshold"] == 0.70
    assert result["status"] == "success"
