
from pathlib import Path

from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_root():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json()["message"] == "AcousticSpace API is running"


def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "healthy"}


def test_valid_audio_upload():
    response = client.post(
        "/upload",
        files={"file": ("test.wav", b"sample audio data", "audio/wav")},
    )
    assert response.status_code == 200
    assert response.json()["status"] == "uploaded"
    assert response.json()["size_bytes"] == len(b"sample audio data")

    # Remove the test upload.
    uploaded_file = Path("uploads") / response.json()["filename"]
    uploaded_file.unlink(missing_ok=True)


def test_unsupported_format():
    response = client.post(
        "/upload",
        files={"file": ("test.webm", b"sample data", "audio/webm")},
    )
    assert response.status_code == 400
    assert "Unsupported audio format" in response.json()["detail"]


def test_empty_audio_upload():
    response = client.post(
        "/upload",
        files={"file": ("empty.wav", b"", "audio/wav")},
    )
    assert response.status_code == 400
    assert "empty" in response.json()["detail"].lower()


def test_file_too_large():
    large_content = b"a" * (20 * 1024 * 1024 + 1)

    response = client.post(
        "/upload",
        files={"file": ("large.wav", large_content, "audio/wav")},
    )
    assert response.status_code == 413
    assert "too large" in response.json()["detail"].lower()
