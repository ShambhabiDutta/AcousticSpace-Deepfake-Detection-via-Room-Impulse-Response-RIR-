
import numpy as np
import pytest
from fastapi.testclient import TestClient

from backend import main as api


@pytest.fixture
def client():
    with TestClient(api.app) as test_client:
        yield test_client


def test_root_endpoint(client):
    response = client.get("/")

    assert response.status_code == 200
    assert response.json() == {
        "message": "AcousticSpace API is running"
    }


def test_logistic_regression_prediction(client, monkeypatch):
    # Mock audio feature extraction.
    monkeypatch.setattr(
        api,
        "extract_fixed_length_spectrogram",
        lambda path: np.zeros((64, 400), dtype=np.float32),
    )

    # Mock feature summarisation.
    monkeypatch.setattr(
        api,
        "summarize_spectrograms",
        lambda batch: np.zeros((1, 128), dtype=np.float32),
    )

    # Mock the Logistic Regression model.
    class FakeModel:
        def predict_proba(self, features):
            return np.array([[0.2, 0.8]])

    monkeypatch.setattr(api, "model", FakeModel())

    response = client.post(
        "/predict",
        files={
            "file": ("sample.wav", b"fake audio data", "audio/wav")
        },
    )

    assert response.status_code == 200

    result = response.json()

    assert result["filename"] == "sample.wav"
    assert result["model"] == "Logistic Regression"
    assert result["prediction"] == "spoof"
    assert result["status"] == "success"
    assert result["class_probabilities"]["real"] == 0.2
    assert result["class_probabilities"]["spoof"] == 0.8
    assert result["analysis"]["threshold"] == api.PREDICTION_THRESHOLD


def test_ast_prediction(client, monkeypatch):
    # Mock AST inference so the test does not load the AST checkpoint.
    monkeypatch.setattr(
        api,
        "predict_ast_file",
        lambda path: 0.8,
    )

    response = client.post(
        "/predict/ast",
        files={
            "file": ("sample.wav", b"fake audio data", "audio/wav")
        },
    )

    assert response.status_code == 200

    result = response.json()

    assert result["filename"] == "sample.wav"
    assert result["model"] == "AST"
    assert result["prediction"] == "spoof"
    assert result["status"] == "success"
    assert result["class_probabilities"]["spoof"] == 0.8
    assert result["analysis"]["threshold"] == api.AST_PREDICTION_THRESHOLD


def test_unsupported_file_format(client):
    response = client.post(
        "/predict/ast",
        files={
            "file": ("sample.txt", b"not audio", "text/plain")
        },
    )

    assert response.status_code == 400


def test_empty_audio_upload(client):
    response = client.post(
        "/predict/ast",
        files={
            "file": ("empty.wav", b"", "audio/wav")
        },
    )

    assert response.status_code == 400


def test_file_size_limit(client, monkeypatch):
    # Use a small limit for this test instead of uploading 20 MB.
    monkeypatch.setattr(api, "MAX_FILE_SIZE", 3)

    response = client.post(
        "/predict/ast",
        files={
            "file": ("large.wav", b"1234", "audio/wav")
        },
    )

    assert response.status_code == 413


def test_invalid_audio_returns_bad_request(client, monkeypatch):
    # Simulate an audio decoding failure in the AST inference function.
    def raise_decode_error(path):
        raise api.AudioDecodeError("Invalid audio file.")

    monkeypatch.setattr(
        api,
        "predict_ast_file",
        raise_decode_error,
    )

    response = client.post(
        "/predict/ast",
        files={
            "file": ("broken.wav", b"invalid audio", "audio/wav")
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid audio file."
def test_cnn_prediction(client, monkeypatch):
    # Mock CNN inference so the test does not need to process real audio.
    monkeypatch.setattr(
        api,
        "predict_cnn_file",
        lambda path: 0.8,
    )

    response = client.post(
        "/predict/cnn",
        files={
            "file": ("sample.wav", b"fake audio data", "audio/wav")
        },
    )

    assert response.status_code == 200

    result = response.json()

    assert result["filename"] == "sample.wav"
    assert result["model"] == "CNN"
    assert result["prediction"] == "spoof"
    assert result["status"] == "success"
    assert result["class_probabilities"]["spoof"] == 0.8
    assert result["analysis"]["threshold"] == api.CNN_PREDICTION_THRESHOLD


def test_cnn_invalid_audio_returns_bad_request(client, monkeypatch):
    def raise_audio_error(path):
        raise api.CNNAudioError("Invalid CNN audio file.")

    monkeypatch.setattr(
        api,
        "predict_cnn_file",
        raise_audio_error,
    )

    response = client.post(
        "/predict/cnn",
        files={
            "file": ("broken.wav", b"invalid audio", "audio/wav")
        },
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "Invalid CNN audio file."


def test_cnn_model_failure_returns_server_error(client, monkeypatch):
    def raise_model_error(path):
        raise api.CNNModelError("CNN checkpoint unavailable.")

    monkeypatch.setattr(
        api,
        "predict_cnn_file",
        raise_model_error,
    )

    response = client.post(
        "/predict/cnn",
        files={
            "file": ("sample.wav", b"fake audio data", "audio/wav")
        },
    )

    assert response.status_code == 500
    assert response.json()["detail"] == (
        "CNN model is unavailable or could not complete prediction."
    )
