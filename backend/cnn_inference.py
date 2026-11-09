from functools import lru_cache
from pathlib import Path

import numpy as np
import torch

from audio_ml.feature_extractor import extract_fixed_length_spectrogram
from backend.models.cnn_model import BaselineCNN


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CNN_MODEL_PATH = PROJECT_ROOT / "audio_ml" / "models" / "cnn_baseline.pt"

DEVICE = torch.device("cpu")
EXPECTED_INPUT_SHAPE = [1, 64, 400]
EXPECTED_CLASS_MAPPING = {"real": 0, "spoof": 1}


class CNNModelError(RuntimeError):
    """Raised when the CNN checkpoint cannot be loaded."""


class CNNAudioError(ValueError):
    """Raised when uploaded audio cannot be processed."""


@lru_cache(maxsize=1)
def load_cnn_model():
    """Load and cache the trained CNN checkpoint."""

    if not CNN_MODEL_PATH.is_file():
        raise CNNModelError(
            f"CNN checkpoint not found: {CNN_MODEL_PATH}"
        )

    try:
        checkpoint = torch.load(
            CNN_MODEL_PATH,
            map_location=DEVICE,
            weights_only=True,
        )

        if checkpoint.get("num_classes") != 2:
            raise CNNModelError("CNN checkpoint must contain two classes.")

        if checkpoint.get("input_shape") != EXPECTED_INPUT_SHAPE:
            raise CNNModelError("CNN checkpoint input shape is unexpected.")

        if checkpoint.get("class_mapping") != EXPECTED_CLASS_MAPPING:
            raise CNNModelError("CNN checkpoint class mapping is unexpected.")

        model = BaselineCNN(num_classes=2)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.to(DEVICE)
        model.eval()

        return model

    except CNNModelError:
        raise
    except Exception as exc:
        raise CNNModelError(
            "Could not load the trained CNN checkpoint."
        ) from exc


def predict_cnn_file(file_path: str | Path) -> float:
    """Return the CNN's probability that an audio file is spoof."""

    try:
        spectrogram = extract_fixed_length_spectrogram(str(file_path))
    except Exception as exc:
        raise CNNAudioError(
            "Could not decode this file as valid audio."
        ) from exc

    if spectrogram.shape != (64, 400):
        raise CNNAudioError(
            f"Unexpected spectrogram shape: {spectrogram.shape}"
        )

    # Match the normalization used during CNN training.
    normalized = np.clip(
        (np.asarray(spectrogram, dtype=np.float32) + 80.0) / 80.0,
        0.0,
        1.0,
    )

    # Shape: [batch, channel, mel bands, time frames].
    features = torch.from_numpy(normalized).unsqueeze(0).unsqueeze(0)

    model = load_cnn_model()

    with torch.no_grad():
        logits = model(features.to(DEVICE))
        probabilities = torch.softmax(logits, dim=1)[0]

    spoof_probability = float(probabilities[1].item())

    if not np.isfinite(spoof_probability):
        raise CNNModelError("CNN returned an invalid probability.")

    if not 0.0 <= spoof_probability <= 1.0:
        raise CNNModelError("CNN probability is outside the valid range.")

    return spoof_probability