
from functools import lru_cache
from pathlib import Path

import librosa
import numpy as np
import torch
from transformers import ASTFeatureExtractor, ASTForAudioClassification


PROJECT_ROOT = Path(__file__).resolve().parent.parent
AST_MODEL_DIR = PROJECT_ROOT / "audio_ml" / "models" / "ast_binary"

SAMPLE_RATE = 16000
DEVICE = torch.device("cpu")


class AudioDecodeError(Exception):
    """Raised when an uploaded file cannot be decoded as audio."""


@lru_cache(maxsize=1)
def load_ast_components():
    """Load and cache the AST model and feature extractor on first use."""

    if not AST_MODEL_DIR.is_dir():
        raise FileNotFoundError(
            f"AST model directory not found: {AST_MODEL_DIR}"
        )

    extractor = ASTFeatureExtractor.from_pretrained(
        str(AST_MODEL_DIR)
    )

    model = ASTForAudioClassification.from_pretrained(
        str(AST_MODEL_DIR)
    )

    model.to(DEVICE)
    model.eval()

    return extractor, model


def predict_ast_file(audio_path: str | Path) -> float:
    """Return the AST model's probability that an audio recording is spoof."""

    # Decode audio using the same sampling rate and mono setting as training.
    try:
        waveform, _ = librosa.load(
            str(audio_path),
            sr=SAMPLE_RATE,
            mono=True,
        )
    except Exception as exc:
        raise AudioDecodeError(
            "Could not decode this file as valid audio."
        ) from exc

    if waveform.size == 0 or not np.isfinite(waveform).all():
        raise AudioDecodeError(
            "The audio file is empty or contains invalid values."
        )

    # Load the saved model only when AST inference is requested.
    extractor, model = load_ast_components()

    features = extractor(
        waveform,
        sampling_rate=SAMPLE_RATE,
        return_tensors="pt",
    )["input_values"]

    features = features.to(DEVICE)

    with torch.inference_mode():
        outputs = model(input_values=features)
        probabilities = torch.softmax(outputs.logits, dim=1)

    # Label index 1 represents spoof.
    spoof_probability = float(probabilities[0, 1].item())

    if not np.isfinite(spoof_probability):
        raise RuntimeError("AST returned an invalid prediction score.")

    return spoof_probability
