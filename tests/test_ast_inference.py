from pathlib import Path

import pytest
import torch
from transformers import ASTForAudioClassification, ASTFeatureExtractor

from audio_ml.audio_processor import load_audio


MODEL_DIR = Path("audio_ml/models/ast_binary")
AUDIO_PATH = Path(
    r"C:\Users\DELL\Documents\LA\ASVspoof2019_LA_train\flac\LA_T_1138215.flac"
)


def test_ast_inference_returns_valid_probabilities():
    if not MODEL_DIR.is_dir():
        pytest.skip("AST checkpoint is not available locally.")

    if not AUDIO_PATH.is_file():
        pytest.skip("ASVspoof test audio is not available locally.")

    model = ASTForAudioClassification.from_pretrained(MODEL_DIR)
    extractor = ASTFeatureExtractor.from_pretrained(MODEL_DIR)
    model.eval()

    audio, sample_rate = load_audio(str(AUDIO_PATH))
    inputs = extractor(
        audio,
        sampling_rate=sample_rate,
        return_tensors="pt",
    )

    with torch.no_grad():
        logits = model(input_values=inputs["input_values"]).logits
        probabilities = torch.softmax(logits, dim=-1)

    assert logits.shape == (1, 2)
    assert probabilities.shape == (1, 2)
    assert torch.isfinite(probabilities).all()
    assert torch.allclose(
        probabilities.sum(dim=-1),
        torch.ones(1),
        atol=1e-5,
    )