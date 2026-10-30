from pathlib import Path

import pytest
import torch
from transformers import ASTForAudioClassification, ASTFeatureExtractor

from audio_ml.audio_processor import load_audio


MODEL_DIR = Path("audio_ml/models/ast_binary")
AUDIO_DIR = Path(
    r"C:\Users\DELL\Documents\LA\ASVspoof2019_LA_dev\flac"
)

SAMPLES = [
    ("LA_D_1047731", "bonafide"),
    ("LA_D_1008730", "spoof"),
]

LABELS = {0: "bonafide", 1: "spoof"}


def test_ast_dev_samples_return_valid_predictions():
    if not MODEL_DIR.is_dir():
        pytest.skip("AST checkpoint is not available locally.")

    available_samples = [
        (audio_id, actual, AUDIO_DIR / f"{audio_id}.flac")
        for audio_id, actual in SAMPLES
        if (AUDIO_DIR / f"{audio_id}.flac").is_file()
    ]

    if not available_samples:
        pytest.skip("No development audio samples are available.")

    model = ASTForAudioClassification.from_pretrained(MODEL_DIR).eval()
    extractor = ASTFeatureExtractor.from_pretrained(MODEL_DIR)

    for audio_id, actual, audio_path in available_samples:
        waveform, sample_rate = load_audio(str(audio_path))
        inputs = extractor(
            waveform,
            sampling_rate=sample_rate,
            return_tensors="pt",
        )

        with torch.no_grad():
            logits = model(input_values=inputs["input_values"]).logits
            probabilities = torch.softmax(logits, dim=-1)

        assert tuple(logits.shape) == (1, 2)
        assert torch.isfinite(probabilities).all()
        assert torch.allclose(
            probabilities.sum(dim=-1),
            torch.ones(1),
            atol=1e-5,
        )

        predicted_id = int(probabilities[0].argmax())
        predicted = LABELS[predicted_id]

        print(
            f"\nAudio: {audio_id}"
            f"\nActual: {actual}"
            f"\nPredicted: {predicted}"
            f"\nBonafide probability: {probabilities[0, 0].item():.4f}"
            f"\nSpoof probability: {probabilities[0, 1].item():.4f}"
        )