
import torch

from backend.models.cnn_model import BaselineCNN


def test_cnn_output_shape():
    model = BaselineCNN(num_classes=2)
    sample_spectrograms = torch.randn(2, 1, 128, 128)

    outputs = model(sample_spectrograms)

    assert outputs.shape == (2, 2)


def test_cnn_outputs_finite_values():
    model = BaselineCNN(num_classes=2)
    sample_spectrograms = torch.randn(2, 1, 128, 128)

    outputs = model(sample_spectrograms)

    assert torch.isfinite(outputs).all()
