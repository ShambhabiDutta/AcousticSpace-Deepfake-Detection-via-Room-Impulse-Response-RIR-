
from pathlib import Path
import sys

import torch

# Add the project root to Python's import path.
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from backend.models.cnn_model import BaselineCNN


def main():
    models_dir = PROJECT_ROOT / "backend" / "models"

    print("AcousticSpace baseline CNN training setup")
    print(f"PyTorch version: {torch.__version__}")
    print(f"Model directory: {models_dir}")

    model = BaselineCNN(num_classes=2)
    parameter_count = sum(
        parameter.numel() for parameter in model.parameters()
    )

    print(f"Model parameters: {parameter_count:,}")
    print("Model initialized successfully.")
    print("Training has not started: labeled data is required.")


if __name__ == "__main__":
    main()

