
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import ConfusionMatrixDisplay, confusion_matrix


def save_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_path: str | Path,
) -> Path:
    """Save a confusion matrix plot for binary real/spoof predictions."""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    if y_true.size == 0:
        raise ValueError("Evaluation data cannot be empty.")

    if y_true.shape != y_pred.shape:
        raise ValueError("True labels and predictions must have equal lengths.")

    if not set(np.unique(y_true)).issubset({0, 1}):
        raise ValueError("True labels must be 0 (real) or 1 (spoof).")

    if not set(np.unique(y_pred)).issubset({0, 1}):
        raise ValueError("Predictions must be 0 (real) or 1 (spoof).")

    matrix = confusion_matrix(y_true, y_pred, labels=[0, 1])
    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=["Real", "Spoof"],
    )

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    figure, axis = plt.subplots()
    display.plot(ax=axis, values_format="d", colorbar=False)
    axis.set_title("Real vs Spoof — Confusion Matrix")
    figure.tight_layout()
    figure.savefig(output, dpi=150)
    plt.close(figure)

    return output
