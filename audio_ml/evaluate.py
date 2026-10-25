
from pathlib import Path

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)


def evaluate_predictions(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    output_dir: str | Path | None = None,
) -> dict[str, object]:
    """Calculate binary classification metrics and save results."""
    y_true = np.asarray(y_true).reshape(-1)
    y_pred = np.asarray(y_pred).reshape(-1)

    if y_true.size == 0:
        raise ValueError("Evaluation data cannot be empty.")

    if y_true.shape != y_pred.shape:
        raise ValueError("True labels and predictions must have equal lengths.")

    valid_labels = {0, 1}
    if not set(np.unique(y_true)).issubset(valid_labels):
        raise ValueError("True labels must be 0 (real) or 1 (spoof).")
    if not set(np.unique(y_pred)).issubset(valid_labels):
        raise ValueError("Predictions must be 0 (real) or 1 (spoof).")

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=[0, 1],
        average="binary",
        pos_label=1,
        zero_division=0,
    )

    metrics = {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1),
        "confusion_matrix": confusion_matrix(
            y_true, y_pred, labels=[0, 1]
        ).tolist(),
        "classification_report": classification_report(
            y_true,
            y_pred,
            labels=[0, 1],
            target_names=["real", "spoof"],
            zero_division=0,
        ),
    }

    if output_dir is not None:
        directory = Path(output_dir)
        directory.mkdir(parents=True, exist_ok=True)

        np.save(
            directory / "confusion_matrix.npy",
            np.asarray(metrics["confusion_matrix"]),
        )

        with (directory / "evaluation_metrics.txt").open(
            "w", encoding="utf-8"
        ) as file:
            for key, value in metrics.items():
                file.write(f"{key}:\n{value}\n\n")

    return metrics
