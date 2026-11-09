from pathlib import Path
import json

import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)

from backend.models.cnn_model import BaselineCNN


PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "asvspoof2019_la_dev_v2"
)

MODEL_PATH = (
    PROJECT_ROOT
    / "audio_ml"
    / "models"
    / "cnn_baseline.pt"
)

OUTPUT_PATH = (
    PROJECT_ROOT
    / "audio_ml"
    / "evaluation"
    / "cnn_dev_evaluation.json"
)

BATCH_SIZE = 8
DEVICE = torch.device("cpu")

THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70]


def main():
    torch.set_num_threads(2)

    # Load development data using memory mapping.
    features = np.load(
        DATA_DIR / "features.npy",
        mmap_mode="r",
    )

    labels = np.load(DATA_DIR / "labels.npy")

    if features.shape[1:] != (64, 400):
        raise ValueError(f"Unexpected feature shape: {features.shape}")

    if len(features) != len(labels):
        raise ValueError("Feature and label counts do not match.")

    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must be 0 (real) or 1 (spoof).")

    # Load the existing trained CNN.
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {MODEL_PATH}")

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
        weights_only=True,
    )

    model = BaselineCNN(num_classes=2)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(DEVICE)
    model.eval()

    actual = []
    spoof_probabilities = []

    # Evaluate every development recording.
    with torch.no_grad():
        for start in range(0, len(labels), BATCH_SIZE):
            end = min(start + BATCH_SIZE, len(labels))

            batch = np.array(
                features[start:end],
                dtype=np.float32,
                copy=True,
            )

            # Use the same preprocessing as CNN training.
            batch = np.clip(
                (batch + 80.0) / 80.0,
                0.0,
                1.0,
            )

            inputs = torch.from_numpy(batch).unsqueeze(1)

            logits = model(inputs.to(DEVICE))
            probabilities = torch.softmax(logits, dim=1)

            actual.extend(labels[start:end].tolist())

            spoof_probabilities.extend(
                probabilities[:, 1].cpu().numpy().tolist()
            )

    actual = np.asarray(actual, dtype=np.int64)
    spoof_probabilities = np.asarray(
        spoof_probabilities,
        dtype=np.float64,
    )

    if not np.isfinite(spoof_probabilities).all():
        raise ValueError("CNN returned invalid probabilities.")

    if not np.isin(actual, [0, 1]).all():
        raise ValueError("Unexpected development labels.")

    auc = float(
        roc_auc_score(actual, spoof_probabilities)
    )

    results = []

    for threshold in THRESHOLDS:
        predicted = (
            spoof_probabilities >= threshold
        ).astype(np.int64)

        tn, fp, fn, tp = confusion_matrix(
            actual,
            predicted,
            labels=[0, 1],
        ).ravel()

        results.append({
            "threshold": threshold,
            "accuracy": float(
                accuracy_score(actual, predicted)
            ),
            "balanced_accuracy": float(
                balanced_accuracy_score(actual, predicted)
            ),
            "spoof_precision": float(
                precision_score(actual, predicted, zero_division=0)
            ),
            "spoof_recall": float(
                recall_score(actual, predicted, zero_division=0)
            ),
            "spoof_f1": float(
                f1_score(actual, predicted, zero_division=0)
            ),
            "real_recall": float(
                recall_score(
                    actual,
                    predicted,
                    pos_label=0,
                    zero_division=0,
                )
            ),
            "confusion_matrix": {
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
            },
        })

    output = {
        "model": "CNN",
        "dataset": "ASVspoof2019 LA development",
        "total_recordings": int(len(actual)),
        "real_recordings": int(np.sum(actual == 0)),
        "spoof_recordings": int(np.sum(actual == 1)),
        "roc_auc": auc,
        "checkpoint_validation_balanced_accuracy": float(
            checkpoint["validation_balanced_accuracy"]
        ),
        "threshold_results": results,
        "note": (
            "This is a separate development-set evaluation. "
            "It is not an independent final test-set result."
        ),
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=4)

    print("\nCNN DEVELOPMENT-SET EVALUATION")
    print("--------------------------------")
    print("Total recordings:", len(actual))
    print("Real recordings:", int(np.sum(actual == 0)))
    print("Spoof recordings:", int(np.sum(actual == 1)))
    print("ROC-AUC:", round(auc, 4))

    print("\nTHRESHOLD COMPARISON")

    for result in results:
        print(
            f"Threshold: {result['threshold']:.2f} | "
            f"Accuracy: {result['accuracy']:.4f} | "
            f"Balanced accuracy: {result['balanced_accuracy']:.4f} | "
            f"Real recall: {result['real_recall']:.4f} | "
            f"Spoof recall: {result['spoof_recall']:.4f}"
        )

    print("\nSaved report:", OUTPUT_PATH)


if __name__ == "__main__":
    main()