from pathlib import Path
import json

import numpy as np
import torch

from sklearn.model_selection import train_test_split
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
    / "asvspoof2019_la_train_v2"
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
    / "cnn_threshold_analysis.json"
)

SEED = 42
BATCH_SIZE = 8
DEVICE = torch.device("cpu")

THRESHOLDS = [0.30, 0.40, 0.50, 0.60, 0.70]


def main():
    torch.set_num_threads(2)

    # Load the existing labelled dataset.
    features = np.load(
        DATA_DIR / "features.npy",
        mmap_mode="r",
    )
    labels = np.load(DATA_DIR / "labels.npy")

    if features.shape[1:] != (64, 400):
        raise ValueError(f"Unexpected feature shape: {features.shape}")

    if len(features) != len(labels):
        raise ValueError("Feature and label counts do not match.")

    # Reproduce the exact training/validation split.
    indices = np.arange(len(labels))

    _, val_indices = train_test_split(
        indices,
        test_size=0.2,
        random_state=SEED,
        stratify=labels,
    )

    # Load the saved best CNN checkpoint.
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

    # Run inference on the original validation samples.
    with torch.no_grad():
        for start in range(0, len(val_indices), BATCH_SIZE):
            batch_indices = val_indices[start:start + BATCH_SIZE]

            batch = np.array(
                features[batch_indices],
                dtype=np.float32,
                copy=True,
            )

            # Match training preprocessing exactly.
            batch = np.clip(
                (batch + 80.0) / 80.0,
                0.0,
                1.0,
            )

            inputs = torch.from_numpy(batch).unsqueeze(1)

            logits = model(inputs.to(DEVICE))
            probabilities = torch.softmax(logits, dim=1)

            actual.extend(labels[batch_indices].tolist())
            spoof_probabilities.extend(
                probabilities[:, 1].cpu().numpy().tolist()
            )

    actual = np.asarray(actual, dtype=np.int64)
    spoof_probabilities = np.asarray(
        spoof_probabilities,
        dtype=np.float64,
    )

    if not np.isfinite(spoof_probabilities).all():
        raise ValueError("Invalid probabilities returned by CNN.")

    # Evaluate the requested thresholds.
    threshold_results = []

    for threshold in THRESHOLDS:
        predicted = (
            spoof_probabilities >= threshold
        ).astype(np.int64)

        tn, fp, fn, tp = confusion_matrix(
            actual,
            predicted,
            labels=[0, 1],
        ).ravel()

        result = {
            "threshold": threshold,
            "accuracy": float(accuracy_score(actual, predicted)),
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
            "confusion_matrix": {
                "true_negative": int(tn),
                "false_positive": int(fp),
                "false_negative": int(fn),
                "true_positive": int(tp),
            },
        }

        threshold_results.append(result)

    auc = float(
        roc_auc_score(actual, spoof_probabilities)
    )

    best_result = max(
        threshold_results,
        key=lambda item: item["balanced_accuracy"],
    )

    output = {
        "model": "CNN",
        "validation_samples": int(len(actual)),
        "random_seed": SEED,
        "validation_split": "stratified 80/20",
        "class_mapping": {
            "real": 0,
            "spoof": 1,
        },
        "checkpoint_validation_balanced_accuracy": float(
            checkpoint["validation_balanced_accuracy"]
        ),
        "roc_auc": auc,
        "threshold_results": threshold_results,
        "best_threshold_among_tested_values": best_result,
        "note": (
            "Threshold selection is based on the validation set. "
            "Independent test data is needed for an unbiased final "
            "performance estimate."
        ),
    }

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with OUTPUT_PATH.open("w", encoding="utf-8") as file:
        json.dump(output, file, indent=4)

    print("\nCNN VALIDATION RESULTS")
    print("---------------------")
    print("Validation samples:", len(actual))
    print("ROC-AUC:", round(auc, 4))

    print("\nTHRESHOLD COMPARISON")

    for result in threshold_results:
        print(
            f"Threshold: {result['threshold']:.2f} | "
            f"Accuracy: {result['accuracy']:.4f} | "
            f"Balanced accuracy: "
            f"{result['balanced_accuracy']:.4f} | "
            f"Spoof precision: "
            f"{result['spoof_precision']:.4f} | "
            f"Spoof recall: "
            f"{result['spoof_recall']:.4f}"
        )

    print("\nBest tested threshold:", best_result["threshold"])
    print(
        "Best balanced accuracy:",
        round(best_result["balanced_accuracy"], 4),
    )
    print("\nSaved results to:", OUTPUT_PATH)


if __name__ == "__main__":
    main()