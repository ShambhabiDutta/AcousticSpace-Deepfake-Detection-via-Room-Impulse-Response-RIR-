import csv
import json
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from transformers import ASTFeatureExtractor, ASTForAudioClassification

from train_ast import (
    DATA_ROOT,
    OUTPUT_DIR,
    balanced_subset,
    read_protocol,
    ASVspoofDataset,
)

ROOT = Path(__file__).resolve().parents[2]

PROTOCOL = (
    DATA_ROOT / "ASVspoof2019_LA_cm_protocols"
    / "ASVspoof2019.LA.cm.dev.trl.txt"
)
AUDIO_DIR = DATA_ROOT / "ASVspoof2019_LA_dev" / "flac"

OUTPUT_DIR_200 = ROOT / "audio_ml" / "evaluation"
MANIFEST = OUTPUT_DIR_200 / "ast_200_manifest.csv"
PREDICTIONS = OUTPUT_DIR_200 / "ast_200_predictions.csv"
METRICS = OUTPUT_DIR_200 / "ast_200_metrics.json"

SAMPLES_PER_CLASS = 100
SEED = 42
BATCH_SIZE = 1
THRESHOLDS = [0.30, 0.50, 0.70]
DEVICE = torch.device("cpu")

NAMES = {0: "bonafide", 1: "spoof"}


def save_csv(path, fields, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")

    with temporary.open("w", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(file, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
        file.flush()

    temporary.replace(path)


def calculate_metrics(actual, probabilities, threshold):
    predicted = (probabilities >= threshold).astype(np.int64)

    return {
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
        "confusion_matrix": confusion_matrix(
            actual, predicted, labels=[0, 1]
        ).tolist(),
    }


def main():
    torch.set_num_threads(2)
    OUTPUT_DIR_200.mkdir(parents=True, exist_ok=True)

    print("AST 200-recording evaluation")
    print("Device:", DEVICE)

    if not PROTOCOL.is_file():
        raise FileNotFoundError(PROTOCOL)
    if not AUDIO_DIR.is_dir():
        raise FileNotFoundError(AUDIO_DIR)
    if not OUTPUT_DIR.is_dir():
        raise FileNotFoundError(OUTPUT_DIR)

    records = read_protocol(PROTOCOL)

    if len({audio_id for audio_id, _ in records}) != len(records):
        raise ValueError("Duplicate audio IDs in protocol.")

    print("Full dev protocol records:", len(records))

    # Deterministic selection: 100 bonafide + 100 spoof.
    records = balanced_subset(records, SAMPLES_PER_CLASS, SEED)

    if len(records) != 200:
        raise ValueError(f"Expected 200 recordings; got {len(records)}.")

    labels = [label for _, label in records]

    if labels.count(0) != 100 or labels.count(1) != 100:
        raise ValueError("The selected subset is not balanced.")

    # Save the exact selected recording list.
    manifest_rows = []

    for index, (audio_id, label) in enumerate(records, start=1):
        manifest_rows.append({
            "sample_order": index,
            "audio_id": audio_id,
            "actual_label": label,
            "actual_class": NAMES[label],
            "seed": SEED,
            "source_split": "dev",
        })

    save_csv(
        MANIFEST,
        [
            "sample_order", "audio_id", "actual_label",
            "actual_class", "seed", "source_split",
        ],
        manifest_rows,
    )

    print("Manifest saved:", MANIFEST)

    # Load the already-trained checkpoint; no training occurs here.
    extractor = ASTFeatureExtractor.from_pretrained(str(OUTPUT_DIR))
    model = ASTForAudioClassification.from_pretrained(str(OUTPUT_DIR))
    model.to(DEVICE)
    model.eval()

    dataset = ASVspoofDataset(records, AUDIO_DIR, extractor)

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    actual = []
    probabilities = []

    print("Starting inference...")

    with torch.inference_mode():
        for index, (features, labels) in enumerate(loader):
            logits = model(
                input_values=features.to(DEVICE)
            ).logits

            probability = float(
                torch.softmax(logits, dim=1)[0, 1].cpu().item()
            )

            actual_label = int(labels[0].item())
            expected_id, expected_label = records[index]

            if actual_label != expected_label:
                raise ValueError(
                    f"Label mismatch for {expected_id}"
                )

            if not np.isfinite(probability) or not 0 <= probability <= 1:
                raise ValueError(
                    f"Invalid probability for {expected_id}"
                )

            actual.append(actual_label)
            probabilities.append(probability)

            completed = index + 1

            if completed % 20 == 0 or completed == 200:
                print(f"Progress: {completed}/200", flush=True)

    actual = np.asarray(actual, dtype=np.int64)
    probabilities = np.asarray(probabilities, dtype=np.float64)

    if len(actual) != 200 or len(probabilities) != 200:
        raise RuntimeError("Prediction count is not 200.")

    auc = float(roc_auc_score(actual, probabilities))

    threshold_results = [
        calculate_metrics(actual, probabilities, threshold)
        for threshold in THRESHOLDS
    ]

    # Save per-recording predictions at threshold 0.50.
    predicted = (probabilities >= 0.50).astype(np.int64)

    prediction_rows = []

    for index, (audio_id, label) in enumerate(records):
        prediction_rows.append({
            "audio_id": audio_id,
            "actual_label": label,
            "predicted_label": int(predicted[index]),
            "spoof_probability": round(
                float(probabilities[index]), 8
            ),
            "threshold": 0.50,
        })

    save_csv(
        PREDICTIONS,
        [
            "audio_id", "actual_label", "predicted_label",
            "spoof_probability", "threshold",
        ],
        prediction_rows,
    )

    report = classification_report(
        actual,
        predicted,
        labels=[0, 1],
        target_names=["bonafide", "spoof"],
        zero_division=0,
        output_dict=True,
    )

    results = {
        "evaluation_status": "recreated_evaluation",
        "dataset": "ASVspoof2019 LA dev",
        "total_recordings": 200,
        "bonafide_count": 100,
        "spoof_count": 100,
        "seed": SEED,
        "samples_per_class": SAMPLES_PER_CLASS,
        "device": str(DEVICE),
        "label_mapping": NAMES,
        "roc_auc": auc,
        "threshold_metrics": threshold_results,
        "classification_report_at_0.50": report,
        "manifest_file": str(MANIFEST),
        "predictions_file": str(PREDICTIONS),
        "note": (
            "Recreated deterministic balanced subset from the official "
            "development protocol. Not verified as the exact original "
            "run behind the earlier report and not an independent "
            "held-out test set."
        ),
    }

    temporary = METRICS.with_suffix(".json.tmp")
    temporary.write_text(
        json.dumps(results, indent=2),
        encoding="utf-8",
    )
    temporary.replace(METRICS)

    print("\nROC-AUC:", round(auc, 4))

    for result in threshold_results:
        print("\nThreshold:", result["threshold"])
        print("Accuracy:", round(result["accuracy"], 4))
        print("Balanced accuracy:", round(result["balanced_accuracy"], 4))
        print("Spoof precision:", round(result["spoof_precision"], 4))
        print("Spoof recall:", round(result["spoof_recall"], 4))
        print("Spoof F1:", round(result["spoof_f1"], 4))
        print("Confusion matrix:", result["confusion_matrix"])

    print("\n200-recording evaluation completed.")
    print("Manifest:", MANIFEST)
    print("Predictions:", PREDICTIONS)
    print("Metrics:", METRICS)


if __name__ == "__main__":
    main()