
import csv
from pathlib import Path

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
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
    read_protocol,
    ASVspoofDataset,
)


def main():
    torch.set_num_threads(2)
    device = torch.device("cpu")

    protocol = (
        DATA_ROOT
        / "ASVspoof2019_LA_cm_protocols"
        / "ASVspoof2019.LA.cm.dev.trl.txt"
    )
    audio_dir = DATA_ROOT / "ASVspoof2019_LA_dev" / "flac"

    records = read_protocol(protocol)

    if len(records) != 24844:
        raise ValueError(
            f"Expected 24,844 development records, found {len(records)}"
        )

    if len(set(audio_id for audio_id, _ in records)) != len(records):
        raise ValueError("Duplicate audio IDs found in development protocol")

    print("Development recordings:", len(records))
    print("Loading AST model...")

    extractor = ASTFeatureExtractor.from_pretrained(OUTPUT_DIR)
    model = ASTForAudioClassification.from_pretrained(OUTPUT_DIR)
    model.to(device)
    model.eval()

    loader = DataLoader(
        ASVspoofDataset(records, audio_dir, extractor),
        batch_size=1,
        shuffle=False,
        num_workers=0,
    )

    actual = []
    spoof_probabilities = []

    print("Starting full development evaluation...")

    with torch.inference_mode():
        for index, (features, labels) in enumerate(loader, start=1):
            logits = model(input_values=features.to(device)).logits
            probabilities = torch.softmax(logits, dim=1)

            actual.extend(labels.tolist())
            spoof_probabilities.extend(
                probabilities[:, 1].cpu().tolist()
            )

            if index % 100 == 0:
                print(f"Processed {index:,}/{len(records):,}")

    actual = np.asarray(actual)
    spoof_probabilities = np.asarray(spoof_probabilities)

    if len(actual) != len(records):
        raise ValueError("Evaluation record count does not match protocol")

    if not np.isfinite(spoof_probabilities).all():
        raise ValueError("Non-finite prediction probabilities detected")

    output_dir = Path("audio_ml/evaluation")
    output_dir.mkdir(parents=True, exist_ok=True)

    predictions_file = output_dir / "ast_full_dev_predictions.csv"

    with predictions_file.open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.writer(file)
        writer.writerow([
            "audio_id",
            "actual_label",
            "spoof_probability",
        ])

        for (audio_id, _), label, probability in zip(
            records, actual, spoof_probabilities
        ):
            writer.writerow([
                audio_id,
                int(label),
                float(probability),
            ])

    print("\nFULL DEVELOPMENT RESULTS")
    print("Recordings:", len(actual))
    print("Class counts:", {
        "bonafide": int(np.sum(actual == 0)),
        "spoof": int(np.sum(actual == 1)),
    })
    print("ROC-AUC:", round(roc_auc_score(
        actual, spoof_probabilities
    ), 4))

    print("\nTHRESHOLD METRICS")
    print(
        "Threshold | Accuracy | Balanced Acc | "
        "Spoof Precision | Spoof Recall | Spoof F1"
    )

    for threshold in [0.30, 0.40, 0.50, 0.60, 0.70]:
        predicted = (spoof_probabilities >= threshold).astype(int)

        print(
            f"{threshold:9.2f} | "
            f"{accuracy_score(actual, predicted):8.4f} | "
            f"{balanced_accuracy_score(actual, predicted):12.4f} | "
            f"{precision_score(actual, predicted, zero_division=0):15.4f} | "
            f"{recall_score(actual, predicted, zero_division=0):12.4f} | "
            f"{f1_score(actual, predicted, zero_division=0):7.4f}"
        )

        if threshold == 0.50:
            print("\nCONFUSION MATRIX AT 0.50")
            print("Rows: actual [bonafide, spoof]")
            print("Columns: predicted [bonafide, spoof]")
            print(confusion_matrix(
                actual,
                predicted,
                labels=[0, 1],
            ))

    print("\nPredictions saved to:", predictions_file)


if __name__ == "__main__":
    main()
