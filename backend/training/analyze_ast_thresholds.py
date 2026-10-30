from pathlib import Path

import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
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


def main():
    protocol = (
        DATA_ROOT
        / "ASVspoof2019_LA_cm_protocols"
        / "ASVspoof2019.LA.cm.dev.trl.txt"
    )
    audio_dir = DATA_ROOT / "ASVspoof2019_LA_dev" / "flac"

    records = balanced_subset(read_protocol(protocol), 2500, 42)

    extractor = ASTFeatureExtractor.from_pretrained(OUTPUT_DIR)
    model = ASTForAudioClassification.from_pretrained(OUTPUT_DIR).eval()

    loader = DataLoader(
        ASVspoofDataset(records, audio_dir, extractor),
        batch_size=2,
        shuffle=False,
    )

    actual, spoof_probabilities = [], []

    with torch.no_grad():
        for features, labels in loader:
            logits = model(input_values=features).logits
            probabilities = torch.softmax(logits, dim=1)

            actual.extend(labels.tolist())
            spoof_probabilities.extend(probabilities[:, 1].tolist())

    print("Threshold analysis - 1,000 development samples")
    print("Threshold | Accuracy | Balanced Acc | Spoof Precision | Spoof Recall | Spoof F1")

    for threshold in [0.30, 0.40, 0.50, 0.60, 0.70]:
        predicted = [
            1 if probability >= threshold else 0
            for probability in spoof_probabilities
        ]

        print(
            f"{threshold:9.2f} | "
            f"{accuracy_score(actual, predicted):8.3f} | "
            f"{balanced_accuracy_score(actual, predicted):12.3f} | "
            f"{precision_score(actual, predicted, zero_division=0):15.3f} | "
            f"{recall_score(actual, predicted, zero_division=0):12.3f} | "
            f"{f1_score(actual, predicted, zero_division=0):7.3f}"
        )

        if threshold == 0.50:
            print("Confusion matrix at 0.50:")
            print(confusion_matrix(actual, predicted, labels=[0, 1]))
            print(confusion_matrix(actual, predicted, labels=[0, 1]))


if __name__ == "__main__":
    main()



