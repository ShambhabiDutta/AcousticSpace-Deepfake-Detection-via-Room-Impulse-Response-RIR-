from pathlib import Path
import argparse

import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-per-class", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    device = torch.device("cpu")

    protocol = (
        DATA_ROOT
        / "ASVspoof2019_LA_cm_protocols"
        / "ASVspoof2019.LA.cm.dev.trl.txt"
    )
    audio_dir = DATA_ROOT / "ASVspoof2019_LA_dev" / "flac"

    records = read_protocol(protocol)
    records = balanced_subset(
        records, args.samples_per_class, args.seed
    )

    print("Evaluation samples:", len(records))
    print("Bonafide samples:", sum(label == 0 for _, label in records))
    print("Spoof samples:", sum(label == 1 for _, label in records))
    print("Loading saved AST checkpoint...")

    extractor = ASTFeatureExtractor.from_pretrained(OUTPUT_DIR)
    model = ASTForAudioClassification.from_pretrained(OUTPUT_DIR)
    model.to(device)
    model.eval()

    loader = DataLoader(
        ASVspoofDataset(records, audio_dir, extractor),
        batch_size=args.batch_size,
        shuffle=False,
    )

    actual, predicted = [], []

    with torch.no_grad():
        for features, labels in loader:
            logits = model(input_values=features.to(device)).logits
            predictions = logits.argmax(dim=1)

            actual.extend(labels.tolist())
            predicted.extend(predictions.cpu().tolist())

    print("\nAccuracy:", round(accuracy_score(actual, predicted), 4))
    print(
        "Balanced accuracy:",
        round(balanced_accuracy_score(actual, predicted), 4),
    )
    print("\nConfusion matrix (rows=actual, columns=predicted):")
    print(confusion_matrix(actual, predicted, labels=[0, 1]))
    print("\nClassification report:")
    print(
        classification_report(
            actual,
            predicted,
            labels=[0, 1],
            target_names=["bonafide", "spoof"],
            zero_division=0,
        )
    )


if __name__ == "__main__":
    main()
