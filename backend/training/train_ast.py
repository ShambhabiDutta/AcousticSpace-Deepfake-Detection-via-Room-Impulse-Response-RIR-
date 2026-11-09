from pathlib import Path
import argparse
import random

import librosa
import numpy as np
import torch

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
)

from torch.utils.data import Dataset, DataLoader
from transformers import ASTFeatureExtractor, ASTForAudioClassification


ROOT = Path(__file__).resolve().parents[2]

DATA_ROOT = Path(r"C:\Users\DELL\Documents\LA")

MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"

# Keep this experiment separate from the existing model.
OUTPUT_DIR = ROOT / "audio_ml" / "models" / "ast_attack_aware_experiment"

SAMPLE_RATE = 16000

LABELS = {
    "bonafide": 0,
    "spoof": 1,
}

ATTACKS = ["A01", "A02", "A03", "A04", "A05", "A06"]


def read_protocol(path):
    """Read audio ID, binary label, and attack ID."""

    records = []

    with path.open(encoding="utf-8") as file:
        for line in file:
            fields = line.split()

            if len(fields) < 5:
                continue

            audio_id = fields[1]
            attack_id = fields[3].upper()
            label = fields[-1].lower()

            if label not in LABELS:
                continue

            if label == "bonafide":
                attack_id = "bonafide"

            elif attack_id not in ATTACKS:
                continue

            records.append(
                (audio_id, LABELS[label], attack_id)
            )

    return records


def attack_aware_subset(
    records,
    bonafide_count,
    spoof_per_attack,
    seed
):
    """Select equal bonafide and spoof totals while representing every attack."""

    rng = random.Random(seed)

    bonafide = [
        record for record in records
        if record[1] == 0
    ]

    selected = []

    if len(bonafide) < bonafide_count:
        raise ValueError(
            f"Requested {bonafide_count} bonafide recordings, "
            f"but only {len(bonafide)} are available."
        )

    selected.extend(
        rng.sample(bonafide, bonafide_count)
    )

    for attack in ATTACKS:

        attack_records = [
            record for record in records
            if record[2] == attack
        ]

        if len(attack_records) < spoof_per_attack:
            raise ValueError(
                f"Attack {attack}: requested {spoof_per_attack}, "
                f"but only {len(attack_records)} are available."
            )

        selected.extend(
            rng.sample(attack_records, spoof_per_attack)
        )

    rng.shuffle(selected)

    return selected


def print_distribution(records, title):

    print(f"\n{title}")

    print("Total recordings:", len(records))

    print(
        "Bonafide:",
        sum(record[1] == 0 for record in records)
    )

    for attack in ATTACKS:

        count = sum(
            record[2] == attack
            for record in records
        )

        print(f"{attack}: {count}")


class ASVspoofDataset(Dataset):

    def __init__(self, records, audio_dir, extractor):

        self.records = records
        self.audio_dir = Path(audio_dir)
        self.extractor = extractor

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):

        audio_id, label, attack_id = self.records[index]

        path = self.audio_dir / f"{audio_id}.flac"

        if not path.is_file():
            raise FileNotFoundError(
                f"Audio file missing: {path}"
            )

        waveform, _ = librosa.load(
            str(path),
            sr=SAMPLE_RATE,
            mono=True
        )

        features = self.extractor(
            waveform,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt"
        )["input_values"].squeeze(0)

        return (
            features,
            torch.tensor(label, dtype=torch.long),
            attack_id
        )


def evaluate(model, loader, device):

    model.eval()

    actual = []
    predicted = []
    attack_results = {}

    with torch.no_grad():

        for features, labels, attack_ids in loader:

            features = features.to(device)

            logits = model(
                input_values=features
            ).logits

            predictions = logits.argmax(dim=1).cpu().tolist()

            actual.extend(labels.tolist())
            predicted.extend(predictions)

            for attack_id, prediction in zip(
                attack_ids, predictions
            ):

                if attack_id == "bonafide":
                    continue

                attack_results.setdefault(
                    attack_id, {"total": 0, "detected": 0}
                )

                attack_results[attack_id]["total"] += 1

                if prediction == 1:
                    attack_results[attack_id]["detected"] += 1

    print(
        "Accuracy:",
        round(accuracy_score(actual, predicted), 4)
    )

    print(
        "Balanced accuracy:",
        round(balanced_accuracy_score(actual, predicted), 4)
    )

    print("\nClassification report:")

    print(
        classification_report(
            actual,
            predicted,
            labels=[0, 1],
            target_names=["bonafide", "spoof"],
            zero_division=0
        )
    )

    print("Spoof detection by attack:")

    for attack in ATTACKS:

        result = attack_results.get(attack)

        if result and result["total"] > 0:

            recall = (
                result["detected"] / result["total"]
            )

            print(
                f"{attack}: "
                f"{result['detected']}/{result['total']} "
                f"detected | recall={recall:.4f}"
            )


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        type=Path,
        default=DATA_ROOT
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DIR
    )

    parser.add_argument(
        "--bonafide-samples",
        type=int,
        default=2580
    )

    parser.add_argument(
        "--spoof-samples-per-attack",
        type=int,
        default=430
    )

    parser.add_argument(
        "--dev-bonafide-samples",
        type=int,
        default=300
    )

    parser.add_argument(
        "--dev-spoof-samples-per-attack",
        type=int,
        default=50
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=3
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=4
    )

    parser.add_argument(
        "--num-workers",
        type=int,
        default=2
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42
    )

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)

    if torch.cuda.is_available():

        device = torch.device("cuda")

        torch.cuda.manual_seed_all(args.seed)

        print("GPU available:", torch.cuda.get_device_name(0))

    else:

        device = torch.device("cpu")

        print("WARNING: GPU unavailable. Training will use CPU.")

    print("Device:", device)

    data_root = args.data_root

    protocols = data_root / "ASVspoof2019_LA_cm_protocols"

    train_protocol = (
        protocols / "ASVspoof2019.LA.cm.train.trn.txt"
    )

    dev_protocol = (
        protocols / "ASVspoof2019.LA.cm.dev.trl.txt"
    )

    if not train_protocol.is_file():
        raise FileNotFoundError(train_protocol)

    if not dev_protocol.is_file():
        raise FileNotFoundError(dev_protocol)

    train_records = read_protocol(train_protocol)
    dev_records = read_protocol(dev_protocol)

    print("Original training recordings:", len(train_records))
    print("Original development recordings:", len(dev_records))

    train_records = attack_aware_subset(
        train_records,
        args.bonafide_samples,
        args.spoof_samples_per_attack,
        args.seed
    )

    dev_records = attack_aware_subset(
        dev_records,
        args.dev_bonafide_samples,
        args.dev_spoof_samples_per_attack,
        args.seed
    )

    print_distribution(train_records, "Training distribution")
    print_distribution(dev_records, "Development distribution")

    train_audio = data_root / "ASVspoof2019_LA_train" / "flac"
    dev_audio = data_root / "ASVspoof2019_LA_dev" / "flac"

    extractor = ASTFeatureExtractor.from_pretrained(
        MODEL_NAME
    )

    model = ASTForAudioClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label={
            0: "bonafide",
            1: "spoof"
        },
        label2id={
            "bonafide": 0,
            "spoof": 1
        },
        ignore_mismatched_sizes=True
    ).to(device)

    train_dataset = ASVspoofDataset(
        train_records,
        train_audio,
        extractor
    )

    dev_dataset = ASVspoofDataset(
        dev_records,
        dev_audio,
        extractor
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )

    dev_loader = DataLoader(
        dev_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=(device.type == "cuda")
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2e-5
    )

    for epoch in range(args.epochs):

        model.train()

        total_loss = 0.0

        for features, labels, _ in train_loader:

            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            output = model(
                input_values=features,
                labels=labels
            )

            output.loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            total_loss += output.loss.item()

        average_loss = total_loss / len(train_loader)

        print(
            f"\nEpoch {epoch + 1}/{args.epochs} "
            f"| Training loss: {average_loss:.4f}"
        )

        evaluate(model, dev_loader, device)

    args.output_dir.mkdir(
        parents=True,
        exist_ok=True
    )

    model.save_pretrained(args.output_dir)
    extractor.save_pretrained(args.output_dir)

    print("\nNew model saved to:")
    print(args.output_dir)


if __name__ == "__main__":
    main()