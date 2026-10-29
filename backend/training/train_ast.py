from pathlib import Path
import argparse
import random

import librosa
import numpy as np
import torch
from sklearn.metrics import accuracy_score, balanced_accuracy_score, classification_report
from torch.utils.data import Dataset, DataLoader
from transformers import ASTFeatureExtractor, ASTForAudioClassification

ROOT = Path(__file__).resolve().parents[2]
DATA_ROOT = Path(r"C:\Users\DELL\Documents\LA")
MODEL_NAME = "MIT/ast-finetuned-audioset-10-10-0.4593"
OUTPUT_DIR = ROOT / "audio_ml" / "models" / "ast_binary"
SAMPLE_RATE = 16000
LABELS = {"bonafide": 0, "spoof": 1}


def read_protocol(path):
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            fields = line.split()
            if len(fields) < 5:
                continue
            audio_id, label = fields[1], fields[-1].lower()
            if label in LABELS:
                records.append((audio_id, LABELS[label]))
    return records


def balanced_subset(records, per_class, seed):
    rng = random.Random(seed)
    result = []
    for label in (0, 1):
        group = [r for r in records if r[1] == label]
        rng.shuffle(group)
        result.extend(group[:per_class])
    rng.shuffle(result)
    return result


class ASVspoofDataset(Dataset):
    def __init__(self, records, audio_dir, extractor):
        self.records = records
        self.audio_dir = audio_dir
        self.extractor = extractor

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        audio_id, label = self.records[index]
        path = self.audio_dir / f"{audio_id}.flac"
        if not path.is_file():
            raise FileNotFoundError(f"Audio file missing: {path}")
        waveform, _ = librosa.load(
            str(path), sr=SAMPLE_RATE, mono=True
        )
        features = self.extractor(
            waveform,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt"
        )["input_values"].squeeze(0)
        return features, torch.tensor(label, dtype=torch.long)


def evaluate(model, loader, device):
    model.eval()
    actual, predicted = [], []
    with torch.no_grad():
        for features, labels in loader:
            logits = model(input_values=features.to(device)).logits
            predicted.extend(logits.argmax(dim=1).cpu().tolist())
            actual.extend(labels.tolist())
    print("Accuracy:", round(accuracy_score(actual, predicted), 4))
    print("Balanced accuracy:", round(balanced_accuracy_score(actual, predicted), 4))
    print(classification_report(
        actual, predicted, labels=[0, 1],
        target_names=["bonafide", "spoof"], zero_division=0
    ))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples-per-class", type=int, default=16)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    random.seed(args.seed)
    np.random.seed(args.seed)
    device = torch.device("cpu")
    print("Device:", device)

    protocols = DATA_ROOT / "ASVspoof2019_LA_cm_protocols"
    train_records = read_protocol(
        protocols / "ASVspoof2019.LA.cm.train.trn.txt"
    )
    dev_records = read_protocol(
        protocols / "ASVspoof2019.LA.cm.dev.trl.txt"
    )

    train_records = balanced_subset(
        train_records, args.samples_per_class, args.seed
    )
    dev_records = balanced_subset(
        dev_records, max(1, args.samples_per_class // 2), args.seed
    )
    print("Train samples:", len(train_records))
    print("Dev samples:", len(dev_records))

    train_audio = DATA_ROOT / "ASVspoof2019_LA_train" / "flac"
    dev_audio = DATA_ROOT / "ASVspoof2019_LA_dev" / "flac"

    extractor = ASTFeatureExtractor.from_pretrained(MODEL_NAME)
    model = ASTForAudioClassification.from_pretrained(
        MODEL_NAME,
        num_labels=2,
        id2label={0: "bonafide", 1: "spoof"},
        label2id={"bonafide": 0, "spoof": 1},
        ignore_mismatched_sizes=True
    ).to(device)

    train_loader = DataLoader(
        ASVspoofDataset(train_records, train_audio, extractor),
        batch_size=args.batch_size, shuffle=True
    )
    dev_loader = DataLoader(
        ASVspoofDataset(dev_records, dev_audio, extractor),
        batch_size=args.batch_size
    )

    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-5)
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for features, labels in train_loader:
            optimizer.zero_grad()
            output = model(
                input_values=features.to(device),
                labels=labels.to(device)
            )
            output.loss.backward()
            optimizer.step()
            total_loss += output.loss.item()
        print(f"Epoch {epoch + 1}: loss={total_loss / len(train_loader):.4f}")
        evaluate(model, dev_loader, device)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(OUTPUT_DIR)
    extractor.save_pretrained(OUTPUT_DIR)
    print("Checkpoint saved to:", OUTPUT_DIR)


if __name__ == "__main__":
    main()
