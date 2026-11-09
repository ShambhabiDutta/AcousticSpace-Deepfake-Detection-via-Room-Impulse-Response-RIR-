
from pathlib import Path
import random
import sys

import numpy as np
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))
DATA_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "asvspoof2019_la_train_v2"
)

MODEL_PATH = PROJECT_ROOT / "audio_ml" / "models" / "cnn_baseline.pt"

SEED = 42
BATCH_SIZE = 8
EPOCHS = 3
LEARNING_RATE = 0.001

DEVICE = torch.device("cpu")


class SpectrogramDataset(Dataset):
    """Load spectrograms one at a time using memory mapping."""

    def __init__(self, features_path, labels, indices):
        self.features_path = features_path
        self.labels = labels
        self.indices = indices
        self.features = np.load(features_path, mmap_mode="r")

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, index):
        actual_index = self.indices[index]

        feature = np.array(
            self.features[actual_index],
            dtype=np.float32,
            copy=True,
        )

        # Convert the log-mel range [-80, 0] to [0, 1].
        feature = np.clip((feature + 80.0) / 80.0, 0.0, 1.0)

        feature = torch.from_numpy(feature).unsqueeze(0)
        label = torch.tensor(
            int(self.labels[actual_index]),
            dtype=torch.long,
        )

        return feature, label


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    torch.set_num_threads(2)

    if not (DATA_DIR / "features.npy").is_file():
        raise FileNotFoundError("features.npy not found.")

    if not (DATA_DIR / "labels.npy").is_file():
        raise FileNotFoundError("labels.npy not found.")

    if MODEL_PATH.exists():
        raise FileExistsError(
            f"Checkpoint already exists: {MODEL_PATH}"
        )

    features = np.load(
        DATA_DIR / "features.npy",
        mmap_mode="r",
    )
    labels = np.load(DATA_DIR / "labels.npy")

    if features.shape[1:] != (64, 400):
        raise ValueError(
            f"Unexpected feature shape: {features.shape}"
        )

    if len(features) != len(labels):
        raise ValueError("Feature and label counts do not match.")

    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must be 0 (real) or 1 (spoof).")

    indices = np.arange(len(labels))

    train_indices, val_indices = train_test_split(
        indices,
        test_size=0.2,
        random_state=SEED,
        stratify=labels,
    )

    train_dataset = SpectrogramDataset(
        DATA_DIR / "features.npy",
        labels,
        train_indices,
    )

    val_dataset = SpectrogramDataset(
        DATA_DIR / "features.npy",
        labels,
        val_indices,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    model = BaselineCNN(num_classes=2).to(DEVICE)

    # Compensate for the imbalanced training labels.
    counts = np.bincount(labels[train_indices], minlength=2)

    class_weights = len(train_indices) / (
        2.0 * counts.astype(np.float32)
    )

    weights = torch.tensor(
        class_weights,
        dtype=torch.float32,
        device=DEVICE,
    )

    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
    )

    best_balanced_accuracy = -1.0
    best_state = None

    print("Starting CNN baseline training")
    print("Device:", DEVICE)
    print("Training recordings:", len(train_indices))
    print("Validation recordings:", len(val_indices))
    print("Class weights:", class_weights)

    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0.0

        for batch_features, batch_labels in train_loader:
            batch_features = batch_features.to(DEVICE)
            batch_labels = batch_labels.to(DEVICE)

            optimizer.zero_grad()

            outputs = model(batch_features)
            loss = criterion(outputs, batch_labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        model.eval()

        actual = []
        predicted = []
        probabilities = []

        with torch.no_grad():
            for batch_features, batch_labels in val_loader:
                outputs = model(batch_features)
                probs = torch.softmax(outputs, dim=1)

                actual.extend(batch_labels.numpy().tolist())
                predicted.extend(
                    outputs.argmax(dim=1).cpu().numpy().tolist()
                )
                probabilities.extend(probs[:, 1].cpu().numpy())

        accuracy = accuracy_score(actual, predicted)
        balanced_accuracy = balanced_accuracy_score(
            actual,
            predicted,
        )

        auc = roc_auc_score(actual, probabilities)

        average_loss = total_loss / len(train_loader)

        print(f"\nEpoch {epoch + 1}/{EPOCHS}")
        print(f"Training loss: {average_loss:.4f}")
        print(f"Validation accuracy: {accuracy:.4f}")
        print(f"Balanced accuracy: {balanced_accuracy:.4f}")
        print(f"ROC-AUC: {auc:.4f}")

        if balanced_accuracy > best_balanced_accuracy:
            best_balanced_accuracy = balanced_accuracy
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    model.load_state_dict(best_state)

    model.eval()
    actual = []
    predicted = []
    probabilities = []

    with torch.no_grad():
        for batch_features, batch_labels in val_loader:
            outputs = model(batch_features)
            probs = torch.softmax(outputs, dim=1)

            actual.extend(batch_labels.numpy().tolist())
            predicted.extend(
                outputs.argmax(dim=1).cpu().numpy().tolist()
            )
            probabilities.extend(probs[:, 1].cpu().numpy())

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)

    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "num_classes": 2,
            "input_shape": [1, 64, 400],
            "class_mapping": {
                "real": 0,
                "spoof": 1,
            },
            "seed": SEED,
            "epochs": EPOCHS,
            "batch_size": BATCH_SIZE,
            "class_weights": class_weights.tolist(),
            "validation_balanced_accuracy": (
                best_balanced_accuracy
            ),
        },
        MODEL_PATH,
    )

    print("\nFinal validation results")
    print("Accuracy:", accuracy_score(actual, predicted))
    print(
        "Balanced accuracy:",
        balanced_accuracy_score(actual, predicted),
    )
    print("ROC-AUC:", roc_auc_score(actual, probabilities))
    print("Confusion matrix:")
    print(confusion_matrix(actual, predicted))
    print(
        classification_report(
            actual,
            predicted,
            target_names=["real", "spoof"],
            zero_division=0,
        )
    )

    print("\nSaved CNN checkpoint:", MODEL_PATH)


if __name__ == "__main__":
    from backend.models.cnn_model import BaselineCNN

    main()
