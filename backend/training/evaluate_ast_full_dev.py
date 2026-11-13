
import csv
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
from transformers import (
    ASTFeatureExtractor,
    ASTForAudioClassification,
)

from train_ast import (
    DATA_ROOT,
    OUTPUT_DIR,
    read_protocol,
    ASVspoofDataset,
)


# ============================================================
# CONFIGURATION
# ============================================================

ROOT = Path(__file__).resolve().parents[2]

OUTPUT_DIR = ROOT / "audio_ml" / "models" / "ast_binary"

PROTOCOL_PATH = (
    DATA_ROOT
    / "ASVspoof2019_LA_cm_protocols"
    / "ASVspoof2019.LA.cm.dev.trl.txt"
)

DEV_AUDIO_DIR = (
    DATA_ROOT
    / "ASVspoof2019_LA_dev"
    / "flac"
)

EVALUATION_DIR = ROOT / "audio_ml" / "evaluation"

CHECKPOINT_PATH = (
    EVALUATION_DIR / "ast_full_dev_checkpoint.csv"
)

PREDICTIONS_PATH = (
    EVALUATION_DIR / "ast_full_dev_predictions.csv"
)

EXPECTED_RECORDS = 24844

BATCH_SIZE = 1
NUM_WORKERS = 0

CHECKPOINT_INTERVAL = 100

THRESHOLDS = [0.30, 0.50, 0.70]

DEVICE = torch.device("cpu")

# IMPORTANT:
# Keep these three columns compatible with the
# existing ast_full_dev_checkpoint.csv.
CHECKPOINT_FIELDS = [
    "audio_id",
    "actual_label",
    "spoof_probability",
]


# ============================================================
# CHECKPOINT LOADING AND VALIDATION
# ============================================================

def load_checkpoint(records):
    """
    Load previously completed predictions.

    Validate each row against the development protocol.
    Resume from the next unprocessed recording.
    """

    completed = {}

    if not CHECKPOINT_PATH.exists():
        print("No existing checkpoint found.")
        print("A new checkpoint will be created.")
        return completed

    print("Existing checkpoint found.")
    print("Validating checkpoint...")

    with CHECKPOINT_PATH.open(
        "r",
        newline="",
        encoding="utf-8",
    ) as file:

        reader = csv.DictReader(file)

        if reader.fieldnames != CHECKPOINT_FIELDS:
            raise ValueError(
                "Checkpoint header does not match the expected "
                f"format. Found: {reader.fieldnames}"
            )

        for index, row in enumerate(reader):

            if index >= len(records):
                raise ValueError(
                    "Checkpoint contains more rows than "
                    "the development dataset."
                )

            expected_id, expected_label = records[index]

            if row["audio_id"] != expected_id:
                raise ValueError(
                    f"Audio ID mismatch at row {index}: "
                    f"{row['audio_id']} != {expected_id}"
                )

            if int(row["actual_label"]) != expected_label:
                raise ValueError(
                    f"Label mismatch at row {index}."
                )

            probability = float(
                row["spoof_probability"]
            )

            if (
                not np.isfinite(probability)
                or not 0.0 <= probability <= 1.0
            ):
                raise ValueError(
                    f"Invalid spoof probability at row {index}."
                )

            completed[index] = probability

    print("Checkpoint validation successful.")
    print("Completed recordings:", len(completed))

    return completed


# ============================================================
# CHECKPOINT SAVING
# ============================================================

def append_checkpoint(rows):
    """
    Append completed predictions without changing
    the original three-column checkpoint format.
    """

    if not rows:
        return

    file_exists = CHECKPOINT_PATH.exists()

    with CHECKPOINT_PATH.open(
        "a",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=CHECKPOINT_FIELDS,
        )

        if not file_exists:
            writer.writeheader()

        writer.writerows(rows)

        file.flush()


# ============================================================
# DATASET VALIDATION
# ============================================================

def validate_records(records):

    if len(records) != EXPECTED_RECORDS:
        raise ValueError(
            f"Expected {EXPECTED_RECORDS} recordings, "
            f"but found {len(records)}."
        )

    audio_ids = [record[0] for record in records]

    if len(audio_ids) != len(set(audio_ids)):
        raise ValueError(
            "Duplicate audio IDs found in the development protocol."
        )

    labels = [record[1] for record in records]

    if not set(labels).issubset({0, 1}):
        raise ValueError(
            "Unexpected labels found. Expected labels 0 and 1."
        )

    print("\nDevelopment dataset information")
    print("--------------------------------")
    print("Total recordings:", len(records))
    print("Bonafide recordings:", labels.count(0))
    print("Spoof recordings:", labels.count(1))


# ============================================================
# THRESHOLD METRICS
# ============================================================

def evaluate_threshold(actual, probabilities, threshold):

    predicted = (
        probabilities >= threshold
    ).astype(np.int64)

    return {
        "threshold": threshold,

        "accuracy": accuracy_score(
            actual,
            predicted,
        ),

        "balanced_accuracy": balanced_accuracy_score(
            actual,
            predicted,
        ),

        "spoof_precision": precision_score(
            actual,
            predicted,
            pos_label=1,
            zero_division=0,
        ),

        "spoof_recall": recall_score(
            actual,
            predicted,
            pos_label=1,
            zero_division=0,
        ),

        "spoof_f1": f1_score(
            actual,
            predicted,
            pos_label=1,
            zero_division=0,
        ),

        "confusion_matrix": confusion_matrix(
            actual,
            predicted,
            labels=[0, 1],
        ),
    }


# ============================================================
# FINAL EVALUATION REPORT
# ============================================================

def print_final_metrics(records, probabilities):

    actual = np.asarray(
        [label for _, label in records],
        dtype=np.int64,
    )

    probabilities = np.asarray(
        probabilities,
        dtype=np.float64,
    )

    if len(actual) != len(probabilities):
        raise ValueError(
            "Prediction count does not match actual label count."
        )

    if not np.isfinite(probabilities).all():
        raise ValueError(
            "Non-finite prediction probabilities found."
        )

    if not np.all(
        (probabilities >= 0.0)
        & (probabilities <= 1.0)
    ):
        raise ValueError(
            "Prediction probabilities must be between 0 and 1."
        )

    if len(np.unique(actual)) != 2:
        raise ValueError(
            "ROC-AUC requires both classes in the dataset."
        )

    auc = roc_auc_score(
        actual,
        probabilities,
    )

    print("\n============================================")
    print("          AST FULL DEV EVALUATION")
    print("============================================")

    print("Total recordings:", len(actual))
    print("ROC-AUC:", round(auc, 4))

    for threshold in THRESHOLDS:

        result = evaluate_threshold(
            actual,
            probabilities,
            threshold,
        )

        print("\n--------------------------------------------")
        print("Threshold:", result["threshold"])

        print(
            "Accuracy:",
            round(result["accuracy"], 4),
        )

        print(
            "Balanced accuracy:",
            round(result["balanced_accuracy"], 4),
        )

        print(
            "Spoof precision:",
            round(result["spoof_precision"], 4),
        )

        print(
            "Spoof recall:",
            round(result["spoof_recall"], 4),
        )

        print(
            "Spoof F1:",
            round(result["spoof_f1"], 4),
        )

        print("\nConfusion matrix")
        print("Rows: actual [bonafide, spoof]")
        print("Columns: predicted [bonafide, spoof]")
        print(result["confusion_matrix"])

    # Detailed report at threshold 0.50.
    predicted = (
        probabilities >= 0.50
    ).astype(np.int64)

    print("\n============================================")
    print("CLASSIFICATION REPORT AT THRESHOLD 0.50")
    print("============================================")

    print(
        classification_report(
            actual,
            predicted,
            labels=[0, 1],
            target_names=["bonafide", "spoof"],
            zero_division=0,
        )
    )

    return actual, probabilities


# ============================================================
# SAVE FINAL PREDICTIONS
# ============================================================

def save_predictions(records, actual, probabilities):

    predicted = (
        probabilities >= 0.50
    ).astype(np.int64)

    temporary_path = PREDICTIONS_PATH.with_suffix(".tmp")

    fields = [
        "audio_id",
        "actual_label",
        "predicted_label",
        "spoof_probability",
        "threshold",
    ]

    with temporary_path.open(
        "w",
        newline="",
        encoding="utf-8",
    ) as file:

        writer = csv.DictWriter(
            file,
            fieldnames=fields,
        )

        writer.writeheader()

        for index, (audio_id, _) in enumerate(records):

            writer.writerow({
                "audio_id": audio_id,
                "actual_label": int(actual[index]),
                "predicted_label": int(predicted[index]),
                "spoof_probability": round(
                    float(probabilities[index]),
                    8,
                ),
                "threshold": 0.50,
            })

    temporary_path.replace(PREDICTIONS_PATH)

    print("\nFinal predictions saved:")
    print(PREDICTIONS_PATH)


# ============================================================
# MAIN EVALUATION
# ============================================================

def main():

    torch.set_num_threads(2)

    EVALUATION_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    print("AST full development-set evaluation")
    print("Device:", DEVICE)

    # --------------------------------------------------------
    # Validate paths
    # --------------------------------------------------------

    if not PROTOCOL_PATH.is_file():
        raise FileNotFoundError(
            f"Development protocol not found: {PROTOCOL_PATH}"
        )

    if not DEV_AUDIO_DIR.is_dir():
        raise FileNotFoundError(
            f"Development audio directory not found: {DEV_AUDIO_DIR}"
        )

    if not OUTPUT_DIR.is_dir():
        raise FileNotFoundError(
            f"AST checkpoint directory not found: {OUTPUT_DIR}"
        )

    # --------------------------------------------------------
    # Load development protocol
    # --------------------------------------------------------

    print("\nLoading development protocol...")

    records = read_protocol(PROTOCOL_PATH)

    validate_records(records)

    # --------------------------------------------------------
    # Load and validate existing checkpoint
    # --------------------------------------------------------

    completed = load_checkpoint(records)

    start_index = len(completed)

    print("\nResume information")
    print("------------------")
    print("Already completed:", start_index)
    print("Remaining:", len(records) - start_index)

    # --------------------------------------------------------
    # If all records are already evaluated
    # --------------------------------------------------------

    if start_index == len(records):

        print("\nAll recordings are already evaluated.")

        probabilities = [
            completed[index]
            for index in range(len(records))
        ]

        actual, probabilities = print_final_metrics(
            records,
            probabilities,
        )

        save_predictions(
            records,
            actual,
            probabilities,
        )

        return

    # --------------------------------------------------------
    # Load AST feature extractor
    # --------------------------------------------------------

    print("\nLoading AST feature extractor...")

    extractor = ASTFeatureExtractor.from_pretrained(
        str(OUTPUT_DIR)
    )

    # --------------------------------------------------------
    # Load AST model
    # --------------------------------------------------------

    print("Loading fine-tuned AST model...")

    model = ASTForAudioClassification.from_pretrained(
        str(OUTPUT_DIR)
    )

    model.to(DEVICE)
    model.eval()

    print("AST model loaded successfully.")

    # --------------------------------------------------------
    # Prepare only remaining records
    # --------------------------------------------------------

    remaining_records = records[start_index:]

    dataset = ASVspoofDataset(
        remaining_records,
        DEV_AUDIO_DIR,
        extractor,
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    pending_rows = []

    probabilities = dict(completed)

    print("\nStarting evaluation...")
    print("Checkpoint interval:", CHECKPOINT_INTERVAL)
    print("Batch size:", BATCH_SIZE)
    print("--------------------------------------------")

    # --------------------------------------------------------
    # AST inference
    # --------------------------------------------------------

    with torch.inference_mode():

        for batch_index, (features, labels) in enumerate(loader):

            original_index = start_index + batch_index

            features = features.to(DEVICE)

            outputs = model(
                input_values=features
            )

            logits = outputs.logits

            # Class 1 = spoof.
            batch_probabilities = torch.softmax(
                logits,
                dim=1,
            )[:, 1]

            spoof_probability = float(
                batch_probabilities[0].cpu().item()
            )

            if (
                not np.isfinite(spoof_probability)
                or not 0.0 <= spoof_probability <= 1.0
            ):
                raise ValueError(
                    f"Invalid probability at index {original_index}."
                )

            audio_id, expected_label = remaining_records[
                batch_index
            ]

            actual_label = int(
                labels[0].item()
            )

            if actual_label != expected_label:
                raise ValueError(
                    f"Dataset label mismatch for {audio_id}."
                )

            probabilities[original_index] = spoof_probability

            # Keep the original three-column CSV format.
            pending_rows.append({
                "audio_id": audio_id,
                "actual_label": expected_label,
                "spoof_probability": round(
                    spoof_probability,
                    8,
                ),
            })

            completed_count = original_index + 1

            # ------------------------------------------------
            # Save every 100 recordings.
            # ------------------------------------------------

            if (
                len(pending_rows) >= CHECKPOINT_INTERVAL
                or completed_count == len(records)
            ):

                append_checkpoint(pending_rows)

                pending_rows.clear()

                print(
                    f"Checkpoint saved: "
                    f"{completed_count}/{len(records)} "
                    f"({completed_count / len(records) * 100:.1f}%)",
                    flush=True,
                )

    # --------------------------------------------------------
    # Verify prediction count
    # --------------------------------------------------------

    if len(probabilities) != len(records):
        raise RuntimeError(
            f"Evaluation incomplete: "
            f"{len(probabilities)}/{len(records)} predictions."
        )

    ordered_probabilities = [
        probabilities[index]
        for index in range(len(records))
    ]

    # --------------------------------------------------------
    # Final metrics
    # --------------------------------------------------------

    actual, ordered_probabilities = print_final_metrics(
        records,
        ordered_probabilities,
    )

    # --------------------------------------------------------
    # Save predictions
    # --------------------------------------------------------

    save_predictions(
        records,
        actual,
        ordered_probabilities,
    )

    print("\n============================================")
    print("FULL AST DEVELOPMENT EVALUATION COMPLETED")
    print("============================================")

    print("Checkpoint:", CHECKPOINT_PATH)
    print("Predictions:", PREDICTIONS_PATH)


if __name__ == "__main__":
    main()

