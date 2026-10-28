
from pathlib import Path

import joblib
import numpy as np
from sklearn.metrics import (
    balanced_accuracy_score,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
)

from audio_ml.train_baseline import summarize_spectrograms


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "processed" / "asvspoof2019_la_dev_v2"
MODEL_PATH = PROJECT_ROOT / "audio_ml" / "models" / "baseline_logreg.joblib"


def main():
    features_path = DATA_DIR / "features.npy"
    labels_path = DATA_DIR / "labels.npy"

    if not features_path.is_file() or not labels_path.is_file():
        raise FileNotFoundError(
            f"Development arrays not found in {DATA_DIR}"
        )

    if not MODEL_PATH.is_file():
        raise FileNotFoundError(f"Model not found: {MODEL_PATH}")

    # Load development data and summarize the spectrograms.
    features = np.load(features_path, mmap_mode="r")
    X = summarize_spectrograms(features)
    y = np.load(labels_path)

    model = joblib.load(MODEL_PATH)
    probabilities = model.predict_proba(X)[:, 1]

    print("AcousticSpace — Threshold Evaluation")
    print(f"Development recordings: {len(y)}")
    print("\nSpoof is the positive class (label 1).")
    print("Lower thresholds flag more recordings as spoof.\n")

    print(
        f"{'Threshold':>9} {'Balanced Acc.':>14} "
        f"{'Real Recall':>12} {'Spoof Recall':>13} "
        f"{'Spoof Precision':>16} {'Spoof F1':>10}"
    )
    print("-" * 80)

    for threshold in [0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]:
        predictions = (probabilities >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(
            y, predictions, labels=[0, 1]
        ).ravel()

        print(
            f"{threshold:9.2f} "
            f"{balanced_accuracy_score(y, predictions):14.4f} "
            f"{recall_score(y, predictions, pos_label=0):12.4f} "
            f"{recall_score(y, predictions, pos_label=1):13.4f} "
            f"{precision_score(y, predictions, pos_label=1, zero_division=0):16.4f} "
            f"{f1_score(y, predictions, pos_label=1, zero_division=0):10.4f}"
        )

        print(f"           TN={tn}, FP={fp}, FN={fn}, TP={tp}")


if __name__ == "__main__":
    main()
