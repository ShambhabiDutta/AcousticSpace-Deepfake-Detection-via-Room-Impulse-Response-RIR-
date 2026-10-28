
from pathlib import Path

import joblib
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


DATA_DIR = Path("data/processed/asvspoof2019_la_train_v2")
MODEL_DIR = Path("audio_ml/models")
MODEL_PATH = MODEL_DIR / "baseline_logreg.joblib"
RANDOM_STATE = 42


def summarize_spectrograms(features: np.ndarray) -> np.ndarray:
    """Summarize each mel band using its mean and standard deviation."""
    means = features.mean(axis=2)
    stds = features.std(axis=2)
    return np.concatenate((means, stds), axis=1).astype(np.float32)


def main() -> None:
    features_path = DATA_DIR / "features.npy"
    labels_path = DATA_DIR / "labels.npy"

    if not features_path.is_file() or not labels_path.is_file():
        raise FileNotFoundError(
            f"Training arrays not found in {DATA_DIR.resolve()}"
        )

    # Memory-map the large spectrogram array instead of loading it all
    # into RAM at once.
    features = np.load(features_path, mmap_mode="r")
    labels = np.load(labels_path)

    if features.ndim != 3:
        raise ValueError(
            f"Expected 3D spectrograms; received {features.shape}"
        )

    if len(features) != len(labels):
        raise ValueError("Feature and label counts do not match.")

    if not np.isfinite(features).all():
        raise ValueError("Features contain NaN or infinite values.")

    if not np.isin(labels, [0, 1]).all():
        raise ValueError("Labels must be 0 (real) or 1 (spoof).")

    print(f"Loaded {len(labels)} recordings.")
    print(f"Original feature shape: {features.shape}")

    # Create a compact matrix with 128 values per recording.
    X = summarize_spectrograms(features)
    y = labels

    print(f"Summarized feature shape: {X.shape}")
    print("Label counts:", dict(zip(*np.unique(y, return_counts=True))))

    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
    )

    print("\nTraining logistic regression baseline...")
    model.fit(X_train, y_train)

    predictions = model.predict(X_val)
    probabilities = model.predict_proba(X_val)[:, 1]

    print("\nValidation results (stratified holdout from training set)")
    print(f"Accuracy:          {accuracy_score(y_val, predictions):.4f}")
    print(
        f"Balanced accuracy: {balanced_accuracy_score(y_val, predictions):.4f}"
    )
    print(f"ROC-AUC:            {roc_auc_score(y_val, probabilities):.4f}")
    print("\nConfusion matrix (rows=true, columns=predicted):")
    print(confusion_matrix(y_val, predictions))
    print("\nClassification report:")
    print(
        classification_report(
            y_val,
            predictions,
            labels=[0, 1],
            target_names=["real", "spoof"],
            zero_division=0,
        )
    )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"\nSaved model to: {MODEL_PATH.resolve()}")


if __name__ == "__main__":
    main()
