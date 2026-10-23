
from pathlib import Path

import numpy as np

from audio_ml.config import REAL_LABEL, SPOOF_LABEL
from audio_ml.feature_extractor import extract_log_mel_spectrogram


LABEL_MAP = {
    "real": REAL_LABEL,
    "bonafide": REAL_LABEL,
    "spoof": SPOOF_LABEL,
    "fake": SPOOF_LABEL,
}


def get_label(label_name: str) -> int:
    """Convert a supported class name to its numeric label."""
    normalized = label_name.strip().lower()

    if normalized not in LABEL_MAP:
        raise ValueError(
            f"Unknown label '{label_name}'. "
            f"Supported labels: {list(LABEL_MAP)}"
        )

    return LABEL_MAP[normalized]


def prepare_sample(
    file_path: str | Path,
    label_name: str,
) -> tuple[np.ndarray, int]:
    """Extract features and return them with the numeric label."""
    path = Path(file_path)

    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    features = extract_log_mel_spectrogram(str(path))
    label = get_label(label_name)

    return features, label


if __name__ == "__main__":
    print("Dataset utilities loaded successfully.")
    print(f"Label map: {LABEL_MAP}")
