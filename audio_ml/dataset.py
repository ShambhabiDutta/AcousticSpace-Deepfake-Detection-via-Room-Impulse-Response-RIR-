
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


def parse_protocol(protocol_path: str | Path) -> list[dict[str, str | int]]:
    """
    Parse an ASVspoof protocol file.

    Expected line format:
    speaker_id audio_id environment channel attack_type label

    Example:
    LA_0069 LA_T_1000137 - - A07 spoof
    LA_0069 LA_T_1000138 - - - bonafide

    Returns records containing audio_id, speaker_id, and numeric label.
    """
    path = Path(protocol_path)

    if not path.is_file():
        raise FileNotFoundError(f"Protocol file not found: {path}")

    records = []

    with path.open("r", encoding="utf-8") as protocol_file:
        for line_number, line in enumerate(protocol_file, start=1):
            line = line.strip()

            if not line:
                continue

            fields = line.split()

            if len(fields) != 6:
                raise ValueError(
                    f"Malformed protocol line {line_number}: "
                    f"expected 6 fields, got {len(fields)}"
                )

            speaker_id, audio_id, _, _, _, label_name = fields

            try:
                label = get_label(label_name)
            except ValueError as error:
                raise ValueError(
                    f"Invalid label on protocol line {line_number}: "
                    f"{label_name}"
                ) from error

            records.append({
                "speaker_id": speaker_id,
                "audio_id": audio_id,
                "label": label,
            })

    return records

def resolve_audio_path(
    audio_id: str,
    audio_dir: str | Path,
    extensions: tuple[str, ...] = (".flac", ".wav"),
) -> Path:
    """Find an audio file by ID inside the dataset audio directory."""
    directory = Path(audio_dir)

    if not directory.is_dir():
        raise NotADirectoryError(
            f"Audio directory does not exist: {directory}"
        )

    if not audio_id.strip():
        raise ValueError("Audio ID cannot be empty.")

    # Prevent audio IDs from pointing outside the audio directory.
    if Path(audio_id).name != audio_id or "/" in audio_id or "\\" in audio_id:
        raise ValueError("Audio ID must be a filename stem, not a path.")

    for extension in extensions:
        candidate = directory / f"{audio_id}{extension}"
        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"No audio file found for '{audio_id}' in '{directory}'. "
        f"Checked extensions: {extensions}"
    )


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
