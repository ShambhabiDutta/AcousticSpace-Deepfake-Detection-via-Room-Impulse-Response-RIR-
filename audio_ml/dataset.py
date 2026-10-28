
from pathlib import Path

import numpy as np

from audio_ml.config import REAL_LABEL, SPOOF_LABEL
from audio_ml.feature_extractor import extract_fixed_length_spectrogram


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


def parse_protocol(
    protocol_path: str | Path,
) -> list[dict[str, str | int]]:
    """Parse five-field ASVspoof and legacy six-field protocol files."""
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

            if len(fields) not in (5, 6):
                raise ValueError(
                    f"Malformed protocol line {line_number}: "
                    f"expected 5 or 6 fields, got {len(fields)}"
                )

            speaker_id = fields[0]
            audio_id = fields[1]
            label_name = fields[-1]

            try:
                label = get_label(label_name)
            except ValueError as error:
                raise ValueError(
                    f"Invalid label on protocol line {line_number}: "
                    f"{label_name}"
                ) from error

            records.append(
                {
                    "speaker_id": speaker_id,
                    "audio_id": audio_id,
                    "label": label,
                }
            )

    return records


def resolve_audio_path(
    audio_id: str,
    audio_dir: str | Path,
    extensions: tuple[str, ...] = (".flac", ".wav"),
) -> Path:
    """Find an audio file by ID using supported extensions."""
    directory = Path(audio_dir)

    for extension in extensions:
        candidate = directory / f"{audio_id}{extension}"

        if candidate.is_file():
            return candidate

    raise FileNotFoundError(
        f"Audio file '{audio_id}' not found in '{directory}' "
        f"with extensions {extensions}"
    )


def validate_audio_file(audio_path: str | Path) -> bool:
    """Check that an audio file exists and is readable."""
    path = Path(audio_path)

    if not path.is_file():
        return False

    try:
        with path.open("rb") as audio_file:
            return bool(audio_file.read(1))
    except OSError:
        return False


def prepare_sample(
    audio_id: str,
    label: int,
    audio_dir: str | Path,
) -> dict[str, object]:
    """Load an audio file and extract its fixed-length spectrogram."""
    audio_path = resolve_audio_path(audio_id, audio_dir)

    if not validate_audio_file(audio_path):
        raise ValueError(
            f"Audio file is empty or unreadable: {audio_path}"
        )

    features = extract_fixed_length_spectrogram(str(audio_path))

    return {
        "audio_id": audio_id,
        "label": label,
        "features": np.asarray(features, dtype=np.float32),
        "audio_path": str(audio_path),
    }


def export_feature_arrays(
    records: list[dict[str, str | int]],
    audio_dir: str | Path,
    output_dir: str | Path,
) -> tuple[Path, Path]:
    """
    Extract features incrementally and save them using memory-mapped arrays.

    Only one recording's feature array is prepared at a time, rather than
    accumulating every recording in Python lists before saving.
    """
    if not records:
        raise ValueError(
            "Cannot export features: protocol contains no records."
        )

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    features_file = output_path / "features.npy"
    labels_file = output_path / "labels.npy"

    # Process the first record to determine the feature shape.
    first = prepare_sample(
        audio_id=str(records[0]["audio_id"]),
        label=int(records[0]["label"]),
        audio_dir=audio_dir,
    )

    first_features = np.asarray(first["features"], dtype=np.float32)

    # Create disk-backed arrays with their final shapes.
    features_array = np.lib.format.open_memmap(
        features_file,
        mode="w+",
        dtype=np.float32,
        shape=(len(records), *first_features.shape),
    )

    labels_array = np.lib.format.open_memmap(
        labels_file,
        mode="w+",
        dtype=np.int64,
        shape=(len(records),),
    )

    try:
        features_array[0] = first_features
        labels_array[0] = int(first["label"])

        # Process each remaining recording individually.
        for index, record in enumerate(records[1:], start=1):
            sample = prepare_sample(
                audio_id=str(record["audio_id"]),
                label=int(record["label"]),
                audio_dir=audio_dir,
            )

            sample_features = np.asarray(
                sample["features"],
                dtype=np.float32,
            )

            if sample_features.shape != first_features.shape:
                raise ValueError(
                    f"Inconsistent feature shape for "
                    f"{record['audio_id']}: {sample_features.shape}; "
                    f"expected {first_features.shape}"
                )

            features_array[index] = sample_features
            labels_array[index] = int(sample["label"])

            if (index + 1) % 1000 == 0:
                features_array.flush()
                labels_array.flush()
                print(
                    f"Processed {index + 1}/{len(records)} recordings"
                )

        features_array.flush()
        labels_array.flush()

    finally:
        # Release memory-mapped array handles.
        del features_array
        del labels_array

    return features_file, labels_file


if __name__ == "__main__":
    print("Supported labels:")

    for name, label in LABEL_MAP.items():
        print(f"  {name}: {label}")
