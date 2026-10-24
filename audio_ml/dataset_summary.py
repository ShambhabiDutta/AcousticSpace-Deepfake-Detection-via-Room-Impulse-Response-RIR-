
from collections import Counter
from pathlib import Path

from audio_ml.dataset import parse_protocol, resolve_audio_path


def summarize_dataset(
    protocol_path: str | Path,
    audio_dir: str | Path,
) -> dict[str, object]:
    """Summarize dataset labels and audio-file availability."""
    records = parse_protocol(protocol_path)
    label_counts = Counter(record["label"] for record in records)

    available = 0
    missing = []

    for record in records:
        audio_id = str(record["audio_id"])

        try:
            resolve_audio_path(audio_id, audio_dir)
            available += 1
        except FileNotFoundError:
            missing.append(audio_id)

    return {
        "total_records": len(records),
        "real_count": label_counts.get(0, 0),
        "spoof_count": label_counts.get(1, 0),
        "available_audio": available,
        "missing_audio_count": len(missing),
        "missing_audio_ids": missing,
    }


if __name__ == "__main__":
    print(
        "Dataset summary utility ready. "
        "Call summarize_dataset() with your protocol and audio directory."
    )
