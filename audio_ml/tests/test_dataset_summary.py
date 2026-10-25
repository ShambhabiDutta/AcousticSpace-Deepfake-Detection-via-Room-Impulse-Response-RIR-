
from audio_ml.dataset_summary import summarize_dataset


def test_summary_counts_labels_and_missing_audio(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()

    (audio_dir / "real1.wav").touch()
    (audio_dir / "spoof1.wav").touch()

    protocol = tmp_path / "protocol.txt"
    protocol.write_text(
        "spk1 real1 - - - bonafide\n"
        "spk2 spoof1 - - A01 spoof\n"
        "spk3 missing1 - - A02 spoof\n",
        encoding="utf-8",
    )

    summary = summarize_dataset(protocol, audio_dir)

    assert summary["total_records"] == 3
    assert summary["real_count"] == 1
    assert summary["spoof_count"] == 2
    assert summary["available_audio"] == 2
    assert summary["missing_audio_count"] == 1
    assert summary["missing_audio_ids"] == ["missing1"]


def test_summary_handles_empty_protocol(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()

    protocol = tmp_path / "protocol.txt"
    protocol.write_text("", encoding="utf-8")

    summary = summarize_dataset(protocol, audio_dir)

    assert summary["total_records"] == 0
    assert summary["real_count"] == 0
    assert summary["spoof_count"] == 0
    assert summary["available_audio"] == 0
