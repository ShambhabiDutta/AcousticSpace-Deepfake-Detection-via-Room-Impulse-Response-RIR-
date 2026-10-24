
import numpy as np
import pytest

from audio_ml.dataset import export_feature_arrays


def test_export_feature_arrays_saves_files(tmp_path, monkeypatch):
    audio_dir = tmp_path / "audio"
    output_dir = tmp_path / "features"
    audio_dir.mkdir()

    (audio_dir / "sample1.wav").touch()
    (audio_dir / "sample2.wav").touch()

    def fake_extract(file_path):
        return np.ones((64, 400), dtype=np.float32)

    monkeypatch.setattr(
        "audio_ml.dataset.extract_fixed_length_spectrogram",
        fake_extract,
    )

    records = [
        {"audio_id": "sample1", "label": 0},
        {"audio_id": "sample2", "label": 1},
    ]

    summary = export_feature_arrays(records, audio_dir, output_dir)

    X = np.load(output_dir / "X.npy")
    y = np.load(output_dir / "y.npy")
    sample_ids = np.load(output_dir / "sample_ids.npy")

    assert X.shape == (2, 64, 400)
    assert X.dtype == np.float32
    assert y.tolist() == [0, 1]
    assert sample_ids.tolist() == ["sample1", "sample2"]
    assert summary["sample_count"] == 2
    assert summary["real_count"] == 1
    assert summary["spoof_count"] == 1


def test_export_rejects_empty_records(tmp_path):
    with pytest.raises(ValueError, match="empty dataset"):
        export_feature_arrays([], tmp_path, tmp_path / "features")


def test_export_rejects_missing_audio(tmp_path):
    audio_dir = tmp_path / "audio"
    audio_dir.mkdir()

    records = [{"audio_id": "missing", "label": 0}]

    with pytest.raises(FileNotFoundError):
        export_feature_arrays(records, audio_dir, tmp_path / "features")
