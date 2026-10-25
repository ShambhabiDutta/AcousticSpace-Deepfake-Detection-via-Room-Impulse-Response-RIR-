
import numpy as np
import pytest
import soundfile as sf

from audio_ml.config import SAMPLE_RATE
from audio_ml.rir_features import extract_reverberation_features


def test_reverberation_features_from_audio(tmp_path):
    audio_path = tmp_path / "sample.wav"
    t = np.arange(SAMPLE_RATE) / SAMPLE_RATE
    audio = (0.5 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    sf.write(audio_path, audio, SAMPLE_RATE)

    features = extract_reverberation_features(str(audio_path))

    assert set(features) == {
        "duration_seconds",
        "mean_rms",
        "tail_energy_ratio",
        "zero_crossing_rate",
    }
    assert features["duration_seconds"] == pytest.approx(1.0, abs=0.02)
    assert features["mean_rms"] > 0
    assert 0 <= features["tail_energy_ratio"] <= 1
    assert 0 <= features["zero_crossing_rate"] <= 1


def test_silent_audio_is_rejected(tmp_path):
    audio_path = tmp_path / "silent.wav"
    sf.write(audio_path, np.zeros(SAMPLE_RATE, dtype=np.float32), SAMPLE_RATE)

    with pytest.raises(ValueError, match="silent audio"):
        extract_reverberation_features(str(audio_path))


def test_missing_audio_is_rejected(tmp_path):
    with pytest.raises(FileNotFoundError):
        extract_reverberation_features(str(tmp_path / "missing.wav"))

