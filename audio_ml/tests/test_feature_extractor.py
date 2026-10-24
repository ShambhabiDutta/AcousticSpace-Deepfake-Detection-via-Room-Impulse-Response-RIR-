
import numpy as np
import soundfile as sf

from audio_ml.config import N_MELS, SAMPLE_RATE
from audio_ml.feature_extractor import extract_log_mel_spectrogram


def test_spectrogram_from_real_audio_file(tmp_path):
    """Check feature extraction from a valid audio file."""
    audio_path = tmp_path / "sample.wav"

    # One second of synthetic audio for a repeatable pipeline test.
    audio = np.sin(
        2 * np.pi * 440 * np.arange(SAMPLE_RATE) / SAMPLE_RATE
    ).astype(np.float32)

    sf.write(audio_path, audio, SAMPLE_RATE)

    features = extract_log_mel_spectrogram(str(audio_path))

    assert features.ndim == 2
    assert features.shape[0] == N_MELS
    assert features.shape[1] > 0
    assert features.dtype == np.float32
    assert np.all(np.isfinite(features))


def test_spectrogram_rejects_missing_file(tmp_path):
    """A missing audio file should raise FileNotFoundError."""
    missing_path = tmp_path / "missing.wav"

    try:
        extract_log_mel_spectrogram(str(missing_path))
    except FileNotFoundError:
        pass
    else:
        raise AssertionError("Expected FileNotFoundError")
