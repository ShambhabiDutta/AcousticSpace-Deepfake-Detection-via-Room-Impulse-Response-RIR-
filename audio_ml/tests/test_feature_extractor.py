
import numpy as np
import pytest
import soundfile as sf

from audio_ml.config import N_MELS, SAMPLE_RATE
from audio_ml.feature_extractor import (
    extract_fixed_length_spectrogram,
    extract_log_mel_spectrogram,
)


def create_test_audio(file_path, duration_seconds=1.0):
    """Create a temporary sine-wave audio file for testing."""
    sample_count = int(SAMPLE_RATE * duration_seconds)
    time = np.arange(sample_count) / SAMPLE_RATE

    audio = np.sin(2 * np.pi * 440 * time).astype(np.float32)

    sf.write(file_path, audio, SAMPLE_RATE)


def test_spectrogram_from_audio_file(tmp_path):
    """Check log-mel extraction from a valid audio file."""
    audio_path = tmp_path / "sample.wav"
    create_test_audio(audio_path)

    features = extract_log_mel_spectrogram(str(audio_path))

    assert features.ndim == 2
    assert features.shape[0] == N_MELS
    assert features.shape[1] > 0
    assert features.dtype == np.float32
    assert np.all(np.isfinite(features))


def test_fixed_length_spectrogram_shape(tmp_path):
    """Check that spectrograms have a consistent shape."""
    audio_path = tmp_path / "sample.wav"
    create_test_audio(audio_path)

    features = extract_fixed_length_spectrogram(
        str(audio_path),
        max_frames=400,
    )

    assert features.shape == (N_MELS, 400)
    assert features.dtype == np.float32
    assert np.all(np.isfinite(features))


def test_short_audio_is_padded(tmp_path):
    """Check that short audio features are padded to the requested size."""
    audio_path = tmp_path / "short.wav"
    create_test_audio(audio_path, duration_seconds=0.1)

    features = extract_fixed_length_spectrogram(
        str(audio_path),
        max_frames=400,
    )

    assert features.shape == (N_MELS, 400)


def test_long_spectrogram_is_truncated(tmp_path):
    """Check that long audio features are truncated."""
    audio_path = tmp_path / "long.wav"
    create_test_audio(audio_path, duration_seconds=3.0)

    features = extract_fixed_length_spectrogram(
        str(audio_path),
        max_frames=100,
    )

    assert features.shape == (N_MELS, 100)


def test_fixed_length_rejects_zero_frames():
    """A non-positive frame count should raise ValueError."""
    with pytest.raises(ValueError):
        extract_fixed_length_spectrogram(
            "unused.wav",
            max_frames=0,
        )


def test_missing_audio_file_raises_error(tmp_path):
    """A missing audio file should raise FileNotFoundError."""
    missing_path = tmp_path / "missing.wav"

    with pytest.raises(FileNotFoundError):
        extract_log_mel_spectrogram(str(missing_path))
