from pathlib import Path
import numpy as np
import soundfile as sf

from audio_ml.cadence_analysis import analyze_breathing_and_syllable_cadence
from audio_ml.feature_extractor import (
    CNN_PADDING_DB,
    extract_fixed_length_spectrogram,
    normalize_cnn_spectrogram,
    verify_preprocessing_parity,
)
from audio_ml.rir_features import extract_reverberation_analysis, extract_reverberation_features


def _make_speech_like_wav(path: Path, duration: float = 2.0, sr: int = 16000) -> Path:
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    carrier = np.sin(2 * np.pi * 220.0 * t) + 0.5 * np.sin(2 * np.pi * 440.0 * t)
    syllable_env = 0.5 * (1.0 + np.sin(2 * np.pi * 3.5 * t))
    sig = (0.2 * carrier * syllable_env).astype(np.float32)
    sf.write(str(path), sig, sr)
    return path


def test_preprocessing_parity_and_padding(tmp_path: Path):
    wav = _make_speech_like_wav(tmp_path / "short.wav", duration=1.0)
    spec = extract_fixed_length_spectrogram(str(wav))
    assert spec.shape == (64, 400)
    assert np.isclose(spec[:, -1].mean(), CNN_PADDING_DB)

    norm = normalize_cnn_spectrogram(spec)
    assert norm.min() >= 0.0 and norm.max() <= 1.0

    parity = verify_preprocessing_parity(str(wav))
    assert parity["cnn_parity_verified"] is True


def test_rir_measured_vs_proxies(tmp_path: Path):
    wav = _make_speech_like_wav(tmp_path / "rir.wav", duration=2.0)
    res = extract_reverberation_analysis(str(wav))
    assert "acoustic_proxies" in res
    assert "blind_rir_estimates" in res
    assert "blind_t60_seconds" in res["blind_rir_estimates"]
    assert "drr_proxy_db" in res["blind_rir_estimates"]


def test_breathing_and_syllable_cadence(tmp_path: Path):
    wav = _make_speech_like_wav(tmp_path / "cadence.wav", duration=2.5)
    cad = analyze_breathing_and_syllable_cadence(str(wav))
    assert cad["syllable_count"] >= 4
    assert cad["segment_localization_supported"] is True
    assert "experimentally_supported" in cad["validation_status"]
    assert "exploratory" in cad["validation_status"]
