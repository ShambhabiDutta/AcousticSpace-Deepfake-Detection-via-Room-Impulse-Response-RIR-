
from pathlib import Path

import librosa
import numpy as np

from audio_ml.audio_processor import load_audio


def extract_reverberation_features(
    file_path: str,
) -> dict[str, float]:
    """Extract basic acoustic proxies related to reverberation."""

    # Check that the audio file exists before attempting to load it.
    if not Path(file_path).is_file():
        raise FileNotFoundError(f"Audio file not found: {file_path}")

    audio, sample_rate = load_audio(file_path)

    if len(audio) < 2:
        raise ValueError("Audio is too short to analyze.")

    rms = librosa.feature.rms(y=audio)[0]
    energy = audio.astype(np.float64) ** 2

    total_energy = float(np.sum(energy))

    if total_energy <= 0:
        raise ValueError("Cannot analyze silent audio.")

    # Fraction of energy occurring in the final 20% of the recording.
    tail_start = int(len(energy) * 0.8)
    tail_energy_ratio = float(
        np.sum(energy[tail_start:]) / total_energy
    )

    # Zero-crossing rate: a general signal descriptor, not an RIR measure.
    zero_crossing_rate = float(
        np.mean(librosa.feature.zero_crossing_rate(y=audio))
    )

    return {
        "duration_seconds": float(len(audio) / sample_rate),
        "mean_rms": float(np.mean(rms)),
        "tail_energy_ratio": tail_energy_ratio,
        "zero_crossing_rate": zero_crossing_rate,
    }
