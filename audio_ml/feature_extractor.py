
import librosa
import numpy as np

from audio_ml.config import HOP_LENGTH, N_FFT, N_MELS, SAMPLE_RATE
from audio_ml.audio_processor import load_audio


DEFAULT_MAX_FRAMES = 400


def extract_log_mel_spectrogram(
    file_path: str,
) -> np.ndarray:
    """
    Convert audio to a log-mel spectrogram.

    Returns:
        Float32 array with shape (n_mels, time_frames).
    """
    audio, sample_rate = load_audio(
        file_path,
        sample_rate=SAMPLE_RATE,
    )

    mel_spectrogram = librosa.feature.melspectrogram(
        y=audio,
        sr=sample_rate,
        n_fft=N_FFT,
        hop_length=HOP_LENGTH,
        n_mels=N_MELS,
        power=2.0,
    )

    log_mel_spectrogram = librosa.power_to_db(
        mel_spectrogram,
        ref=np.max,
    )

    return log_mel_spectrogram.astype(np.float32)


def extract_fixed_length_spectrogram(
    file_path: str,
    max_frames: int = DEFAULT_MAX_FRAMES,
) -> np.ndarray:
    """
    Return a fixed-size log-mel spectrogram.

    Short spectrograms are padded with zeros.
    Long spectrograms are truncated to max_frames.

    Returns:
        Float32 array with shape (N_MELS, max_frames).
    """
    if max_frames <= 0:
        raise ValueError("max_frames must be greater than zero.")

    features = extract_log_mel_spectrogram(file_path)

    if features.shape[1] >= max_frames:
        features = features[:, :max_frames]
    else:
        padding_width = max_frames - features.shape[1]
        features = np.pad(
            features,
            ((0, 0), (0, padding_width)),
            mode="constant",
            constant_values=0,
        )

    return features.astype(np.float32)


if __name__ == "__main__":
    print("Log-mel spectrogram extractor loaded successfully.")
    print(f"Target sample rate: {SAMPLE_RATE}")
    print(f"Number of mel bands: {N_MELS}")
    print(f"Default maximum frames: {DEFAULT_MAX_FRAMES}")
