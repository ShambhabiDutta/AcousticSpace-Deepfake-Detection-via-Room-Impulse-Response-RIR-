
import librosa
import numpy as np

from audio_ml.config import HOP_LENGTH, N_FFT, N_MELS, SAMPLE_RATE
from audio_ml.audio_processor import load_audio


def extract_log_mel_spectrogram(
    file_path: str,
) -> np.ndarray:
    """
    Convert an audio file into a log-mel spectrogram.

    Returns:
        A float32 array with shape (n_mels, time_frames).
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


if __name__ == "__main__":
    print("Log-mel spectrogram extractor loaded successfully.")
    print(f"Target sample rate: {SAMPLE_RATE}")
    print(f"Number of mel bands: {N_MELS}")
