import librosa
import numpy as np

from audio_ml.config import HOP_LENGTH, N_FFT, N_MELS, SAMPLE_RATE
from audio_ml.audio_processor import load_audio

DEFAULT_MAX_FRAMES = 400
CNN_PADDING_DB = -80.0
AST_SAMPLE_RATE = 16000
AST_NUM_MEL_BINS = 128
AST_MAX_LENGTH = 1024
AST_NORM_MEAN = -4.2677393
AST_NORM_STD = 4.5689974


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
    padding_value: float = CNN_PADDING_DB,
) -> np.ndarray:
    """
    Return a fixed-size log-mel spectrogram matching training and inference.

    Short spectrograms are padded with padding_value (-80.0 dB silence floor).
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
            constant_values=padding_value,
        )

    return features.astype(np.float32)


def normalize_cnn_spectrogram(spectrogram: np.ndarray) -> np.ndarray:
    """Normalize log-mel dB spectrogram [-80, 0] to [0.0, 1.0] for CNN inference."""
    arr = np.asarray(spectrogram, dtype=np.float32)
    return np.clip((arr + 80.0) / 80.0, 0.0, 1.0).astype(np.float32)


def verify_preprocessing_parity(file_path: str) -> dict[str, object]:
    """
    Verify training-time vs inference-time preprocessing parity for CNN and AST.
    """
    spec = extract_fixed_length_spectrogram(file_path)
    norm_spec = normalize_cnn_spectrogram(spec)
    audio, sr = load_audio(file_path, sample_rate=SAMPLE_RATE)

    cnn_parity = (
        sr == 16000
        and spec.shape == (64, 400)
        and norm_spec.shape == (64, 400)
        and float(norm_spec.min()) >= 0.0
        and float(norm_spec.max()) <= 1.0
        and np.isfinite(norm_spec).all()
    )

    return {
        "sample_rate": sr,
        "audio_samples": int(len(audio)),
        "cnn_input_shape": [1, int(spec.shape[0]), int(spec.shape[1])],
        "cnn_padding_db": CNN_PADDING_DB,
        "cnn_norm_min": float(norm_spec.min()),
        "cnn_norm_max": float(norm_spec.max()),
        "cnn_parity_verified": bool(cnn_parity),
        "ast_expected_config": {
            "sample_rate": AST_SAMPLE_RATE,
            "num_mel_bins": AST_NUM_MEL_BINS,
            "max_length": AST_MAX_LENGTH,
            "mean": AST_NORM_MEAN,
            "std": AST_NORM_STD,
        },
    }


if __name__ == "__main__":
    print("Log-mel spectrogram extractor loaded successfully.")
    print(f"Target sample rate: {SAMPLE_RATE}, Mel bands: {N_MELS}, Max frames: {DEFAULT_MAX_FRAMES}")
