
import librosa
import numpy as np


def load_audio(
    file_path: str,
    sample_rate: int = 16000,
) -> tuple[np.ndarray, int]:
    """
    Load an audio file as mono audio at the target sample rate.

    Returns:
        audio: One-dimensional floating-point audio array.
        sample_rate: The sample rate of the returned audio.
    """
    if sample_rate <= 0:
        raise ValueError("Sample rate must be greater than zero.")

    audio, actual_sample_rate = librosa.load(
        file_path,
        sr=sample_rate,
        mono=True,
    )

    if audio.size == 0:
        raise ValueError("The audio file contains no audio samples.")

    if not np.all(np.isfinite(audio)):
        raise ValueError("Audio contains invalid numeric values.")

    return audio.astype(np.float32), actual_sample_rate


if __name__ == "__main__":
    print("Audio processor module loaded successfully.")
    print("Use load_audio(file_path) to process an audio file.")

