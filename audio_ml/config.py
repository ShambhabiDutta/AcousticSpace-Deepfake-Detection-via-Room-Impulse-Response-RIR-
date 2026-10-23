
from pathlib import Path

# Audio ML module directories
BASE_DIR = Path(__file__).resolve().parent

DATA_DIR = BASE_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
FEATURES_DIR = BASE_DIR / "features"
MODELS_DIR = BASE_DIR / "models"
TESTS_DIR = BASE_DIR / "tests"

# Audio preprocessing configuration
SAMPLE_RATE = 16000
N_FFT = 1024
HOP_LENGTH = 256
N_MELS = 64

# Dataset labels
REAL_LABEL = 0
SPOOF_LABEL = 1


def create_directories():
    """Create required directories if they do not exist."""
    for directory in (
        RAW_DATA_DIR,
        PROCESSED_DATA_DIR,
        FEATURES_DIR,
        MODELS_DIR,
        TESTS_DIR,
    ):
        directory.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    create_directories()
    print("Audio ML directories are ready.")
    print(f"Raw audio directory: {RAW_DATA_DIR}")
    print(f"Processed data directory: {PROCESSED_DATA_DIR}")
    print(f"Features directory: {FEATURES_DIR}")
