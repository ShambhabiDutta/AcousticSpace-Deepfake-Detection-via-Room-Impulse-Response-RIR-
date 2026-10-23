
import numpy as np
import pytest

from audio_ml.audio_processor import load_audio
from audio_ml.dataset import get_label


def test_real_audio_label():
    assert get_label("bonafide") == 0


def test_spoof_audio_label():
    assert get_label("spoof") == 1


def test_unknown_label_raises_error():
    with pytest.raises(ValueError):
        get_label("unknown")


def test_missing_audio_file_raises_error():
    with pytest.raises(Exception):
        load_audio("this_file_does_not_exist.wav")


def test_invalid_sample_rate_raises_error():
    with pytest.raises(ValueError):
        load_audio("dummy.wav", sample_rate=0)
