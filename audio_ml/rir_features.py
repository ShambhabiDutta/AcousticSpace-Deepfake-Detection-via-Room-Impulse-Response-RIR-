from pathlib import Path

import librosa
import numpy as np
from scipy.signal import butter, filtfilt, hilbert

from audio_ml.audio_processor import load_audio


def _estimate_blind_decay_metrics(audio: np.ndarray, sample_rate: int) -> dict[str, float | bool]:
    """
    Estimate blind reverberation decay metrics via Schroeder backward integration
    on high-energy decay tails, explicitly flagging reliability limits.
    """
    nyq = 0.5 * sample_rate
    b, a = butter(4, [250.0 / nyq, min(4000.0 / nyq, 0.99)], btype="band")
    filtered = filtfilt(b, a, audio.astype(np.float64))

    envelope = np.abs(hilbert(filtered))
    frame_len = int(0.01 * sample_rate)
    if len(envelope) < frame_len * 10:
        return {
            "blind_edt_seconds": 0.0,
            "blind_t30_seconds": 0.0,
            "blind_t60_seconds": 0.0,
            "drr_proxy_db": 0.0,
            "c50_clarity_db": 0.0,
            "rir_estimate_reliable": False,
        }

    n_frames = len(envelope) // frame_len
    trimmed = envelope[: n_frames * frame_len].reshape(n_frames, frame_len)
    frame_energy = np.mean(trimmed ** 2, axis=1) + 1e-12

    peak_idx = int(np.argmax(frame_energy))
    decay_energy = frame_energy[peak_idx:]
    if len(decay_energy) < 8:
        return {
            "blind_edt_seconds": 0.0,
            "blind_t30_seconds": 0.0,
            "blind_t60_seconds": 0.0,
            "drr_proxy_db": 0.0,
            "c50_clarity_db": 0.0,
            "rir_estimate_reliable": False,
        }

    edc = np.cumsum(decay_energy[::-1])[::-1]
    edc_db = 10.0 * np.log10(edc / edc[0] + 1e-12)
    times = np.arange(len(edc_db)) * 0.01

    idx_10 = np.where(edc_db <= -10.0)[0]
    if len(idx_10) > 0 and idx_10[0] >= 2:
        end = idx_10[0] + 1
        slope_edt, _ = np.polyfit(times[:end], edc_db[:end], 1)
        edt = float(np.clip(-60.0 / min(slope_edt, -1e-3), 0.02, 5.0))
    else:
        edt = 0.0

    idx_5 = np.where(edc_db <= -5.0)[0]
    idx_35 = np.where(edc_db <= -35.0)[0]
    if len(idx_5) > 0 and len(idx_35) > 0 and idx_35[0] > idx_5[0] + 2:
        s, e = idx_5[0], idx_35[0] + 1
        slope_t30, _ = np.polyfit(times[s:e], edc_db[s:e], 1)
        t30 = float(np.clip(-30.0 / min(slope_t30, -1e-3), 0.02, 5.0))
        t60 = float(np.clip(-60.0 / min(slope_t30, -1e-3), 0.04, 10.0))
        reliable = True
    else:
        t30 = round(edt * 0.5, 4) if edt > 0 else 0.0
        t60 = round(edt, 4) if edt > 0 else 0.0
        reliable = False

    direct_samples = max(1, int(0.0025 * sample_rate))
    early_50ms_samples = max(1, int(0.050 * sample_rate))
    peak_sample = int(np.argmax(np.abs(filtered)))
    tail_audio = filtered[peak_sample:]

    e_direct = float(np.sum(tail_audio[:direct_samples] ** 2)) + 1e-12
    e_reverb = float(np.sum(tail_audio[direct_samples:] ** 2)) + 1e-12
    drr_db = float(10.0 * np.log10(e_direct / e_reverb))

    e_early50 = float(np.sum(tail_audio[:early_50ms_samples] ** 2)) + 1e-12
    e_late50 = float(np.sum(tail_audio[early_50ms_samples:] ** 2)) + 1e-12
    c50_db = float(10.0 * np.log10(e_early50 / e_late50))

    return {
        "blind_edt_seconds": round(edt, 4),
        "blind_t30_seconds": round(t30, 4),
        "blind_t60_seconds": round(t60, 4),
        "drr_proxy_db": round(drr_db, 4),
        "c50_clarity_db": round(c50_db, 4),
        "rir_estimate_reliable": reliable,
    }


def extract_reverberation_features(
    file_path: str,
) -> dict[str, float | bool | dict[str, float | bool]]:
    """
    Extract both simple acoustic proxies and blind RIR decay estimates,
    explicitly distinguishing proxy descriptors from blind RIR decay measures.
    """
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

    tail_start = int(len(energy) * 0.8)
    tail_energy_ratio = float(np.sum(energy[tail_start:]) / total_energy)
    zero_crossing_rate = float(np.mean(librosa.feature.zero_crossing_rate(y=audio)))
    spectral_centroid_hz = float(
        np.mean(librosa.feature.spectral_centroid(y=audio, sr=sample_rate))
    )

    blind_rir = _estimate_blind_decay_metrics(audio, sample_rate)

    return {
        "duration_seconds": float(len(audio) / sample_rate),
        "mean_rms": float(np.mean(rms)),
        "tail_energy_ratio": tail_energy_ratio,
        "zero_crossing_rate": zero_crossing_rate,
    }


def extract_reverberation_analysis(file_path: str) -> dict:
    base = extract_reverberation_features(file_path)
    audio, sample_rate = load_audio(file_path)
    return {
    }


def extract_reverberation_analysis(file_path: str) -> dict:
    base = extract_reverberation_features(file_path)
    audio, sample_rate = load_audio(file_path)
    return {
        "acoustic_proxies": base,
        "blind_rir_estimates": _estimate_blind_decay_metrics(audio, sample_rate),
    }
