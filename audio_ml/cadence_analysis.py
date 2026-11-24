from pathlib import Path

import numpy as np
from scipy.signal import butter, filtfilt, find_peaks, hilbert

from audio_ml.audio_processor import load_audio


def analyze_breathing_and_syllable_cadence(
    file_path: str | Path,
    min_segment_duration: float = 0.25,
) -> dict[str, object]:
    """
    Analyze spoken-syllable cadence, inter-syllable timing regularity,
    and inter-phrase breath/pause windows.

    Returns segment timestamps ONLY when the recording has sufficient duration
    and SNR/energy modulation to localize anomalous windows reliably.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Audio file not found: {path}")

    audio, sr = load_audio(str(path), sample_rate=16000)
    duration = float(len(audio) / sr)

    if len(audio) < int(0.2 * sr):
        raise ValueError("Audio is too short for cadence analysis (< 0.2s).")

    rms_total = float(np.sqrt(np.mean(audio ** 2)))
    if rms_total <= 1e-7:
        raise ValueError("Cannot analyze silent audio.")

    nyq = 0.5 * sr
    b, a = butter(4, [300.0 / nyq, 3400.0 / nyq], btype="band")
    speech_band = filtfilt(b, a, audio.astype(np.float64))

    envelope = np.abs(hilbert(speech_band))
    win = max(1, int(0.04 * sr))
    smooth_env = np.convolve(envelope, np.ones(win) / win, mode="same")

    peak_thresh = float(np.percentile(smooth_env, 90) * 0.25)
    min_dist = int(0.11 * sr)
    peaks, _ = find_peaks(smooth_env, distance=min_dist, prominence=max(peak_thresh * 0.4, 1e-5))

    syllable_times = peaks / float(sr)
    syllable_count = int(len(syllable_times))
    syllable_rate_hz = float(syllable_count / max(duration, 1e-6))

    if syllable_count >= 2:
        isis = np.diff(syllable_times)
        mean_isi = float(np.mean(isis))
        isi_cv = float(np.std(isis) / max(mean_isi, 1e-6))
    else:
        mean_isi = 0.0
        isi_cv = 0.0

    frame_len = int(0.02 * sr)
    n_frames = len(smooth_env) // frame_len
    frame_env = smooth_env[: n_frames * frame_len].reshape(n_frames, frame_len).mean(axis=1)
    active_thresh = float(np.percentile(frame_env, 75) * 0.22)
    is_pause = frame_env < active_thresh

    pause_windows = []
    in_pause = False
    start_f = 0
    for idx, flag in enumerate(is_pause):
        if flag and not in_pause:
            in_pause = True
            start_f = idx
        elif not flag and in_pause:
            in_pause = False
            dur_p = (idx - start_f) * 0.02
            if 0.18 <= dur_p <= 1.50:
                pause_windows.append((round(start_f * 0.02, 3), round(idx * 0.02, 3)))

    breath_window_count = len(pause_windows)

    segment_localization_supported = duration >= 1.0 and syllable_count >= 3
    suspicious_segments = []

    if segment_localization_supported:
        isis = np.diff(syllable_times)
        for i, isi in enumerate(isis):
            seg_start = float(syllable_times[i])
            seg_end = float(syllable_times[i + 1])
            if (seg_end - seg_start) >= min_segment_duration and isi > 0.65:
                suspicious_segments.append({
                    "start_sec": round(seg_start, 3),
                    "end_sec": round(seg_end, 3),
                    "reason": "abrupt_inter_syllable_gap",
                    "confidence": round(min(0.95, 0.55 + (isi - 0.65) * 0.5), 3),
                })

        peak_amps = smooth_env[peaks]
        amp_cv = float(np.std(peak_amps) / max(np.mean(peak_amps), 1e-6))
        if amp_cv < 0.18 and syllable_count >= 5:
            suspicious_segments.append({
                "start_sec": round(float(syllable_times[0]), 3),
                "end_sec": round(float(syllable_times[-1]), 3),
                "reason": "low_syllable_amplitude_variance",
                "confidence": round(min(0.90, 0.60 + (0.18 - amp_cv) * 1.5), 3),
            })

    return {
        "duration_seconds": round(duration, 3),
        "syllable_count": syllable_count,
        "syllable_rate_per_sec": round(syllable_rate_hz, 3),
        "mean_inter_syllable_interval_sec": round(mean_isi, 4),
        "isi_coefficient_of_variation": round(isi_cv, 4),
        "breath_pause_window_count": breath_window_count,
        "segment_localization_supported": segment_localization_supported,
        "suspicious_segments": suspicious_segments,
        "validation_status": {
            "experimentally_supported": [
                "syllable_count",
                "syllable_rate_per_sec",
                "mean_inter_syllable_interval_sec",
                "isi_coefficient_of_variation",
            ],
            "exploratory": [
                "breath_pause_window_count",
                "suspicious_segments",
            ],
        },
    }
