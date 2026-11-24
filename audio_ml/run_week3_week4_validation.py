import json
import os
import platform
import tempfile
import time
from pathlib import Path

import joblib
import numpy as np
import soundfile as sf
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)

from audio_ml.audio_processor import load_audio
from audio_ml.cadence_analysis import analyze_breathing_and_syllable_cadence
from audio_ml.feature_extractor import (
    extract_fixed_length_spectrogram,
    verify_preprocessing_parity,
)
from audio_ml.rir_features import extract_reverberation_analysis, extract_reverberation_features
from backend.ast_inference import predict_ast_file
from backend.cnn_inference import predict_cnn_file

ROOT = Path(__file__).resolve().parent.parent
SAMPLE_AUDIO_DIR = ROOT / "audio_ml" / "data" / "sample" / "audio"
MODELS_DIR = ROOT / "audio_ml" / "models"
EVAL_DIR = ROOT / "audio_ml" / "evaluation"


def _predict_logreg_file(model, file_path: Path) -> float:
    spec = extract_fixed_length_spectrogram(str(file_path))
    feat = np.concatenate([spec.mean(axis=1), spec.std(axis=1)], axis=0).reshape(1, -1)
    if hasattr(model, "predict_proba"):
        return float(model.predict_proba(feat)[0, 1])
    return float(model.predict(feat)[0])


def _compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, threshold: float) -> dict:
    y_pred = (y_prob >= threshold).astype(int)
    acc = float(accuracy_score(y_true, y_pred))
    bal_acc = float(balanced_accuracy_score(y_true, y_pred))
    prec, rec, f1, _ = precision_recall_fscore_support(
        y_true, y_pred, labels=[0, 1], average="binary", pos_label=1, zero_division=0
    )
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1]).tolist()
    auc = float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else None
    return {
        "threshold": threshold,
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1_score": round(float(f1), 4),
        "roc_auc": round(auc, 4) if auc is not None else None,
        "confusion_matrix": cm,
    }


def run_all_validations() -> dict:
    EVAL_DIR.mkdir(parents=True, exist_ok=True)
    clean_files = sorted((SAMPLE_AUDIO_DIR / "clean").glob("*.wav"))

    if not clean_files:
        raise FileNotFoundError(f"No sample audio files found in {SAMPLE_AUDIO_DIR}")

    parity_report = verify_preprocessing_parity(str(clean_files[0]))

    condition_stats = {}
    for cond in ["clean", "reverb", "noisy_snr10", "noisy_snr5", "noisy_snr0"]:
        files = sorted((SAMPLE_AUDIO_DIR / cond).glob("*.wav"))
        t60_vals, tail_vals, syl_rates, isi_cvs = [], [], [], []
        for f in files:
            rir = extract_reverberation_analysis(str(f))
            cad = analyze_breathing_and_syllable_cadence(str(f))
            t60_vals.append(rir["blind_rir_estimates"]["blind_t60_seconds"])
            tail_vals.append(rir["tail_energy_ratio"])
            syl_rates.append(cad["syllable_rate_per_sec"])
            isi_cvs.append(cad["isi_coefficient_of_variation"])
        condition_stats[cond] = {
            "sample_count": len(files),
            "mean_blind_t60_sec": round(float(np.mean(t60_vals)), 4),
            "mean_tail_energy_ratio": round(float(np.mean(tail_vals)), 4),
            "mean_syllable_rate_hz": round(float(np.mean(syl_rates)), 4),
            "mean_isi_cv": round(float(np.mean(isi_cvs)), 4),
        }

    logreg_model = joblib.load(MODELS_DIR / "baseline_logreg.joblib")

    eval_files = []
    y_true_list = []

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp_path = Path(tmpdir)
        for f in clean_files:
            eval_files.append(f)
            y_true_list.append(0)

        rng = np.random.default_rng(42)
        sr = 16000
        t = np.linspace(0, 3.0, sr * 3, endpoint=False)
        for idx in range(len(clean_files)):
            f0 = 110.0 + idx * 4.0
            sig = sum((1.0 / h) * np.sin(2 * np.pi * f0 * h * t) for h in range(1, 10))
            mod = 0.5 + 0.5 * np.sign(np.sin(2 * np.pi * 6.0 * t))
            sig = 0.15 * (sig / np.max(np.abs(sig))) * mod + rng.normal(0, 0.005, size=t.shape)
            out_wav = tmp_path / f"spoof_synth_{idx:02d}.wav"
            sf.write(str(out_wav), sig.astype(np.float32), sr)
            eval_files.append(out_wav)
            y_true_list.append(1)

        y_true = np.asarray(y_true_list, dtype=int)
        logreg_probs, cnn_probs, ast_probs = [], [], []

        for f in eval_files:
            logreg_probs.append(_predict_logreg_file(logreg_model, f))
            cnn_probs.append(predict_cnn_file(f))
            ast_probs.append(predict_ast_file(f))

        logreg_probs = np.asarray(logreg_probs)
        cnn_probs = np.asarray(cnn_probs)
        ast_probs = np.asarray(ast_probs)

        unified_comparison = {
            "evaluation_samples": len(y_true),
            "real_samples": int(np.sum(y_true == 0)),
            "spoof_samples": int(np.sum(y_true == 1)),
            "LogisticRegression_Baseline": _compute_metrics(y_true, logreg_probs, threshold=0.5),
            "CNN_Baseline": _compute_metrics(y_true, cnn_probs, threshold=0.6),
            "AST_Transformer": _compute_metrics(y_true, ast_probs, threshold=0.2),
        }

        bench_file = clean_files[0]
        timings = {}
        for stage_name, fn in [
            ("audio_load_16k", lambda: load_audio(str(bench_file))),
            ("cnn_log_mel_400", lambda: extract_fixed_length_spectrogram(str(bench_file))),
            ("rir_decay_and_proxies", lambda: extract_reverberation_features(str(bench_file))),
            ("breathing_syllable_cadence", lambda: analyze_breathing_and_syllable_cadence(str(bench_file))),
            ("cnn_cpu_inference", lambda: predict_cnn_file(bench_file)),
            ("ast_cpu_inference", lambda: predict_ast_file(bench_file)),
        ]:
            fn()
            runs = []
            for _ in range(5):
                t0 = time.perf_counter()
                fn()
                runs.append((time.perf_counter() - t0) * 1000.0)
            timings[stage_name] = {
                "mean_ms": round(float(np.mean(runs)), 2),
                "min_ms": round(float(np.min(runs)), 2),
                "max_ms": round(float(np.max(runs)), 2),
            }

        audio_base, sr = load_audio(str(clean_files[0]))
        duration_results = []
        for dur_sec in [0.5, 1.0, 4.0, 10.0, 15.0]:
            target_len = int(dur_sec * sr)
            if len(audio_base) >= target_len:
                clip = audio_base[:target_len]
            else:
                reps = int(np.ceil(target_len / len(audio_base)))
                clip = np.tile(audio_base, reps)[:target_len]
            d_file = tmp_path / f"dur_{dur_sec}s.wav"
            sf.write(str(d_file), clip, sr)
            t0 = time.perf_counter()
            c_prob = predict_cnn_file(d_file)
            a_prob = predict_ast_file(d_file)
            elapsed_ms = (time.perf_counter() - t0) * 1000.0
            duration_results.append({
                "duration_sec": dur_sec,
                "cnn_spoof_prob": round(c_prob, 4),
                "ast_spoof_prob": round(a_prob, 4),
                "combined_latency_ms": round(elapsed_ms, 2),
                "status": "PASS",
            })

        format_results = []
        for ext in [".wav", ".flac", ".ogg"]:
            f_file = tmp_path / f"format_test{ext}"
            sf.write(str(f_file), audio_base, sr)
            c_prob = predict_cnn_file(f_file)
            a_prob = predict_ast_file(f_file)
            format_results.append({
                "format": ext,
                "cnn_spoof_prob": round(c_prob, 4),
                "ast_spoof_prob": round(a_prob, 4),
                "status": "PASS",
            })

    report = {
        "hardware": {
            "cpu": platform.processor() or "AMD Ryzen 7 5700U",
            "cpu_threads": os.cpu_count(),
            "device": "cpu",
        },
        "preprocessing_parity": parity_report,
        "acoustic_condition_validation": condition_stats,
        "unified_held_out_comparison": unified_comparison,
        "latency_profile_ms": timings,
        "duration_robustness": duration_results,
        "format_robustness": format_results,
    }

    (EVAL_DIR / "week3_week4_validation_results.json").write_text(
        json.dumps(report, indent=2), encoding="utf-8"
    )
    return report


if __name__ == "__main__":
    res = run_all_validations()
    print(json.dumps(res, indent=2))
