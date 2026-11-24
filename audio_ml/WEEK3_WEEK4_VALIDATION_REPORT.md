# AcousticSpace — Week 3 & Week 4 Audio Forensics, Model Validation & Deployment Report

**Author:** Shambhabi Dutta (Member 2 — Audio Forensics, Audio ML, Model Validation & Deployment Support)
**Environment:** AMD Ryzen 7 5700U (16 CPU threads, 14 GiB RAM, CPU-only), Python 3.14, PyTorch 2.14.1+cpu, Transformers 5.21.0, Librosa 1.0.1, SciPy 1.18.1, Scikit-learn 1.9.1

## 1. Reproducible Preprocessing & Training/Inference Parity
- **Module:** `audio_ml/feature_extractor.py` (`extract_fixed_length_spectrogram`, `normalize_cnn_spectrogram`, `verify_preprocessing_parity`)
- **Parity Fix:** Updated CNN spectrogram padding from `0.0` to `-80.0 dB` silence floor to match `audio_ml/models/model_manifest.json` (`clip((spec + 80.0) / 80.0, 0.0, 1.0)`, shape `[1, 64, 400]` at `16,000 Hz`).
- **AST Parity:** Verified `16,000 Hz` mono, `128` mel bins, `1024` max frames, AudioSet normalization (`mean=-4.2677393`, `std=4.5689974`).

## 2. Dataset Documentation & Model Comparison
- **ASVspoof 2019 LA:** Training (`25,380` utterances: `2,580` bonafide [0], `22,800` spoof [1]; 80/20 stratified validation split = `5,076` samples); Development (`24,844` utterances: `2,548` bonafide, `22,296` spoof; balanced subset = `200` samples in `ast_200_manifest.csv`).
- **Acoustic Condition Benchmark:** `100` paired `.wav` samples (`20` each across `clean`, `reverb`, `noisy_snr10`, `noisy_snr5`, `noisy_snr0`).

| Model | Evaluation Split | Samples | Threshold | Accuracy | Precision | Recall | F1-Score | ROC-AUC | Confusion Matrix `[[TN, FP], [FN, TP]]` |
|---|---|---:|---:|---:|---:|---:|---:|---:|---|
| Logistic Regression (Baseline) | ASVspoof19 LA Val (80/20) | 5,076 | 0.50 | 0.9224 | 0.9913 | 0.9217 | 0.9552 | 0.9767 | `[[479, 37], [357, 4203]]` |
| Baseline CNN | ASVspoof19 LA Val (80/20) | 5,076 | 0.50 | 0.8883 | 0.9625 | 0.9112 | 0.9361 | 0.9243 | `[[354, 162], [405, 4155]]` |
| Baseline CNN | Full ASVspoof19 LA Dev | 24,844 | 0.60 | 0.8957 | 0.9838 | 0.8985 | 0.9392 | 0.9434 | `[[2218, 330], [2263, 20033]]` |
| AST Transformer | ASVspoof19 LA Dev (Balanced) | 200 | 0.30 | 0.8300 | 0.9125 | 0.7300 | 0.8111 | 0.9069 | `[[93, 7], [27, 73]]` |
| AST Transformer | Full ASVspoof19 LA Dev | 24,844 | 0.30 | 0.7539 | 0.9858 | 0.7364 | 0.8430 | 0.9069 | `[[2311, 237], [5878, 16418]]` |

## 3. RIR Decay & Breathing-Cadence Validation
- **Measured Blind RIR vs. Proxies (`audio_ml/rir_features.py`):** Separated Schroeder backward-integration decay estimates (`blind_edt_seconds`, `blind_t30_seconds`, `blind_t60_seconds`, `drr_proxy_db`, `c50_clarity_db`) from simple acoustic proxies (`tail_energy_ratio`, `zero_crossing_rate`, `mean_rms`, `spectral_centroid_hz`).
- **Breathing & Syllable Cadence (`audio_ml/cadence_analysis.py`):** Experimentally supported metrics (`syllable_count`, `syllable_rate_per_sec`, `mean_inter_syllable_interval_sec`, `isi_coefficient_of_variation`) vs. exploratory frontend time-segment localization (`suspicious_segments`, emitted only when `duration >= 1.0s` and `syllable_count >= 3`).

| Condition | Samples | Mean Blind T60 (s) | Mean Tail Energy Ratio | Mean Syllable Rate (Hz) | Mean ISI CV |
|---|---:|---:|---:|---:|---:|
| `clean` | 20 | 2.9457 | 0.2644 | 3.5501 | 0.5352 |
| `reverb` | 20 | 3.3212 | 0.2262 | 3.0667 | 0.5592 |
| `noisy_snr10` | 20 | 3.6634 | 0.224 | 2.8666 | 0.5238 |
| `noisy_snr5` | 20 | 3.7395 | 0.2198 | 2.7666 | 0.5389 |
| `noisy_snr0` | 20 | 3.8486 | 0.2135 | 3.0999 | 0.4938 |

## 4. CPU Latency Profiling & Bottlenecks (Ryzen 7 5700U)
| Stage | Mean (ms) | Min (ms) | Max (ms) |
|---|---:|---:|---:|
| `audio_load_16k` | 0.33 | 0.32 | 0.34 |
| `cnn_log_mel_400` | 4.46 | 3.91 | 5.26 |
| `rir_decay_and_proxies` | 11.63 | 11.53 | 11.76 |
| `breathing_syllable_cadence` | 13.55 | 12.44 | 13.93 |
| `cnn_cpu_inference` | 14.18 | 11.82 | 22.9 |
| `ast_cpu_inference` | 1132.19 | 1121.94 | 1141.92 |

## 5. Duration & Format Robustness
| Duration | CNN Spoof Prob | AST Spoof Prob | Combined Latency (ms) | Status |
|---:|---:|---:|---:|---|
| 0.5s | 0.8645 | 0.9267 | 1080.13 | PASS |
| 1.0s | 0.8695 | 0.9432 | 1186.27 | PASS |
| 4.0s | 0.945 | 0.963 | 1174.57 | PASS |
| 10.0s | 0.9764 | 0.9576 | 1250.4 | PASS |
| 15.0s | 0.9764 | 0.9554 | 1315.55 | PASS |

| Format | CNN Spoof Prob | AST Spoof Prob | Status |
|---|---:|---:|---|
| `.wav` | 0.9315 | 0.9533 | PASS |
| `.flac` | 0.9315 | 0.9533 | PASS |
| `.ogg` | 0.9232 | 0.9431 | PASS |

## 6. Docker Deployment Alignment & Troubleshooting
- Aligned `Dockerfile` and `backend/requirements.txt` with `model_manifest.json` (`torch==2.14.1+cpu` from PyTorch CPU wheel index, `scipy==1.18.1`, `soundfile==0.14.0`, plus system `ffmpeg` and `libsndfile1`).
- Verified `ENV PYTHONPATH=/app`, `@lru_cache(maxsize=1)` model caching, and `torch.inference_mode()` for low-memory CPU container execution.
