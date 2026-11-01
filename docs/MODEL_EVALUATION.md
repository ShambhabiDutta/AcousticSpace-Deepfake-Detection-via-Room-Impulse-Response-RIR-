# AcousticSpace – Model Evaluation Report

## 1. Experiment Overview

This experiment compares two audio deepfake detection models:

- Baseline: Logistic Regression
- Advanced model: Fine-tuned Audio Spectrogram Transformer (AST)

Both models were evaluated using the same balanced subset of the ASVspoof2019 LA development dataset.

## 2. Dataset and Sample Selection

- Dataset: ASVspoof2019 LA
- Split: Development
- Total recordings: 200
- Bonafide recordings: 100
- Spoof recordings: 100
- Selection method: Balanced random sampling
- Random seed: 42

Both models received the same recordings and ground-truth labels.

## 3. Evaluation Results

### ROC-AUC

| Model | ROC-AUC |
|---|---:|
| Baseline Logistic Regression | 0.8804 |
| Fine-tuned AST | 0.9069 |

AST achieved a higher ROC-AUC in this experiment.

### Classification Results at Threshold 0.30

| Metric | Baseline | AST |
|---|---:|---:|
| Accuracy | 0.770 | 0.830 |
| Balanced Accuracy | 0.770 | 0.830 |
| Spoof Precision | 0.711 | 0.912 |
| Spoof Recall | 0.910 | 0.730 |
| Spoof F1-score | 0.798 | 0.811 |

### Classification Results at Threshold 0.50

| Metric | Baseline | AST |
|---|---:|---:|
| Accuracy | 0.790 | 0.745 |
| Balanced Accuracy | 0.790 | 0.745 |
| Spoof Precision | 0.759 | 0.962 |
| Spoof Recall | 0.850 | 0.510 |
| Spoof F1-score | 0.802 | 0.667 |

### Classification Results at Threshold 0.70

| Metric | Baseline | AST |
|---|---:|---:|
| Accuracy | 0.785 | 0.715 |
| Balanced Accuracy | 0.785 | 0.715 |
| Spoof Precision | 0.788 | 1.000 |
| Spoof Recall | 0.780 | 0.430 |
| Spoof F1-score | 0.784 | 0.601 |

## 4. Confusion Matrices

Rows represent actual classes and columns represent predicted classes.

Class order: Bonafide, Spoof.

### Baseline Logistic Regression

Threshold 0.30:

| Actual / Predicted | Bonafide | Spoof |
|---|---:|---:|
| Bonafide | 63 | 37 |
| Spoof | 9 | 91 |

Threshold 0.50:

| Actual / Predicted | Bonafide | Spoof |
|---|---:|---:|
| Bonafide | 73 | 27 |
| Spoof | 15 | 85 |

Threshold 0.70:

| Actual / Predicted | Bonafide | Spoof |
|---|---:|---:|
| Bonafide | 79 | 21 |
| Spoof | 22 | 78 |

### Fine-tuned AST

Threshold 0.30:

| Actual / Predicted | Bonafide | Spoof |
|---|---:|---:|
| Bonafide | 93 | 7 |
| Spoof | 27 | 73 |

Threshold 0.50:

| Actual / Predicted | Bonafide | Spoof |
|---|---:|---:|
| Bonafide | 98 | 2 |
| Spoof | 49 | 51 |

Threshold 0.70:

| Actual / Predicted | Bonafide | Spoof |
|---|---:|---:|
| Bonafide | 100 | 0 |
| Spoof | 57 | 43 |

## 5. AST Preprocessing Configuration

The saved AST feature extractor uses:

- Sampling rate: 16,000 Hz
- Mel bins: 128
- Maximum input length: 1,024
- Normalization: Enabled

The saved AST model configuration uses:

- Patch size: 16
- Frequency stride: 10
- Time stride: 10

A preprocessing warning was observed indicating that one mel filter is entirely zero.

Diagnostic findings:

- Filter-bank dimensions: 257 × 128
- Completely zero filter indices: [3]
- Number of completely zero filters: 1

Feature extraction and AST inference completed successfully. The warning remains under investigation.

## 6. Findings

1. Both models completed inference on the same 200 development recordings.
2. AST achieved a higher ROC-AUC than the baseline in this experiment.
3. At threshold 0.30, AST achieved higher accuracy and balanced accuracy.
4. At threshold 0.50, the baseline achieved higher spoof recall and F1-score.
5. AST showed higher spoof precision but lower spoof recall at thresholds 0.50 and 0.70.
6. Classification performance changes substantially with the prediction threshold.

## 7. Limitations

- This is a preliminary experiment using 200 development recordings.
- Results from this subset do not establish generalization to unseen data.
- Threshold selection requires separate validation.
- The AST mel-filter warning has not yet been fully investigated.
- No production-readiness claim is made based on this experiment.

## 8. Future Work

- Evaluate both models on a separate held-out dataset.
- Investigate the zero-valued AST mel filter.
- Perform further threshold calibration using validation data.
- Compare false-positive and false-negative cases.
- Validate performance before integrating AST into the production prediction API.

## 9. Conclusion

The preliminary comparison indicates that the fine-tuned AST model has a higher ROC-AUC than the baseline Logistic Regression model on the selected development subset. However, the baseline achieves stronger spoof recall at the 0.50 threshold.

Further validation is required before choosing a final model or prediction threshold for AcousticSpace.


