# AcousticSpace – Model Evaluation Report

## 1. Experiment Overview

This report documents the development and evaluation of audio deepfake detection models for AcousticSpace.

The experiment compares:

- **Baseline model:** Logistic Regression
- **Advanced model:** Fine-tuned Audio Spectrogram Transformer (AST)

The preliminary comparison uses the same balanced subset of the ASVspoof2019 LA development dataset for both models.

A separate full development-set evaluation of AST is currently being performed.

The objective is to evaluate model performance and identify a suitable model for the AcousticSpace prediction API.

## 2. Dataset and Sample Selection

### 2.1 Preliminary Evaluation Dataset

- Dataset: ASVspoof2019 LA
- Split: Development
- Total recordings: 200
- Bonafide recordings: 100
- Spoof recordings: 100
- Selection method: Balanced random sampling
- Random seed: 42

Both models received the same recordings and ground-truth labels.

### 2.2 Full Development Dataset

The separate full AST evaluation uses the following dataset:

| Dataset detail | Count |
|---|---:|
| Total recordings | 24,844 |
| Bonafide recordings | 2,548 |
| Spoof recordings | 22,296 |

The full development dataset is imbalanced. Therefore, evaluation should consider balanced accuracy, precision, recall, F1-score, ROC-AUC and confusion matrices rather than relying on accuracy alone.

## 3. Model Architecture

### 3.1 Baseline Logistic Regression

The existing baseline uses summarized log-mel spectrogram features with a StandardScaler and Logistic Regression classifier.

Configuration:

- Algorithm: Logistic Regression
- Maximum iterations: 1,000
- Class weighting: Balanced
- Random state: 42
- Input features: 128 summarized features
- Saved model: `audio_ml/models/baseline_logreg.joblib`

The model provides a classical machine-learning baseline for comparison with AST.

### 3.2 Advanced AST Model

The advanced model uses the Hugging Face Audio Spectrogram Transformer.

Pretrained model:

`MIT/ast-finetuned-audioset-10-10-0.4593`

Verified architecture configuration:

| Component | Configuration |
|---|---|
| Architecture | ASTForAudioClassification |
| Model type | Audio Spectrogram Transformer |
| Hidden size | 768 |
| Transformer layers | 12 |
| Attention heads | 12 |
| Intermediate size | 3,072 |
| Activation | GELU |
| Mel frequency bins | 128 |
| Patch size | 16 |
| Frequency stride | 10 |
| Time stride | 10 |
| Maximum sequence length | 1,024 |
| Classification type | Single-label binary classification |
| Data type | float32 |

### 3.3 Classification Labels

| Label | Class |
|---|---|
| 0 | Bonafide (genuine audio) |
| 1 | Spoof (fake or manipulated audio) |

The label mapping is consistent across the AST configuration and training pipeline.

## 4. AST Fine-Tuning Configuration

The AST training pipeline is implemented in:

`backend/training/train_ast.py`

The pipeline initializes a pretrained AST model and adapts it for binary audio classification.

### Training configuration

| Setting | Value |
|---|---|
| Framework | PyTorch |
| Model | ASTForAudioClassification |
| Optimizer | AdamW |
| Learning rate | 0.00002 |
| Device | CPU |
| Random seed | 42 |
| Default samples per training class | 16 |
| Default samples per development class | 8 |
| Default epochs | 1 |
| Default batch size | 2 |
| Number of classes | 2 |

### Training pipeline steps

1. Load ASVspoof2019 LA training and development protocols.
2. Read bonafide and spoof labels.
3. Select balanced subsets using random sampling.
4. Load audio recordings using librosa.
5. Resample audio to 16 kHz and convert to mono.
6. Extract AST input features using ASTFeatureExtractor.
7. Initialize the pretrained AST model for binary classification.
8. Fine-tune the model using AdamW.
9. Evaluate the model on the selected development subset.
10. Save the model and feature extractor configuration.

**Training limitation:** The script defaults to a small smoke-scale configuration of 16 recordings per class for training and 8 per class for development. These defaults do not represent full-dataset training. The exact settings used to produce the existing checkpoint must be verified from the training run before making stronger claims.

## 5. AST Checkpoint and Saved Files

The AST checkpoint directory is:

`audio_ml/models/ast_binary/`

The following files have been verified:

| File | Purpose |
|---|---|
| `config.json` | Model architecture and classification configuration |
| `model.safetensors` | Saved model weights |
| `preprocessor_config.json` | Audio preprocessing configuration |

The AST checkpoint was successfully loaded during the full development-set evaluation.

The model weights loaded successfully, with all 203 loading entries reported as loaded.

## 6. AST Preprocessing Configuration

The saved AST feature extractor uses the following settings:

| Setting | Value |
|---|---|
| Feature extractor | ASTFeatureExtractor |
| Sampling rate | 16,000 Hz |
| Feature size | 1 |
| Mel bins | 128 |
| Maximum input length | 1,024 |
| Normalization | Enabled |
| Mean | -4.2677393 |
| Standard deviation | 4.5689974 |
| Padding side | Right |
| Padding value | 0.0 |
| Attention mask | Disabled |

Audio preprocessing converts recordings into the expected representation for the AST model.

### 6.1 Preprocessing Diagnostic

A warning was observed indicating that at least one mel filter has all zero values.

Recorded diagnostic findings:

- Filter-bank dimensions: 257 × 128
- Completely zero filter indices: [3]
- Number of completely zero filters: 1

Feature extraction and AST inference proceeded successfully. However, the warning remains under investigation and should be resolved or explained before production deployment.

## 7. Preliminary Evaluation Results

The following results are from the preliminary balanced 200-recording development subset.

These results must be interpreted separately from the full development-set evaluation.

### 7.1 ROC-AUC

| Model | ROC-AUC |
|---|---:|
| Baseline Logistic Regression | 0.8804 |
| Fine-tuned AST | 0.9069 |

AST achieved a higher ROC-AUC in this preliminary experiment.

### 7.2 Classification Results at Threshold 0.30

| Metric | Baseline | AST |
|---|---:|---:|
| Accuracy | 0.770 | 0.830 |
| Balanced Accuracy | 0.770 | 0.830 |
| Spoof Precision | 0.711 | 0.912 |
| Spoof Recall | 0.910 | 0.730 |
| Spoof F1-score | 0.798 | 0.811 |

### 7.3 Classification Results at Threshold 0.50

| Metric | Baseline | AST |
|---|---:|---:|
| Accuracy | 0.790 | 0.745 |
| Balanced Accuracy | 0.790 | 0.745 |
| Spoof Precision | 0.759 | 0.962 |
| Spoof Recall | 0.850 | 0.510 |
| Spoof F1-score | 0.802 | 0.667 |

### 7.4 Classification Results at Threshold 0.70

| Metric | Baseline | AST |
|---|---:|---:|
| Accuracy | 0.785 | 0.715 |
| Balanced Accuracy | 0.785 | 0.715 |
| Spoof Precision | 0.788 | 1.000 |
| Spoof Recall | 0.780 | 0.430 |
| Spoof F1-score | 0.784 | 0.601 |

## 8. Preliminary Confusion Matrices

Rows represent actual classes and columns represent predicted classes.

Class order: Bonafide, Spoof.

### 8.1 Baseline Logistic Regression

**Threshold 0.30**

| Actual / Predicted | Bonafide | Spoof |
|---|---:|---:|
| Bonafide | 63 | 37 |
| Spoof | 9 | 91 |

**Threshold 0.50**

| Actual / Predicted | Bonafide | Spoof |
|---|---:|---:|
| Bonafide | 73 | 27 |
| Spoof | 15 | 85 |

**Threshold 0.70**

| Actual / Predicted | Bonafide | Spoof |
|---|---:|---:|
| Bonafide | 79 | 21 |
| Spoof | 22 | 78 |

### 8.2 Fine-Tuned AST

**Threshold 0.30**

| Actual / Predicted | Bonafide | Spoof |
|---|---:|---:|
| Bonafide | 93 | 7 |
| Spoof | 27 | 73 |

**Threshold 0.50**

| Actual / Predicted | Bonafide | Spoof |
|---|---:|---:|
| Bonafide | 98 | 2 |
| Spoof | 49 | 51 |

**Threshold 0.70**

| Actual / Predicted | Bonafide | Spoof |
|---|---:|---:|
| Bonafide | 100 | 0 |
| Spoof | 57 | 43 |

## 9. Findings

The preliminary evaluation indicates:

1. Both models completed inference on the same 200 development recordings.
2. AST achieved a higher ROC-AUC than Logistic Regression.
3. At threshold 0.30, AST achieved higher accuracy and balanced accuracy.
4. At thresholds 0.50 and 0.70, Logistic Regression achieved higher spoof recall and F1-score.
5. AST showed higher spoof precision but lower spoof recall at thresholds 0.50 and 0.70.
6. Classification performance changes substantially depending on the prediction threshold.

These findings are preliminary and should not be treated as final model-selection evidence.

## 10. Limitations

- The preliminary comparison uses only 200 development recordings.
- The preliminary results do not establish generalization to unseen data.
- Threshold selection requires separate validation.
- The AST mel-filter warning has not been fully investigated.
- The full development-set AST evaluation is being performed separately.
- The training script's default settings use a very small balanced subset.
- The CNN architecture exists, but a trained CNN checkpoint has not been established as an available comparison baseline.
- Production readiness has not been established.

## 11. Future Work

1. Complete the full AST development-set evaluation.
2. Compare AST performance with the Logistic Regression baseline.
3. Investigate the zero-valued AST mel filter.
4. Perform threshold calibration using validation data.
5. Examine false-positive and false-negative cases.
6. Evaluate performance on a separate held-out test dataset.
7. Update the prediction API after model selection.
8. Verify the model's inference performance and reliability before deployment.

## 12. Full Development-Set Evaluation Status

The full AST evaluation uses:

- Dataset: ASVspoof2019 LA development set
- Total recordings: 24,844
- Bonafide recordings: 2,548
- Spoof recordings: 22,296
- Device: CPU
- Batch size: 1
- Checkpoint interval: Every 100 recordings

The evaluation script supports checkpoint-based resumption to avoid repeating completed recordings after an interruption.

**Status:** In progress.

Final metrics, including accuracy, balanced accuracy, precision, recall, F1-score, ROC-AUC and confusion matrices, will be added after the evaluation completes.

## 13. Conclusion

The preliminary comparison indicates that the fine-tuned AST model achieves a higher ROC-AUC than the baseline Logistic Regression model on the selected balanced development subset.

However, Logistic Regression achieves stronger spoof recall at the 0.50 threshold.

The AST architecture, preprocessing configuration, training pipeline and saved checkpoint have been reviewed.

The full development-set evaluation and additional validation are necessary before selecting a final model or prediction threshold for the AcousticSpace prediction API.

No production-readiness claim is made at this stage.