# AcousticSpace – Model Evaluation Report

## 1. Experiment Overview

This report documents the development and evaluation of audio deepfake detection models for AcousticSpace.

The project evaluates three approaches:

- **Baseline:** Logistic Regression
- **CNN model:** Convolutional Neural Network
- **Advanced model:** Audio Spectrogram Transformer (AST)

The Logistic Regression and CNN models were evaluated using the same stratified validation split from the processed training dataset.

AST was evaluated separately using:
- A balanced subset of 200 recordings from the official ASVspoof2019 LA development set.
- The complete ASVspoof2019 LA development set containing 24,844 recordings.

The CNN was also evaluated on the complete development set using the saved development features.

The evaluation aims to compare model behaviour, examine false-positive and false-negative predictions, and support future model selection for the AcousticSpace prediction API.

**Important:** Results from different evaluation datasets should not be treated as directly comparable.

## 2. Dataset and Sample Selection

### 2.1 Processed Training Dataset

The processed training dataset contains:

| Class | Recordings |
|---|---:|
| Bonafide | 2,580 |
| Spoof | 22,800 |
| **Total** | **25,380** |

The Logistic Regression and CNN validation experiments used a stratified 80/20 split with random seed 42.

The resulting validation set contained 5,076 recordings.

### 2.2 Preliminary AST Evaluation Dataset

The preliminary AST evaluation used a deterministic balanced subset of the official ASVspoof2019 LA development protocol.

| Dataset detail | Count |
|---|---:|
| Total recordings | 200 |
| Bonafide recordings | 100 |
| Spoof recordings | 100 |

Selection settings:

- Dataset: ASVspoof2019 LA
- Split: Development
- Selection method: Balanced random sampling
- Random seed: 42
- Samples per class: 100

The exact recording list is saved in `audio_ml/evaluation/ast_200_manifest.csv`.

### 2.3 Full Development Dataset

The full development dataset contains:

| Dataset detail | Count |
|---|---:|
| Total recordings | 24,844 |
| Bonafide recordings | 2,548 |
| Spoof recordings | 22,296 |

The full development dataset is imbalanced. Therefore, balanced accuracy, precision, recall, F1-score, ROC-AUC and confusion matrices are important when interpreting model performance.

The development dataset is not an independent held-out test set.

## 3. Model Architectures

### 3.1 Baseline Logistic Regression

The baseline uses summarized log-mel spectrogram features with a StandardScaler and Logistic Regression classifier.

Configuration:

- Algorithm: Logistic Regression
- Maximum iterations: 1,000
- Class weighting: Balanced
- Random state: 42
- Input features: 128 summarized features
- Saved model: `audio_ml/models/baseline_logreg.joblib`

This model provides a classical machine-learning baseline for comparison.

### 3.2 Convolutional Neural Network

The CNN baseline is implemented in:

`backend/training/train_cnn.py`

Configuration:

- Framework: PyTorch
- Input representation: Log-mel spectrogram
- Input shape: `(1, 64, 400)`
- Batch size: 8
- Epochs: 3
- Optimizer: Adam
- Learning rate: 0.001
- Loss function: Weighted CrossEntropyLoss
- Random seed: 42
- Device: CPU

Saved checkpoint:

`audio_ml/models/cnn_baseline.pt`

The checkpoint was saved and successfully reloaded for verification.

### 3.3 Advanced AST Model

The advanced model uses the Hugging Face Audio Spectrogram Transformer.

Base model:

`MIT/ast-finetuned-audioset-10-10-0.4593`

The verified architecture configuration includes:

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

### 3.4 Classification Labels

| Label | Class |
|---|---|
| 0 | Bonafide (genuine audio) |
| 1 | Spoof (fake or manipulated audio) |

The label mapping is consistent across the AST configuration and inference pipeline.

## 4. AST Fine-Tuning Pipeline

The AST training pipeline is implemented in:

`backend/training/train_ast.py`

The pipeline is designed to initialize a pretrained AST model and adapt it for binary audio classification.

### Training configuration in the historical script

| Setting | Value |
|---|---|
| Framework | PyTorch |
| Model | ASTForAudioClassification |
| Optimizer | AdamW |
| Learning rate | 0.00002 |
| Random seed | 42 |
| Default training samples per class | 16 |
| Default development samples per class | 8 |
| Default epochs | 1 |
| Default batch size | 2 |
| Number of classes | 2 |

### Training pipeline steps

1. Load ASVspoof2019 LA training and development protocols.
2. Read bonafide and spoof labels.
3. Select balanced subsets using random sampling.
4. Load audio recordings.
5. Resample audio to 16 kHz and convert to mono.
6. Extract AST input features using ASTFeatureExtractor.
7. Initialize the pretrained AST model for binary classification.
8. Fine-tune the model using AdamW.
9. Evaluate the model on the selected development subset.
10. Save the model and feature extractor configuration.

### Training status and limitation

The fine-tuning pipeline is implemented, but the available evidence does not independently establish that the existing `ast_binary` checkpoint was produced by a completed fine-tuning run.

The historical training script contains a fine-tuning loop and uses `audio_ml/models/ast_binary` as its output directory. However, the existence of that script and directory alone does not prove that training was executed.

The exact training history and provenance of the evaluated weights remain unverified.

## 5. AST Checkpoint and Saved Files

The existing AST checkpoint directory is:

`audio_ml/models/ast_binary/`

The following files have been verified:

| File | Purpose |
|---|---|
| `config.json` | Model architecture and classification configuration |
| `model.safetensors` | Saved model weights |
| `preprocessor_config.json` | Audio preprocessing configuration |

The AST checkpoint loaded successfully during the saved evaluation workflow.

The full-development evaluation used the existing checkpoint directory. However, the precise origin of its weights—original pretrained weights or weights produced by additional fine-tuning—has not been independently verified.

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

A warning was observed indicating that at least one mel filter has all-zero values.

Recorded diagnostic findings:

- Filter-bank dimensions: 257 × 128
- Completely zero filter indices: `[3]`
- Number of completely zero filters: 1

Feature extraction and AST inference proceeded successfully. However, the warning remains under investigation and should be resolved or explained before production deployment.

## 7. Logistic Regression and CNN Validation Results

Both models were evaluated using the same stratified validation split from the processed training dataset.

- Validation recordings: 5,076
- Split: Stratified 80/20
- Random seed: 42

### 7.1 Validation Performance

| Metric | Logistic Regression | CNN |
|---|---:|---:|
| Accuracy | 0.9224 | 0.8883 |
| Balanced Accuracy | 0.9250 | 0.7986 |
| ROC-AUC | 0.9767 | 0.9243 |

### 7.2 Logistic Regression Confusion Matrix

Rows represent actual classes and columns represent predicted classes.

Class order: Bonafide, Spoof.

| Actual / Predicted | Bonafide | Spoof |
|---|---:|---:|
| Bonafide | 479 | 37 |
| Spoof | 357 | 4,203 |

### 7.3 CNN Training Progress

| Epoch | Training Loss | Validation Accuracy | Balanced Accuracy | ROC-AUC |
|---|---:|---:|---:|---:|
| 1 | 0.6567 | 0.8983 | 0.5000 | 0.5393 |
| 2 | 0.6362 | 0.8950 | 0.5024 | 0.8395 |
| 3 | 0.4668 | 0.8883 | 0.7986 | 0.9243 |

Final CNN validation confusion matrix:

| Actual / Predicted | Bonafide | Spoof |
|---|---:|---:|
| Bonafide | 354 | 162 |
| Spoof | 405 | 4,155 |

### 7.4 Validation Findings

Logistic Regression achieved higher accuracy, balanced accuracy and ROC-AUC than the CNN on their shared validation split.

The CNN's balanced accuracy and ROC-AUC improved over the three training epochs.

These results are validation results from the processed training dataset, not independent test-set results.

## 8. Preliminary AST Evaluation Results

The preliminary AST evaluation used the balanced 200-recording development subset described in Section 2.2.

### 8.1 ROC-AUC

| Model | ROC-AUC |
|---|---:|
| Logistic Regression | 0.8804 |
| AST checkpoint | 0.9069 |

The AST checkpoint achieved a higher ROC-AUC in this preliminary experiment.

The AST result should not be described as definitively fine-tuned because the provenance of the evaluated checkpoint remains unverified.

### 8.2 Classification Results at Threshold 0.30

| Metric | Logistic Regression | AST |
|---|---:|---:|
| Accuracy | 0.770 | 0.830 |
| Balanced Accuracy | 0.770 | 0.830 |
| Spoof Precision | 0.711 | 0.9125 |
| Spoof Recall | 0.910 | 0.730 |
| Spoof F1-score | 0.798 | 0.8111 |

### 8.3 Classification Results at Threshold 0.50

| Metric | Logistic Regression | AST |
|---|---:|---:|
| Accuracy | 0.790 | 0.745 |
| Balanced Accuracy | 0.790 | 0.745 |
| Spoof Precision | 0.759 | 0.9623 |
| Spoof Recall | 0.850 | 0.510 |
| Spoof F1-score | 0.802 | 0.6667 |

### 8.4 Classification Results at Threshold 0.70

| Metric | Logistic Regression | AST |
|---|---:|---:|
| Accuracy | 0.785 | 0.715 |
| Balanced Accuracy | 0.785 | 0.715 |
| Spoof Precision | 0.788 | 1.000 |
| Spoof Recall | 0.780 | 0.430 |
| Spoof F1-score | 0.784 | 0.6014 |

## 9. Preliminary AST Confusion Matrices

Rows represent actual classes and columns represent predicted classes.

Class order: Bonafide, Spoof.

### 9.1 AST at Threshold 0.30

| Actual / Predicted | Bonafide | Spoof |
|---|---:|---:|
| Bonafide | 93 | 7 |
| Spoof | 27 | 73 |

### 9.2 AST at Threshold 0.50

| Actual / Predicted | Bonafide | Spoof |
|---|---:|---:|
| Bonafide | 98 | 2 |
| Spoof | 49 | 51 |

### 9.3 AST at Threshold 0.70

| Actual / Predicted | Bonafide | Spoof |
|---|---:|---:|
| Bonafide | 100 | 0 |
| Spoof | 57 | 43 |

## 10. Full Development-Set AST Evaluation

The full AST evaluation has been completed on the official ASVspoof2019 LA development set.

### 10.1 Dataset and Evaluation Details

| Detail | Value |
|---|---|
| Dataset | ASVspoof2019 LA |
| Split | Development |
| Total recordings | 24,844 |
| Bonafide recordings | 2,548 |
| Spoof recordings | 22,296 |
| Device | CPU |
| Batch size | 1 |
| Evaluation status | Completed |
| Checkpoint directory | `audio_ml/models/ast_binary` |

### 10.2 Overall Performance

| Metric | Result |
|---|---:|
| ROC-AUC | 0.9069 |

### 10.3 Threshold-Based Performance

| Metric | Threshold 0.30 | Threshold 0.50 | Threshold 0.70 |
|---|---:|---:|---:|
| Accuracy | 0.7539 | 0.6249 | 0.4998 |
| Balanced Accuracy | 0.8217 | 0.7780 | 0.7185 |
| Spoof Precision | 0.9858 | 0.9943 | 0.9984 |
| Spoof Recall | 0.7364 | 0.5854 | 0.4434 |

### 10.4 Confusion Matrices

Rows represent actual classes and columns represent predicted classes.

Class order: Bonafide, Spoof.

**Threshold 0.30**

| Actual / Predicted | Bonafide | Spoof |
|---|---:|---:|
| Bonafide | 2,311 | 237 |
| Spoof | 5,878 | 16,418 |

**Threshold 0.50**

| Actual / Predicted | Bonafide | Spoof |
|---|---:|---:|
| Bonafide | 2,473 | 75 |
| Spoof | 9,243 | 13,053 |

**Threshold 0.70**

| Actual / Predicted | Bonafide | Spoof |
|---|---:|---:|
| Bonafide | 2,532 | 16 |
| Spoof | 12,411 | 9,885 |

### 10.5 Prediction Consistency Check

The 200 recordings in the balanced subset were compared with their corresponding predictions in the full-development checkpoint.

| Check | Result |
|---|---:|
| Matched recordings | 200 |
| Matching probabilities | 200 |
| Different probabilities | 0 |

All 200 spoof probabilities matched between the two prediction files within the comparison tolerance of 0.000001.

This confirms prediction consistency across the two saved evaluations. It supports the conclusion that the same scoring model state was used for the overlapping recordings, but it does not independently establish whether the checkpoint was pretrained or fine-tuned.

### 10.6 Model Provenance and Limitations

The full-development evaluation used the existing checkpoint in `audio_ml/models/ast_binary/`.

The historical evaluation and training scripts used this directory as the AST checkpoint output and loading location. However, the available evidence does not independently establish whether the evaluated weights were the original pretrained weights or weights produced by an actual fine-tuning run.

Therefore, the full-development results are reported as results from the existing AST checkpoint. Its fine-tuning status remains unverified.

The development dataset is not an independent held-out test set. These results should not be treated as final evidence of generalization to unseen recordings.

## 11. Full Development-Set CNN Evaluation

The trained CNN checkpoint was evaluated on the complete ASVspoof2019 LA development dataset using the saved development features.

### 11.1 Dataset and Evaluation Details

| Detail | Value |
|---|---|
| Dataset | ASVspoof2019 LA |
| Split | Development |
| Total recordings | 24,844 |
| Bonafide recordings | 2,548 |
| Spoof recordings | 22,296 |
| Model | Baseline CNN |
| Device | CPU |
| Batch size | 8 |
| Evaluation status | Completed |
| Checkpoint | `audio_ml/models/cnn_baseline.pt` |

### 11.2 Overall Performance

| Metric | Result |
|---|---:|
| ROC-AUC | **0.9434** |

### 11.3 Threshold-Based Performance

The CNN was evaluated at five classification thresholds.

| Metric | 0.30 | 0.40 | 0.50 | 0.60 | 0.70 |
|---|---:|---:|---:|---:|---:|
| Accuracy | 0.9042 | 0.9050 | 0.9033 | 0.8957 | 0.8735 |
| Balanced Accuracy | 0.7228 | 0.7816 | 0.8363 | 0.8845 | 0.9119 |
| Real Recall | 0.4945 | 0.6264 | 0.7520 | 0.8705 | 0.9604 |
| Spoof Recall | 0.9511 | 0.9368 | 0.9206 | 0.8985 | 0.8635 |

### 11.4 Threshold Analysis

Among the five tested thresholds, 0.70 achieved the highest balanced accuracy of 0.9119 (91.19%) and real recall of 0.9604 (96.04%).

At threshold 0.30, spoof recall was highest at 0.9511 (95.11%), while real recall was 0.4945 (49.45%).

At threshold 0.50, the CNN achieved 0.8363 (83.63%) balanced accuracy, with 0.7520 (75.20%) real recall and 0.9206 (92.06%) spoof recall.

Increasing the threshold improved real recall but reduced spoof recall. This demonstrates the trade-off between falsely identifying genuine recordings as spoof and missing spoof recordings.

The threshold of 0.50 remains the provisional CNN API setting pending further review and threshold selection based on application requirements.

### 11.5 Evaluation Limitations

- The CNN checkpoint was selected using balanced accuracy on the training validation split.
- The full development-set evaluation provides additional evaluation evidence but is not an independent final test-set result.
- The dataset is imbalanced, so accuracy alone does not fully describe classification performance.
- Performance on unseen datasets and recording conditions has not yet been established.

### 11.6 Saved Evaluation Artifact

| File | Description |
|---|---|
| `audio_ml/evaluation/cnn_dev_evaluation.json` | Full CNN development-set metrics and threshold comparison |

The evaluation processed all 24,844 development recordings successfully and saved the results.

## 12. Findings and Model Comparison

The evaluation provides the following findings:

1. Logistic Regression achieved stronger accuracy, balanced accuracy and ROC-AUC than the CNN on their shared validation split.
2. CNN validation performance improved across the three training epochs.
3. On the full development dataset, the CNN achieved a ROC-AUC of 0.9434.
4. Among the five tested CNN thresholds, 0.70 achieved the highest balanced accuracy of 91.19%.
5. CNN spoof recall was highest at threshold 0.30, while real recall was highest at threshold 0.70.
6. The AST checkpoint achieved a ROC-AUC of 0.9069 on both the preliminary 200-recording subset and the full development set.
7. On the preliminary 200-recording subset, AST achieved higher spoof precision than Logistic Regression at the evaluated thresholds.
8. Logistic Regression achieved stronger spoof recall at thresholds 0.50 and 0.70 in the preliminary comparison.
9. On the full development set, AST balanced accuracy was highest among its three evaluated thresholds at 0.30.
10. All 200 overlapping AST recordings had matching probabilities in the preliminary and full-development prediction files.
11. The AST checkpoint's fine-tuning provenance remains unverified.

The Logistic Regression and CNN validation results are based on a different evaluation split from the full development-set results. Therefore, their metrics should not be used to declare a definitive overall model winner.

Although the CNN and AST were both evaluated on the full development set, their ROC-AUC values alone are not sufficient to establish an overall winner. A consistent comparison should also consider threshold-specific metrics, error costs, and independent test-set performance.

## 13. Limitations

- The preliminary AST evaluation uses only 200 development recordings.
- The full AST and CNN evaluations use the development split, not an independent test set.
- The AST checkpoint's fine-tuning history has not been independently verified.
- The historical AST training script defaults to a small smoke-scale configuration.
- Threshold selection requires additional validation based on the intended operating point.
- The AST mel-filter warning has not been fully investigated.
- The Logistic Regression and CNN validation metrics are not directly comparable with their full-development metrics because the evaluation datasets differ.
- Performance on unseen recording conditions and attack types has not been fully established.
- Production readiness has not been established.

## 14. Future Work

1. Verify the AST checkpoint's training history and provenance.
2. Run a reproducible AST fine-tuning experiment when suitable GPU resources are available.
3. Save any fine-tuned model to a separate checkpoint directory so the existing checkpoint and evaluation results are preserved.
4. Investigate the zero-valued AST mel filter.
5. Perform threshold calibration using a clearly defined validation protocol.
6. Examine false-positive and false-negative predictions, including attack-specific errors.
7. Evaluate all candidate models using a consistent evaluation protocol.
8. Evaluate performance on a separate held-out test dataset.
9. Validate the prediction API against multiple known bonafide and spoof recordings.
10. Update model selection and API configuration only after sufficient validation.

## 15. Saved Evaluation Artifacts

The following evaluation artifacts have been identified:

| File | Description |
|---|---|
| `audio_ml/evaluation/ast_200_manifest.csv` | Exact 200-recording balanced subset manifest |
| `audio_ml/evaluation/ast_200_predictions.csv` | Per-recording AST predictions for the 200-sample subset |
| `audio_ml/evaluation/ast_200_metrics.json` | Preliminary AST evaluation metrics |
| `audio_ml/evaluation/ast_full_dev_checkpoint.csv` | Per-recording full-development AST probabilities |
| `audio_ml/evaluation/ast_full_dev_predictions.csv` | Full-development predictions at the configured threshold |
| `audio_ml/evaluation/ast_full_dev_metrics.json` | Full-development AST evaluation metrics |
| `audio_ml/evaluation/cnn_validation_metrics.json` | CNN validation metrics |
| `audio_ml/evaluation/cnn_threshold_analysis.json` | CNN validation threshold analysis |
| `audio_ml/evaluation/cnn_dev_evaluation.json` | Full CNN development-set metrics and threshold comparison |
| `audio_ml/evaluation/model_comparison_summary.json` | Preliminary model comparison summary |

The full-development prediction checkpoint contains 24,844 recordings. The saved AST prediction files were compared against the 200-sample subset, and all 200 overlapping probabilities matched.

## 16. Conclusion

The AcousticSpace project has established Logistic Regression and CNN validation baselines and completed preliminary and full-development-set evaluations of both CNN and AST.

Logistic Regression achieved stronger validation performance than the CNN on their shared validation split. The CNN achieved a ROC-AUC of 0.9434 on the full ASVspoof2019 LA development set, with balanced accuracy reaching 91.19% at threshold 0.70 among the tested operating points.

The AST checkpoint achieved a ROC-AUC of 0.9069 on the full ASVspoof2019 LA development set. Its full-development predictions were also checked for consistency with the preliminary evaluation, with all 200 overlapping recording probabilities matching.

The AST training pipeline is implemented, but the provenance of the existing evaluated checkpoint remains unverified. Consequently, the report does not claim that the full-development result is definitively from a fine-tuned AST model.

Further work is required to verify checkpoint provenance, complete a reproducible fine-tuning experiment, validate the API across multiple samples, calibrate thresholds using a clearly defined protocol, and evaluate the models under a consistent independent test protocol.

No production-readiness claim is made at this stage.