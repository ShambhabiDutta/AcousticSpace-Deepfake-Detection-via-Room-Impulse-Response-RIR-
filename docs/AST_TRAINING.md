# AcousticSpace — AST Training Pipeline

## 1. Overview

AcousticSpace uses the Hugging Face Audio Spectrogram Transformer (AST) as an experimental deepfake audio classification model.

The AST training pipeline fine-tunes a pretrained audio classification model using the ASVspoof2019 Logical Access (LA) dataset.

The objective is to distinguish genuine (bonafide) audio from spoofed audio.

## 2. Base Model

| Setting | Configuration |
|---|---|
| Model | MIT AST AudioSet |
| Hugging Face identifier | `MIT/ast-finetuned-audioset-10-10-0.4593` |
| Model architecture | ASTForAudioClassification |
| Number of classes | 2 |
| Class 0 | Bonafide |
| Class 1 | Spoof |
| Sample rate | 16,000 Hz |
| Mel frequency bins | 128 |
| Maximum input length | 1,024 |

## 3. Dataset

The training pipeline reads the ASVspoof2019 LA protocol files and uses the corresponding audio recordings.

Dataset protocol directory:

`C:\Users\DELL\Documents\LA\ASVspoof2019_LA_cm_protocols`

Protocol files:

- `ASVspoof2019.LA.cm.train.trn.txt`
- `ASVspoof2019.LA.cm.dev.trl.txt`

Audio directories:

- `ASVspoof2019_LA_train\flac`
- `ASVspoof2019_LA_dev\flac`

The dataset labels are mapped as follows:

| Label | Class |
|---|---|
| 0 | Bonafide / genuine |
| 1 | Spoof / fake |

The training script selects balanced subsets from the training and development records to keep the initial experiment manageable on CPU hardware.

## 4. Training Configuration

The current defaults in `backend/training/train_ast.py` are:

| Parameter | Default value |
|---|---:|
| Training samples per class | 16 |
| Development samples per class | 8 |
| Total training recordings | 32 |
| Total development recordings | 16 |
| Epochs | 1 |
| Batch size | 2 |
| Random seed | 42 |
| Device | CPU |
| Learning rate | `2e-5` |

The random seed is applied to PyTorch, Python's random module and NumPy to support reproducibility.

**Important:** This is a smoke-test configuration intended to verify that the AST fine-tuning pipeline works. It is not full-scale training on the complete ASVspoof2019 dataset.

## 5. Training Workflow

The training pipeline performs the following steps:

1. Reads the ASVspoof2019 LA training and development protocols.
2. Selects balanced subsets of bonafide and spoof recordings.
3. Loads the pretrained AST feature extractor.
4. Loads the pretrained AST audio classification model with two output classes.
5. Converts audio recordings into AST-compatible input features.
6. Creates training and development data loaders.
7. Fine-tunes the model using the configured number of epochs.
8. Evaluates the model on the development subset after each epoch.
9. Saves the trained model and feature extractor.

## 6. Optimization and Training Process

The AST model is fine-tuned using the AdamW optimizer.

| Component | Implementation |
|---|---|
| Optimizer | AdamW |
| Learning rate | `2e-5` |
| Loss function | AST model's classification loss |
| Gradient calculation | Backpropagation |
| Parameter updates | AdamW optimizer |

For each training batch, the pipeline:

1. Clears previously accumulated gradients using `optimizer.zero_grad()`.
2. Passes audio features and labels to the AST model.
3. Calculates the classification loss.
4. Performs backpropagation using `loss.backward()`.
5. Updates model parameters using `optimizer.step()`.
6. Accumulates training loss for epoch reporting.

The model is placed in training mode using `model.train()` during training.

After each epoch, the script reports average training loss and evaluates the model using the development data loader.

## 7. Checkpoint Saving

After training, the model and feature extractor are saved using Hugging Face's `save_pretrained()` functionality.

Output directory:

`audio_ml/models/ast_binary`

The saved model directory contains:

| File | Purpose |
|---|---|
| `config.json` | Model architecture and classification configuration |
| `model.safetensors` | Saved model weights |
| `preprocessor_config.json` | Audio feature extractor configuration |

The current script saves the final epoch's model and feature extractor.

It does not yet implement best-checkpoint selection, early stopping or automatic recovery from interrupted training.

## 8. Running the Training Script

Run the following command from the project root:

```powershell
python backend/training/train_ast.py
```

Optional training arguments:

```powershell
python backend/training/train_ast.py --samples-per-class 16 --epochs 1 --batch-size 2 --seed 42
```

The available arguments control:

- `--samples-per-class`: Number of recordings selected per class for training.
- `--epochs`: Number of training epochs.
- `--batch-size`: Number of recordings processed in each batch.
- `--seed`: Random seed for reproducibility.

## 9. Evaluation

The AST model is evaluated separately using:

`backend/training/evaluate_ast_full_dev.py`

The full ASVspoof2019 LA development dataset contains 24,844 recordings:

| Class | Recordings |
|---|---:|
| Bonafide | 2,548 |
| Spoof | 22,296 |
| **Total** | **24,844** |

The full development evaluation is separate from the small smoke-test fine-tuning subset.

The evaluation pipeline processes the development recordings and calculates classification metrics at multiple probability thresholds, including:

- Accuracy
- Precision
- Recall
- F1-score
- ROC-AUC
- Confusion matrix

The results help assess whether AST is suitable for integration into the AcousticSpace prediction API.

## 10. Limitations

- The current training defaults use only 32 training recordings and 16 development recordings.
- Training is performed on CPU.
- One epoch is insufficient to establish the best possible AST performance.
- The development dataset is class-imbalanced.
- The current script saves the final epoch's checkpoint rather than selecting the best checkpoint.
- Evaluation results must be interpreted with appropriate class-sensitive metrics.
- A successful checkpoint save does not by itself establish model accuracy or generalisation.
- The AST model requires its own compatible audio preprocessing pipeline; it cannot directly replace the existing Logistic Regression model in the API.

## 11. Future Improvements

- Increase the training dataset size.
- Experiment with additional training epochs.
- Evaluate learning-rate and batch-size configurations.
- Introduce validation-based checkpoint selection.
- Implement early stopping and checkpoint recovery.
- Investigate the AST feature-extractor warning about an all-zero mel filter.
- Compare AST performance with the existing Logistic Regression baseline and, if trained, the CNN model.
- Integrate the selected model into the prediction API with compatible preprocessing.
- Evaluate performance on an independent test dataset.

## 12. Summary

The AST fine-tuning pipeline provides an initial experimental deepfake audio classification model for AcousticSpace.

The current configuration verifies the training workflow using a small balanced subset and CPU-based processing.

Full-scale training, comprehensive evaluation, model comparison and final API integration remain important steps before selecting AST as the production prediction model.