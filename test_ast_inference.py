from pathlib import Path
import torch
from transformers import ASTForAudioClassification, ASTFeatureExtractor
from audio_ml.audio_processor import load_audio

model_dir = Path("audio_ml/models/ast_binary")
audio_path = Path(r"C:\Users\DELL\Documents\LA\ASVspoof2019_LA_train\flac\LA_T_1138215.flac")

model = ASTForAudioClassification.from_pretrained(model_dir).eval()
extractor = ASTFeatureExtractor.from_pretrained(model_dir)

audio, sr = load_audio(str(audio_path))
inputs = extractor(audio, sampling_rate=sr, return_tensors="pt")

with torch.no_grad():
    logits = model(input_values=inputs["input_values"]).logits
    probabilities = torch.softmax(logits, dim=-1)[0]

print("Checkpoint loaded:", model_dir.resolve())
print("Class labels:", model.config.id2label)
print("Logits:", logits.tolist())
print("Probabilities:", probabilities.tolist())
