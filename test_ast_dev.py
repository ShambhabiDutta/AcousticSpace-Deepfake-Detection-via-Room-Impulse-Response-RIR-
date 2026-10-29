from pathlib import Path
import torch
from transformers import ASTForAudioClassification, ASTFeatureExtractor
from audio_ml.audio_processor import load_audio

model_dir = Path("audio_ml/models/ast_binary")
audio_dir = Path(r"C:\Users\DELL\Documents\LA\ASVspoof2019_LA_dev\flac")

samples = [
    ("LA_D_1047731", "bonafide"),
    ("LA_D_1008730", "spoof"),
]

model = ASTForAudioClassification.from_pretrained(model_dir).eval()
extractor = ASTFeatureExtractor.from_pretrained(model_dir)

for audio_id, actual in samples:
    path = audio_dir / f"{audio_id}.flac"
    if not path.exists():
        print(f"Missing audio: {path}")
        continue

    waveform, sr = load_audio(str(path))
    inputs = extractor(
        waveform,
        sampling_rate=sr,
        return_tensors="pt"
    )

    with torch.no_grad():
        logits = model(input_values=inputs["input_values"]).logits
        probs = torch.softmax(logits, dim=-1)[0]

    predicted_id = int(probs.argmax())
    predicted = model.config.id2label[predicted_id]

    print(f"\nAudio: {audio_id}")
    print(f"Actual: {actual}")
    print(f"Predicted: {predicted}")
    print(f"Bonafide probability: {probs[0].item():.4f}")
    print(f"Spoof probability: {probs[1].item():.4f}")
