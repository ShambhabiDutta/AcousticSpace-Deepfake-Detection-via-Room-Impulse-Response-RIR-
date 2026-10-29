import torch
from transformers import ASTForAudioClassification

name = "MIT/ast-finetuned-audioset-10-10-0.4593"
model = ASTForAudioClassification.from_pretrained(
    name,
    num_labels=2,
    ignore_mismatched_sizes=True
)
model.eval()

x = torch.zeros(1, 1024, 128)
with torch.no_grad():
    output = model(input_values=x)

print("Number of classes:", model.config.num_labels)
print("Logits shape:", tuple(output.logits.shape))
print("Binary AST head test passed.")
