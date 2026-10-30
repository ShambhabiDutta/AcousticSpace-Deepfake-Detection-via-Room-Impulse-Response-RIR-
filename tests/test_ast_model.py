import torch
from transformers import ASTForAudioClassification


def test_ast_model_loads_and_returns_two_logits():
    name = "MIT/ast-finetuned-audioset-10-10-0.4593"

    model = ASTForAudioClassification.from_pretrained(
        name,
        num_labels=2,
        ignore_mismatched_sizes=True,
    )
    model.eval()

    x = torch.zeros(1, 1024, 128)

    with torch.no_grad():
        output = model(input_values=x)

    assert model.config.num_labels == 2
    assert tuple(output.logits.shape) == (1, 2)
