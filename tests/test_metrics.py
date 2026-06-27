import torch

from src.training.losses import masked_cross_entropy_loss
from src.training.metrics import token_accuracy


def test_masked_cross_entropy_loss_ignores_padding_positions():
    logits = torch.tensor([[[0.0, 4.0], [4.0, 0.0]]])
    tag_ids = torch.tensor([[1, 1]])
    mask = torch.tensor([[True, False]])

    loss = masked_cross_entropy_loss(logits, tag_ids, mask)
    expected = torch.nn.functional.cross_entropy(
        logits[:, :1, :].reshape(1, 2),
        tag_ids[:, :1].reshape(1),
    )

    assert torch.isclose(loss, expected)


def test_token_accuracy_ignores_padding_positions():
    logits = torch.tensor([[[0.0, 3.0], [3.0, 0.0]]])
    tag_ids = torch.tensor([[1, 1]])
    mask = torch.tensor([[True, False]])

    accuracy = token_accuracy(logits, tag_ids, mask)

    assert accuracy == 1.0
