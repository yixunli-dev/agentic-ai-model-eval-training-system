import torch

from src.models.transformer_decoder import DecoderBlock, TransformerDecoderModel
from src.training.losses import masked_language_model_loss


def test_decoder_block_output_shape():
    block = DecoderBlock(embedding_dim=12, num_heads=3, feedforward_dim=24, dropout=0.0)
    hidden_states = torch.randn(2, 4, 12)

    output = block(hidden_states)

    assert output.shape == (2, 4, 12)


def test_transformer_decoder_model_forward_output_shape():
    model = TransformerDecoderModel(
        vocab_size=20,
        sequence_length=6,
        embedding_dim=12,
        num_heads=3,
        num_layers=2,
        feedforward_dim=24,
        dropout=0.0,
    )
    token_ids = torch.tensor([[2, 3, 4, 0], [5, 6, 0, 0]])

    logits = model(token_ids, pad_token_id=0)

    assert logits.shape == (2, 4, 20)


def test_transformer_training_loss_decreases_on_tiny_dataset():
    torch.manual_seed(7)
    token_ids = torch.tensor([[2, 3, 2], [3, 2, 3]], dtype=torch.long)
    target_ids = torch.tensor([[3, 2, 3], [2, 3, 2]], dtype=torch.long)
    mask = torch.ones_like(token_ids, dtype=torch.bool)
    model = TransformerDecoderModel(
        vocab_size=4,
        sequence_length=3,
        embedding_dim=8,
        num_heads=2,
        num_layers=1,
        feedforward_dim=16,
        dropout=0.0,
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=0.03)

    losses = []
    for _ in range(25):
        optimizer.zero_grad()
        logits = model(token_ids, pad_token_id=0)
        loss = masked_language_model_loss(logits, target_ids, mask)
        loss.backward()
        optimizer.step()
        losses.append(loss.item())

    assert losses[-1] < losses[0]
