import torch

from src.models.transformer_decoder import (
    MultiHeadSelfAttention,
    create_causal_mask,
    scaled_dot_product_attention,
)


def test_scaled_dot_product_attention_output_shape():
    queries = torch.randn(2, 3, 4, 5)
    keys = torch.randn(2, 3, 4, 5)
    values = torch.randn(2, 3, 4, 5)

    context, attention_weights = scaled_dot_product_attention(queries, keys, values)

    assert context.shape == (2, 3, 4, 5)
    assert attention_weights.shape == (2, 3, 4, 4)


def test_causal_mask_shape_and_correctness():
    mask = create_causal_mask(sequence_length=4)

    assert mask.shape == (4, 4)
    assert mask[0, 0].item() == 0.0
    assert mask[0, 1].item() == float("-inf")
    assert mask[3, 0].item() == 0.0


def test_future_tokens_are_masked_by_attention():
    queries = torch.ones(1, 1, 3, 2)
    keys = torch.ones(1, 1, 3, 2)
    values = torch.tensor([[[[1.0, 0.0], [10.0, 0.0], [100.0, 0.0]]]])
    mask = create_causal_mask(sequence_length=3).view(1, 1, 3, 3)

    context, _ = scaled_dot_product_attention(queries, keys, values, attention_mask=mask)

    assert torch.isclose(context[0, 0, 0, 0], torch.tensor(1.0))


def test_multi_head_attention_output_shape():
    attention = MultiHeadSelfAttention(embedding_dim=12, num_heads=3, dropout=0.0)
    input_embeddings = torch.randn(2, 5, 12)

    output = attention(input_embeddings)

    assert output.shape == (2, 5, 12)
