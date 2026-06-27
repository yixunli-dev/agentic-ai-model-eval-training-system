import math

import torch
import torch.nn as nn


def create_causal_mask(sequence_length, device=None):
    mask = torch.full((sequence_length, sequence_length), float("-inf"), device=device)
    return torch.triu(mask, diagonal=1)


def create_padding_mask(token_ids, pad_token_id=0):
    return token_ids == pad_token_id


def scaled_dot_product_attention(queries, keys, values, attention_mask=None):
    head_dim = queries.size(-1)
    scores = torch.matmul(queries, keys.transpose(-2, -1))
    scores = scores / math.sqrt(head_dim)

    if attention_mask is not None:
        scores = scores + attention_mask

    attention_weights = torch.softmax(scores, dim=-1)
    context = torch.matmul(attention_weights, values)
    return context, attention_weights


class PositionalEncoding(nn.Module):
    def __init__(self, embedding_dim, sequence_length):
        super().__init__()
        positions = torch.arange(sequence_length).unsqueeze(1)
        scale_values = torch.exp(
            torch.arange(0, embedding_dim, 2) * (-math.log(10000.0) / embedding_dim)
        )

        positional_encoding = torch.zeros(sequence_length, embedding_dim)
        positional_encoding[:, 0::2] = torch.sin(positions * scale_values)
        positional_encoding[:, 1::2] = torch.cos(positions * scale_values)
        self.register_buffer("positional_encoding", positional_encoding.unsqueeze(0))

    def forward(self, token_embeddings):
        sequence_length = token_embeddings.size(1)
        return token_embeddings + self.positional_encoding[:, :sequence_length, :]


class MultiHeadSelfAttention(nn.Module):
    def __init__(self, embedding_dim, num_heads, dropout=0.0):
        super().__init__()
        if embedding_dim % num_heads != 0:
            raise ValueError("embedding_dim must be divisible by num_heads")

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.query_layer = nn.Linear(embedding_dim, embedding_dim)
        self.key_layer = nn.Linear(embedding_dim, embedding_dim)
        self.value_layer = nn.Linear(embedding_dim, embedding_dim)
        self.output_layer = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def split_heads(self, tensor):
        batch_size, sequence_length, _ = tensor.shape
        tensor = tensor.view(batch_size, sequence_length, self.num_heads, self.head_dim)
        return tensor.transpose(1, 2)

    def combine_heads(self, tensor):
        batch_size, _, sequence_length, _ = tensor.shape
        tensor = tensor.transpose(1, 2).contiguous()
        return tensor.view(batch_size, sequence_length, self.embedding_dim)

    def forward(self, hidden_states, attention_mask=None):
        queries = self.split_heads(self.query_layer(hidden_states))
        keys = self.split_heads(self.key_layer(hidden_states))
        values = self.split_heads(self.value_layer(hidden_states))

        context, _ = scaled_dot_product_attention(
            queries,
            keys,
            values,
            attention_mask=attention_mask,
        )
        context = self.combine_heads(context)
        return self.output_layer(self.dropout(context))


class FeedForwardNetwork(nn.Module):
    def __init__(self, embedding_dim, feedforward_dim, dropout=0.0):
        super().__init__()
        self.layers = nn.Sequential(
            nn.Linear(embedding_dim, feedforward_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, embedding_dim),
        )

    def forward(self, hidden_states):
        return self.layers(hidden_states)


class DecoderBlock(nn.Module):
    def __init__(self, embedding_dim, num_heads, feedforward_dim, dropout=0.0):
        super().__init__()
        self.self_attention = MultiHeadSelfAttention(embedding_dim, num_heads, dropout)
        self.feed_forward = FeedForwardNetwork(embedding_dim, feedforward_dim, dropout)
        self.attention_norm = nn.LayerNorm(embedding_dim)
        self.feed_forward_norm = nn.LayerNorm(embedding_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states, attention_mask=None):
        attention_output = self.self_attention(hidden_states, attention_mask)
        hidden_states = self.attention_norm(hidden_states + self.dropout(attention_output))

        feed_forward_output = self.feed_forward(hidden_states)
        hidden_states = self.feed_forward_norm(
            hidden_states + self.dropout(feed_forward_output)
        )
        return hidden_states


class TransformerDecoderModel(nn.Module):
    def __init__(
        self,
        vocab_size,
        sequence_length,
        embedding_dim,
        num_heads,
        num_layers,
        feedforward_dim,
        dropout=0.0,
    ):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.positional_encoding = PositionalEncoding(embedding_dim, sequence_length)
        self.decoder_blocks = nn.ModuleList(
            [
                DecoderBlock(embedding_dim, num_heads, feedforward_dim, dropout)
                for _ in range(num_layers)
            ]
        )
        self.dropout = nn.Dropout(dropout)
        self.output_layer = nn.Linear(embedding_dim, vocab_size)

    def build_attention_mask(self, token_ids, pad_token_id=0):
        batch_size, sequence_length = token_ids.shape
        causal_mask = create_causal_mask(sequence_length, device=token_ids.device)
        causal_mask = causal_mask.view(1, 1, sequence_length, sequence_length)

        padding_mask = create_padding_mask(token_ids, pad_token_id)
        padding_mask = padding_mask.view(batch_size, 1, 1, sequence_length)
        padding_mask = padding_mask.float().masked_fill(padding_mask, float("-inf"))
        return causal_mask + padding_mask

    def forward(self, token_ids, pad_token_id=0):
        hidden_states = self.token_embedding(token_ids)
        hidden_states = self.positional_encoding(hidden_states)
        hidden_states = self.dropout(hidden_states)

        attention_mask = self.build_attention_mask(token_ids, pad_token_id)
        for decoder_block in self.decoder_blocks:
            hidden_states = decoder_block(hidden_states, attention_mask)

        return self.output_layer(hidden_states)
