import torch

from src.models.lstm_ner import LSTMNERTagger


def test_lstm_ner_forward_returns_logits_for_each_token():
    model = LSTMNERTagger(vocab_size=12, tag_count=4, embedding_dim=8, hidden_dim=10)
    token_ids = torch.tensor([[2, 3, 0], [4, 5, 6]])

    logits = model(token_ids)

    assert logits.shape == (2, 3, 4)
