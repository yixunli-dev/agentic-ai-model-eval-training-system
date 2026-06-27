import torch.nn as nn


class LSTMNERTagger(nn.Module):
    def __init__(self, vocab_size, tag_count, embedding_dim, hidden_dim):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=0)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            batch_first=True,
            bidirectional=True,
        )
        self.classifier = nn.Linear(hidden_dim * 2, tag_count)

    def forward(self, token_ids):
        embeddings = self.embedding(token_ids)
        lstm_outputs, _ = self.lstm(embeddings)
        logits = self.classifier(lstm_outputs)
        return logits
