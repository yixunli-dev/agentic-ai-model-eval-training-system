import torch

from src.data.language_modeling_dataset import pad_language_modeling_batch
from src.data.ner_preprocess import pad_ner_batch
from src.models.lstm_ner import LSTMNERTagger
from src.models.transformer_decoder import TransformerDecoderModel
from src.training.losses import masked_cross_entropy_loss, masked_language_model_loss
from src.training.metrics import token_accuracy


def train_lstm_ner_model(dataset, config):
    token_ids, tag_ids, mask = pad_ner_batch(
        dataset["encoded_sentences"],
        dataset["encoded_tags"],
        token_pad_id=dataset["token_to_id"]["<pad>"],
        tag_pad_id=dataset["tag_to_id"]["<pad>"],
    )

    model = LSTMNERTagger(
        vocab_size=len(dataset["token_to_id"]),
        tag_count=len(dataset["tag_to_id"]),
        embedding_dim=config.get("embedding_dim", 32),
        hidden_dim=config.get("hidden_dim", 64),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("learning_rate", 0.001))

    history = []
    for epoch in range(config.get("epochs", 5)):
        model.train()
        optimizer.zero_grad()
        logits = model(token_ids)
        loss = masked_cross_entropy_loss(logits, tag_ids, mask)
        loss.backward()
        optimizer.step()

        accuracy = token_accuracy(logits, tag_ids, mask)
        history.append(
            {
                "epoch": epoch + 1,
                "loss": float(loss.item()),
                "token_accuracy": float(accuracy),
            }
        )

    return model, history, {"token_ids": token_ids, "tag_ids": tag_ids, "mask": mask}


def train_transformer_decoder_model(dataset, config):
    input_ids, target_ids, mask = pad_language_modeling_batch(
        dataset["input_sequences"],
        dataset["target_sequences"],
        pad_token_id=dataset["pad_token_id"],
    )

    model = TransformerDecoderModel(
        vocab_size=len(dataset["token_to_id"]),
        sequence_length=config.get("sequence_length", 8),
        embedding_dim=config.get("embedding_dim", 32),
        num_heads=config.get("num_heads", 4),
        num_layers=config.get("num_layers", 2),
        feedforward_dim=config.get("feedforward_dim", 64),
        dropout=config.get("dropout", 0.0),
    )
    optimizer = torch.optim.Adam(model.parameters(), lr=config.get("learning_rate", 0.001))

    history = []
    for epoch in range(config.get("epochs", 5)):
        model.train()
        optimizer.zero_grad()
        logits = model(input_ids, pad_token_id=dataset["pad_token_id"])
        loss = masked_language_model_loss(logits, target_ids, mask)
        loss.backward()
        optimizer.step()

        history.append(
            {
                "epoch": epoch + 1,
                "loss": float(loss.item()),
                "perplexity": float(torch.exp(loss).item()),
            }
        )

    return model, history, {"input_ids": input_ids, "target_ids": target_ids, "mask": mask}
