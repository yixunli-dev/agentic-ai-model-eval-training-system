import argparse

import torch

from src.data.ner_preprocess import encode_sentence
from src.models.lstm_ner import LSTMNERTagger
from src.models.transformer_decoder import TransformerDecoderModel


def parse_args():
    parser = argparse.ArgumentParser(description="Run predictions with a saved checkpoint.")
    parser.add_argument("--task", choices=["ner", "transformer"], default="ner")
    parser.add_argument("--checkpoint", default="outputs/runs/latest/model.pt", help="Path to model.pt.")
    parser.add_argument("--sentence", required=False, help="Sentence to tag for NER.")
    parser.add_argument("--prompt", required=False, help="Prompt for Transformer generation.")
    parser.add_argument("--max-new-tokens", type=int, default=8)
    return parser.parse_args()


def predict_ner(checkpoint, sentence):
    token_to_id = checkpoint["token_to_id"]
    id_to_tag = {int(index): tag for index, tag in checkpoint["id_to_tag"].items()}
    config = checkpoint["config"]

    model = LSTMNERTagger(
        vocab_size=len(token_to_id),
        tag_count=len(checkpoint["tag_to_id"]),
        embedding_dim=config.get("embedding_dim", 32),
        hidden_dim=config.get("hidden_dim", 64),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    tokens, token_ids = encode_sentence(sentence, token_to_id)
    with torch.no_grad():
        logits = model(torch.tensor([token_ids], dtype=torch.long))
        predicted_tag_ids = torch.argmax(logits, dim=-1).squeeze(0).tolist()

    for token, tag_id in zip(tokens, predicted_tag_ids):
        print(f"{token}\t{id_to_tag[tag_id]}")


def predict_transformer(checkpoint, prompt, max_new_tokens):
    token_to_id = checkpoint["token_to_id"]
    id_to_token = {int(index): token for index, token in checkpoint["id_to_token"].items()}
    config = checkpoint["config"]
    pad_token_id = checkpoint.get("pad_token_id", token_to_id["<pad>"])
    unk_token_id = checkpoint.get("unk_token_id", token_to_id["<unk>"])

    model = TransformerDecoderModel(
        vocab_size=len(token_to_id),
        sequence_length=config.get("sequence_length", 8),
        embedding_dim=config.get("embedding_dim", 32),
        num_heads=config.get("num_heads", 4),
        num_layers=config.get("num_layers", 2),
        feedforward_dim=config.get("feedforward_dim", 64),
        dropout=config.get("dropout", 0.0),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    generated_token_ids = [
        token_to_id.get(token, unk_token_id) for token in prompt.lower().strip().split()
    ]
    sequence_length = config.get("sequence_length", 8)

    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = generated_token_ids[-sequence_length:]
            token_tensor = torch.tensor([context], dtype=torch.long)
            logits = model(token_tensor, pad_token_id=pad_token_id)
            next_token_id = int(torch.argmax(logits[0, -1]).item())
            generated_token_ids.append(next_token_id)

    generated_tokens = [id_to_token.get(token_id, "<unk>") for token_id in generated_token_ids]
    print(" ".join(generated_tokens))


def main():
    args = parse_args()
    checkpoint = torch.load(args.checkpoint, map_location="cpu")
    if args.task == "transformer":
        if not args.prompt:
            raise ValueError("--prompt is required when --task transformer")
        predict_transformer(checkpoint, args.prompt, args.max_new_tokens)
        return

    if not args.sentence:
        raise ValueError("--sentence is required when --task ner")
    predict_ner(checkpoint, args.sentence)


if __name__ == "__main__":
    main()
