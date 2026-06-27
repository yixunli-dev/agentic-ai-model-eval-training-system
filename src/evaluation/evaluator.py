import json
from pathlib import Path

import torch

from src.training.losses import masked_cross_entropy_loss, masked_language_model_loss
from src.training.metrics import token_accuracy


def evaluate_model(model, batch, dataset, predictions_path):
    model.eval()
    token_ids = batch["token_ids"]
    tag_ids = batch["tag_ids"]
    mask = batch["mask"]

    with torch.no_grad():
        logits = model(token_ids)
        loss = masked_cross_entropy_loss(logits, tag_ids, mask)
        accuracy = token_accuracy(logits, tag_ids, mask)
        predicted_ids = torch.argmax(logits, dim=-1)

    predictions = []
    for sentence_index, tokens in enumerate(dataset["sentences"]):
        gold_tags = dataset["tags"][sentence_index]
        predicted_tags = []
        for token_index in range(len(tokens)):
            predicted_tag_id = int(predicted_ids[sentence_index, token_index].item())
            predicted_tags.append(dataset["id_to_tag"][predicted_tag_id])

        predictions.append(
            {
                "tokens": tokens,
                "gold_tags": gold_tags,
                "predicted_tags": predicted_tags,
            }
        )

    path = Path(predictions_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for prediction in predictions:
            file.write(json.dumps(prediction) + "\n")

    return {"eval_loss": float(loss.item()), "token_accuracy": float(accuracy)}, predictions


def generate_text(model, prompt, dataset, sequence_length, max_new_tokens=8):
    token_to_id = dataset["token_to_id"]
    id_to_token = dataset["id_to_token"]
    pad_token_id = dataset["pad_token_id"]
    unk_token_id = dataset["unk_token_id"]
    generated_token_ids = [
        token_to_id.get(token, unk_token_id) for token in prompt.lower().strip().split()
    ]

    model.eval()
    with torch.no_grad():
        for _ in range(max_new_tokens):
            context = generated_token_ids[-sequence_length:]
            token_tensor = torch.tensor([context], dtype=torch.long)
            logits = model(token_tensor, pad_token_id=pad_token_id)
            next_token_id = int(torch.argmax(logits[0, -1]).item())
            generated_token_ids.append(next_token_id)

    return " ".join(id_to_token.get(token_id, "<unk>") for token_id in generated_token_ids)


def evaluate_language_model(model, batch, dataset, predictions_path, config):
    model.eval()
    input_ids = batch["input_ids"]
    target_ids = batch["target_ids"]
    mask = batch["mask"]

    with torch.no_grad():
        logits = model(input_ids, pad_token_id=dataset["pad_token_id"])
        loss = masked_language_model_loss(logits, target_ids, mask)
        perplexity = torch.exp(loss)

    prompt = config.get("sample_prompt", "machine learning")
    generated_text = generate_text(
        model,
        prompt,
        dataset,
        sequence_length=config.get("sequence_length", 8),
        max_new_tokens=config.get("max_new_tokens", 8),
    )
    prediction = {"prompt": prompt, "generated_text": generated_text}

    path = Path(predictions_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        file.write(json.dumps(prediction) + "\n")

    return {
        "eval_loss": float(loss.item()),
        "perplexity": float(perplexity.item()),
    }, [prediction]
