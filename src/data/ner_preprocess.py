from pathlib import Path

import torch

from src.data.tokenizer import SimpleTokenizer


def read_ner_data(file_path):
    path = Path(file_path)
    sentences = []
    tags = []
    current_tokens = []
    current_tags = []

    with path.open("r", encoding="utf-8") as file:
        for line in file:
            stripped_line = line.strip()
            if not stripped_line:
                if current_tokens:
                    sentences.append(current_tokens)
                    tags.append(current_tags)
                    current_tokens = []
                    current_tags = []
                continue

            token, tag = stripped_line.split()
            current_tokens.append(token)
            current_tags.append(tag)

    if current_tokens:
        sentences.append(current_tokens)
        tags.append(current_tags)

    return sentences, tags


def build_tag_vocabulary(tag_sequences):
    tag_to_id = {"<pad>": 0}
    for tag in sorted({tag for sequence in tag_sequences for tag in sequence}):
        if tag not in tag_to_id:
            tag_to_id[tag] = len(tag_to_id)
    return tag_to_id


def build_ner_dataset(sentences, tags):
    tokenizer = SimpleTokenizer()
    tokenizer.build_vocab(sentences)
    tag_to_id = build_tag_vocabulary(tags)

    encoded_sentences = [tokenizer.encode_tokens(sentence) for sentence in sentences]
    encoded_tags = [[tag_to_id[tag] for tag in tag_sequence] for tag_sequence in tags]

    return {
        "sentences": sentences,
        "tags": tags,
        "token_to_id": tokenizer.token_to_id,
        "id_to_token": tokenizer.id_to_token,
        "tag_to_id": tag_to_id,
        "id_to_tag": {index: tag for tag, index in tag_to_id.items()},
        "encoded_sentences": encoded_sentences,
        "encoded_tags": encoded_tags,
    }


def pad_ner_batch(token_sequences, tag_sequences, token_pad_id=0, tag_pad_id=0):
    max_length = max(len(sequence) for sequence in token_sequences)
    padded_tokens = []
    padded_tags = []
    mask = []

    for token_sequence, tag_sequence in zip(token_sequences, tag_sequences):
        pad_count = max_length - len(token_sequence)
        padded_tokens.append(token_sequence + [token_pad_id] * pad_count)
        padded_tags.append(tag_sequence + [tag_pad_id] * pad_count)
        mask.append([True] * len(token_sequence) + [False] * pad_count)

    return (
        torch.tensor(padded_tokens, dtype=torch.long),
        torch.tensor(padded_tags, dtype=torch.long),
        torch.tensor(mask, dtype=torch.bool),
    )


def encode_sentence(sentence, token_to_id):
    tokens = sentence.strip().split()
    unk_id = token_to_id.get("<unk>", 1)
    token_ids = [token_to_id.get(token, unk_id) for token in tokens]
    return tokens, token_ids
