from pathlib import Path

import torch


def read_text_corpus(corpus_path):
    path = Path(corpus_path)
    with path.open("r", encoding="utf-8") as file:
        return [line.strip() for line in file if line.strip()]


def tokenize_text(text):
    return text.lower().strip().split()


def build_token_vocabulary(corpus_lines):
    token_to_id = {"<pad>": 0, "<unk>": 1}
    for line in corpus_lines:
        for token in tokenize_text(line):
            if token not in token_to_id:
                token_to_id[token] = len(token_to_id)
    id_to_token = {index: token for token, index in token_to_id.items()}
    return token_to_id, id_to_token


def encode_corpus(corpus_lines, token_to_id):
    unk_id = token_to_id["<unk>"]
    token_ids = []
    for line in corpus_lines:
        token_ids.extend([token_to_id.get(token, unk_id) for token in tokenize_text(line)])
    return token_ids


def create_language_modeling_sequences(token_ids, sequence_length):
    input_sequences = []
    target_sequences = []
    if len(token_ids) < 2:
        return input_sequences, target_sequences

    step_size = max(1, sequence_length)
    for start_index in range(0, len(token_ids) - 1, step_size):
        chunk = token_ids[start_index : start_index + sequence_length + 1]
        if len(chunk) < 2:
            continue
        input_sequences.append(chunk[:-1])
        target_sequences.append(chunk[1:])

    return input_sequences, target_sequences


def pad_language_modeling_batch(input_sequences, target_sequences, pad_token_id=0):
    max_length = max(len(sequence) for sequence in input_sequences)
    padded_inputs = []
    padded_targets = []
    mask = []

    for input_sequence, target_sequence in zip(input_sequences, target_sequences):
        pad_count = max_length - len(input_sequence)
        padded_inputs.append(input_sequence + [pad_token_id] * pad_count)
        padded_targets.append(target_sequence + [pad_token_id] * pad_count)
        mask.append([True] * len(input_sequence) + [False] * pad_count)

    return (
        torch.tensor(padded_inputs, dtype=torch.long),
        torch.tensor(padded_targets, dtype=torch.long),
        torch.tensor(mask, dtype=torch.bool),
    )


def build_language_modeling_dataset(corpus_lines, sequence_length):
    token_to_id, id_to_token = build_token_vocabulary(corpus_lines)
    token_ids = encode_corpus(corpus_lines, token_to_id)
    input_sequences, target_sequences = create_language_modeling_sequences(
        token_ids,
        sequence_length,
    )
    return {
        "corpus_lines": corpus_lines,
        "token_to_id": token_to_id,
        "id_to_token": id_to_token,
        "token_ids": token_ids,
        "input_sequences": input_sequences,
        "target_sequences": target_sequences,
        "pad_token_id": token_to_id["<pad>"],
        "unk_token_id": token_to_id["<unk>"],
    }
