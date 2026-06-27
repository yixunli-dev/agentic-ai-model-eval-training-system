from src.data.language_modeling_dataset import (
    build_language_modeling_dataset,
    create_language_modeling_sequences,
    pad_language_modeling_batch,
    read_text_corpus,
)


def test_language_modeling_dataset_creates_input_target_pairs(tmp_path):
    corpus_path = tmp_path / "sample_corpus.txt"
    corpus_path.write_text("machine learning models predict tokens\n", encoding="utf-8")

    corpus_lines = read_text_corpus(corpus_path)
    dataset = build_language_modeling_dataset(corpus_lines, sequence_length=4)

    assert dataset["input_sequences"][0][1:] == dataset["target_sequences"][0][:-1]
    assert "machine" in dataset["token_to_id"]
    assert dataset["pad_token_id"] == 0


def test_create_language_modeling_sequences_supports_fixed_length_padding():
    token_ids = [2, 3, 4]

    inputs, targets = create_language_modeling_sequences(token_ids, sequence_length=5)
    padded_inputs, padded_targets, mask = pad_language_modeling_batch(
        inputs,
        targets,
        pad_token_id=0,
    )

    assert inputs == [[2, 3]]
    assert targets == [[3, 4]]
    assert padded_inputs.tolist() == [[2, 3]]
    assert padded_targets.tolist() == [[3, 4]]
    assert mask.tolist() == [[True, True]]
