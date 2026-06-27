from src.data.ner_preprocess import (
    build_ner_dataset,
    pad_ner_batch,
    read_ner_data,
)


def test_read_ner_data_uses_blank_lines_as_sentence_boundaries(tmp_path):
    data_path = tmp_path / "sample_ner.txt"
    data_path.write_text(
        "Tesla B-ORG\nbuilds O\n\nCalifornia B-LOC\n",
        encoding="utf-8",
    )

    sentences, tags = read_ner_data(data_path)

    assert sentences == [["Tesla", "builds"], ["California"]]
    assert tags == [["B-ORG", "O"], ["B-LOC"]]


def test_build_ner_dataset_creates_token_and_tag_vocabularies():
    sentences = [["Tesla", "builds"], ["California"]]
    tags = [["B-ORG", "O"], ["B-LOC"]]

    dataset = build_ner_dataset(sentences, tags)

    assert dataset["token_to_id"]["<pad>"] == 0
    assert dataset["token_to_id"]["<unk>"] == 1
    assert dataset["tag_to_id"]["<pad>"] == 0
    assert "Tesla" in dataset["token_to_id"]
    assert "B-ORG" in dataset["tag_to_id"]
    assert len(dataset["encoded_sentences"]) == 2
    assert len(dataset["encoded_tags"]) == 2


def test_pad_ner_batch_adds_padding_and_boolean_mask():
    padded_tokens, padded_tags, mask = pad_ner_batch(
        token_sequences=[[2, 3], [4]],
        tag_sequences=[[1, 2], [3]],
        token_pad_id=0,
        tag_pad_id=0,
    )

    assert padded_tokens.tolist() == [[2, 3], [4, 0]]
    assert padded_tags.tolist() == [[1, 2], [3, 0]]
    assert mask.tolist() == [[True, True], [True, False]]
