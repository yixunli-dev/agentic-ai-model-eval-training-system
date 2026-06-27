from collections import Counter


class SimpleTokenizer:
    def __init__(self, min_frequency=1):
        self.min_frequency = min_frequency
        self.pad_token = "<pad>"
        self.unk_token = "<unk>"
        self.token_to_id = {self.pad_token: 0, self.unk_token: 1}
        self.id_to_token = {0: self.pad_token, 1: self.unk_token}

    @property
    def pad_id(self):
        return self.token_to_id[self.pad_token]

    @property
    def unk_id(self):
        return self.token_to_id[self.unk_token]

    def build_vocab(self, token_sequences):
        counts = Counter()
        for tokens in token_sequences:
            counts.update(tokens)

        for token in sorted(counts):
            if counts[token] >= self.min_frequency and token not in self.token_to_id:
                token_id = len(self.token_to_id)
                self.token_to_id[token] = token_id
                self.id_to_token[token_id] = token

    def encode_tokens(self, tokens):
        return [self.token_to_id.get(token, self.unk_id) for token in tokens]

    def __len__(self):
        return len(self.token_to_id)
