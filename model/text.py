import re
import numpy as np

TOKEN_RE = re.compile(r"[a-z0-9]+(?:'[a-z]+)?")

def tokenize(text):
    return TOKEN_RE.findall(text.lower())

class Vocabulary:
    def __init__(self, tokens):
        self.words = ["<unk>"] + sorted(set(tokens) - {"<unk>"})
        self.word_to_id = {word: i for i, word in enumerate(self.words)}
    def encode(self, tokens):
        return [self.word_to_id.get(t, 0) for t in tokens]
    def decode(self, index):
        return self.words[int(index)]
    def __len__(self):
        return len(self.words)

def make_sequences(ids, length):
    if length < 1: raise ValueError("sequence length must be positive")
    return [(ids[i:i+length], ids[i+length]) for i in range(max(0, len(ids)-length))]

def split_token_stream(ids, train_fraction=0.70, validation_fraction=0.15):
    """Deterministically split ordered tokens before windowing to prevent boundary leakage."""
    if not 0 < train_fraction < 1 or not 0 < validation_fraction < 1:
        raise ValueError("split fractions must be between zero and one")
    if train_fraction + validation_fraction >= 1:
        raise ValueError("training and validation fractions must sum to less than one")
    train_end = int(len(ids) * train_fraction)
    validation_end = train_end + int(len(ids) * validation_fraction)
    return ids[:train_end], ids[train_end:validation_end], ids[validation_end:]

def one_hot(index, size):
    x = np.zeros((size, 1)); x[index, 0] = 1.0
    return x
