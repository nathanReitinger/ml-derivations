"""Port of ../js/data.js. Same normalization / analog-bits (signed one-hot)
encoding, so a text grid encodes identically on both sides of the bridge."""
import json

import numpy as np
import torch


def normalize_text(text):
    return text.replace("\r\n", "\n").replace("\r", "\n").rstrip("\n")


def text_to_indices(text, vocab, height, width):
    rows = normalize_text(text).split("\n")
    if len(rows) != height:
        raise ValueError(f"expected {height} rows, got {len(rows)}")
    vmap = {c: i for i, c in enumerate(vocab)}
    idx = np.zeros((height, width), dtype=np.int64)
    for r, row in enumerate(rows):
        if len(row) != width:
            raise ValueError(f"expected {width} cols, got {len(row)} in row {row!r}")
        for c, ch in enumerate(row):
            if ch not in vmap:
                raise ValueError(f'character "{ch}" not in vocab {vocab}')
            idx[r, c] = vmap[ch]
    return idx


def indices_to_text(idx, vocab, height, width):
    rows = ["".join(vocab[idx[r, c]] for c in range(width)) for r in range(height)]
    return "\n".join(rows)


def text_to_analog(text, vocab, height, width):
    """Signed one-hot ('analog bits'): [H,W,V] in {-1,+1}, matching data.js's textToAnalogTensor."""
    idx = text_to_indices(text, vocab, height, width)
    one_hot = np.eye(len(vocab), dtype=np.float32)[idx]
    return one_hot * 2 - 1


def analog_to_text(tensor, vocab, height, width):
    """tensor: [H,W,V] (numpy or torch). Decodes via per-cell argmax, matching data.js."""
    if isinstance(tensor, torch.Tensor):
        idx = tensor.argmax(dim=-1).cpu().numpy()
    else:
        idx = tensor.argmax(axis=-1)
    return indices_to_text(idx, vocab, height, width)


def text_to_binary(text, vocab, height, width):
    """0/1 grid, vocab[0]==background -> 0, else 1. Matches js/similarity.js's textToBinary."""
    idx = text_to_indices(text, vocab, height, width)
    return (idx != 0).astype(np.float32)


class Dataset:
    """Loads data/dataset.json exactly as app.js's init() does: two arrays
    that never mix. `train_tensor` is what training ever sees; `test_bin` is
    the only thing the search script is allowed to score against."""

    def __init__(self, path):
        with open(path) as f:
            d = json.load(f)
        self.height, self.width, self.vocab = d["height"], d["width"], d["vocab"]

        self.train_files = [r["file"] for r in d["train"]]
        self.train_texts = [r["text"] for r in d["train"]]
        self.test_files = [r["file"] for r in d["test"]]
        self.test_texts = [r["text"] for r in d["test"]]

        self.train_bin = np.stack(
            [text_to_binary(t, self.vocab, self.height, self.width) for t in self.train_texts]
        )
        self.test_bin = np.stack(
            [text_to_binary(t, self.vocab, self.height, self.width) for t in self.test_texts]
        )

    def train_tensor(self, device):
        analog = np.stack(
            [text_to_analog(t, self.vocab, self.height, self.width) for t in self.train_texts]
        )
        return torch.tensor(analog, dtype=torch.float32, device=device)

    def __repr__(self):
        return (f"Dataset(train={len(self.train_texts)}, test={len(self.test_texts)}, "
                f"grid={self.width}x{self.height}, vocab={self.vocab})")
