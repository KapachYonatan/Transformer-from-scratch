from __future__ import annotations
from typing import Iterator
import json
import torch
from torch import nn
import torch.nn.functional as F
import random
import glob

class CharTokenizer:
    def __init__(self):
        self.symbols = ["<PAD>"]
        self.tokens = set()
        self.vocab = list(self.symbols)
        self.stoi = {s:i for i, s in enumerate(self.vocab)}

    def pad_id(self): return self.stoi["<PAD>"]

    def get_id(self, tok: str): return self.stoi[tok]

    def vocab_size(self): return len(self.vocab)
        
    def train(self, sequences: list[str]) -> None:
        for seq in sequences:
            for symbol in self._tokenize_to_symbols(seq):
                self.tokens.add(symbol)

        self.vocab = list(self.symbols) + list(sorted(self.tokens))
        self.stoi = {s:i for i, s in enumerate(self.vocab)}


    def _tokenize_to_symbols(self, text: str) -> list[str]:
        return list(text)

    def tokenize(self, text: str) -> list[int]:
        seq: list[str] = self._tokenize_to_symbols(text)
        return [self.stoi[s] for s in seq]

    def detokenize(self, tokens: list[int], keep_symbols = True) -> str:
        strs: list[str] = [self.vocab[t] for t in tokens]
        if not keep_symbols:
            strs = [s for s in strs if len(s) == 1]
        return "".join(strs)

    def save(self, path: str) -> None:
        with open(path, "w") as file:
            data = {"vocab": self.vocab}
            json.dump(data, file)

    @staticmethod
    def load(path: str) -> CharTokenizer:
        tokenizer = CharTokenizer()
        with open(path, "r") as file:
            loaded_data = json.load(file)
        tokenizer.vocab = loaded_data["vocab"]
        tokenizer.stoi = {s: i for i, s in enumerate(tokenizer.vocab)}
        tokenizer.tokens = set(tokenizer.vocab[1:])
        return tokenizer

class RandomOrderDataIterator:
    def __init__(self, data, desired_length):
        self.desired_length = desired_length
        self.data: list[list[int]] = [seq for seq in data if len(seq) > self.desired_length]

    def __iter__(self):
        if len(self.data) == 0: return
        while True:
            seq = random.choice(self.data)
            idx = random.randint(0, len(seq) - self.desired_length)
            yield seq[idx:idx + self.desired_length]


# This both creates the tokenizer and uses it to tokenize the data.
# In a real system you'd like to split it to two separate functions.
# Feel free to separate it to two functions also in this code.
def load_data(path: str) -> tuple[CharTokenizer, list[list[int]]]:
    print(f"Starting data loading from {path}...", flush=True)
    tokenizer = CharTokenizer()
    file_names = glob.glob(f"{path}/*.txt")
    total_characters = 0
    loaded_documents = 0

    for fname in file_names:
        print(f"Tokenizing file: {fname}...", flush=True)
        with open(fname) as fh:
            text = fh.read()
            total_characters += len(text)
            loaded_documents += 1
            tokenizer.train(text)

    data: list[list[int]] = []
    for fname in file_names:
        with open(fname) as fh:
            text = fh.read()
            data.append(tokenizer.tokenize(text))

    print(
        f"Total documents loaded: {loaded_documents}. Total characters: {total_characters}.",
        flush=True,
    )

    return (tokenizer, data)

def batch_items(data_iter: Iterator[list[int]], batch_size: int = 2) -> Iterator[torch.LongTensor]:
    batch = []
    for seq in data_iter:
        idx = 0
        batch.append(seq)
        if len(batch) >= batch_size:
            yield torch.tensor(batch, dtype=torch.long)
            batch = []
    if len(batch) > 0:
        yield torch.tensor(batch, dtype=torch.long)

