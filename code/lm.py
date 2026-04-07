from __future__ import annotations
import torch
from torch import nn
import torch.nn.functional as F

def batch_to_labeled_samples(batch: torch.IntTensor) -> tuple[torch.IntTensor, torch.IntTensor]:
    # The batches that we get from the reader have corpus-sequences of length max-context + 1.
    # We need to translate them to input/output examples, each of which is shorter by one.
    # That is, if our input is of dimension (b x n) our output is two tensors, each of dimension (b x n-1)
    inputs = batch[:, :-1]
    labels = batch[:, 1:]
    return (inputs, labels)

def compute_loss(logits: torch.Tensor, labels: torch.Tensor, pad_id: int = 0):
    # logits size is (batch, seq_len, vocab_size)
    # labels size is (batch, seq_len)
    logits_flat = logits.reshape(-1, logits.size(-1))
    labels_flat = labels.reshape(-1)
    return F.cross_entropy(logits_flat, labels_flat, ignore_index=pad_id)

