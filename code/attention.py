from typing import Optional
from torch import nn
import torch
import torch.nn.functional as F
import math


def create_kqv_matrix(input_vector_dim, n_heads = 1):
    # One fused projection: [k | q | v] in the last dimension.
    return nn.Linear(input_vector_dim, 3 * (input_vector_dim // n_heads))


def create_kqv_parameters(d: int):
    # Compatibility helper matching the assignment wording.
    return create_kqv_matrix(d)

def kqv(x, linear):
    k, q, v = linear(x).chunk(3, dim=-1)
    return k, q, v

def attention_scores(q, k):
    B1, N1, D1 = q.size()
    B2, N2, D2 = k.size()
    assert B1 == B2
    assert D1 == D2

    # Scaled dot-product attention scores: QK^T / sqrt(d).
    A = (q @ k.transpose(-2, -1)) / math.sqrt(D1)
    return A

def create_causal_mask(max_length: int):
    # Lower-triangular causal mask with batch dimension.
    mask = torch.tril(torch.ones((max_length, max_length), dtype=torch.float32))
    mask = mask.unsqueeze(0)
    return mask

def self_attention(v, attention_scores, mask = None):
    # Apply optional causal mask, then softmax and weighted sum.
    if mask is not None:
        n_q = attention_scores.size(-2)
        n_k = attention_scores.size(-1)
        sliced_mask = mask[:, :n_q, :n_k]
        attention_scores = attention_scores.masked_fill(sliced_mask == 0, float("-inf"))

    weights = F.softmax(attention_scores, dim=-1)
    sa = weights @ v
    return sa


def self_attention_layer(x, kqv_matrix, attention_mask):
    k, q, v = kqv(x, kqv_matrix)
    att = attention_scores(q, k)
    sa = self_attention(v, att, attention_mask)
    return sa

def multi_head_attention_layer(x, kqv_matrices, mask):
    head_outputs = []
    for matrix in kqv_matrices:
        head_outputs.append(self_attention_layer(x, matrix, mask))

    sa = torch.cat(head_outputs, dim=-1)
    return sa


class CausalSelfAttention(nn.Module):
    def __init__(self, embed_dim, n_heads, max_context_len):
        super().__init__()
        assert embed_dim % n_heads == 0
        # the linear layers used for k, q, v computations:
        # each linear is for a different head, but for all of k, q and v for this head.
        self.kqv_matrices = nn.ModuleList([create_kqv_matrix(embed_dim, n_heads) for i in range(n_heads)])
        self.o_matrix = nn.Linear(embed_dim, embed_dim)
        # For use in the causal part.  "register_buffer" is used to store a tensor which is fixed but is not a parameter of the model.
        # You can then access it with: self.mask
        mask = create_causal_mask(max_context_len)
        self.register_buffer("mask", mask)
        self.n_heads = n_heads
        self.embed_dim = embed_dim

    def forward(self, x):
        sa = multi_head_attention_layer(x, self.kqv_matrices, self.mask)
        sa = self.o_matrix(sa)
        return sa
