from torch import nn
import torch
import torch.nn.functional as F
import attention
import mlp

class TransformerDecoderBlock(nn.Module):
    def __init__(self, n_heads: int, embed_size: int, mlp_hidden_size: int, max_context_len, with_residuals: bool = False, use_pre_norm: bool = True, attention_dropout: float = 0.0, self_attention_dropout: float = 0.0):
        super().__init__()
        self.causal_attention = attention.CausalSelfAttention(embed_size, n_heads, max_context_len, attention_dropout=attention_dropout)
        self.mlp = mlp.MLP(embed_size, mlp_hidden_size)
        self.layer_norm_1 = nn.LayerNorm(embed_size)
        self.layer_norm_2 = nn.LayerNorm(embed_size)
        self.self_attention_dropout = nn.Dropout(self_attention_dropout)
        self.with_residuals = with_residuals
        self.use_pre_norm = use_pre_norm

    def forward(self, inputs):
        x = inputs
        if self.use_pre_norm:
            # PRE-NORM LOGIC
            x = x + self.self_attention_dropout(self.causal_attention(self.layer_norm_1(x)))
            x = x + self.mlp(self.layer_norm_2(x))
        else:
            # POST-NORM LOGIC
            x = self.layer_norm_1(x + self.self_attention_dropout(self.causal_attention(x)))
            x = self.layer_norm_2(x + self.mlp(x))
        return x

class Embed(nn.Module):
    def __init__(self, vocab_size: int, embed_size: int, max_context_len, embedding_dropout: float = 0.0):
        super().__init__()
        self.token_embeddings = nn.Embedding(vocab_size, embed_size)
        self.position_embeddings = nn.Embedding(max_context_len, embed_size)
        self.dropout = nn.Dropout(embedding_dropout)
        self.max_context_len = max_context_len

    def forward(self, x):
        # x has shape (b, n) where b is batch and n is sequence length
        # Output shape should be (b, n, d) where d is embedding dimension
        n = x.size(1)
        tok_embeddings = self.token_embeddings(x)
        positions = torch.arange(n, device=x.device)
        pos_embeddings = self.position_embeddings(positions)
        return self.dropout(tok_embeddings + pos_embeddings)


class TransformerLM(nn.Module):
    def __init__(
            self,
            n_layers: int,
            n_heads: int,
            embed_size: int,
            max_context_len: int,
            vocab_size: int,
            mlp_hidden_size: int,
            with_residuals: bool,
            use_pre_norm: bool = True,
            init_scheme: str = "xavier_uniform",
            embedding_dropout: float = 0.0,
            attention_dropout: float = 0.0,
            self_attention_dropout: float = 0.0,
            ):
        super().__init__()
        self.init_scheme = init_scheme
        self.embed = Embed(vocab_size, embed_size, max_context_len, embedding_dropout=embedding_dropout)
        self.layers = nn.ModuleList([
            TransformerDecoderBlock(
                n_heads,
                embed_size,
                mlp_hidden_size,
                max_context_len,
                with_residuals,
                use_pre_norm=use_pre_norm,
                attention_dropout=attention_dropout,
                self_attention_dropout=self_attention_dropout,
            )
            for _ in range(n_layers)
        ])
        self.layer_norm = nn.LayerNorm(embed_size)
        self.word_prediction = nn.Linear(embed_size, vocab_size)
        self.max_context_len = max_context_len

        self.init_weights()

        n_params = sum(p.numel() for p in self.parameters())
        print("Parameter count: %.2fM" % (n_params/1e6,))

    def forward(self, inputs):
        x = self.embed(inputs)
        for layer in self.layers:
            x = layer(x)
        x = self.layer_norm(x)
        logits = self.word_prediction(x)
        return logits

    def init_weights(self):
        for module in self.modules():
            if isinstance(module, nn.LayerNorm):
                torch.nn.init.ones_(module.weight)
                torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Linear):
                if self.init_scheme == "xavier_uniform":
                    torch.nn.init.xavier_uniform_(module.weight)
                elif self.init_scheme == "kaiming_normal":
                    torch.nn.init.kaiming_normal_(module.weight, nonlinearity="linear")
                elif self.init_scheme == "normal_0p02":
                    torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                else:
                    raise ValueError(f"Unsupported init_scheme: {self.init_scheme}")

                if module.bias is not None:
                    torch.nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                if self.init_scheme in {"xavier_uniform", "kaiming_normal", "normal_0p02"}:
                    torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
                else:
                    raise ValueError(f"Unsupported init_scheme: {self.init_scheme}")


    def sample_continuation(self, prefix: list[int], max_tokens_to_generate: int) -> list[int]:
        device = next(self.parameters()).device
        feed_to_lm = prefix[:]
        generated = []
        with torch.no_grad():
            while len(generated) < max_tokens_to_generate:
                if len(feed_to_lm) > self.max_context_len:
                    # if we have more tokens than context length, trim it to context length.
                    feed_to_lm = feed_to_lm[-self.max_context_len:]
                logits = self(torch.tensor([feed_to_lm], dtype=torch.long, device=device))
                logits_for_last_token = logits[0][-1]
                distribution_for_last_token = F.softmax(logits_for_last_token, dim=-1)
                sampled_token = torch.multinomial(distribution_for_last_token, num_samples=1)
                sampled_token_id = int(sampled_token.item())
                generated.append(sampled_token_id)
                feed_to_lm.append(sampled_token_id)
        return generated

    def better_sample_continuation(self, prefix: list[int], max_tokens_to_generate: int, temperature: float, topK: int) -> list[int]:
        if temperature <= 0:
            raise ValueError("temperature must be > 0")
        if topK < 1:
            raise ValueError("topK must be >= 1")

        device = next(self.parameters()).device
        feed_to_lm = prefix[:]
        generated = []
        with torch.no_grad():
            while len(generated) < max_tokens_to_generate:
                if len(feed_to_lm) > self.max_context_len:
                    # if we have more tokens than context length, trim it to context length.
                    feed_to_lm = feed_to_lm[-self.max_context_len:]

                logits = self(torch.tensor([feed_to_lm], dtype=torch.long, device=device))
                logits_for_last_token = logits[0][-1] / temperature

                k = min(topK, logits_for_last_token.size(-1))
                topk_logits, topk_indices = torch.topk(logits_for_last_token, k)
                topk_distribution = F.softmax(topk_logits, dim=-1)

                sampled_topk_pos = torch.multinomial(topk_distribution, num_samples=1)
                sampled_token_id = int(topk_indices[sampled_topk_pos].item())

                generated.append(sampled_token_id)
                feed_to_lm.append(sampled_token_id)
        return generated

