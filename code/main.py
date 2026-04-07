from __future__ import annotations

import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import optim

import data
import lm
from transformer import TransformerLM

def run_experiment(config: dict, base_data_path: str, base_save_path: str = "experiments"):
    seq_len = config["seq_len"]
    batch_size = config["batch_size"]
    n_layers = config["n_layers"]
    n_heads = config["n_heads"]
    embed_size = config["embed_size"]
    mlp_hidden_size = config.get("mlp_hidden_size", embed_size * 4)
    learning_rate = config["learning_rate"]
    gradient_clipping = config["gradient_clipping"]
    num_batches_to_train = config["num_batches_to_train"]
    with_residuals = config.get("with_residuals", True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    exp_dir = Path(base_save_path) / config["exp_name"]
    exp_dir.mkdir(parents=True, exist_ok=True)

    with open(exp_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)

    tokenizer, tokenized_data = data.load_data(base_data_path)
    n_total = len(tokenized_data)
    n_train = int(0.8 * n_total)
    n_val = int(0.1 * n_total)

    train_data = tokenized_data[:n_train]
    val_data = tokenized_data[n_train:n_train + n_val]
    test_data = tokenized_data[n_train + n_val:]

    # NOTE: are data items are longer by one than the sequence length,
    # They will be shortened by 1 when converted to training examples.
    data_iter = iter(data.RandomOrderDataIterator(train_data, seq_len + 1))
    val_iter = iter(data.RandomOrderDataIterator(val_data, seq_len + 1))

    model: torch.nn.Module = TransformerLM(
        n_layers,
        n_heads,
        embed_size,
        seq_len,
        tokenizer.vocab_size(),
        mlp_hidden_size,
        with_residuals=with_residuals,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=[0.9, 0.95])

    model.train()

    train_losses = []
    val_losses = []
    val_steps = []
    best_val_loss = float("inf")

    num_batches = 0
    while num_batches < num_batches_to_train:
        # Refresh the iterator every time we finish the data
        data_iter = iter(data.RandomOrderDataIterator(train_data, seq_len + 1))

        for batch in data.batch_items(data_iter, batch_size):
            if num_batches >= num_batches_to_train:
                break
            num_batches = num_batches + 1

            batch_x, batch_y = lm.batch_to_labeled_samples(batch)
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            logits = model(batch_x)

            loss = lm.compute_loss(logits, batch_y, pad_id=tokenizer.pad_id())

            # parameters update
            model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            optimizer.step()

            if num_batches % 10 == 0:
                print(f"Seen {num_batches} batches. last loss is: {loss.item()}")
                if num_batches % 100 == 0:
                    for _ in range(1):
                        model.eval()
                        sampled = tokenizer.detokenize(
                            model.sample_continuation(tokenizer.tokenize("Hello"), 500)
                        )
                        model.train()
                        print(f"Model sample: '''{sampled}'''")
                    print("")

            if num_batches % config["val_interval"] == 0:
                model.eval()
                val_loss_values = []
                with torch.no_grad():
                    for val_batch_idx, val_batch in enumerate(data.batch_items(val_iter, batch_size)):
                        if val_batch_idx >= 50:
                            break
                        val_x, val_y = lm.batch_to_labeled_samples(val_batch)
                        val_x, val_y = val_x.to(device), val_y.to(device)
                        val_logits = model(val_x)
                        val_loss = lm.compute_loss(val_logits, val_y, pad_id=tokenizer.pad_id())
                        val_loss_values.append(val_loss.item())

                avg_val_loss = sum(val_loss_values) / len(val_loss_values)
                train_losses.append(loss.item())
                val_losses.append(avg_val_loss)
                val_steps.append(num_batches)

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save(model.state_dict(), exp_dir / "best_model.pth")

                model.train()

    tokenizer.save(str(exp_dir / "tokenizer.json"))

    plt.figure(figsize=(8, 5))
    plt.plot(val_steps, train_losses, label="Train Loss")
    plt.plot(val_steps, val_losses, label="Validation Loss")
    plt.xlabel("Training Steps")
    plt.ylabel("Loss")
    plt.title(f"Loss Curves: {config['exp_name']}")
    plt.grid(True)
    plt.legend()
    plt.savefig(exp_dir / "loss_plot.png")
    plt.close()

    return best_val_loss


if __name__ == "__main__":
    DATA_PATH = "../data/en/"
    experiments = [
        {
            "exp_name": "deep_narrow",
            "seq_len": 128,
            "batch_size": 64,
            "n_layers": 8,
            "n_heads": 8,
            "embed_size": 128,
            "learning_rate": 5e-4,
            "gradient_clipping": 1.0,
            "num_batches_to_train": 50000,
            "val_interval": 100,
            "with_residuals": True,
        },
        {
            "exp_name": "shallow_wide",
            "seq_len": 128,
            "batch_size": 64,
            "n_layers": 4,
            "n_heads": 8,
            "embed_size": 256,
            "learning_rate": 4e-4,
            "gradient_clipping": 1.0,
            "num_batches_to_train": 50000,
            "val_interval": 100,
            "with_residuals": True,
        },
    ]

    for config in experiments:
        best_val_loss = run_experiment(config, DATA_PATH)
        print(f"Experiment {config['exp_name']} best validation loss: {best_val_loss:.4f}")
        torch.cuda.empty_cache()
