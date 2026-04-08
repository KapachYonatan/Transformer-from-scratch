from __future__ import annotations

import copy
import json
import os
import time
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch import optim
from torch.optim.lr_scheduler import CosineAnnealingLR, LambdaLR

import data
import lm
from transformer import TransformerLM


def build_experiments(base_config: dict, variant_overrides: list[dict]) -> list[dict]:
    experiments = []
    for override in variant_overrides:
        if "exp_name" not in override:
            raise ValueError("Each experiment override must include 'exp_name'.")
        cfg = copy.deepcopy(base_config)
        cfg.update(override)
        experiments.append(cfg)
    return experiments

def run_experiment(config: dict, tokenizer, tokenized_data, base_save_path: str = "experiments"):
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
    use_pre_norm = config.get("use_pre_norm", True)
    init_scheme = config.get("init_scheme", "xavier_uniform")
    embedding_dropout = config.get("embedding_dropout", 0.0)
    attention_dropout = config.get("attention_dropout", 0.0)
    self_attention_dropout = config.get("self_attention_dropout", 0.0)
    weight_decay = config.get("weight_decay", 0.0)
    adam_betas = tuple(config.get("adam_betas", [0.9, 0.95]))
    resume_from = config.get("resume_from", None)
    reset_optimizer_on_resume = config.get("reset_optimizer_on_resume", False)
    scheduler_type = config.get("scheduler_type", "none")
    warmup_steps = config.get("warmup_steps", 0)
    min_lr_ratio = config.get("min_lr_ratio", 0.1)
    reset_scheduler_on_resume = config.get("reset_scheduler_on_resume", False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Initializing experiment: {config['exp_name']} on {device}...", flush=True)
    exp_dir = Path(base_save_path) / config["exp_name"]
    exp_dir.mkdir(parents=True, exist_ok=True)
    metrics_path = exp_dir / "metrics.jsonl"
    if metrics_path.exists():
        metrics_path.unlink()

    with open(exp_dir / "config.json", "w") as f:
        json.dump(config, f, indent=4)

    n_total = len(tokenized_data)
    if n_total == 1:
        full_seq = tokenized_data[0]
        split = int(0.9 * len(full_seq))
        train_data = [full_seq[:split]]
        val_data = [full_seq[split:]]
    else:
        n_train = int(0.8 * n_total)
        n_val = int(0.1 * n_total)
        train_data = tokenized_data[:n_train]
        val_data = tokenized_data[n_train:n_train + n_val]

    train_tokens = sum(len(seq) for seq in train_data)
    val_tokens = sum(len(seq) for seq in val_data)
    print(f"Train tokens: {train_tokens}, Val tokens: {val_tokens}", flush=True)

    if len(train_data) == 0:
        raise ValueError("train_data is empty after splitting. Please provide more data or adjust the split.")

    # NOTE: are data items are longer by one than the sequence length,
    # They will be shortened by 1 when converted to training examples.
    data_iter = iter(data.RandomOrderDataIterator(train_data, seq_len + 1))
    val_iter = iter(data.RandomOrderDataIterator(val_data, seq_len + 1))

    print("Constructing Transformer model architecture...", flush=True)
    model: torch.nn.Module = TransformerLM(
        n_layers,
        n_heads,
        embed_size,
        seq_len,
        tokenizer.vocab_size(),
        mlp_hidden_size,
        with_residuals=with_residuals,
        use_pre_norm=use_pre_norm,
        init_scheme=init_scheme,
        embedding_dropout=embedding_dropout,
        attention_dropout=attention_dropout,
        self_attention_dropout=self_attention_dropout,
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, betas=adam_betas, weight_decay=weight_decay)

    scheduler = None
    if scheduler_type == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=max(1, num_batches_to_train),
            eta_min=learning_rate * min_lr_ratio,
        )
    elif scheduler_type == "linear":
        def lr_lambda(step: int):
            if warmup_steps > 0 and step < warmup_steps:
                return float(step + 1) / float(warmup_steps)

            decay_steps = max(1, num_batches_to_train - warmup_steps)
            decay_step = max(0, step - warmup_steps)
            return max(0.0, 1.0 - (decay_step / decay_steps))

        scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)
    elif scheduler_type != "none":
        raise ValueError(f"Unsupported scheduler_type: {scheduler_type}")

    if resume_from is not None:
        print(f"Resuming from checkpoint: {resume_from}", flush=True)
        checkpoint = torch.load(resume_from, map_location=device)
        if isinstance(checkpoint, dict) and "model_state" in checkpoint:
            model.load_state_dict(checkpoint["model_state"])
            if (not reset_optimizer_on_resume) and ("optimizer_state" in checkpoint):
                optimizer.load_state_dict(checkpoint["optimizer_state"])
                for state in optimizer.state.values():
                    for key, value in state.items():
                        if torch.is_tensor(value):
                            state[key] = value.to(device)
            if scheduler is not None and (not reset_scheduler_on_resume) and ("scheduler_state" in checkpoint):
                scheduler.load_state_dict(checkpoint["scheduler_state"])
        else:
            model.load_state_dict(checkpoint)

    model.train()

    train_losses = []
    val_losses = []
    val_steps = []
    val_lrs = []
    best_val_loss = float("inf")
    best_step = -1
    experiment_start_time = time.perf_counter()

    print(f"Entering training loop for {num_batches_to_train} batches...", flush=True)
    print("Model successfully moved to GPU. Starting training loop...", flush=True)
    last_log_time = time.perf_counter()
    last_log_batches = 0

    num_batches = 0
    while num_batches < num_batches_to_train:
        # Refresh the iterator each epoch so training can continue for multiple epochs.
        data_iter = iter(data.RandomOrderDataIterator(train_data, seq_len + 1))
        saw_batch = False

        for batch in data.batch_items(data_iter, batch_size):
            saw_batch = True
            if num_batches >= num_batches_to_train:
                break
            num_batches += 1

            if num_batches <= 5:
                print(f"Batch [{num_batches}] reached", flush=True)

            batch_x, batch_y = lm.batch_to_labeled_samples(batch)
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            logits = model(batch_x)

            loss = lm.compute_loss(logits, batch_y, pad_id=tokenizer.pad_id())

            # parameters update
            model.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), gradient_clipping)
            optimizer.step()
            if scheduler is not None:
                scheduler.step()

            if num_batches % 10 == 0:
                current_time = time.perf_counter()
                elapsed = max(current_time - last_log_time, 1e-12)
                batches_per_sec = (num_batches - last_log_batches) / elapsed
                print(
                    f"Seen {num_batches} batches. last loss is: {loss.item():.4f}. lr: {optimizer.param_groups[0]['lr']:.6g}. speed: {batches_per_sec:.2f} batches/sec",
                    flush=True,
                )
                last_log_time = current_time
                last_log_batches = num_batches

                if num_batches != 0 and num_batches % 100 == 0:
                    print("Generating qualitative sample...", flush=True)
                    for _ in range(1):
                        model.eval()
                        sampled = tokenizer.detokenize(
                            model.sample_continuation(tokenizer.tokenize("Hello"), 500)
                        )
                        model.train()
                        print(f"Model sample: '''{sampled}'''", flush=True)
                    print("Sample generation complete.", flush=True)
                    print("", flush=True)

            if num_batches % config["val_interval"] == 0:
                print("Starting validation on 50 batches...", flush=True)
                model.eval()
                val_loss_values = []
                val_iter = iter(data.RandomOrderDataIterator(val_data, seq_len + 1))
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
                val_lrs.append(optimizer.param_groups[0]["lr"])

                metric_record = {
                    "step": num_batches,
                    "train_loss": loss.item(),
                    "val_loss": avg_val_loss,
                    "lr": optimizer.param_groups[0]["lr"],
                }
                with open(metrics_path, "a") as metrics_file:
                    metrics_file.write(json.dumps(metric_record) + "\n")

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    best_step = num_batches
                    torch.save(model.state_dict(), exp_dir / "best_model.pth")

                torch.save(
                    {
                        "model_state": model.state_dict(),
                        "optimizer_state": optimizer.state_dict(),
                        "scheduler_state": None if scheduler is None else scheduler.state_dict(),
                        "num_batches": num_batches,
                        "best_val_loss": best_val_loss,
                        "config": config,
                    },
                    exp_dir / "last_checkpoint.pth",
                )

                model.train()

        if not saw_batch:
            raise ValueError(
                "No training batches were produced. Ensure training sequences are longer than seq_len + 1."
            )

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

    summary = {
        "exp_name": config["exp_name"],
        "best_val_loss": best_val_loss,
        "best_step": best_step,
        "num_batches_trained": num_batches,
        "elapsed_sec": time.perf_counter() - experiment_start_time,
        "last_lr": optimizer.param_groups[0]["lr"],
        "num_val_points": len(val_steps),
    }
    with open(exp_dir / "summary.json", "w") as summary_file:
        json.dump(summary, summary_file, indent=4)

    return best_val_loss


if __name__ == "__main__":
    DATA_PATH = "../data/en/"
    tokenized_data_path = Path("tokenized_data.pth")
    tokenizer_path = Path("tokenizer.json")

    if tokenized_data_path.exists():
        tokenized_data = torch.load(tokenized_data_path)
        tokenizer = data.CharTokenizer.load(str(tokenizer_path))
    else:
        tokenizer, tokenized_data = data.load_data(DATA_PATH)
        torch.save(tokenized_data, tokenized_data_path)
        tokenizer.save(str(tokenizer_path))

    base_config = {
        "seq_len": 128,
        "batch_size": 64,
        "n_layers": 6,
        "n_heads": 8,
        "embed_size": 192,
        "learning_rate": 5e-4,
        "gradient_clipping": 1.0,
        "num_batches_to_train": 50000,
        "val_interval": 100,
        "with_residuals": True,
        "use_pre_norm": True,
        "init_scheme": "xavier_uniform",
        "weight_decay": 0.0,
        "embedding_dropout": 0.0,
        "attention_dropout": 0.0,
        "self_attention_dropout": 0.0,
        "scheduler_type": "none",
    }

    experiments = build_experiments(
        base_config,
        [
            {
                "exp_name": "deep_narrow",
                "n_layers": 8,
                "embed_size": 128,
                "learning_rate": 5e-4,
                "embedding_dropout": 0.05,
                "attention_dropout": 0.0,
                "self_attention_dropout": 0.1,
                "scheduler_type": "cosine",
                "min_lr_ratio": 0.1,
            },
            {
                "exp_name": "shallow_wide",
                "n_layers": 4,
                "embed_size": 256,
                "learning_rate": 4e-4,
                "embedding_dropout": 0.1,
                "attention_dropout": 0.05,
                "self_attention_dropout": 0.05,
                "scheduler_type": "linear",
                "warmup_steps": 500,
            },
        ],
    )

    leaderboard = []

    for config in experiments:
        best_val_loss = run_experiment(config, tokenizer, tokenized_data)
        print(f"Experiment {config['exp_name']} best validation loss: {best_val_loss:.4f}", flush=True)
        leaderboard.append(
            {
                "exp_name": config["exp_name"],
                "best_val_loss": best_val_loss,
                "summary_path": str(Path("experiments") / config["exp_name"] / "summary.json"),
            }
        )
        torch.cuda.empty_cache()

    leaderboard = sorted(leaderboard, key=lambda x: x["best_val_loss"])
    with open(Path("experiments") / "leaderboard.json", "w") as leaderboard_file:
        json.dump(leaderboard, leaderboard_file, indent=4)
