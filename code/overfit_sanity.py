from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import torch
from torch import optim

import data
import lm
from transformer import TransformerLM


DEFAULT_DATA_PATH = str((Path(__file__).resolve().parent.parent / "data" / "he").resolve())


DEFAULT_ARCH = {
    "seq_len": 128,
    "n_layers": 8,
    "n_heads": 8,
    "embed_size": 256,
    "with_residuals": True,
    "use_pre_norm": True,
    "init_scheme": "xavier_uniform",
    "embedding_dropout": 0.0,
    "attention_dropout": 0.0,
    "self_attention_dropout": 0.0,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Overfit sanity check: train on one tiny fixed batch to verify optimization works."
    )
    parser.add_argument("--data_path", type=str, default=DEFAULT_DATA_PATH, help="Folder with .txt files")
    parser.add_argument("--batch_size", type=int, default=8, help="Fixed tiny batch size")
    parser.add_argument("--steps", type=int, default=300, help="Optimization steps on the same batch")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--seed", type=int, default=7, help="Random seed")
    parser.add_argument(
        "--arch_json",
        type=str,
        default=None,
        help="JSON string overriding architecture keys, e.g. '{\"n_layers\":4,\"embed_size\":192}'",
    )
    parser.add_argument(
        "--arch_file",
        type=str,
        default=None,
        help="Path to JSON file with architecture overrides",
    )
    parser.add_argument(
        "--print_every",
        type=int,
        default=25,
        help="How often to print training loss",
    )
    return parser.parse_args()


def build_arch(args: argparse.Namespace) -> dict:
    arch = dict(DEFAULT_ARCH)

    if args.arch_file is not None:
        with open(args.arch_file, "r") as f:
            file_arch = json.load(f)
        arch.update(file_arch)

    if args.arch_json is not None:
        inline_arch = json.loads(args.arch_json)
        arch.update(inline_arch)

    required = ["seq_len", "n_layers", "n_heads", "embed_size"]
    for key in required:
        if key not in arch:
            raise ValueError(f"Missing required architecture key: {key}")

    return arch


def make_fixed_batch(tokenized_data: list[list[int]], seq_len: int, batch_size: int, seed: int) -> torch.LongTensor:
    rng = random.Random(seed)
    windows = []
    needed_len = seq_len + 1

    selected_sequence = None
    for seq in tokenized_data:
        if len(seq) >= needed_len:
            selected_sequence = seq
            break

    if selected_sequence is None:
        raise ValueError(
            f"No sequence long enough for seq_len={seq_len}. Need at least {needed_len} tokens in a single sequence. "
            "Use a smaller seq_len for this sanity check."
        )

    while len(windows) < batch_size:
        start = rng.randint(0, len(selected_sequence) - needed_len)
        windows.append(selected_sequence[start : start + needed_len])

    return torch.tensor(windows, dtype=torch.long)


def main() -> None:
    args = parse_args()
    arch = build_arch(args)

    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    data_path = Path(args.data_path).expanduser().resolve()
    txt_files = list(data_path.glob("*.txt"))
    if len(txt_files) == 0:
        raise ValueError(
            f"No .txt files found under data_path={data_path}. "
            "Pass --data_path explicitly (e.g., --data_path ../data/he when running from code/)."
        )

    print(f"Loading data and tokenizer from: {data_path}", flush=True)
    tokenizer, tokenized_data = data.load_data(str(data_path))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}", flush=True)
    print(f"Architecture: {json.dumps(arch, indent=2)}", flush=True)

    model = TransformerLM(
        n_layers=arch["n_layers"],
        n_heads=arch["n_heads"],
        embed_size=arch["embed_size"],
        max_context_len=arch["seq_len"],
        vocab_size=tokenizer.vocab_size(),
        mlp_hidden_size=arch.get("mlp_hidden_size", arch["embed_size"] * 4),
        with_residuals=arch.get("with_residuals", True),
        use_pre_norm=arch.get("use_pre_norm", True),
        init_scheme=arch.get("init_scheme", "xavier_uniform"),
        embedding_dropout=arch.get("embedding_dropout", 0.0),
        attention_dropout=arch.get("attention_dropout", 0.0),
        self_attention_dropout=arch.get("self_attention_dropout", 0.0),
    ).to(device)

    optimizer = optim.AdamW(model.parameters(), lr=args.lr)

    fixed_batch = make_fixed_batch(tokenized_data, arch["seq_len"], args.batch_size, args.seed).to(device)
    batch_x, batch_y = lm.batch_to_labeled_samples(fixed_batch)

    model.train()
    with torch.no_grad():
        init_logits = model(batch_x)
        init_loss = lm.compute_loss(init_logits, batch_y, pad_id=tokenizer.pad_id()).item()

    print(
        f"Overfit sanity check starts. fixed batch size={args.batch_size}, seq_len={arch['seq_len']}, steps={args.steps}",
        flush=True,
    )
    print(f"Initial loss: {init_loss:.6f}", flush=True)

    last_loss = init_loss
    for step in range(1, args.steps + 1):
        logits = model(batch_x)
        loss = lm.compute_loss(logits, batch_y, pad_id=tokenizer.pad_id())

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        last_loss = loss.item()
        if step % args.print_every == 0 or step == 1 or step == args.steps:
            print(f"step={step:4d} loss={last_loss:.6f}", flush=True)

    print("", flush=True)
    print(f"Final loss: {last_loss:.6f}", flush=True)
    print(f"Absolute drop: {init_loss - last_loss:.6f}", flush=True)
    if last_loss < init_loss:
        print("PASS: loss decreased on the same tiny batch.", flush=True)
    else:
        print("WARN: loss did not decrease. Try higher steps or lr, or smaller seq_len.", flush=True)


if __name__ == "__main__":
    main()