from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

import attention
import main


PUNCTUATION = set(",.;:!?\"'()[]{}-_")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze attention head behavior on text samples and generate visual artifacts."
    )
    parser.add_argument(
        "--experiment_dir",
        type=str,
        required=True,
        help="Path to experiment directory containing config.json, best_model.pth, and tokenizer.json.",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Directory for analysis artifacts. Defaults to <experiment_dir>/attention_analysis.",
    )
    parser.add_argument(
        "--samples_file",
        type=str,
        default=None,
        help="Optional text file containing one input sample per line.",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default=None,
        help="Optional .txt file or directory of .txt files used to draw random text snippets.",
    )
    parser.add_argument(
        "--num_samples",
        type=int,
        default=8,
        help="Maximum number of tokenizable samples to analyze.",
    )
    parser.add_argument(
        "--max_chars",
        type=int,
        default=128,
        help="Maximum characters per sample before tokenization.",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default="all",
        help="Layer selection (e.g., 'all', '0,1,3', '0-2').",
    )
    parser.add_argument(
        "--heads",
        type=str,
        default="all",
        help="Head selection (e.g., 'all', '0,2', '1-3').",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        help="Torch device string, e.g. 'cpu', 'cuda', or 'auto'.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for snippet selection.",
    )
    return parser.parse_args()


def resolve_device(device_arg: str) -> torch.device:
    if device_arg == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device_arg)


def parse_indices(spec: str, upper_bound: int, name: str) -> list[int]:
    if upper_bound <= 0:
        raise ValueError(f"{name} upper_bound must be positive, got {upper_bound}.")

    spec = spec.strip().lower()
    if spec == "all":
        return list(range(upper_bound))

    selected: set[int] = set()
    for part in spec.split(","):
        token = part.strip()
        if not token:
            continue
        if "-" in token:
            start_str, end_str = token.split("-", 1)
            start = int(start_str)
            end = int(end_str)
            if start > end:
                raise ValueError(f"Invalid {name} range '{token}': start must be <= end.")
            for idx in range(start, end + 1):
                selected.add(idx)
        else:
            selected.add(int(token))

    if not selected:
        raise ValueError(f"No {name} selected from spec '{spec}'.")

    sorted_indices = sorted(selected)
    for idx in sorted_indices:
        if idx < 0 or idx >= upper_bound:
            raise ValueError(f"{name} index out of bounds: {idx} (valid range 0..{upper_bound - 1}).")
    return sorted_indices


def load_samples_file(samples_file: Path) -> list[str]:
    with open(samples_file, "r", encoding="utf-8", errors="ignore") as file:
        return [line.rstrip("\n") for line in file if line.strip()]


def collect_text_files(data_path: Path) -> list[Path]:
    if data_path.is_file():
        return [data_path]
    if data_path.is_dir():
        return sorted(data_path.glob("*.txt"))
    raise FileNotFoundError(f"data_path does not exist: {data_path}")


def draw_snippets_from_data(data_path: Path, num_candidates: int, max_chars: int, rng: random.Random) -> list[str]:
    files = collect_text_files(data_path)
    if not files:
        raise ValueError(f"No .txt files found in data_path: {data_path}")

    texts: list[str] = []
    for path in files:
        with open(path, "r", encoding="utf-8", errors="ignore") as file:
            text = file.read()
            if text:
                texts.append(text)

    if not texts:
        raise ValueError(f"No readable text content found in data_path: {data_path}")

    snippets: list[str] = []
    attempts = 0
    max_attempts = max(1, num_candidates * 10)

    while len(snippets) < num_candidates and attempts < max_attempts:
        attempts += 1
        source_text = rng.choice(texts)
        if len(source_text) < 2:
            continue

        if len(source_text) <= max_chars:
            snippet = source_text
        else:
            start = rng.randint(0, len(source_text) - max_chars)
            snippet = source_text[start:start + max_chars]

        snippet = snippet.replace("\n", " ").strip()
        if len(snippet) >= 2:
            snippets.append(snippet)

    return snippets


def tokenize_candidates(
    candidates: list[str],
    tokenizer,
    max_chars: int,
    max_samples: int,
) -> tuple[list[dict], list[str]]:
    selected: list[dict] = []
    skipped: list[str] = []

    for text in candidates:
        if len(selected) >= max_samples:
            break

        clipped = text[:max_chars]
        try:
            token_ids = tokenizer.tokenize(clipped)
        except KeyError as exc:
            skipped.append(f"Skipped sample due to unknown character {exc!s}: {clipped[:40]!r}")
            continue

        if len(token_ids) < 2:
            skipped.append(f"Skipped sample shorter than 2 tokens: {clipped[:40]!r}")
            continue

        selected.append({"text": clipped, "tokens": token_ids})

    return selected, skipped


def token_to_label(symbol: str) -> str:
    if symbol == " ":
        return "[SP]"
    if symbol == "\n":
        return "[NL]"
    if symbol == "\t":
        return "[TAB]"
    return symbol


def capture_model_attentions(model, token_ids: list[int], device: torch.device) -> list[torch.Tensor]:
    input_tensor = torch.tensor([token_ids], dtype=torch.long, device=device)
    n_layers = len(model.layers)
    n_heads = model.layers[0].causal_attention.n_heads
    captured_weights: list[torch.Tensor] = []
    original_self_attention = attention.self_attention

    def capturing_self_attention(v, attention_scores, mask = None, dropout_p: float = 0.0, training: bool = False):
        if mask is not None:
            n_q = attention_scores.size(-2)
            n_k = attention_scores.size(-1)
            sliced_mask = mask[:, :n_q, :n_k]
            attention_scores = attention_scores.masked_fill(sliced_mask == 0, float("-inf"))

        weights = F.softmax(attention_scores, dim=-1)
        captured_weights.append(weights.detach().cpu())
        if dropout_p > 0.0:
            weights = F.dropout(weights, p=dropout_p, training=training)
        return weights @ v

    attention.self_attention = capturing_self_attention

    try:
        with torch.no_grad():
            _ = model(input_tensor)
    finally:
        attention.self_attention = original_self_attention

    expected_captures = n_layers * n_heads
    if len(captured_weights) != expected_captures:
        raise RuntimeError(
            f"Expected {expected_captures} head captures but got {len(captured_weights)}."
        )

    by_layer: list[torch.Tensor] = []
    idx = 0
    for _layer_idx in range(n_layers):
        layer_heads = []
        for _head_idx in range(n_heads):
            layer_heads.append(captured_weights[idx])
            idx += 1
        by_layer.append(torch.stack(layer_heads, dim=1))

    return by_layer


def plot_heatmap(matrix: torch.Tensor, labels: list[str], title: str, save_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix.numpy(), cmap="magma", aspect="auto")
    ax.set_title(title)
    ax.set_xlabel("Key position")
    ax.set_ylabel("Query position")
    fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    n = len(labels)
    if n <= 40:
        ticks = list(range(n))
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        ax.set_xticklabels(labels, rotation=90, fontsize=7)
        ax.set_yticklabels(labels, fontsize=7)

    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def initialize_metric_table(n_layers: int, n_heads: int) -> dict[tuple[int, int], dict[str, float]]:
    table: dict[tuple[int, int], dict[str, float]] = {}
    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            table[(layer_idx, head_idx)] = {
                "rows": 0.0,
                "prev_char_hits": 0.0,
                "prev_space_hits": 0.0,
                "punctuation_hits": 0.0,
                "distance_sum": 0.0,
                "entropy_sum": 0.0,
            }
    return table


def initialize_snippet_table(n_layers: int, n_heads: int) -> dict[tuple[int, int], dict[str, list[dict[str, object]]]]:
    table: dict[tuple[int, int], dict[str, list[dict[str, object]]]] = {}
    for layer_idx in range(n_layers):
        for head_idx in range(n_heads):
            table[(layer_idx, head_idx)] = {
                "prev_char": [],
                "prev_space": [],
                "punctuation": [],
            }
    return table


def build_snippet_record(
    symbols: list[str],
    sample_idx: int,
    query_pos: int,
    key_pos: int,
    radius: int = 6,
    max_chars: int = 80,
) -> dict[str, object]:
    start = max(0, min(query_pos, key_pos) - radius)
    end = min(len(symbols), max(query_pos, key_pos) + radius + 1)

    context = "".join(token_to_label(symbol) for symbol in symbols[start:end])
    if len(context) > max_chars:
        context = context[: max_chars - 3] + "..."

    return {
        "sample_index": sample_idx,
        "query_pos": query_pos,
        "key_pos": key_pos,
        "query_symbol": token_to_label(symbols[query_pos]),
        "key_symbol": token_to_label(symbols[key_pos]),
        "context": context,
    }


def maybe_add_snippet(
    snippet_table: dict[tuple[int, int], dict[str, list[dict[str, object]]]],
    layer_idx: int,
    head_idx: int,
    role_name: str,
    snippet_record: dict[str, object],
    max_per_role: int,
) -> None:
    slot = snippet_table[(layer_idx, head_idx)][role_name]
    snippet_key = (
        snippet_record["sample_index"],
        snippet_record["query_pos"],
        snippet_record["key_pos"],
    )
    for existing in slot:
        existing_key = (
            existing["sample_index"],
            existing["query_pos"],
            existing["key_pos"],
        )
        if existing_key == snippet_key:
            return
    if len(slot) < max_per_role:
        slot.append(snippet_record)


def select_diverse_snippets(
    role_snippets: list[dict[str, object]],
    snippets_per_head: int,
    used_positions: set[tuple[int, int, int]],
    used_contexts: set[str],
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    local_samples: set[int] = set()

    # Pass 1: prefer unseen sample + unseen context + unseen (sample,q,k).
    for snippet in role_snippets:
        pos_key = (int(snippet["sample_index"]), int(snippet["query_pos"]), int(snippet["key_pos"]))
        context_key = str(snippet["context"])
        sample_idx = int(snippet["sample_index"])

        if pos_key in used_positions or context_key in used_contexts:
            continue
        if sample_idx in local_samples:
            continue

        selected.append(snippet)
        local_samples.add(sample_idx)
        used_positions.add(pos_key)
        used_contexts.add(context_key)
        if len(selected) >= snippets_per_head:
            return selected

    # Pass 2: relax sample uniqueness, keep global deduplication.
    for snippet in role_snippets:
        pos_key = (int(snippet["sample_index"]), int(snippet["query_pos"]), int(snippet["key_pos"]))
        context_key = str(snippet["context"])
        if pos_key in used_positions or context_key in used_contexts:
            continue

        selected.append(snippet)
        used_positions.add(pos_key)
        used_contexts.add(context_key)
        if len(selected) >= snippets_per_head:
            return selected

    # Pass 3: fallback to fill quota if necessary.
    for snippet in role_snippets:
        if snippet in selected:
            continue
        selected.append(snippet)
        if len(selected) >= snippets_per_head:
            return selected

    return selected


def update_metrics(
    metric_table: dict[tuple[int, int], dict[str, float]],
    snippet_table: dict[tuple[int, int], dict[str, list[dict[str, object]]]],
    attention_by_layer: list[torch.Tensor],
    symbols: list[str],
    sample_idx: int,
    max_snippets_per_role: int = 8,
) -> None:
    for layer_idx, layer_tensor in enumerate(attention_by_layer):
        # layer_tensor shape: [1, heads, seq_len, seq_len]
        head_tensor = layer_tensor[0]
        n_heads = head_tensor.size(0)
        seq_len = head_tensor.size(1)

        for head_idx in range(n_heads):
            mat = head_tensor[head_idx]
            entry = metric_table[(layer_idx, head_idx)]

            for query_pos in range(1, seq_len):
                valid_row = mat[query_pos, : query_pos + 1]
                if valid_row.numel() == 0:
                    continue

                key_pos = int(torch.argmax(valid_row).item())
                pointed_symbol = symbols[key_pos]
                snippet_record = build_snippet_record(symbols, sample_idx, query_pos, key_pos)

                entry["rows"] += 1.0
                if key_pos == query_pos - 1:
                    entry["prev_char_hits"] += 1.0
                    maybe_add_snippet(
                        snippet_table,
                        layer_idx,
                        head_idx,
                        "prev_char",
                        snippet_record,
                        max_snippets_per_role,
                    )
                if pointed_symbol == " ":
                    entry["prev_space_hits"] += 1.0
                    maybe_add_snippet(
                        snippet_table,
                        layer_idx,
                        head_idx,
                        "prev_space",
                        snippet_record,
                        max_snippets_per_role,
                    )
                if pointed_symbol in PUNCTUATION:
                    entry["punctuation_hits"] += 1.0
                    maybe_add_snippet(
                        snippet_table,
                        layer_idx,
                        head_idx,
                        "punctuation",
                        snippet_record,
                        max_snippets_per_role,
                    )

                entry["distance_sum"] += float(query_pos - key_pos)
                probs = valid_row.clamp_min(1e-12)
                entropy = float((-(probs * torch.log(probs))).sum().item())
                entry["entropy_sum"] += entropy


def finalize_metrics(metric_table: dict[tuple[int, int], dict[str, float]]) -> list[dict]:
    results: list[dict] = []

    for (layer_idx, head_idx), entry in sorted(metric_table.items()):
        rows = entry["rows"]
        if rows <= 0:
            results.append(
                {
                    "layer": layer_idx,
                    "head": head_idx,
                    "rows": 0,
                    "prev_char_rate": 0.0,
                    "prev_space_rate": 0.0,
                    "punctuation_rate": 0.0,
                    "avg_distance": 0.0,
                    "avg_entropy": 0.0,
                }
            )
            continue

        results.append(
            {
                "layer": layer_idx,
                "head": head_idx,
                "rows": int(rows),
                "prev_char_rate": entry["prev_char_hits"] / rows,
                "prev_space_rate": entry["prev_space_hits"] / rows,
                "punctuation_rate": entry["punctuation_hits"] / rows,
                "avg_distance": entry["distance_sum"] / rows,
                "avg_entropy": entry["entropy_sum"] / rows,
            }
        )

    return results


def top_heads(stats: list[dict], metric_name: str, top_n: int = 5) -> list[dict]:
    ranked = sorted(stats, key=lambda item: item[metric_name], reverse=True)
    return ranked[: min(top_n, len(ranked))]


def append_role_section(
    lines: list[str],
    title: str,
    top_items: list[dict],
    role_name: str,
    main_metric: str,
    secondary_metric: str,
    heatmap_examples: dict[tuple[int, int], str],
    snippet_table: dict[tuple[int, int], dict[str, list[dict[str, object]]]],
    snippets_per_head: int,
) -> None:
    lines.append(title)
    used_positions: set[tuple[int, int, int]] = set()
    used_contexts: set[str] = set()

    for item in top_items:
        layer = item["layer"]
        head = item["head"]
        key = (layer, head)
        example = heatmap_examples.get(key, "(no heatmap generated for this head)")
        lines.append(
            f"- layer={layer} head={head} {main_metric}={item[main_metric]:.3f} {secondary_metric}={item[secondary_metric]:.3f} example={example}"
        )

        role_snippets = snippet_table[key][role_name]
        if not role_snippets:
            lines.append("  snippets: none captured")
            continue

        diverse_snippets = select_diverse_snippets(
            role_snippets,
            snippets_per_head=snippets_per_head,
            used_positions=used_positions,
            used_contexts=used_contexts,
        )

        for snippet in diverse_snippets:
            lines.append(
                "  "
                f"snippet sample={snippet['sample_index']} q={snippet['query_pos']}({snippet['query_symbol']}) "
                f"-> k={snippet['key_pos']}({snippet['key_symbol']}): {snippet['context']}"
            )
    lines.append("")


def write_report(
    report_path: Path,
    stats: list[dict],
    sample_count: int,
    heatmap_examples: dict[tuple[int, int], str],
    snippet_table: dict[tuple[int, int], dict[str, list[dict[str, object]]]],
    snippets_per_head: int = 2,
) -> None:
    prev_char_top = top_heads(stats, "prev_char_rate")
    prev_space_top = top_heads(stats, "prev_space_rate")
    punctuation_top = top_heads(stats, "punctuation_rate")

    lines: list[str] = []
    lines.append("# Attention Interpretability Report")
    lines.append("")
    lines.append(f"Analyzed samples: {sample_count}")
    lines.append(f"Analyzed head combinations: {len(stats)}")
    lines.append("")

    append_role_section(
        lines,
        title="## Top Heads For Previous-Character Behavior",
        top_items=prev_char_top,
        role_name="prev_char",
        main_metric="prev_char_rate",
        secondary_metric="avg_distance",
        heatmap_examples=heatmap_examples,
        snippet_table=snippet_table,
        snippets_per_head=snippets_per_head,
    )
    append_role_section(
        lines,
        title="## Top Heads For Previous-Space Behavior",
        top_items=prev_space_top,
        role_name="prev_space",
        main_metric="prev_space_rate",
        secondary_metric="avg_entropy",
        heatmap_examples=heatmap_examples,
        snippet_table=snippet_table,
        snippets_per_head=snippets_per_head,
    )
    append_role_section(
        lines,
        title="## Top Heads For Punctuation-Targeting Behavior",
        top_items=punctuation_top,
        role_name="punctuation",
        main_metric="punctuation_rate",
        secondary_metric="avg_entropy",
        heatmap_examples=heatmap_examples,
        snippet_table=snippet_table,
        snippets_per_head=snippets_per_head,
    )

    lines.append("## Interpretation Notes")
    lines.append("- High prev_char_rate suggests a head focusing strongly on immediate predecessor tokens.")
    lines.append("- High prev_space_rate suggests boundary-tracking behavior (word segmentation cues).")
    lines.append("- High punctuation_rate may indicate structural or formatting tracking.")

    with open(report_path, "w", encoding="utf-8") as file:
        file.write("\n".join(lines) + "\n")


def ensure_experiment_files(experiment_dir: Path) -> None:
    required = ["config.json", "best_model.pth", "tokenizer.json"]
    missing = [name for name in required if not (experiment_dir / name).exists()]
    if missing:
        joined = ", ".join(missing)
        raise FileNotFoundError(f"experiment_dir is missing required files: {joined}")


def main_cli() -> None:
    args = parse_args()

    if args.num_samples < 1:
        raise ValueError("num_samples must be >= 1")
    if args.max_chars < 2:
        raise ValueError("max_chars must be >= 2")

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)

    experiment_dir = Path(args.experiment_dir)
    ensure_experiment_files(experiment_dir)

    output_dir = Path(args.output_dir) if args.output_dir is not None else (experiment_dir / "attention_analysis")
    heatmaps_dir = output_dir / "heatmaps"
    text_outputs_dir = output_dir / "sample_outputs"
    output_dir.mkdir(parents=True, exist_ok=True)
    heatmaps_dir.mkdir(parents=True, exist_ok=True)
    text_outputs_dir.mkdir(parents=True, exist_ok=True)

    device = resolve_device(args.device)
    print(f"Loading model/tokenizer from {experiment_dir} on {device}...", flush=True)
    model, tokenizer = main.load_experiment_artifacts(experiment_dir, device=device)

    n_layers = len(model.layers)
    if n_layers == 0:
        raise ValueError("Model has no layers.")
    n_heads = model.layers[0].causal_attention.n_heads

    selected_layers = parse_indices(args.layers, n_layers, "layer")
    selected_heads = parse_indices(args.heads, n_heads, "head")

    if args.samples_file is None and args.data_path is None:
        raise ValueError("Provide at least one sample source: --samples_file and/or --data_path.")

    candidate_samples: list[str] = []
    if args.samples_file is not None:
        candidate_samples.extend(load_samples_file(Path(args.samples_file)))
    if args.data_path is not None:
        candidate_samples.extend(
            draw_snippets_from_data(
                Path(args.data_path),
                num_candidates=max(args.num_samples * 4, 16),
                max_chars=args.max_chars,
                rng=rng,
            )
        )

    if not candidate_samples:
        raise ValueError("No candidate samples could be loaded from the provided sources.")

    sample_items, skipped = tokenize_candidates(
        candidate_samples,
        tokenizer,
        max_chars=min(args.max_chars, model.max_context_len),
        max_samples=args.num_samples,
    )

    if not sample_items:
        raise ValueError("No valid tokenizable samples were found.")

    for skip_msg in skipped:
        print(skip_msg, flush=True)

    metric_table = initialize_metric_table(n_layers, n_heads)
    snippet_table = initialize_snippet_table(n_layers, n_heads)
    heatmap_examples: dict[tuple[int, int], str] = {}

    print(f"Analyzing {len(sample_items)} samples...", flush=True)
    for sample_idx, item in enumerate(sample_items):
        text = item["text"]
        token_ids = item["tokens"]
        symbols = [tokenizer.vocab[token_id] for token_id in token_ids]
        labels = [token_to_label(symbol) for symbol in symbols]

        captured = capture_model_attentions(model, token_ids, device=device)
        generated_token_ids = model.better_sample_continuation(
            token_ids,
            max_tokens_to_generate=500,
            temperature=1.0,
            topK=5,
        )
        network_output_text = tokenizer.detokenize(generated_token_ids)
        update_metrics(metric_table, snippet_table, captured, symbols, sample_idx)

        text_output_path = text_outputs_dir / f"sample_{sample_idx:03d}.txt"
        with open(text_output_path, "w", encoding="utf-8") as file:
            file.write("Input sample:\n")
            file.write(text)
            file.write("\n\n")
            file.write("Network output (sampled continuation):\n")
            file.write(network_output_text)
            file.write("\n")

        for layer_idx in selected_layers:
            for head_idx in selected_heads:
                matrix = captured[layer_idx][0, head_idx]
                heatmap_name = f"sample_{sample_idx:03d}_layer_{layer_idx:02d}_head_{head_idx:02d}.png"
                heatmap_path = heatmaps_dir / heatmap_name
                title = f"Sample {sample_idx} | Layer {layer_idx} Head {head_idx}"
                plot_heatmap(matrix, labels, title, heatmap_path)

                key = (layer_idx, head_idx)
                if key not in heatmap_examples:
                    heatmap_examples[key] = str(Path("heatmaps") / heatmap_name)

    stats = finalize_metrics(metric_table)

    stats_payload = {
        "experiment_dir": str(experiment_dir),
        "num_samples_analyzed": len(sample_items),
        "num_layers": n_layers,
        "num_heads": n_heads,
        "selected_layers": selected_layers,
        "selected_heads": selected_heads,
        "stats": stats,
    }

    stats_path = output_dir / "attention_stats.json"
    with open(stats_path, "w", encoding="utf-8") as file:
        json.dump(stats_payload, file, indent=2)

    report_path = output_dir / "report.md"
    readme_path = output_dir / "README.md"
    write_report(report_path, stats, len(sample_items), heatmap_examples, snippet_table)
    write_report(readme_path, stats, len(sample_items), heatmap_examples, snippet_table)

    print(f"Saved stats: {stats_path}", flush=True)
    print(f"Saved report: {report_path}", flush=True)
    print(f"Saved README: {readme_path}", flush=True)
    print(f"Saved heatmaps under: {heatmaps_dir}", flush=True)
    print(f"Saved sample text outputs under: {text_outputs_dir}", flush=True)


if __name__ == "__main__":
    main_cli()
