"""Compute diff-of-means steering vectors from synthetic contrastive pairs.

Pipeline:
  1. Load synthetic pairs (output of generate_synthetic_pairs.py).
  2. Compute steering vectors at ALL decoder layers.
  3. Run gradient-based layer attribution across ALL layers to find the best
     steering layer per category (argmax restricted to the 25%–75% range).
  4. Save attribution curve plots (full layer range).
  5. Generate raw and steered responses (steering on the best layer).
  6. Auto-grade responses via the API model.

Usage:
    cd train-vectors
    python compute_diff_of_means.py \
        --model Qwen/Qwen2.5-7B \
        --pairs_file results/synthetic_pairs/synthetic_pairs_qwen2.5-7b_10clusters.json \
        --n_eval_questions 20
"""

import argparse
import asyncio
import json
import os
import re
import sys
import random
import gc
from contextlib import contextmanager
from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from tqdm import tqdm

# Allow imports from parent directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import utils.utils as utils

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Compute diff-of-means steering vectors from synthetic pairs"
    )
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B")
    p.add_argument("--thinking_model", type=str,
                   default="Open-Reasoner-Zero/Open-Reasoner-Zero-7B")
    p.add_argument("--pairs_file", type=str,
                   default="results/synthetic_pairs/synthetic_pairs_qwen2.5-7b_10clusters.json")
    p.add_argument("--segment_tokens", type=int, default=0,
                   help="Max tokens of the labelled segment to include (0 = full range)")
    p.add_argument("--save_dir", type=str, default="results/diff_of_means")
    p.add_argument("--n_eval_questions", type=int, default=20)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--load_in_8bit", action="store_true", default=False)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch_size", type=int, default=8,
                   help="Batch size for activation caching forward passes")
    p.add_argument("--gen_batch_size", type=int, default=16,
                   help="Batch size for generation")
    p.add_argument("--n_attribution_examples", type=int, default=30,
                   help="Number of examples per category for attribution analysis")
    p.add_argument("--skip_vectors", action="store_true", default=False,
                   help="Skip vector computation; load pre-saved vectors")
    p.add_argument("--skip_attribution", action="store_true", default=False,
                   help="Skip attribution; load pre-saved attribution")
    p.add_argument("--use_raw_norm", action="store_true", default=False,
                   help="Use raw diff-of-means norm instead of rescaling to mean activation norm")
    p.add_argument("--steer_coeffs", type=str, default="0.5,1.0,1.5,2.0",
                   help="Comma-separated steering coefficients to sweep over")
    p.add_argument("--eval_max_tokens", type=int, default=32,
                   help="Max new tokens to generate for steered continuations (and to keep from raw)")
    p.add_argument("--eval_cats", type=str, default="",
                   help="Comma-separated category IDs to evaluate (default: all)")
    p.add_argument("--api_model", type=str, default="gpt-4.1",
                   help="API model for auto-grading")
    p.add_argument("--regrade_only", type=str, default="",
                   help="Path to existing eval JSON — skip generation, just re-grade and re-select best coeffs")
    p.add_argument("--eval_pairs_file", type=str, default="",
                   help="Separate pairs file for eval (unseen data). If empty, eval samples from --pairs_file.")
    p.add_argument("--n_train_pairs", type=int, default=0,
                   help="Use only the first N pairs for vector computation (0 = all)")
    p.add_argument("--min_layer_frac", type=float, default=0.0,
                   help="Exclude the first X%% of layers when selecting the best layer "
                        "(e.g. 0.2 = skip first 20%% of layers)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CATEGORY_START_TAG = "[CATEGORY_START]"
CATEGORY_END_TAG = "[CATEGORY_END]"


def strip_markers(text: str) -> tuple[str, Optional[str], Optional[int]]:
    """Remove [CATEGORY_START]/[CATEGORY_END] markers from text.

    Returns (cleaned_text, segment_content, segment_char_start_in_cleaned).
    segment_char_start_in_cleaned is the char offset in cleaned_text where
    the segment content begins.
    """
    start_match = re.search(re.escape(CATEGORY_START_TAG), text)
    end_match = re.search(re.escape(CATEGORY_END_TAG), text)
    if start_match is None or end_match is None:
        # No markers — return text as-is
        return text, None, None

    segment_content = text[start_match.end():end_match.start()]
    # Remove both markers
    cleaned = text[:start_match.start()] + segment_content + text[end_match.end():]
    seg_char_start = start_match.start()
    return cleaned, segment_content, seg_char_start


def find_segment_token_positions(
    cleaned_text: str, seg_char_start: int, seg_char_end: int, tokenizer,
) -> Optional[dict]:
    """Find token positions of the contrastive segment in *marker-free* text.

    Args:
        cleaned_text: text with [CATEGORY_START]/[CATEGORY_END] removed.
        seg_char_start: char offset in cleaned_text where segment begins.
        seg_char_end: char offset in cleaned_text one past the segment end.
        tokenizer: HF tokenizer.

    Returns dict with segment_start, segment_end (token indices, end exclusive).
    """
    if seg_char_start is None or seg_char_start <= 0:
        return None

    char_to_token = utils.get_char_to_token_map(cleaned_text, tokenizer)

    # Segment start: first token of the segment content
    seg_start_tok = char_to_token.get(seg_char_start)
    # Segment end: token of last char of segment + 1 (exclusive)
    if seg_char_end is not None and seg_char_end > seg_char_start:
        last_seg_char = seg_char_end - 1
        seg_end_tok_last = char_to_token.get(last_seg_char)
        seg_end_tok = (seg_end_tok_last + 1) if seg_end_tok_last is not None else None
    else:
        seg_end_tok = None

    if seg_start_tok is None:
        return None
    if seg_end_tok is None:
        seg_end_tok = len(tokenizer.encode(cleaned_text))

    return {
        "segment_start": seg_start_tok,
        "segment_end": seg_end_tok,
    }


def get_segment_positions(pos_info: dict) -> list[int]:
    """Return all segment token positions [start, end)."""
    start = pos_info["segment_start"]
    end = pos_info["segment_end"]
    return list(range(start, end))


def _extract_hidden(saved) -> torch.Tensor:
    """Extract hidden state tensor from nnsight saved output (stays on device)."""
    raw = saved[0] if isinstance(saved, tuple) else saved
    return raw.detach()


# ---------------------------------------------------------------------------
# Target layer range
# ---------------------------------------------------------------------------

def get_all_layers(model) -> list[int]:
    """Return ALL decoder layers."""
    n = model.config.num_hidden_layers
    return list(range(n))



# ---------------------------------------------------------------------------
# Multi-layer diff-of-means  (streaming running averages)
# Uses raw HF forward pass with output_hidden_states to avoid nnsight
# trace overhead.  Activation processing stays on GPU; only the running-
# average scalars are kept on CPU in float64.
# ---------------------------------------------------------------------------

_BASELINE = "baseline"
_EDITED = "edited"


def _hf_forward_layers(
    hf_model, input_ids, attention_mask, target_layers: list[int],
) -> dict[int, torch.Tensor]:
    """Run a plain HF forward pass and return hidden states for target layers.
    Returns dict layer -> (batch, seq, d_model) tensors on the model device.
    """
    with torch.no_grad():
        out = hf_model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
    # out.hidden_states is a tuple of (n_layers+1) tensors: embedding + each layer
    hidden = {}
    for l in target_layers:
        hidden[l] = out.hidden_states[l + 1]  # +1 because index 0 = embedding
    del out
    return hidden


def _find_response_start_tok(prompt: str, tokenizer) -> int:
    """Return the token index where the response starts (after assistant template)."""
    return len(tokenizer.encode(prompt))


def compute_steering_vectors_multilayer(
    model,
    tokenizer,
    pairs_data: dict,
    target_layers: list[int],
    batch_size: int = 32,
) -> dict:
    """Compute diff-of-means steering vectors for each category at each layer.

    **Contrastive diff-of-means:**
      D+ = mean hidden state at *edited* segment token positions (the GPT-4.1
           inserted behaviour).
      D- = mean hidden state at the *same character span* in the **baseline**
           response (the original raw continuation that was replaced).

    Both D+ and D- are per-category running averages.  For each valid edit
    we process both the edited text (for D+) and the baseline text (for D-),
    extracting activations at the aligned segment positions.

    Vectors are saved in their **raw** diff-of-means norm.

    Returns dict  cat_id -> {
        "vectors": {layer: tensor},   # raw u = D+ - D-
        "title": str,
        "n_positive": int,
        "overall_mean_norms": {layer: float},  # norm of D- per layer
        "raw_norms": {layer: float},
    }
    """
    pairs = pairs_data["pairs"]
    categories = pairs_data["categories"]
    hf_model = getattr(model, "_model", model)
    d_model = hf_model.config.hidden_size
    device = next(hf_model.parameters()).device

    # 1.  Collect every text + metadata
    #     For each valid edit we add TWO items:
    #       - _EDITED: the cleaned edited text  (for D+)
    #       - _BASELINE_SEG: the baseline text   (for D- at the same span)
    items: list[dict] = []
    for pair_idx, pair in enumerate(pairs):
        prompt = pair["baseline_prompt"]
        resp_start = _find_response_start_tok(prompt, tokenizer)
        baseline_text = prompt + pair["baseline_response"]

        for cat_id, cat_edit in pair.get("category_edits", {}).items():
            if cat_id not in categories:
                continue
            if not cat_edit.get("validation", {}).get("valid", False):
                continue
            raw_edited = prompt + cat_edit["edited_response"]
            cleaned, seg_content, seg_char_start = strip_markers(raw_edited)
            seg_char_end = (seg_char_start + len(seg_content)
                           if seg_char_start is not None and seg_content else None)
            if seg_char_start is None or seg_char_end is None:
                continue

            # D+ item — edited text, segment positions from edited markers
            items.append({
                "text": cleaned, "kind": _EDITED,
                "pair_idx": pair_idx, "cat_id": cat_id,
                "seg_char_start": seg_char_start,
                "seg_char_end": seg_char_end,
                "resp_start": resp_start,
            })

            # D- item — baseline text, same character span
            # The text before seg_char_start is identical between baseline and
            # edited (the edit only replaces the segment).  So the same char
            # offsets in the baseline give us the original text that was replaced.
            # The baseline may be shorter/longer than the edited at the segment,
            # so we clamp seg_char_end to the baseline length.
            bl_seg_end = min(seg_char_end, len(baseline_text))
            bl_seg_start = min(seg_char_start, len(baseline_text))
            if bl_seg_start >= bl_seg_end:
                # Degenerate: segment position beyond baseline length — skip
                items.pop()  # remove the D+ item we just added
                continue
            items.append({
                "text": baseline_text, "kind": _BASELINE,
                "pair_idx": pair_idx, "cat_id": cat_id,
                "seg_char_start": bl_seg_start,
                "seg_char_end": bl_seg_end,
                "resp_start": resp_start,
            })

    # Pre-compute segment token positions (CPU, once) for BOTH D+ and D- items
    for item in items:
        item["_pos_info"] = find_segment_token_positions(
            item["text"], item.get("seg_char_start"),
            item.get("seg_char_end"), tokenizer,
        )

    n_total = len(items)
    n_edited = sum(1 for it in items if it["kind"] == _EDITED)
    n_baseline = sum(1 for it in items if it["kind"] == _BASELINE)
    print(f"\nCollected {n_total} texts for forward pass "
          f"({n_edited} edited D+ + {n_baseline} baseline D-) — "
          f"batch_size={batch_size}, {len(target_layers)} layers")

    # 2.  Sort by length to minimise padding
    sorted_indices = sorted(range(n_total), key=lambda i: len(items[i]["text"]))

    # 3.  Per-category running-average accumulators (CPU, float64)
    cat_plus_mean = {
        cid: {l: torch.zeros(d_model, dtype=torch.float64) for l in target_layers}
        for cid in categories
    }
    cat_minus_mean = {
        cid: {l: torch.zeros(d_model, dtype=torch.float64) for l in target_layers}
        for cid in categories
    }
    cat_plus_count: dict[str, int] = {cid: 0 for cid in categories}
    cat_minus_count: dict[str, int] = {cid: 0 for cid in categories}

    # 4.  Stream batches
    n_batches = (n_total + batch_size - 1) // batch_size
    for batch_start in tqdm(range(0, n_total, batch_size),
                            desc="Forward passes",
                            total=n_batches):
        batch_idx = sorted_indices[batch_start : batch_start + batch_size]
        batch_items = [items[i] for i in batch_idx]
        batch_texts = [it["text"] for it in batch_items]

        encodings = tokenizer(
            batch_texts, return_tensors="pt", padding=True,
            truncation=True, max_length=4096,
        ).to(device)
        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]

        # Forward — plain HF, no nnsight
        hidden_by_layer = _hf_forward_layers(
            hf_model, input_ids, attention_mask, target_layers)

        # Process on GPU, update running averages on CPU
        for j, item in enumerate(batch_items):
            seq_len = int(attention_mask[j].sum().item())
            pos_info = item.get("_pos_info")
            if pos_info is None:
                continue

            selected = get_segment_positions(pos_info)

            for l in target_layers:
                # acts aligned to actual tokens (strip left padding)
                acts = hidden_by_layer[l][j, -seq_len:, :]  # (seq_len, d) on GPU

                max_pos = acts.shape[0] - 1
                sel = [p for p in selected if 0 <= p <= max_pos]
                if not sel:
                    continue

                seg_vec = acts[sel, :].mean(dim=0).cpu().to(torch.float64)
                cid = item["cat_id"]

                if item["kind"] == _EDITED:
                    # D+
                    if l == target_layers[0]:
                        cat_plus_count[cid] += 1
                    cat_plus_mean[cid][l] += (seg_vec - cat_plus_mean[cid][l]) / cat_plus_count[cid]
                else:
                    # D- (baseline at same span)
                    if l == target_layers[0]:
                        cat_minus_count[cid] += 1
                    cat_minus_mean[cid][l] += (seg_vec - cat_minus_mean[cid][l]) / cat_minus_count[cid]

        del hidden_by_layer, input_ids, attention_mask, encodings
        del batch_items, batch_texts
        torch.cuda.empty_cache()
        gc.collect()

    del items, sorted_indices
    gc.collect()

    # 5.  Compute vectors — saved in raw norm
    steering_vectors = {}
    for cat_id, cat_info in sorted(categories.items()):
        n_plus = cat_plus_count[cat_id]
        n_minus = cat_minus_count[cat_id]
        if n_plus == 0 or n_minus == 0:
            print(f"  Category {cat_id} ({cat_info['title']}): "
                  f"n+={n_plus}, n-={n_minus} — skipping.")
            continue

        vectors = {}
        overall_mean_norms = {}
        raw_norms = {}
        for l in target_layers:
            d_minus_norm = cat_minus_mean[cat_id][l].norm().item()
            overall_mean_norms[l] = d_minus_norm
            u_raw = cat_plus_mean[cat_id][l] - cat_minus_mean[cat_id][l]
            raw_norm = u_raw.norm().item()
            raw_norms[l] = raw_norm
            # Save in raw norm (direction × raw magnitude)
            vectors[l] = u_raw.to(torch.float32)

        steering_vectors[cat_id] = {
            "vectors": vectors,
            "title": cat_info["title"],
            "n_positive": n_plus,
            "overall_mean_norms": overall_mean_norms,
            "raw_norms": raw_norms,
        }
        print(f"  Category {cat_id} ({cat_info['title']}): "
              f"{n_plus} D+ / {n_minus} D- examples, "
              f"raw_norm@best≈{max(raw_norms.values()):.2f}")

    return steering_vectors


# ---------------------------------------------------------------------------
# Layer attribution  (gradient-based, à la vector-layer-attribution)
# ---------------------------------------------------------------------------

def compute_kl_metric(logits):
    probs = F.log_softmax(logits, dim=-1)
    detached_probs = F.log_softmax(logits.detach(), dim=-1)
    return F.kl_div(probs, detached_probs, reduction="batchmean")


def compute_layer_attribution(
    model, tokenizer, pairs_data: dict,
    steering_vectors: dict,
    target_layers: list[int],
    n_examples_per_cat: int = 30,
    batch_size: int = 8,
) -> dict:
    """Gradient-based attribution: for each layer and category, compute
    grad · normalized_steering_vector at the decision point (= segment_start - 1).

    Uses batched HF forward+backward passes with hooks to capture gradients.
    Since examples in a batch don't interact (causal attention + padding),
    per-example gradients are correct even when computing a summed loss.

    Returns: cat_id -> {layer: [per-example scores]}
    """
    pairs = pairs_data["pairs"]
    categories = pairs_data["categories"]
    hf_model = getattr(model, "_model", model)
    device = next(hf_model.parameters()).device

    # Collect edited items per category
    cat_items: dict[str, list[dict]] = {cid: [] for cid in categories}
    for pair in pairs:
        for cat_id, cat_edit in pair.get("category_edits", {}).items():
            if cat_id not in categories or cat_id not in steering_vectors:
                continue
            if not cat_edit.get("validation", {}).get("valid", False):
                continue
            raw_text = pair["baseline_prompt"] + cat_edit["edited_response"]
            cleaned, seg_content, seg_char_start = strip_markers(raw_text)
            seg_char_end = (seg_char_start + len(seg_content)
                           if seg_char_start is not None and seg_content else None)
            pos_info = find_segment_token_positions(
                cleaned, seg_char_start, seg_char_end, tokenizer)
            if pos_info is None:
                continue
            cat_items[cat_id].append({"text": cleaned, "pos_info": pos_info})

    attribution = {}  # cat_id -> {layer: list[float]}

    for cat_id, items in sorted(cat_items.items()):
        if not items:
            continue
        title = steering_vectors[cat_id]["title"]
        items = items[:n_examples_per_cat]
        n_batches = (len(items) + batch_size - 1) // batch_size
        print(f"  Attribution for cat {cat_id} ({title}): "
              f"{len(items)} examples, {n_batches} batches (bs={batch_size})")

        layer_scores = {l: [] for l in target_layers}

        # Pre-normalise steering vectors for this category (CPU)
        sv_normed = {}
        for l in target_layers:
            sv = steering_vectors[cat_id]["vectors"][l]
            sv_normed[l] = sv / (sv.norm() + 1e-12)

        for batch_start in tqdm(range(0, len(items), batch_size),
                                desc=f"Cat {cat_id}", total=n_batches,
                                leave=False):
            batch_items = items[batch_start : batch_start + batch_size]
            batch_texts = [it["text"] for it in batch_items]
            batch_dps = []
            for it in batch_items:
                dp = max(0, it["pos_info"]["segment_start"] - 1)
                batch_dps.append(dp)

            # Tokenise & pad
            encodings = tokenizer(
                batch_texts, return_tensors="pt", padding=True,
                truncation=True, max_length=4096,
            ).to(device)
            input_ids = encodings["input_ids"]       # (B, S)
            attention_mask = encodings["attention_mask"]  # (B, S)
            B, S = input_ids.shape

            # Compute actual seq lengths & adjust decision points for left-padding
            seq_lens = attention_mask.sum(dim=1).tolist()  # list[int]
            adj_dps = []  # decision points adjusted for left-padding
            valid_mask = []
            for j in range(B):
                pad_len = S - int(seq_lens[j])
                adj_dp = batch_dps[j] + pad_len
                ok = adj_dp < S
                adj_dps.append(adj_dp if ok else 0)
                valid_mask.append(ok)

            # Register hooks to capture hidden states with gradients
            captured = {}
            handles = []

            def make_hook(layer_idx):
                def hook_fn(module, inp, out):
                    hs = out[0] if isinstance(out, tuple) else out
                    hs.retain_grad()
                    captured[layer_idx] = hs
                return hook_fn

            for l in target_layers:
                h = hf_model.model.layers[l].register_forward_hook(make_hook(l))
                handles.append(h)

            try:
                hf_model.zero_grad()
                out = hf_model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                logits = out.logits  # (B, S, vocab)

                # Per-example KL self-divergence at their decision points
                total_kl = torch.tensor(0.0, device=device, requires_grad=True)
                n_valid = 0
                for j in range(B):
                    if not valid_mask[j]:
                        continue
                    dp = adj_dps[j]
                    lp = F.log_softmax(logits[j, dp:dp+1], dim=-1)
                    dlp = lp.detach()
                    kl_j = F.kl_div(lp, dlp.exp(), reduction="batchmean",
                                    log_target=False)
                    total_kl = total_kl + kl_j
                    n_valid += 1

                if n_valid > 0:
                    total_kl.backward()

                    # Extract per-example, per-layer scores
                    for l in target_layers:
                        if l not in captured or captured[l].grad is None:
                            continue
                        grad = captured[l].grad  # (B, S, d)
                        for j in range(B):
                            if not valid_mask[j]:
                                continue
                            dp = adj_dps[j]
                            g = grad[j, dp, :].detach().cpu().float()
                            score = torch.dot(sv_normed[l], g).abs().item()
                            layer_scores[l].append(score)

            except Exception as e:
                print(f"    [WARN] Attribution batch failed: {e}")
            finally:
                for h in handles:
                    h.remove()
                del captured, input_ids, attention_mask, encodings
                if 'out' in dir():
                    del out
                torch.cuda.empty_cache()

        attribution[cat_id] = layer_scores

    return attribution


def find_best_layers(attribution: dict, all_layers: list[int],
                     min_layer_frac: float = 0.0) -> dict:
    """For each category, find the layer with highest mean attribution.

    Args:
        min_layer_frac: Fraction of early layers to exclude (e.g. 0.2 means
            skip the first 20% of layers when selecting the best).
    """
    n_layers = len(all_layers)
    min_idx = int(n_layers * min_layer_frac)
    candidate_layers = all_layers[min_idx:]
    if not candidate_layers:
        candidate_layers = all_layers  # fallback

    best = {}
    for cat_id, layer_scores in attribution.items():
        means = {}
        for l in all_layers:
            scores = layer_scores.get(l, [])
            means[l] = np.mean(scores) if scores else 0.0
        # Only consider candidate layers for the argmax
        best_layer = max(candidate_layers, key=lambda l: means[l])
        best[cat_id] = {"best_layer": best_layer, "mean_score": means[best_layer],
                        "all_means": means}
    return best


def plot_attribution_curves(
    attribution: dict, steering_vectors: dict,
    target_layers: list[int], save_path: str,
    best_layers: dict,
):
    """Save a per-category subplot of attribution curves."""
    n_cats = len(attribution)
    if n_cats == 0:
        return
    cols = min(4, n_cats)
    rows = (n_cats + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows),
                             facecolor="white", squeeze=False)

    colors = plt.cm.tab10.colors

    for i, (cat_id, layer_scores) in enumerate(sorted(attribution.items())):
        ax = axes[i // cols][i % cols]
        title = steering_vectors[cat_id]["title"]
        means = []
        stds = []
        for l in target_layers:
            scores = layer_scores.get(l, [])
            means.append(np.mean(scores) if scores else 0.0)
            stds.append(np.std(scores) if scores else 0.0)
        means = np.array(means)
        stds = np.array(stds)
        color = colors[i % len(colors)]

        ax.fill_between(target_layers, means - stds, means + stds,
                        alpha=0.2, color=color)
        ax.plot(target_layers, means, color=color, linewidth=2, marker="o",
                markersize=3)
        best_l = best_layers[cat_id]["best_layer"]
        ax.axvline(best_l, color="red", linestyle="--", linewidth=1, alpha=0.7)
        ax.set_title(f"[{cat_id}] {title}", fontsize=10)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Attribution")
        ax.grid(True, alpha=0.3)

    # Hide unused axes
    for j in range(n_cats, rows * cols):
        axes[j // cols][j % cols].set_visible(False)

    fig.suptitle("Layer Attribution (gradient · steering vector)", fontsize=14,
                 weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved attribution plot to {save_path}")


# ---------------------------------------------------------------------------
# Steered generation
# ---------------------------------------------------------------------------

def _make_additive_hook(steering_vector: torch.Tensor, coeff: float = 1.0):
    """Forward hook that adds *coeff * steering_vector* to hidden states.
    Skips BOS in prefill, adds to all in decode."""
    is_first_call = [True]
    scaled_sv = steering_vector * coeff

    def hook_fn(module, input, output):
        if isinstance(output, tuple):
            hs = output[0]
        else:
            hs = output
        if is_first_call[0] and hs.shape[1] > 1:
            hs[:, 1:, :] = hs[:, 1:, :] + scaled_sv
            is_first_call[0] = False
        else:
            hs[:] = hs + scaled_sv
            is_first_call[0] = False
        if isinstance(output, tuple):
            return (hs,) + output[1:]
        return hs

    return hook_fn


def _get_hf_model(model):
    return getattr(model, "_model", model)


def generate_batch(
    hf_model, tokenizer, prompts: list[str], max_new_tokens: int,
    batch_size: int, steering_vector=None, layer=None, steer_coeff: float = 1.0,
) -> list[str]:
    results: list[Optional[str]] = [None] * len(prompts)
    n_batches = (len(prompts) + batch_size - 1) // batch_size
    for bs in tqdm(range(0, len(prompts), batch_size),
                   desc="Generating", total=n_batches):
        batch = prompts[bs : bs + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True,
                           truncation=True).to(hf_model.device)
        handle = None
        if steering_vector is not None and layer is not None:
            sv = steering_vector.to(hf_model.device).to(hf_model.dtype)
            handle = hf_model.model.layers[layer].register_forward_hook(
                _make_additive_hook(sv, coeff=steer_coeff))
        try:
            with torch.no_grad():
                gen_ids = hf_model.generate(
                    **inputs, max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                )
        finally:
            if handle is not None:
                handle.remove()
        pl = inputs["input_ids"].shape[1]
        for j in range(len(batch)):
            results[bs + j] = tokenizer.decode(gen_ids[j, pl:],
                                                skip_special_tokens=True)
        del inputs, gen_ids
        torch.cuda.empty_cache()
    return results  # type: ignore


def load_eval_questions(thinking_model, n_questions, seed, exclude_qids):
    rng = random.Random(seed)
    thinking_short = thinking_model.split("/")[-1].lower()
    responses_path = os.path.join(
        os.path.dirname(__file__), "..",
        "generate-responses", "results", "vars",
        f"responses_{thinking_short}.json",
    )
    if os.path.exists(responses_path):
        print(f"Loading eval questions from {responses_path}")
        with open(responses_path) as f:
            data = json.load(f)
        candidates = [r for r in data if r["question_id"] not in exclude_qids]
        rng.shuffle(candidates)
        return [{"question": r["original_message"]["content"],
                 "question_id": r["question_id"]}
                for r in candidates[:n_questions]]
    from datasets import load_dataset
    ds = load_dataset("TIGER-Lab/MMLU-Pro")
    rows = [r for r in ds["test"] if r["question_id"] not in exclude_qids]
    rng.shuffle(rows)
    return [{"question": r["question"], "question_id": r["question_id"]}
            for r in rows[:n_questions]]


def build_eval_prompt(question: str) -> str:
    """Return a plain-text prompt (no chat template) for the base model."""
    return (
        "Answer the question below. Explain your reasoning step by step.\n\n"
        f"Question:\n{question}\n\nStep by step answer:\n"
    )


# ---------------------------------------------------------------------------
# Auto-grade via API
# ---------------------------------------------------------------------------

GRADE_PROMPT = """\
You are a strict evaluator of reasoning behaviour in language model outputs.

**Question:** {question}

**Short continuation snippet (~1 sentence) to evaluate:**
{response}

**Target reasoning behaviour:**
Title: {cat_title}
Description: {cat_description}

**Examples of this behaviour (from other responses):**
{examples}

Evaluate on two axes:
1. **Output quality** (1-5): Is the text non-degraded? 5=normal fluent text, 3=somewhat off but readable, 1=gibberish, repetition loops, or incoherent fragments. Do NOT judge correctness or reasoning depth — only whether the text is degraded.
2. **Behaviour presence** (1-5): Does this snippet DIRECTLY and UNAMBIGUOUSLY perform the specific behaviour described above, similar to the examples shown? The snippet's primary communicative purpose must match. Merely being on-topic or tangentially related is NOT enough. 1=absent, 3=partial, 5=snippet exists to perform this behaviour.

Return ONLY a JSON object: {{"answer_quality": <int>, "behaviour_presence": <int>}}
"""


def _extract_behaviour_examples(pairs_data: dict, n_examples: int = 3) -> dict[str, str]:
    """Extract example segments per category from synthetic pairs for grading context.

    Returns dict mapping cat_id -> formatted string of example snippets.
    """
    from collections import defaultdict
    cat_segments: dict[str, list[str]] = defaultdict(list)
    for pair in pairs_data.get("pairs", []):
        for cat_id, cat_edit in pair.get("category_edits", {}).items():
            if not cat_edit.get("validation", {}).get("valid", False):
                continue
            edited = cat_edit.get("edited_response", "")
            s = edited.find("[CATEGORY_START]")
            e = edited.find("[CATEGORY_END]")
            if s >= 0 and e > s:
                seg = edited[s + len("[CATEGORY_START]"):e].strip()
                if 20 < len(seg) < 500:
                    cat_segments[cat_id].append(seg)
    result = {}
    for cat_id, segs in cat_segments.items():
        chosen = segs[:n_examples]
        formatted = "\n".join(f"  - \"{seg[:200]}\"" for seg in chosen)
        result[cat_id] = formatted if formatted else "(no examples available)"
    return result


async def autograde_responses(
    eval_results: list[dict],
    categories: dict,
    api_model: str,
) -> list[dict]:
    """Grade raw and steered responses via the API model."""
    prompts = []
    prompt_keys = []  # (q_idx, response_type, cat_id_or_None)

    for q_idx, r in enumerate(eval_results):
        question = r["question"]

        # Grade raw response against each category
        for cat_id, cat_info in sorted(categories.items()):
            prompts.append(GRADE_PROMPT.format(
                question=question, response=r["raw_response"][:3000],
                cat_title=cat_info["title"],
                cat_description=cat_info.get("description", ""),
            ))
            prompt_keys.append((q_idx, "raw", cat_id))

        # Grade each steered response
        for cat_id, sr in sorted(r.get("steered_responses", {}).items()):
            cat_info = categories.get(cat_id, {})
            prompts.append(GRADE_PROMPT.format(
                question=question, response=sr["steered_response"][:3000],
                cat_title=cat_info.get("title", sr.get("category_title", "")),
                cat_description=cat_info.get("description", ""),
            ))
            prompt_keys.append((q_idx, "steered", cat_id))

    print(f"\nAuto-grading {len(prompts)} responses with {api_model}...")
    api_responses = await utils.chat_batch(
        prompts, model=api_model, max_tokens=100, max_concurrent_requests=30
    )

    # Parse scores
    grades = {}  # (q_idx, response_type, cat_id) -> {answer_quality, behaviour_presence}
    for key, resp in zip(prompt_keys, api_responses):
        try:
            # Try to find JSON in the response
            json_match = re.search(r'\{[^}]+\}', resp)
            if json_match:
                parsed = json.loads(json_match.group())
                grades[key] = {
                    "answer_quality": int(parsed.get("answer_quality", 0)),
                    "behaviour_presence": int(parsed.get("behaviour_presence", 0)),
                }
            else:
                grades[key] = {"answer_quality": 0, "behaviour_presence": 0}
        except Exception:
            grades[key] = {"answer_quality": 0, "behaviour_presence": 0}

    # Attach grades to eval_results
    for q_idx, r in enumerate(eval_results):
        r["raw_grades"] = {}
        for cat_id in categories:
            key = (q_idx, "raw", cat_id)
            r["raw_grades"][cat_id] = grades.get(key, {})

        for cat_id, sr in r.get("steered_responses", {}).items():
            key = (q_idx, "steered", cat_id)
            sr["grade"] = grades.get(key, {})

    return eval_results


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(os.path.join(args.save_dir, "figures"), exist_ok=True)
    model_short = args.model.split("/")[-1].lower()

    # ---- Load pairs ----
    print(f"Loading pairs from {args.pairs_file}...")
    with open(args.pairs_file) as f:
        pairs_data = json.load(f)
    print(f"  Model: {pairs_data['model']}")
    print(f"  Pairs: {len(pairs_data['pairs'])}")
    print(f"  Categories: {list(pairs_data['categories'].keys())}")

    categories = pairs_data["categories"]

    # ---- Optional: restrict training pairs & load separate eval pairs ----
    if args.n_train_pairs > 0 and args.n_train_pairs < len(pairs_data["pairs"]):
        print(f"  Using first {args.n_train_pairs} pairs for vector computation (train split).")
        train_pairs_data = {**pairs_data, "pairs": pairs_data["pairs"][:args.n_train_pairs]}
    else:
        train_pairs_data = pairs_data

    if args.eval_pairs_file:
        print(f"  Loading separate eval pairs from {args.eval_pairs_file}...")
        with open(args.eval_pairs_file) as f:
            eval_pairs_data = json.load(f)
        print(f"  Eval pairs: {len(eval_pairs_data['pairs'])}")
    else:
        eval_pairs_data = None  # will sample from pairs_data

    # ---- Load model ----
    print(f"\nLoading model {args.model}...")
    model, tokenizer = utils.load_model(
        model_name=args.model, load_in_8bit=args.load_in_8bit
    )
    print(f"Model loaded on {model.device}")

    all_layers = get_all_layers(model)
    print(f"All layers: 0–{all_layers[-1]} ({len(all_layers)} total)")

    # ---- Compute or load steering vectors ----
    vectors_path = os.path.join(args.save_dir,
                                f"dom_vectors_multilayer_{model_short}.pt")
    metadata_path = os.path.join(args.save_dir,
                                 f"dom_metadata_multilayer_{model_short}.json")

    if args.skip_vectors and os.path.exists(vectors_path):
        print(f"\n--skip_vectors: loading from {vectors_path}")
        vectors_save = torch.load(vectors_path, map_location="cpu")
        with open(metadata_path) as f:
            metadata_save = json.load(f)
        steering_vectors = {}
        for cat_id, layer_vecs in vectors_save.items():
            meta = metadata_save[cat_id]
            steering_vectors[cat_id] = {
                "vectors": {int(l): v for l, v in layer_vecs.items()},
                "title": meta["title"],
                "n_positive": meta["n_positive"],
                "overall_mean_norms": {int(k): v for k, v in meta["overall_mean_norms"].items()},
                "raw_norms": {int(k): v for k, v in meta["raw_norms"].items()},
            }
        print(f"  Loaded {len(steering_vectors)} categories, "
              f"{len(all_layers)} layers each.")
    else:
        steering_vectors = compute_steering_vectors_multilayer(
            model, tokenizer, train_pairs_data, all_layers,
            batch_size=args.batch_size,
        )
        if not steering_vectors:
            print("No steering vectors computed. Exiting.")
            return

        # Save (raw norm)
        vectors_save = {}
        metadata_save = {}
        for cat_id, sv in steering_vectors.items():
            vectors_save[cat_id] = {str(l): v for l, v in sv["vectors"].items()}
            metadata_save[cat_id] = {
                "title": sv["title"],
                "n_positive": sv["n_positive"],
                "overall_mean_norms": {str(l): v for l, v in sv["overall_mean_norms"].items()},
                "raw_norms": {str(l): v for l, v in sv["raw_norms"].items()},
            }
        torch.save(vectors_save, vectors_path)
        with open(metadata_path, "w") as f:
            json.dump(metadata_save, f, indent=2)
        print(f"\nSaved multi-layer vectors (raw norm) to {vectors_path}")

    # ---- Layer attribution ----
    attrib_path = os.path.join(args.save_dir,
                               f"dom_attribution_{model_short}.json")
    best_layers_path = os.path.join(args.save_dir,
                                    f"dom_best_layers_{model_short}.json")

    if args.skip_attribution and os.path.exists(attrib_path):
        print(f"\n--skip_attribution: loading from {attrib_path}")
        with open(attrib_path) as f:
            attribution = json.load(f)
        attribution = {
            cat_id: {int(l): scores for l, scores in layer_scores.items()}
            for cat_id, layer_scores in attribution.items()
        }
        # Re-derive best layers with the (possibly updated) min_layer_frac
        best_layers = find_best_layers(attribution, all_layers,
                                       min_layer_frac=args.min_layer_frac)
        # Re-save best layers
        best_save = {}
        for cat_id, bl in best_layers.items():
            best_save[cat_id] = {
                "best_layer": bl["best_layer"],
                "mean_score": bl["mean_score"],
                "all_means": {str(l): v for l, v in bl["all_means"].items()},
            }
        with open(best_layers_path, "w") as f:
            json.dump(best_save, f, indent=2)
        print(f"  Re-derived best layers with min_layer_frac={args.min_layer_frac}")
    else:
        print("\nComputing layer attribution...")
        attrib_batch_size = max(1, args.batch_size // 4)  # smaller for grad memory
        attribution = compute_layer_attribution(
            model, tokenizer, train_pairs_data, steering_vectors,
            all_layers,
            n_examples_per_cat=args.n_attribution_examples,
            batch_size=attrib_batch_size,
        )

        best_layers = find_best_layers(attribution, all_layers,
                                       min_layer_frac=args.min_layer_frac)

        # Save
        attrib_save = {
            cat_id: {str(l): scores for l, scores in layer_scores.items()}
            for cat_id, layer_scores in attribution.items()
        }
        with open(attrib_path, "w") as f:
            json.dump(attrib_save, f, indent=2)
        best_save = {}
        for cat_id, bl in best_layers.items():
            best_save[cat_id] = {
                "best_layer": bl["best_layer"],
                "mean_score": bl["mean_score"],
                "all_means": {str(l): v for l, v in bl["all_means"].items()},
            }
        with open(best_layers_path, "w") as f:
            json.dump(best_save, f, indent=2)
        print(f"Saved attribution to {attrib_path}")

    # Plot
    plot_path = os.path.join(args.save_dir, "figures",
                             f"dom_attribution_{model_short}.pdf")
    plot_attribution_curves(attribution, steering_vectors, all_layers,
                            plot_path, best_layers)

    print("\nBest layers per category:")
    for cat_id, bl in sorted(best_layers.items()):
        title = steering_vectors[cat_id]["title"]
        print(f"  Cat {cat_id} ({title}): layer {bl['best_layer']} "
              f"(score={bl['mean_score']:.6f})")

    # ---- Rescale vectors for steering ----
    # Vectors are stored in raw norm.  By default, rescale to overall-mean
    # norm (the norm of the response-token mean activation).  With
    # --use_raw_norm, keep the raw diff-of-means magnitude.
    # The steer_coeff multiplier is applied per-coeff at generation time.
    if not args.use_raw_norm:
        print("\nRescaling vectors to overall-mean-activation norm...")
        for cat_id, sv_info in steering_vectors.items():
            for l, vec in sv_info["vectors"].items():
                raw_n = vec.norm().item()
                overall_n = sv_info["overall_mean_norms"][l]
                if raw_n > 1e-12:
                    sv_info["vectors"][l] = vec * (overall_n / raw_n)
            best_l = best_layers[cat_id]["best_layer"]
            if isinstance(best_l, str):
                best_l = int(best_l)
            new_norm = sv_info["vectors"][best_l].norm().item()
            print(f"  Cat {cat_id} ({sv_info['title']}): "
                  f"layer {best_l} → norm {new_norm:.2f} "
                  f"(raw={sv_info['raw_norms'][best_l]:.2f})")
    else:
        print("\n[use_raw_norm] Keeping vectors at raw diff-of-means norm.")
        for cat_id, sv_info in steering_vectors.items():
            best_l = best_layers[cat_id]["best_layer"]
            if isinstance(best_l, str):
                best_l = int(best_l)
            print(f"  Cat {cat_id} ({sv_info['title']}): "
                  f"layer {best_l} norm {sv_info['vectors'][best_l].norm().item():.2f}")
    steer_coeffs = [float(c) for c in args.steer_coeffs.split(",")]
    print(f"\nSteering coefficients to sweep: {steer_coeffs}")

    # ---- Free GPU before generation ----
    torch.cuda.empty_cache()
    gc.collect()

    # ---- Eval: generate from contrastive-pair prefixes ----
    if args.n_eval_questions <= 0 and not args.regrade_only:
        print("No eval questions requested — done.")
        return

    # ---- Regrade-only mode: load existing eval, skip to grading ----
    if args.regrade_only:
        print(f"\n=== Regrade-only mode: loading {args.regrade_only} ===")
        with open(args.regrade_only) as f:
            prev_eval = json.load(f)
        eval_results = prev_eval["results"]
        steer_coeffs_loaded = prev_eval.get("steer_coeffs", steer_coeffs)
        steer_coeffs = [float(c) for c in steer_coeffs_loaded]
        # Strip old grades
        for er in eval_results:
            for k in list(er.keys()):
                if k.endswith("_grade"):
                    del er[k]
        # Build cat_groups
        from collections import defaultdict
        cat_groups: dict[str, list[int]] = defaultdict(list)
        for idx, er in enumerate(eval_results):
            cat_groups[er["cat_id"]].append(idx)
        print(f"  Loaded {len(eval_results)} items, {len(cat_groups)} categories, coeffs={steer_coeffs}")
    else:
        # ---- Normal generation path ----
        hf_model = _get_hf_model(model)
        # Use separate eval pairs file if provided, else sample from training pairs
        if eval_pairs_data is not None:
            eval_source_pairs = eval_pairs_data["pairs"]
            eval_pairs = eval_source_pairs[:args.n_eval_questions] if args.n_eval_questions > 0 else eval_source_pairs
            print(f"\n=== Eval: generating from {len(eval_pairs)} UNSEEN contrastive-pair prefixes ===")
        else:
            pairs = pairs_data["pairs"]
            rng = random.Random(args.seed + 2000)
            eval_pairs = rng.sample(pairs, min(args.n_eval_questions, len(pairs)))
            print(f"\n=== Eval: generating from {len(eval_pairs)} contrastive-pair prefixes ===")

        # Build eval items: for each pair × category, construct prefix
        # Also extract the raw continuation directly from baseline_response
        # (no need to regenerate — it's already in the pairs data).
        eval_items: list[dict] = []
        for pair in eval_pairs:
            prompt = pair["baseline_prompt"]
            baseline_resp = pair.get("baseline_response", "")
            for cat_id, cat_edit in pair.get("category_edits", {}).items():
                if cat_id not in categories or cat_id not in steering_vectors:
                    continue
                if not cat_edit.get("validation", {}).get("valid", False):
                    continue
                edited_resp = cat_edit["edited_response"]
                marker_pos = edited_resp.find(CATEGORY_START_TAG)
                if marker_pos < 0:
                    continue
                prefix_resp = edited_resp[:marker_pos]
                prefix = prompt + prefix_resp
                # Raw continuation = baseline_response after the prefix portion
                raw_cont = baseline_resp[len(prefix_resp):]
                eval_items.append({
                    "question_id": pair["question_id"],
                    "question": pair["question"],
                    "cat_id": cat_id,
                    "prefix": prefix,
                    "raw_continuation": raw_cont,
                })

        print(f"  Built {len(eval_items)} eval items ({len(eval_pairs)} pairs × categories)")

        # Filter to requested categories if --eval_cats is set
        if args.eval_cats:
            eval_cat_set = set(args.eval_cats.split(","))
            eval_items = [ei for ei in eval_items if ei["cat_id"] in eval_cat_set]
            print(f"  Filtered to cats {eval_cat_set}: {len(eval_items)} eval items")

        # Group by category for steered generation
        from collections import defaultdict
        cat_groups: dict[str, list[int]] = defaultdict(list)
        for idx, ei in enumerate(eval_items):
            cat_groups[ei["cat_id"]].append(idx)

        # Raw continuations: truncate to eval_max_tokens tokens so grading
        # compares the same-length window as steered generation.
        all_prefixes = [ei["prefix"] for ei in eval_items]
        eval_tok_limit = args.eval_max_tokens
        print(f"\nTruncating raw continuations to {eval_tok_limit} tokens each.")
        for ei in eval_items:
            raw_tokens = tokenizer.encode(ei["raw_continuation"],
                                          add_special_tokens=False)
            ei["raw_continuation"] = tokenizer.decode(
                raw_tokens[:eval_tok_limit], skip_special_tokens=True)
        print(f"Raw continuations extracted & truncated (no generation needed).")

        # ---- Generate steered continuations for each (category, coeff) ----
        steered_by_coeff: dict[float, list[Optional[str]]] = {
            c: [None] * len(eval_items) for c in steer_coeffs
        }
        for cat_id, indices in sorted(cat_groups.items()):
            best_layer = best_layers[cat_id]["best_layer"]
            if isinstance(best_layer, str):
                best_layer = int(best_layer)
            sv_vec = steering_vectors[cat_id]["vectors"][best_layer]
            cat_prefixes = [all_prefixes[i] for i in indices]
            for coeff in steer_coeffs:
                print(f"Generating steered continuations for cat {cat_id} "
                      f"({steering_vectors[cat_id]['title']}) at layer {best_layer} "
                      f"(coeff={coeff}, {len(cat_prefixes)} items)...")
                cat_results = generate_batch(
                    hf_model, tokenizer, cat_prefixes, eval_tok_limit,
                    args.gen_batch_size, steering_vector=sv_vec, layer=best_layer,
                    steer_coeff=coeff,
                )
                for j, gi in enumerate(indices):
                    steered_by_coeff[coeff][gi] = cat_results[j]

        # ---- Assemble eval results ----
        eval_results = []
        for idx, ei in enumerate(eval_items):
            entry = {
                "question_id": ei["question_id"],
                "question": ei["question"],
                "cat_id": ei["cat_id"],
                "category_title": steering_vectors[ei["cat_id"]]["title"],
                "steer_layer": best_layers[ei["cat_id"]]["best_layer"],
                "prefix": ei["prefix"][-500:],  # truncate for storage
                "raw_continuation": ei["raw_continuation"],
            }
            for coeff in steer_coeffs:
                entry[f"steered_{coeff}"] = steered_by_coeff[coeff][idx]
            eval_results.append(entry)

    # ---- Auto-grade: raw (once) + steered per coeff ----
    print("\n=== Auto-grading continuations ===")
    # Extract behaviour examples per category from synthetic pairs
    behaviour_examples = _extract_behaviour_examples(pairs_data, n_examples=3)
    grade_prompts = []
    grade_keys = []  # (idx, key_name)  e.g. (0, "raw") or (0, "steered_1.0")
    for idx, er in enumerate(eval_results):
        cat_info = categories.get(er["cat_id"], {})
        cat_title = cat_info.get("title", er.get("category_title", ""))
        cat_desc = cat_info.get("description", "")
        cat_examples = behaviour_examples.get(er["cat_id"], "(no examples available)")
        # Raw — grade once
        grade_prompts.append(GRADE_PROMPT.format(
            question=er["question"], response=er["raw_continuation"][:3000],
            cat_title=cat_title, cat_description=cat_desc,
            examples=cat_examples,
        ))
        grade_keys.append((idx, "raw"))
        # Steered — grade for each coeff
        for coeff in steer_coeffs:
            text = er.get(f"steered_{coeff}", "") or ""
            grade_prompts.append(GRADE_PROMPT.format(
                question=er["question"], response=text[:3000],
                cat_title=cat_title, cat_description=cat_desc,
                examples=cat_examples,
            ))
            grade_keys.append((idx, f"steered_{coeff}"))

    print(f"Grading {len(grade_prompts)} continuations with {args.api_model}...")
    try:
        api_responses = asyncio.run(utils.chat_batch(
            grade_prompts, model=args.api_model, max_tokens=100,
            max_concurrent_requests=30,
        ))
        for key, resp in zip(grade_keys, api_responses):
            try:
                json_match = re.search(r'\{[^}]+\}', resp)
                if json_match:
                    parsed = json.loads(json_match.group())
                    grade = {
                        "answer_quality": int(parsed.get("answer_quality", 0)),
                        "behaviour_presence": int(parsed.get("behaviour_presence", 0)),
                    }
                else:
                    grade = {"answer_quality": 0, "behaviour_presence": 0}
            except Exception:
                grade = {"answer_quality": 0, "behaviour_presence": 0}
            idx, key_name = key
            eval_results[idx][f"{key_name}_grade"] = grade
    except Exception as e:
        print(f"[WARN] Auto-grading failed: {e}")

    # ---- Select best coefficient per category ----
    # Best coeff = argmax of behaviour_delta (behaviour presence improvement)
    print("\n=== Selecting best coefficient per category ===")
    best_coeffs: dict[str, dict] = {}  # cat_id -> {best_coeff, scores_by_coeff, ...}
    for cat_id in sorted(cat_groups.keys()):
        indices = cat_groups[cat_id]
        # Raw baseline scores for this category
        raw_beh = [eval_results[i].get("raw_grade", {}).get("behaviour_presence", 0)
                   for i in indices]
        raw_qual = [eval_results[i].get("raw_grade", {}).get("answer_quality", 0)
                    for i in indices]
        mean_raw_beh = np.mean(raw_beh) if raw_beh else 0
        mean_raw_qual = np.mean(raw_qual) if raw_qual else 0

        coeff_scores = {}
        for coeff in steer_coeffs:
            st_beh = [eval_results[i].get(f"steered_{coeff}_grade", {}).get("behaviour_presence", 0)
                      for i in indices]
            st_qual = [eval_results[i].get(f"steered_{coeff}_grade", {}).get("answer_quality", 0)
                       for i in indices]
            mean_st_beh = np.mean(st_beh) if st_beh else 0
            mean_st_qual = np.mean(st_qual) if st_qual else 0
            beh_delta = mean_st_beh - mean_raw_beh
            qual_delta = mean_st_qual - mean_raw_qual
            rep_pct = (sum(1 for q in st_qual if q == 1) / len(st_qual) * 100) if st_qual else 0
            coeff_scores[coeff] = {
                "beh_delta": float(beh_delta),
                "qual_delta": float(qual_delta),
                "mean_beh": float(mean_st_beh),
                "mean_qual": float(mean_st_qual),
                "rep_pct": float(rep_pct),
            }
        best_c = max(steer_coeffs,
                     key=lambda c: coeff_scores[c]["beh_delta"] + min(0, coeff_scores[c]["qual_delta"]))
        best_coeffs[cat_id] = {
            "best_coeff": best_c,
            "scores_by_coeff": coeff_scores,
            "raw_beh": float(mean_raw_beh),
            "raw_qual": float(mean_raw_qual),
        }
        title = steering_vectors[cat_id]["title"]
        print(f"  Cat {cat_id} ({title}): best_coeff={best_c}")
        for c in steer_coeffs:
            s = coeff_scores[c]
            print(f"    coeff={c}: beh Δ={s['beh_delta']:+.2f}, "
                  f"qual Δ={s['qual_delta']:+.2f}, rep={s['rep_pct']:.0f}%")

    # ---- Plot coefficient sweep figure ----
    fig_dir = os.path.join(args.save_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)
    n_cats = len(cat_groups)
    fig, axes = plt.subplots(n_cats, 1, figsize=(8, 3 * n_cats), sharex=True)
    if n_cats == 1:
        axes = [axes]
    for ax_idx, cat_id in enumerate(sorted(cat_groups.keys())):
        ax = axes[ax_idx]
        bc = best_coeffs[cat_id]
        beh_vals = [bc["scores_by_coeff"][c]["beh_delta"] for c in steer_coeffs]
        qual_vals = [bc["scores_by_coeff"][c]["qual_delta"] for c in steer_coeffs]
        x = np.arange(len(steer_coeffs))
        w = 0.35
        bars1 = ax.bar(x - w/2, beh_vals, w, label="Behaviour Δ", color="#2196F3")
        bars2 = ax.bar(x + w/2, qual_vals, w, label="Quality Δ", color="#FF9800")
        ax.axhline(0, color="black", linewidth=0.5)
        # Mark best coeff
        best_idx = steer_coeffs.index(bc["best_coeff"])
        ax.bar(x[best_idx] - w/2, beh_vals[best_idx], w,
               color="#2196F3", edgecolor="red", linewidth=2.5)
        title = steering_vectors[cat_id]["title"]
        ax.set_title(f"Cat {cat_id}: {title}", fontsize=10, fontweight="bold")
        ax.set_ylabel("Δ score")
        ax.set_xticks(x)
        ax.set_xticklabels([str(c) for c in steer_coeffs])
        if ax_idx == 0:
            ax.legend(loc="upper left", fontsize=8)
    axes[-1].set_xlabel("Steering coefficient")
    plt.tight_layout()
    coeff_fig_path = os.path.join(fig_dir, f"dom_coeff_sweep_{model_short}.pdf")
    plt.savefig(coeff_fig_path, bbox_inches="tight")
    plt.close()
    print(f"\nSaved coefficient sweep figure to {coeff_fig_path}")

    # ---- Overall steering effectiveness figure ----
    sorted_cats = sorted(cat_groups.keys(), key=int)
    cat_labels = [f"[{c}] {steering_vectors[c]['title']}" for c in sorted_cats]
    raw_beh_vals = [best_coeffs[c]["raw_beh"] for c in sorted_cats]
    steered_beh_vals = [best_coeffs[c]["scores_by_coeff"][best_coeffs[c]["best_coeff"]]["mean_beh"]
                        for c in sorted_cats]
    raw_qual_vals = [best_coeffs[c]["raw_qual"] for c in sorted_cats]
    steered_qual_vals = [best_coeffs[c]["scores_by_coeff"][best_coeffs[c]["best_coeff"]]["mean_qual"]
                         for c in sorted_cats]

    y = np.arange(len(sorted_cats))
    bar_h = 0.2
    fig_eff, ax_eff = plt.subplots(figsize=(10, max(6, 0.9 * len(sorted_cats))),
                                   facecolor="white")
    ax_eff.barh(y - 1.5 * bar_h, raw_beh_vals, bar_h, label="Raw behaviour", color="#90CAF9")
    ax_eff.barh(y - 0.5 * bar_h, steered_beh_vals, bar_h, label="Steered behaviour", color="#1565C0")
    ax_eff.barh(y + 0.5 * bar_h, raw_qual_vals, bar_h, label="Raw quality", color="#FFCC80")
    ax_eff.barh(y + 1.5 * bar_h, steered_qual_vals, bar_h, label="Steered quality", color="#E65100")

    ax_eff.set_yticks(y)
    ax_eff.set_yticklabels(cat_labels, fontsize=8)
    ax_eff.set_xlabel("Score (1–5)")
    ax_eff.set_xlim(0, 5.5)
    ax_eff.legend(loc="lower right", fontsize=8)
    ax_eff.set_title("Steering Effectiveness: all categories (best coeff)", fontweight="bold")
    ax_eff.grid(axis="x", alpha=0.3)
    ax_eff.invert_yaxis()
    fig_eff.tight_layout()
    eff_fig_path = os.path.join(fig_dir, f"dom_steering_effectiveness_{model_short}.pdf")
    fig_eff.savefig(eff_fig_path, dpi=150, bbox_inches="tight")
    plt.close(fig_eff)
    print(f"Saved steering effectiveness figure to {eff_fig_path}")

    # ---- Save ----
    eval_suffix = f"_cats{'_'.join(args.eval_cats.split(','))}" if args.eval_cats else ""
    eval_path = os.path.join(args.save_dir, f"dom_eval_{model_short}{eval_suffix}.json")
    best_coeffs_path = os.path.join(args.save_dir, f"dom_best_coeffs_{model_short}.json")
    with open(eval_path, "w") as f:
        json.dump({
            "model": args.model,
            "all_layers": all_layers,
            "best_layers": {cat_id: bl["best_layer"]
                            for cat_id, bl in best_layers.items()},
            "norm_mode": "raw" if args.use_raw_norm else "overall_mean",
            "steer_coeffs": steer_coeffs,
            "best_coeffs": {cid: bc["best_coeff"] for cid, bc in best_coeffs.items()},
            "n_eval_items": len(eval_results),
            "results": eval_results,
        }, f, indent=2)
    print(f"\nSaved eval results ({len(eval_results)} items) to {eval_path}")

    with open(best_coeffs_path, "w") as f:
        json.dump(best_coeffs, f, indent=2)
    print(f"Saved best coefficients to {best_coeffs_path}")

    # ---- Print summary table ----
    print(f"\n{'='*80}")
    print("Grade Summary (raw vs. best-coeff steered):")
    for cat_id in sorted(cat_groups.keys()):
        bc = best_coeffs[cat_id]
        best_c = bc["best_coeff"]
        s = bc["scores_by_coeff"][best_c]
        title = steering_vectors[cat_id]["title"]
        n = len(cat_groups[cat_id])
        print(f"  Cat {cat_id} ({title}) [n={n}, coeff={best_c}]: "
              f"beh {bc['raw_beh']:.2f}→{s['mean_beh']:.2f} "
              f"(Δ={s['beh_delta']:+.2f}), "
              f"qual {bc['raw_qual']:.2f}→{s['mean_qual']:.2f} "
              f"(Δ={s['qual_delta']:+.2f}), "
              f"rep={s['rep_pct']:.0f}%")


if __name__ == "__main__":
    main()
