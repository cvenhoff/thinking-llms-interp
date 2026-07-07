#!/usr/bin/env python3
"""Recompute the activation mean that was used during SAE training and patch it
into SAE checkpoints that are missing it.

The activation mean is computed exactly as in
``utils.utils.process_saved_responses``:

    For each response that has a ``<think>`` block, we
    1. Tokenize the full response.
    2. Find the token span covering ALL thinking sentences.
    3. Take the mean hidden-state over that token span at the target layer.
    4. Update a running mean across all responses.

The resulting vector is the centering mean used before L2-normalization in the
SAE activation pipeline.

Usage
-----
    python patch_sae_activation_mean.py \
        --model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
        --layers 6 10 14 18 22 26 \
        --n_examples 100000 \
        [--load_in_8bit] \
        [--dry_run]
"""

import argparse
import glob
import json
import os
import sys
import gc

import torch
import numpy as np
from tqdm import tqdm

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils import utils
from utils.responses import extract_thinking_process, split_into_sentences


def parse_args():
    p = argparse.ArgumentParser(description="Patch SAE checkpoints with missing activation_mean.")
    p.add_argument("--model", type=str, required=True, help="HuggingFace model name")
    p.add_argument("--layers", type=int, nargs="+", required=True, help="Layers to compute means for")
    p.add_argument("--n_examples", type=int, default=100000, help="n_examples used during original training (default 100000)")
    p.add_argument("--load_in_8bit", action="store_true", default=False)
    p.add_argument("--dry_run", action="store_true", default=False, help="Compute means but don't patch checkpoints")
    p.add_argument("--batch_size", type=int, default=1, help="Batch size for forward passes (1 = safe for large models)")
    p.add_argument("--validate_n_clusters", type=int, default=0,
                   help="If >0, run sanity-check annotation on a few traces with this K (SAE cluster count)")
    p.add_argument("--validate_n_traces", type=int, default=5,
                   help="Number of thinking traces to annotate for validation")
    return p.parse_args()


def compute_activation_means(model, tokenizer, model_id, layers, n_examples):
    """Replicate the mean computation from utils.process_saved_responses.

    Returns dict  layer -> np.ndarray (d_model,) float32
    """
    responses_path = os.path.join(
        os.path.dirname(__file__), "..", "generate-responses", "results", "vars",
        f"responses_{model_id}.json"
    )
    print(f"Loading responses from {responses_path} ...")
    with open(responses_path) as f:
        responses_data = json.load(f)

    # Original code does random.shuffle then [:n_examples].
    # With n_examples=100000 >> len(responses_data) for all our models, ALL
    # responses are used and order only affects floating-point running-mean
    # accumulation (negligible).  We iterate in file order for determinism.
    responses_data = responses_data[:n_examples]
    print(f"  Using {len(responses_data)} responses (n_examples cap={n_examples})")

    hf_model = getattr(model, "_model", model)
    hidden_size = hf_model.config.hidden_size

    mean_by_layer = {l: torch.zeros(1, hidden_size, dtype=torch.float64) for l in layers}
    count_by_layer = {l: 0 for l in layers}

    for resp in tqdm(responses_data, desc="Computing activation means"):
        thinking = extract_thinking_process(resp["full_response"])
        if not thinking:
            continue

        sentences = split_into_sentences(thinking)
        full_response = resp["full_response"]

        input_ids = tokenizer.encode(full_response, return_tensors="pt").to(model.device)

        # Forward pass collecting layer outputs
        layer_outputs = {}
        with model.trace({"input_ids": input_ids,
                          "attention_mask": (input_ids != tokenizer.pad_token_id).long()}) as _tracer:
            for l in layers:
                layer_outputs[l] = model.model.layers[l].output.save()

        for l in layers:
            layer_outputs[l] = layer_outputs[l].detach().cpu().to(torch.float32)

        char_to_token = utils.get_char_to_token_map(full_response, tokenizer)

        for l in layers:
            lo = layer_outputs[l]
            min_tok = float("inf")
            max_tok = -float("inf")

            for sent in sentences:
                pos = full_response.find(sent)
                if pos >= 0:
                    ts = char_to_token.get(pos)
                    te = char_to_token.get(pos + len(sent))
                    if ts is not None and te is not None and ts < te:
                        min_tok = min(min_tok, ts)
                        max_tok = max(max_tok, te)

            if min_tok < lo.shape[1] and max_tok > 0:
                vec = lo[:, min_tok:max_tok, :].mean(dim=1).to(torch.float64)
                count_by_layer[l] += 1
                mean_by_layer[l] += (vec - mean_by_layer[l]) / count_by_layer[l]

        del layer_outputs, input_ids
        torch.cuda.empty_cache()

    results = {}
    for l in layers:
        m = mean_by_layer[l].cpu().numpy().reshape(-1).astype(np.float32)
        assert m.shape == (hidden_size,), f"Bad shape: {m.shape}"
        assert np.isfinite(m).all(), f"Non-finite mean for layer {l}"
        results[l] = m
        print(f"  Layer {l}: mean computed from {count_by_layer[l]} responses, "
              f"norm={np.linalg.norm(m):.4f}")

    return results


def patch_checkpoints(model_id, layers, means, n_examples, dry_run=False):
    """Inject activation_mean into all SAE checkpoints for the given model/layers."""
    sae_dir = os.path.join(os.path.dirname(__file__), "results", "vars", "saes")
    patched = 0
    skipped = 0

    for l in layers:
        pattern = os.path.join(sae_dir, f"sae_{model_id}_layer{l}_clusters*.pt")
        ckpt_paths = sorted(glob.glob(pattern))
        if not ckpt_paths:
            print(f"  [WARN] No checkpoints found for {model_id} layer {l}")
            continue

        mean_t = torch.from_numpy(means[l]).to(torch.float32)

        for path in ckpt_paths:
            ckpt = torch.load(path, map_location="cpu", weights_only=False)

            if "activation_mean" in ckpt:
                # Already has it — verify it matches
                existing = ckpt["activation_mean"]
                if isinstance(existing, torch.Tensor) and existing.shape == mean_t.shape:
                    diff = (existing.float() - mean_t).abs().max().item()
                    if diff < 1e-3:
                        skipped += 1
                        continue
                    else:
                        print(f"  [WARN] {os.path.basename(path)}: existing mean differs (max_diff={diff:.6f}), overwriting")

            ckpt["activation_mean"] = mean_t
            ckpt["activation_mean_model_id"] = model_id
            ckpt["activation_mean_layer"] = l
            ckpt["activation_mean_n_examples"] = n_examples

            if dry_run:
                print(f"  [DRY RUN] Would patch: {os.path.basename(path)}")
            else:
                torch.save(ckpt, path)
                print(f"  Patched: {os.path.basename(path)}")
            patched += 1

    print(f"\nDone: patched={patched}, skipped (already correct)={skipped}")


def save_mean_pickle(model_id, layers, means, n_examples, count_vectors=None):
    """Also save the mean as a pickle file (same format as process_saved_responses)
    so load_activation_mean() works for future SAE training runs."""
    import pickle
    vars_dir = os.path.join(os.path.dirname(__file__), "..", "generate-responses", "results", "vars")
    os.makedirs(vars_dir, exist_ok=True)

    for l in layers:
        mean_path = os.path.join(vars_dir, f"activations_{model_id}_{n_examples}_{l}_mean.pkl")
        payload = {
            "model_id": model_id,
            "layer": int(l),
            "n_examples": int(n_examples),
            "count_vectors": int(count_vectors[l]) if count_vectors else 0,
            "activation_mean": means[l],
        }
        with open(mean_path, "wb") as f:
            pickle.dump(payload, f)
        print(f"  Saved mean pickle: {os.path.basename(mean_path)}")


def validate_annotations(model, tokenizer, model_id, layer, n_clusters, mean_vec,
                         n_traces=5):
    """Annotate a few thinking traces with and without mean centering.

    For each sentence we:
      - Extract the per-sentence mean hidden state at *layer*.
      - **With mean**: center by *mean_vec*, L2-normalise, SAE argmax → category.
      - **Without mean**: just L2-normalise (no centering), SAE argmax → category.

    Saves two JSON files:
      results/vars/sanity_<model_id>_layer<L>_k<K>_with_mean.json
      results/vars/sanity_<model_id>_layer<L>_k<K>_no_mean.json
    """
    from utils.sae import SAE as SAEClass

    sae_dir = os.path.join(os.path.dirname(__file__), "results", "vars", "saes")
    sae_path = os.path.join(sae_dir, f"sae_{model_id}_layer{layer}_clusters{n_clusters}.pt")
    assert os.path.exists(sae_path), f"SAE checkpoint not found: {sae_path}"

    ckpt = torch.load(sae_path, map_location="cpu", weights_only=False)
    sae = SAEClass(ckpt["input_dim"], ckpt["num_latents"], k=ckpt.get("topk", 3))
    sae.encoder.weight.data = ckpt["encoder_weight"]
    sae.encoder.bias.data = ckpt["encoder_bias"]
    sae.W_dec.data = ckpt["decoder_weight"]
    sae.b_dec.data = ckpt["b_dec"]
    sae.eval()

    # Load cluster titles from evaluation results
    titles = {}
    eval_path = os.path.join(os.path.dirname(__file__), "results", "vars",
                             f"sae_topk_results_{model_id}_layer{layer}.json")
    if os.path.exists(eval_path):
        with open(eval_path) as f:
            eval_data = json.load(f)
        k_str = str(n_clusters)
        if k_str in eval_data.get("results_by_cluster_size", {}):
            cats = eval_data["results_by_cluster_size"][k_str]["all_results"][0].get("categories", [])
            for cat in cats:
                titles[int(cat[0])] = cat[1]

    mean_t = torch.from_numpy(mean_vec).to(torch.float32)
    hidden_size = mean_t.shape[0]

    responses_path = os.path.join(
        os.path.dirname(__file__), "..", "generate-responses", "results", "vars",
        f"responses_{model_id}.json"
    )
    with open(responses_path) as f:
        responses_data = json.load(f)

    # Pick traces that have thinking blocks
    selected = []
    for resp in responses_data:
        thinking = extract_thinking_process(resp["full_response"])
        if thinking and len(split_into_sentences(thinking)) >= 3:
            selected.append(resp)
        if len(selected) >= n_traces:
            break

    print(f"\n  Validating on {len(selected)} traces, layer={layer}, K={n_clusters}")

    results_with_mean = []
    results_no_mean = []

    for resp in tqdm(selected, desc="Annotating traces"):
        full_response = resp["full_response"]
        thinking = extract_thinking_process(full_response)
        sentences = split_into_sentences(thinking)

        input_ids = tokenizer.encode(full_response, return_tensors="pt").to(model.device)
        with model.trace({"input_ids": input_ids,
                          "attention_mask": (input_ids != tokenizer.pad_token_id).long()}) as _tracer:
            layer_out = model.model.layers[layer].output.save()
        layer_out = layer_out.detach().cpu().to(torch.float32)

        char_to_token = utils.get_char_to_token_map(full_response, tokenizer)

        trace_with = {"question": resp.get("question", "")[:200], "sentences": []}
        trace_no = {"question": resp.get("question", "")[:200], "sentences": []}

        for sent in sentences:
            pos = full_response.find(sent)
            if pos < 0:
                continue
            ts = char_to_token.get(pos)
            te = char_to_token.get(pos + len(sent))
            if ts is None or te is None or ts >= te:
                continue

            seg = layer_out[:, ts - 1:te, :]
            if seg.shape[1] == 0:
                continue
            act = seg.mean(dim=1).squeeze(0)  # (d_model,)

            # --- With mean centering ---
            centered = act - mean_t
            normed_with = centered / centered.norm()
            with torch.no_grad():
                logits_with = sae.encoder(normed_with - sae.b_dec)
            cat_with = int(logits_with.argmax().item())

            # --- Without mean centering (raw L2 norm) ---
            normed_no = act / act.norm()
            with torch.no_grad():
                logits_no = sae.encoder(normed_no - sae.b_dec)
            cat_no = int(logits_no.argmax().item())

            sent_short = sent[:120]
            trace_with["sentences"].append({
                "text": sent_short,
                "category_id": cat_with,
                "category_title": titles.get(cat_with, f"[{cat_with}]"),
                "top_logit": f"{logits_with.max().item():.3f}",
            })
            trace_no["sentences"].append({
                "text": sent_short,
                "category_id": cat_no,
                "category_title": titles.get(cat_no, f"[{cat_no}]"),
                "top_logit": f"{logits_no.max().item():.3f}",
            })

        results_with_mean.append(trace_with)
        results_no_mean.append(trace_no)

        del layer_out, input_ids
        torch.cuda.empty_cache()

    # Count agreement
    total_sents = sum(len(t["sentences"]) for t in results_with_mean)
    agree = 0
    for tw, tn in zip(results_with_mean, results_no_mean):
        for sw, sn in zip(tw["sentences"], tn["sentences"]):
            if sw["category_id"] == sn["category_id"]:
                agree += 1
    pct = 100.0 * agree / total_sents if total_sents else 0
    print(f"  Agreement (with vs without mean): {agree}/{total_sents} sentences ({pct:.1f}%)")

    out_dir = os.path.join(os.path.dirname(__file__), "results", "vars")
    path_with = os.path.join(out_dir, f"sanity_{model_id}_layer{layer}_k{n_clusters}_with_mean.json")
    path_no = os.path.join(out_dir, f"sanity_{model_id}_layer{layer}_k{n_clusters}_no_mean.json")
    with open(path_with, "w") as f:
        json.dump(results_with_mean, f, indent=2)
    with open(path_no, "w") as f:
        json.dump(results_no_mean, f, indent=2)
    print(f"  Saved: {os.path.basename(path_with)}")
    print(f"  Saved: {os.path.basename(path_no)}")


def main():
    args = parse_args()
    model_id = args.model.split("/")[-1].lower()

    print(f"Model: {args.model}  (id: {model_id})")
    print(f"Layers: {args.layers}")
    print(f"n_examples: {args.n_examples}")
    print()

    # Check which layers actually need patching
    sae_dir = os.path.join(os.path.dirname(__file__), "results", "vars", "saes")
    layers_needing_patch = []
    for l in args.layers:
        pattern = os.path.join(sae_dir, f"sae_{model_id}_layer{l}_clusters*.pt")
        ckpt_paths = glob.glob(pattern)
        if not ckpt_paths:
            print(f"  Layer {l}: no checkpoints found, skipping")
            continue
        # Check first checkpoint
        ckpt = torch.load(ckpt_paths[0], map_location="cpu", weights_only=False)
        if "activation_mean" in ckpt:
            print(f"  Layer {l}: already has activation_mean (✓)")
        else:
            print(f"  Layer {l}: MISSING activation_mean (✗) — will compute")
            layers_needing_patch.append(l)

    if not layers_needing_patch:
        print("\nAll checkpoints already have activation_mean. Nothing to do.")
        return

    print(f"\nLayers to compute: {layers_needing_patch}")
    print(f"\nLoading model {args.model} ...")
    model, tokenizer = utils.load_model(model_name=args.model, load_in_8bit=args.load_in_8bit)
    print(f"Model loaded on {model.device}")

    means = compute_activation_means(
        model, tokenizer, model_id, layers_needing_patch, args.n_examples
    )

    print("\nPatching SAE checkpoints ...")
    patch_checkpoints(model_id, layers_needing_patch, means, args.n_examples, dry_run=args.dry_run)

    print("\nSaving mean pickle files ...")
    save_mean_pickle(model_id, layers_needing_patch, means, args.n_examples)

    # Validation: annotate traces with and without mean centering
    if args.validate_n_clusters > 0:
        # Use the first layer in the list for validation
        val_layer = layers_needing_patch[0]
        print(f"\n{'='*60}")
        print(f"VALIDATION: annotating {args.validate_n_traces} traces (layer {val_layer}, K={args.validate_n_clusters})")
        print(f"{'='*60}")
        validate_annotations(
            model, tokenizer, model_id, val_layer, args.validate_n_clusters,
            means[val_layer], n_traces=args.validate_n_traces,
        )

    # Free GPU memory
    del model, tokenizer
    torch.cuda.empty_cache()
    gc.collect()

    print("\n✓ All done.")


if __name__ == "__main__":
    main()

