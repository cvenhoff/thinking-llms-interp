"""Evaluate diff-of-means steering vectors on GSM8K.

For each category, generates steered (and baseline) responses on GSM8K
questions and evaluates correctness via an API judge.

No thinking model is needed at runtime – steering vectors are applied
unconditionally to every token from position 1 onwards (coefficient=1).
Vectors are rescaled to overall-mean-activation norm at the best layer
identified by the attribution analysis.

Usage:
    python eval_gsm8k_steering.py \
        --model Qwen/Qwen2.5-7B \
        --vectors_dir results/diff_of_means \
        --n_tasks 200 --max_new_tokens 2000 --gen_batch_size 16
"""

import argparse
import asyncio
import json
import os
import re
import sys
import random
import gc
import time
from collections import defaultdict
from typing import Optional

import torch
import numpy as np
from tqdm import tqdm
from datasets import load_dataset

# Allow imports from parent directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import utils.utils as utils


# ---------------------------------------------------------------------------
# Args
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description="GSM8K steering evaluation")
    p.add_argument("--model", type=str, required=True,
                   help="Base model to steer (e.g. Qwen/Qwen2.5-7B)")
    p.add_argument("--vectors_dir", type=str, default="results/diff_of_means",
                   help="Directory containing dom_vectors, dom_metadata, dom_best_layers")
    p.add_argument("--n_tasks", type=int, default=200,
                   help="Number of GSM8K questions to evaluate")
    p.add_argument("--max_new_tokens", type=int, default=2000)
    p.add_argument("--gen_batch_size", type=int, default=16)
    p.add_argument("--coefficient", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--api_model", type=str, default="gpt-4.1",
                   help="API model for judging answers")
    p.add_argument("--judge_batch_size", type=int, default=50,
                   help="Batch size for judge API calls")
    p.add_argument("--results_dir", type=str, default="results/gsm8k_steering",
                   help="Directory to save results")
    p.add_argument("--eval_start_idx", type=int, default=0,
                   help="Starting index in the dataset")
    p.add_argument("--load_in_8bit", action="store_true", default=False)
    p.add_argument("--vectors_model_short", type=str, default=None,
                   help="Override model short name for vector file lookup "
                        "(e.g. qwen2.5-7b when running ORZ 7B)")
    p.add_argument("--skip_baseline", action="store_true", default=False,
                   help="Skip baseline generation (load from previous run)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Steering hook (same as compute_diff_of_means.py)
# ---------------------------------------------------------------------------

def _make_additive_hook(steering_vector: torch.Tensor, coefficient: float = 1.0):
    """Forward hook: adds coefficient * steering_vector to hidden states.
    Skips BOS (pos 0) on the first (prefill) call, adds to all on decode."""
    is_first_call = [True]
    scaled_sv = coefficient * steering_vector

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


# ---------------------------------------------------------------------------
# Generation
# ---------------------------------------------------------------------------

def generate_batch(hf_model, tokenizer, prompts, max_new_tokens, batch_size,
                   steering_vector=None, layer=None, coefficient=1.0):
    """Generate responses, optionally with steering."""
    results = [None] * len(prompts)
    for bs in range(0, len(prompts), batch_size):
        batch = prompts[bs: bs + batch_size]
        inputs = tokenizer(batch, return_tensors="pt", padding=True).to(hf_model.device)
        handle = None
        if steering_vector is not None and layer is not None:
            sv = steering_vector.to(hf_model.device).to(hf_model.dtype)
            handle = hf_model.model.layers[layer].register_forward_hook(
                _make_additive_hook(sv, coefficient))
        try:
            with torch.no_grad():
                gen_ids = hf_model.generate(
                    **inputs, max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id,
                    do_sample=False,
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
    return results


# ---------------------------------------------------------------------------
# Judge
# ---------------------------------------------------------------------------

JUDGE_PROMPT = """Please evaluate whether the model arrived at the correct answer for this math problem.

Question: {question}

Correct answer: {correct_answer}

Model's response (including reasoning trace): {model_answer}

Instructions for evaluation:
1. First, identify the correct numerical answer from the "Correct answer" field.
2. Search the ENTIRE model response (including all reasoning steps) for this correct answer.
3. IMPORTANT: If the correct answer appears ANYWHERE in the model's response - whether in the final \\boxed{{}}, after ####, or even mentioned during intermediate reasoning steps - answer YES.
4. The model gets credit if it computed or stated the correct answer at any point, even if:
   - It appeared during "checking" or "verification" steps
   - The final boxed answer is different (due to copying errors, continued generation, etc.)
   - The response continues or becomes garbled after stating the correct answer
5. Only answer NO if the correct numerical answer does not appear anywhere in the response.

Just answer YES if the correct answer appears anywhere in the response, or NO if it doesn't. Nothing else.
"""


def judge_answers(questions, correct_answers, model_answers, api_model, batch_size=50):
    """Judge correctness of model answers via API."""
    prompts = []
    for q, ca, ma in zip(questions, correct_answers, model_answers):
        prompts.append(JUDGE_PROMPT.format(
            question=q, correct_answer=ca,
            model_answer=ma[:6000],  # truncate very long answers
        ))

    all_results = []
    for bs in range(0, len(prompts), batch_size):
        batch = prompts[bs: bs + batch_size]
        max_retries = 3
        for attempt in range(max_retries):
            try:
                responses = asyncio.run(utils.chat_batch(
                    batch, model=api_model, max_tokens=50,
                    max_concurrent_requests=30,
                ))
                for resp in responses:
                    is_correct = "yes" in resp.lower() if isinstance(resp, str) else False
                    all_results.append(is_correct)
                break
            except Exception as e:
                print(f"  Judge API error: {e}, retrying ({attempt+1}/{max_retries})...")
                time.sleep(2 ** attempt)
                if attempt == max_retries - 1:
                    all_results.extend([False] * len(batch))

    return all_results


# ---------------------------------------------------------------------------
# Load vectors
# ---------------------------------------------------------------------------

def load_dom_vectors(vectors_dir, model_short):
    """Load diff-of-means vectors, metadata, and best layers."""
    vectors_path = os.path.join(vectors_dir, f"dom_vectors_multilayer_{model_short}.pt")
    metadata_path = os.path.join(vectors_dir, f"dom_metadata_multilayer_{model_short}.json")
    best_layers_path = os.path.join(vectors_dir, f"dom_best_layers_{model_short}.json")

    vectors_save = torch.load(vectors_path, map_location="cpu")
    with open(metadata_path) as f:
        metadata_save = json.load(f)
    with open(best_layers_path) as f:
        best_layers = json.load(f)

    steering_vectors = {}
    for cat_id, layer_vecs in vectors_save.items():
        meta = metadata_save[cat_id]
        bl = best_layers[cat_id]
        best_layer = int(bl["best_layer"])

        # Get the raw vector at the best layer
        raw_vec = layer_vecs[str(best_layer)]
        raw_norm = raw_vec.norm().item()
        overall_mean_norm = meta["overall_mean_norms"][str(best_layer)]

        # Rescale to overall-mean-activation norm
        if raw_norm > 1e-12:
            rescaled_vec = raw_vec * (overall_mean_norm / raw_norm)
        else:
            rescaled_vec = raw_vec

        steering_vectors[cat_id] = {
            "title": meta["title"],
            "vector": rescaled_vec.to(torch.float32),
            "best_layer": best_layer,
            "raw_norm": raw_norm,
            "rescaled_norm": rescaled_vec.norm().item(),
        }

    return steering_vectors


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    os.makedirs(args.results_dir, exist_ok=True)
    model_short = args.model.split("/")[-1].lower()

    # ---- Load GSM8K ----
    print("Loading GSM8K dataset...")
    dataset = load_dataset("openai/gsm8k", "main")["test"]
    dataset = list(dataset)
    # Select subset
    start = args.eval_start_idx
    end = min(start + args.n_tasks, len(dataset))
    eval_data = dataset[start:end]
    questions = [item["question"] for item in eval_data]
    correct_answers = [item["answer"] for item in eval_data]
    print(f"  Evaluating {len(questions)} questions (idx {start}–{end-1})")

    # ---- Load model ----
    print(f"\nLoading model {args.model}...")
    model, tokenizer = utils.load_model(
        model_name=args.model, load_in_8bit=args.load_in_8bit
    )
    hf_model = getattr(model, "_model", model)
    print(f"Model loaded on {hf_model.device}")

    # ---- Load vectors ----
    vec_model_short = args.vectors_model_short or model_short
    print(f"\nLoading diff-of-means vectors from {args.vectors_dir} "
          f"(model_short={vec_model_short})...")
    steering_vectors = load_dom_vectors(args.vectors_dir, vec_model_short)
    print(f"  Loaded {len(steering_vectors)} categories:")
    for cat_id, sv in sorted(steering_vectors.items()):
        print(f"    Cat {cat_id} ({sv['title']}): layer {sv['best_layer']}, "
              f"norm {sv['rescaled_norm']:.2f} (raw {sv['raw_norm']:.2f})")

    # ---- Format prompts ----
    # Use the same prompt format as the hybrid pipeline
    prompts = []
    for q in questions:
        prompt = (f"Task: Answer the question below. Explain your reasoning "
                  f"step by step.\n\n\n\nQuestion:\n{q}\n\nStep by step answer:\n")
        prompts.append(prompt)

    # ---- Baseline generation ----
    results_path = os.path.join(args.results_dir, f"gsm8k_{model_short}.json")
    baseline_answers = None

    if args.skip_baseline and os.path.exists(results_path):
        print("\n--skip_baseline: loading previous baseline results...")
        with open(results_path) as f:
            prev = json.load(f)
        baseline_answers = prev.get("baseline_answers")

    if baseline_answers is None:
        print(f"\n=== Generating baseline (no steering) responses ===")
        print(f"  {len(prompts)} prompts, batch_size={args.gen_batch_size}, "
              f"max_new_tokens={args.max_new_tokens}")
        baseline_answers = generate_batch(
            hf_model, tokenizer, prompts, args.max_new_tokens,
            args.gen_batch_size,
        )
        print(f"  Done. First answer preview: {baseline_answers[0][:200]}...")

    # ---- Judge baseline ----
    print(f"\nJudging baseline answers with {args.api_model}...")
    baseline_correct = judge_answers(
        questions, correct_answers, baseline_answers,
        args.api_model, args.judge_batch_size,
    )
    baseline_acc = sum(baseline_correct) / len(baseline_correct) * 100
    print(f"  Baseline accuracy: {baseline_acc:.1f}% "
          f"({sum(baseline_correct)}/{len(baseline_correct)})")

    # ---- Steered generation per category ----
    category_results = {}
    for cat_id, sv_info in sorted(steering_vectors.items()):
        title = sv_info["title"]
        vec = sv_info["vector"]
        layer = sv_info["best_layer"]

        print(f"\n=== Cat {cat_id} ({title}) — layer {layer}, "
              f"coeff={args.coefficient}, norm={sv_info['rescaled_norm']:.2f} ===")
        steered_answers = generate_batch(
            hf_model, tokenizer, prompts, args.max_new_tokens,
            args.gen_batch_size,
            steering_vector=vec, layer=layer,
            coefficient=args.coefficient,
        )

        # Judge
        print(f"  Judging steered answers...")
        steered_correct = judge_answers(
            questions, correct_answers, steered_answers,
            args.api_model, args.judge_batch_size,
        )
        steered_acc = sum(steered_correct) / len(steered_correct) * 100
        delta = steered_acc - baseline_acc
        print(f"  Steered accuracy: {steered_acc:.1f}% "
              f"({sum(steered_correct)}/{len(steered_correct)}) "
              f"Δ={delta:+.1f}%")

        category_results[cat_id] = {
            "title": title,
            "layer": layer,
            "rescaled_norm": sv_info["rescaled_norm"],
            "raw_norm": sv_info["raw_norm"],
            "answers": steered_answers,
            "correct": steered_correct,
            "accuracy": steered_acc,
        }

    # ---- Save ----
    save_data = {
        "model": args.model,
        "coefficient": args.coefficient,
        "max_new_tokens": args.max_new_tokens,
        "n_tasks": len(questions),
        "eval_start_idx": args.eval_start_idx,
        "baseline_accuracy": baseline_acc,
        "baseline_correct": baseline_correct,
        "baseline_answers": baseline_answers,
        "questions": questions,
        "correct_answers": correct_answers,
        "categories": {},
    }
    for cat_id, cr in category_results.items():
        save_data["categories"][cat_id] = {
            "title": cr["title"],
            "layer": cr["layer"],
            "rescaled_norm": cr["rescaled_norm"],
            "raw_norm": cr["raw_norm"],
            "accuracy": cr["accuracy"],
            "correct": cr["correct"],
            "answers": cr["answers"],
        }

    with open(results_path, "w") as f:
        json.dump(save_data, f, indent=2)
    print(f"\nSaved results to {results_path}")

    # ---- Summary ----
    print(f"\n{'='*80}")
    print(f"GSM8K Steering Evaluation Summary — {args.model}")
    print(f"{'='*80}")
    print(f"  Baseline: {baseline_acc:.1f}%")
    for cat_id, cr in sorted(category_results.items()):
        delta = cr["accuracy"] - baseline_acc
        print(f"  Cat {cat_id} ({cr['title']}): {cr['accuracy']:.1f}% "
              f"(Δ={delta:+.1f}%) [layer {cr['layer']}]")
    print(f"{'='*80}")


if __name__ == "__main__":
    main()

