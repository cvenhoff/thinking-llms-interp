"""Generate paired synthetic data for diff-of-means steering vectors.

Strategy:
  1. Generate baseline responses from the local model (e.g. Qwen2.5-7B).
  2. For each reasoning category, call an API model (GPT-4.1) to *edit* the
     baseline response by inserting a single, cleanly realised instance of that
     category's behaviour, surrounded by parsable labels:
         [CATEGORY_START] ... [CATEGORY_END]
  3. The resulting paired data (baseline vs. edited) can later be fed through
     the local model to extract activations.  The parsable labels let us locate
     the *exact* contrastive segment so we can compute a targeted diff-of-means.

Category descriptions are loaded dynamically from:
  - SAE clustering results (default): from train-saes/results/vars/
  - 7-category annotation framework (--use_7cat): parsed from utils.py

Usage:
    cd train-vectors
    python generate_synthetic_pairs.py \\
        --model Qwen/Qwen2.5-7B \\
        --n_questions 10 \\
        --max_new_tokens 512 \\
        --use_7cat
"""

import argparse
import asyncio
import json
import os
import re
import sys
import random
import gc
from typing import Optional

import torch
from tqdm import tqdm

# Allow imports from parent directory
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import utils.utils as utils
from messages import messages, eval_messages


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Generate paired synthetic data for diff-of-means steering vectors"
    )
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B",
                   help="Base model to generate baseline responses from")
    p.add_argument("--api_model", type=str, default="gpt-4.1",
                   help="API model used to edit responses")
    p.add_argument("--thinking_model", type=str,
                   default="Open-Reasoner-Zero/Open-Reasoner-Zero-7B",
                   help="Thinking model whose SAE categories to use")
    p.add_argument("--layer", type=int, default=10,
                   help="Layer for SAE category descriptions")
    p.add_argument("--n_clusters", type=int, default=10,
                   help="Number of clusters in SAE taxonomy to use")
    p.add_argument("--n_questions", type=int, default=50,
                   help="Number of questions to use")
    p.add_argument("--max_new_tokens", type=int, default=512,
                   help="Maximum new tokens for local model generation")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--categories", type=str, default="",
                   help="Comma-separated category indices (empty = all)")
    p.add_argument("--dataset", type=str, default="TIGER-Lab/MMLU-Pro",
                   choices=["messages", "TIGER-Lab/MMLU-Pro"])
    p.add_argument("--load_in_8bit", action="store_true", default=False)
    p.add_argument("--batch_size", type=int, default=32,
                   help="Batch size for local model generation")
    p.add_argument("--save_dir", type=str, default="results/synthetic_pairs")
    p.add_argument("--use_7cat", action="store_true", default=False,
                   help="Use the 7-category annotation framework instead of SAE clusters")
    p.add_argument("--question_offset", type=int, default=0,
                   help="Start index for question selection (skip first N questions)")
    p.add_argument("--output_suffix", type=str, default="",
                   help="Optional suffix appended to output filename (e.g. '_eval')")
    return p.parse_args()


# ---------------------------------------------------------------------------
# Category loading — everything is loaded dynamically, nothing hardcoded
# ---------------------------------------------------------------------------

def load_sae_category_descriptions(thinking_model: str, layer: int,
                                    n_clusters: int) -> dict[int, dict]:
    """Load category titles, descriptions, and top-activating examples from SAE
    clustering results.  The examples come from real reasoning model outputs and
    keep the API editor on-policy."""
    model_id = thinking_model.split("/")[-1].lower()
    sae_results_dir = os.path.join(
        os.path.dirname(__file__), "..", "train-saes", "results", "vars"
    )
    candidate_ids = list({
        model_id, model_id.replace("-", "_"), model_id.replace("_", "-")
    })
    for mid in candidate_ids:
        for try_layer in [layer, layer - 2, layer + 2, layer - 4, layer + 4]:
            fn = os.path.join(
                sae_results_dir,
                f"sae_topk_results_{mid}_layer{try_layer}.json",
            )
            if os.path.exists(fn):
                with open(fn) as f:
                    data = json.load(f)
                cluster_key = str(n_clusters)
                if cluster_key in data.get("results_by_cluster_size", {}):
                    results = data["results_by_cluster_size"][cluster_key]
                    categories_raw = results["all_results"][0]["categories"]
                    # Top-activating examples per cluster
                    examples_by_cluster = results.get("examples", {})
                    categories = {}
                    for cat in categories_raw:
                        idx = int(cat[0])
                        categories[idx] = {
                            "title": cat[1],
                            "description": cat[2],
                            "top_examples": examples_by_cluster.get(str(idx), []),
                        }
                    print(f"Loaded {len(categories)} category descriptions "
                          f"from {fn} (cluster_size={n_clusters})")
                    return categories
    print(f"WARNING: Could not find SAE results for "
          f"{thinking_model} layer {layer} n_clusters {n_clusters}")
    return {}


def load_7cat_from_annotation_prompt() -> dict[int, dict]:
    """Parse the 7-category reasoning framework from the annotation prompt in
    utils/utils.py — single source of truth."""
    import inspect
    source = inspect.getsource(utils.process_batch_annotations)
    pattern = re.compile(
        r"(\d+)\.\s+([\w-]+)\s+[–—-]\s+(.+?)\n"
        r"\s+\*Description:\*\s+(.+?)\n"
        r"\s+\*Includes:\*\s+(.+?)\n"
        r"\s+\*Excludes:\*\s+(.+?)\n"
        r"\s+\*Examples:\*\s+(.+?)(?:\n\n|\n\d+\.)",
        re.DOTALL,
    )
    categories: dict[int, dict] = {}
    for m in pattern.finditer(source):
        idx = int(m.group(1)) - 1
        categories[idx] = {
            "title": m.group(3).strip(),
            "description": m.group(4).strip(),
            "includes": m.group(5).strip(),
            "excludes": m.group(6).strip(),
            "examples": m.group(7).strip(),
        }
    if not categories:
        raise RuntimeError(
            "Failed to parse 7-category framework from "
            "utils.process_batch_annotations."
        )
    print(f"Parsed {len(categories)} categories from annotation prompt")
    return categories


def load_categories(use_7cat: bool, thinking_model: str,
                    layer: int, n_clusters: int) -> dict[int, dict]:
    if use_7cat:
        return load_7cat_from_annotation_prompt()
    return load_sae_category_descriptions(thinking_model, layer, n_clusters)


# ---------------------------------------------------------------------------
# API editing prompt — asks the API model to insert one contrastive segment
# ---------------------------------------------------------------------------

EDIT_PROMPT_TEMPLATE = """\
You are an expert at editing reasoning traces to create contrastive training examples.

Below is a step-by-step reasoning response that a language model produced for a question. Your task is to **replace** a portion of this response with a segment that demonstrates the reasoning behaviour described below. The key requirement is MAXIMUM CONTRAST: you must choose a location where the original response is doing something VERY DIFFERENT from the target behaviour, and replace that part with the target behaviour.

## Reasoning behaviour to insert

**Title:** {title}
**Description:** {description}
{extra_metadata}

## Rules

1. **Choose a CONTRASTIVE insertion point.** Find a location in the original response where the model is performing a DIFFERENT type of reasoning than the target behaviour (e.g., if the target is "recalling a formula", find a spot where the model is doing arithmetic, listing data, or drawing conclusions — NOT where it is already recalling formulas). The more different the original text is from the target behaviour, the better.
2. **Replace** 1-3 sentences at that location with a new segment that cleanly demonstrates the target behaviour. Surround it with `[CATEGORY_START]` and `[CATEGORY_END]` labels.
3. The replacement must still feel natural in context — it should flow with the surrounding text, even if it changes the direction of reasoning.
4. Keep the rest of the response IDENTICAL. Only the replaced segment and minor bridging edits should differ.
5. The segment should be a genuine, realistic example of this behaviour — not a meta-comment about it.
6. Do NOT insert at the very beginning of the response. Choose a point at least a few sentences in where the original reasoning has a clear, identifiable character that DIFFERS from the target.
7. Return ONLY the full edited response. No preamble, no explanation, no commentary.
8. You may STOP writing shortly after [CATEGORY_END] — only include enough of the remaining response for the segment to read naturally (one or two sentences after is fine).

## Original question

{question}

## Original response

{response}

## Edited response (with [CATEGORY_START]...[CATEGORY_END] labels)
"""


def build_edit_prompt(question: str, response: str, cat: dict) -> str:
    """Build the API prompt that asks the model to edit the baseline response."""
    extra_parts = []
    if cat.get("includes"):
        extra_parts.append(f"**Includes:** {cat['includes']}")
    if cat.get("excludes"):
        extra_parts.append(f"**Excludes:** {cat['excludes']}")
    if cat.get("examples"):
        extra_parts.append(f"**Example sentences:** {cat['examples']}")
    # Include top-activating examples from real reasoning model outputs
    top_examples = cat.get("top_examples", [])
    if top_examples:
        examples_str = "\n".join(f"  - {ex[:300]}" for ex in top_examples[:5])
        extra_parts.append(
            f"**Real examples of this behaviour from reasoning model traces "
            f"(use these as style reference):**\n{examples_str}"
        )
    extra_metadata = "\n".join(extra_parts)

    return EDIT_PROMPT_TEMPLATE.format(
        title=cat["title"],
        description=cat["description"],
        extra_metadata=extra_metadata,
        question=question,
        response=response,
    )


# ---------------------------------------------------------------------------
# Question loading — uses MMLU-Pro by default (same as the rest of the pipeline)
# ---------------------------------------------------------------------------

def load_questions(dataset: str, n_questions: int, seed: int,
                   thinking_model: str = "",
                   question_offset: int = 0) -> list[dict]:
    """Load questions, preferring the existing responses file so we draw from
    the exact same pool used for SAE training and steering vector optimisation.

    Args:
        question_offset: Skip the first *question_offset* questions (after
            shuffling), useful for generating disjoint train/eval splits with
            the same seed.
    """
    rng = random.Random(seed)

    if dataset == "TIGER-Lab/MMLU-Pro":
        # First try to load from the existing responses file (same pool as
        # optimize_steering_vectors.py and SAE training)
        thinking_short = thinking_model.split("/")[-1].lower() if thinking_model else ""
        responses_path = os.path.join(
            os.path.dirname(__file__), "..",
            "generate-responses", "results", "vars",
            f"responses_{thinking_short}.json",
        )
        if os.path.exists(responses_path):
            print(f"Loading questions from existing responses file: {responses_path}")
            with open(responses_path) as f:
                responses_data = json.load(f)
            rng.shuffle(responses_data)
            selected = responses_data[question_offset:question_offset + n_questions]
            return [{"question": r["original_message"]["content"],
                     "question_id": r["question_id"],
                     "category": r.get("category", "")} for r in selected]

        # Fallback: load directly from HuggingFace
        print("Responses file not found — loading MMLU-Pro from HuggingFace")
        from datasets import load_dataset
        ds = load_dataset("TIGER-Lab/MMLU-Pro")
        rows = list(ds["test"])
        rng.shuffle(rows)
        selected = rows[question_offset:question_offset + n_questions]
        return [{"question": row["question"],
                 "question_id": row["question_id"],
                 "category": row.get("category", "")} for row in selected]

    elif dataset == "messages":
        all_msgs = messages + eval_messages
        rng.shuffle(all_msgs)
        selected = all_msgs[question_offset:question_offset + n_questions]
        return [{"question": msg["content"], "question_id": i}
                for i, msg in enumerate(selected)]

    raise ValueError(f"Unknown dataset: {dataset}")


# ---------------------------------------------------------------------------
# Prompt construction (for local baseline generation)
# ---------------------------------------------------------------------------

BASELINE_TEMPLATE = (
    "Answer the question below. Explain your reasoning step by step.\n\n"
    "Question:\n{question}\n\n"
    "Step by step answer:\n"
)


def build_baseline_prompt(question: str, tokenizer) -> str:
    """Return a plain-text prompt (no chat template) for the base model."""
    return BASELINE_TEMPLATE.format(question=question)


# ---------------------------------------------------------------------------
# Local generation (batched via nnsight)
# ---------------------------------------------------------------------------

def generate_batch(model, tokenizer, prompts: list[str],
                   max_new_tokens: int) -> list[str]:
    encoded = tokenizer(prompts, return_tensors="pt", padding=True)
    input_ids = encoded["input_ids"].to(model.device)
    attention_mask = encoded["attention_mask"].to(model.device)
    input_len = input_ids.shape[1]

    with model.generate(
        {"input_ids": input_ids, "attention_mask": attention_mask},
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
    ) as gen:
        outputs = model.generator.output.save()

    responses = []
    for i in range(outputs.shape[0]):
        new_tokens = outputs[i][input_len:]
        responses.append(tokenizer.decode(new_tokens, skip_special_tokens=True))
    return responses


# ---------------------------------------------------------------------------
# API editing — batch all categories for all questions in parallel
# ---------------------------------------------------------------------------

async def edit_responses_with_api(
    questions: list[str],
    baseline_responses: list[str],
    categories: dict[int, dict],
    api_model: str,
) -> dict[int, dict[int, str]]:
    """For every (question, baseline_response) pair and every category, call
    the API model to produce an edited response with a labeled contrastive
    segment.

    Returns: dict[question_idx][cat_idx] -> edited_response string.
    """
    sorted_cats = sorted(categories.items())

    # Build all prompts
    prompts = []
    prompt_keys = []  # (q_idx, cat_idx) for each prompt
    for q_idx in range(len(questions)):
        for cat_idx, cat in sorted_cats:
            prompt = build_edit_prompt(
                questions[q_idx], baseline_responses[q_idx], cat
            )
            prompts.append(prompt)
            prompt_keys.append((q_idx, cat_idx))

    print(f"\nCalling {api_model} for {len(prompts)} edits "
          f"({len(questions)} questions × {len(sorted_cats)} categories)...")

    # Use the existing chat_batch utility for parallel API calls
    api_responses = await utils.chat_batch(
        prompts, model=api_model, max_tokens=4096, max_concurrent_requests=20
    )

    # Organize results
    results: dict[int, dict[int, str]] = {}
    for (q_idx, cat_idx), resp in zip(prompt_keys, api_responses):
        if q_idx not in results:
            results[q_idx] = {}
        results[q_idx][cat_idx] = resp

    return results


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------

def validate_edit(edited_response: str) -> dict:
    """Check that the edited response has exactly one [CATEGORY_START]...[CATEGORY_END] block."""
    starts = [m.start() for m in re.finditer(r"\[CATEGORY_START\]", edited_response)]
    ends = [m.start() for m in re.finditer(r"\[CATEGORY_END\]", edited_response)]

    if len(starts) == 1 and len(ends) == 1 and starts[0] < ends[0]:
        segment = edited_response[
            starts[0] + len("[CATEGORY_START]"):ends[0]
        ].strip()
        return {"valid": True, "segment": segment}
    return {
        "valid": False,
        "n_starts": len(starts),
        "n_ends": len(ends),
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ---- Load categories ----
    categories = load_categories(
        args.use_7cat, args.thinking_model, args.layer, args.n_clusters
    )
    if not categories:
        print("ERROR: No categories found.")
        sys.exit(1)

    if args.categories:
        requested = [int(x.strip()) for x in args.categories.split(",")]
        categories = {k: v for k, v in categories.items() if k in requested}
        print(f"Filtered to {len(categories)} categories: "
              f"{list(categories.keys())}")

    print("\n=== Categories ===")
    for idx, cat in sorted(categories.items()):
        top_exs = cat.get("top_examples", [])
        print(f"  [{idx}] {cat['title']}  ({len(top_exs)} top examples)")
        for ex in top_exs[:2]:
            print(f"       ↳ {ex[:120]}")
    print()

    # ---- Load questions ----
    questions = load_questions(args.dataset, args.n_questions, args.seed,
                               thinking_model=args.thinking_model,
                               question_offset=args.question_offset)
    print(f"Loaded {len(questions)} questions from {args.dataset}")

    # ---- Load local model and generate baselines ----
    model_name = args.model
    print(f"\nLoading model {model_name}...")
    model, tokenizer = utils.load_model(
        model_name=model_name, load_in_8bit=args.load_in_8bit
    )
    print(f"Model loaded on {model.device}")

    print("\n=== Phase 1: Generating baseline responses (local model) ===")
    baseline_prompts = [
        build_baseline_prompt(q["question"], tokenizer) for q in questions
    ]
    # Generate baselines in batches
    baseline_responses = []
    for i in tqdm(range(0, len(baseline_prompts), args.batch_size),
                  desc="Baseline generation"):
        batch = baseline_prompts[i:i + args.batch_size]
        resps = generate_batch(model, tokenizer, batch, args.max_new_tokens)
        baseline_responses.extend(resps)
        torch.cuda.empty_cache()
        gc.collect()

    for i, resp in enumerate(baseline_responses):
        print(f"  Q{i}: {questions[i]['question'][:60]}...")
        print(f"       Response ({len(resp.split())} words): {resp[:120]}...")

    # ---- Free GPU memory before API calls ----
    del model
    torch.cuda.empty_cache()
    gc.collect()

    # ---- Call API to edit each baseline for each category ----
    print("\n=== Phase 2: Editing responses via API ({}) ===".format(args.api_model))
    question_texts = [q["question"] for q in questions]
    edited = asyncio.run(edit_responses_with_api(
        question_texts, baseline_responses, categories, args.api_model
    ))

    # ---- Assemble and save results ----
    os.makedirs(args.save_dir, exist_ok=True)
    model_short = model_name.split("/")[-1].lower()
    cat_suffix = "7cat" if args.use_7cat else f"{args.n_clusters}clusters"

    all_results = {
        "model": model_name,
        "api_model": args.api_model,
        "thinking_model": args.thinking_model,
        "layer": args.layer,
        "n_clusters": args.n_clusters,
        "n_questions": len(questions),
        "max_new_tokens": args.max_new_tokens,
        "use_7cat": args.use_7cat,
        "categories": {
            str(k): {"title": v["title"], "description": v["description"]}
            for k, v in categories.items()
        },
        "pairs": [],
    }

    n_valid = 0
    n_total = 0
    for q_idx, q_data in enumerate(questions):
        category_edits = {}
        for cat_idx, cat in sorted(categories.items()):
            edited_resp = edited.get(q_idx, {}).get(cat_idx, "")
            validation = validate_edit(edited_resp)
            n_total += 1
            if validation["valid"]:
                n_valid += 1

            category_edits[str(cat_idx)] = {
                "category_title": cat["title"],
                "edited_response": edited_resp,
                "validation": validation,
            }

        pair_entry = {
            "question_id": q_data["question_id"],
            "question": q_data["question"],
            "baseline_prompt": baseline_prompts[q_idx],
            "baseline_response": baseline_responses[q_idx],
            "category_edits": category_edits,
        }
        all_results["pairs"].append(pair_entry)

    suffix = args.output_suffix if args.output_suffix else ""
    save_path = os.path.join(
        args.save_dir,
        f"synthetic_pairs_{model_short}_{cat_suffix}{suffix}.json",
    )
    with open(save_path, "w") as f:
        json.dump(all_results, f, indent=2)

    # ---- Print summary ----
    print(f"\n{'='*80}")
    print(f"Done! {len(questions)} questions × {len(categories)} categories "
          f"= {n_total} edits")
    print(f"Valid edits (with parsable labels): {n_valid}/{n_total} "
          f"({100*n_valid/max(n_total,1):.0f}%)")
    print(f"Saved to {save_path}")

    # Show a few examples
    print(f"\n{'='*80}")
    print("=== Sample outputs ===")
    for q_idx in range(min(2, len(questions))):
        pair = all_results["pairs"][q_idx]
        print(f"\nQ{q_idx}: {pair['question'][:80]}...")
        print(f"  Baseline ({len(pair['baseline_response'].split())} words): "
              f"{pair['baseline_response'][:150]}...")
        for cid, ce in list(pair["category_edits"].items())[:3]:
            v = ce["validation"]
            status = "✓" if v["valid"] else "✗"
            print(f"\n  Cat {cid} [{ce['category_title']}] {status}")
            if v["valid"]:
                print(f"    Contrastive segment: {v['segment'][:200]}...")
            else:
                print(f"    (invalid: {v})")
            print(f"    Full edit (first 200 chars): "
                  f"{ce['edited_response'][:200]}...")


if __name__ == "__main__":
    main()
