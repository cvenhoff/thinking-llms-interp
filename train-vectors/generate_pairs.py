"""Generate contrastive pairs for diff-of-means steering vectors.

For each MMLU-Pro question:
  1. Generate a baseline response from the base model.
  2. For each SAE category, call an API model to edit the baseline by inserting
     a single contrastive segment wrapped in [CATEGORY_START]...[CATEGORY_END].

The paired data (baseline vs. edited) is later used to extract activations
for computing targeted diff-of-means steering vectors.
"""

import argparse
import asyncio
import json
import os
import re
import sys
import random
import gc
from collections import defaultdict

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import utils.utils as utils


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B")
    p.add_argument("--api_model", type=str, default="gpt-4.1")
    p.add_argument("--thinking_model", type=str,
                   default="Open-Reasoner-Zero/Open-Reasoner-Zero-7B")
    p.add_argument("--sae_layer", type=int, default=16)
    p.add_argument("--n_clusters", type=int, default=15)
    p.add_argument("--n_questions", type=int, default=50)
    p.add_argument("--max_new_tokens", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--max_concurrent", type=int, default=40,
                   help="Max concurrent API requests")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--dataset", type=str, default="TIGER-Lab/MMLU-Pro")
    p.add_argument("--question_offset", type=int, default=0,
                   help="Skip first N questions (for train/eval splits)")
    p.add_argument("--output_suffix", type=str, default="")
    p.add_argument("--save_dir", type=str, default="results/synthetic_pairs")
    p.add_argument("--categories", type=str, default="",
                   help="Comma-separated category indices to include (empty=all)")
    return p.parse_args()


# ---------------------------------------------------------------------------
# SAE category loading
# ---------------------------------------------------------------------------

def _load_annotation_pools(thinking_model: str, layer: int,
                           n_clusters: int) -> dict[int, list[str]]:
    """Parse annotated rollouts and return per-category pools of segments
    drawn from the top activation-strength quartile."""
    model_id = thinking_model.split("/")[-1].lower()
    ann_dir = os.path.join(
        os.path.dirname(__file__), "..", "generate-responses", "results", "vars"
    )
    ann_path = os.path.join(
        ann_dir,
        f"annotated_responses_{model_id}_{n_clusters}clusters_layer{layer}.json",
    )
    if not os.path.exists(ann_path):
        print(f"  Annotations file not found: {ann_path}")
        return {}

    with open(ann_path) as f:
        data = json.load(f)

    tag_re = re.compile(r'\["(\d+\.?\d*):idx(\d+)"\]')
    end_re = re.compile(r'\["end-section"\]')
    segments_by_cat: dict[int, list[tuple[float, str]]] = defaultdict(list)

    for entry in data:
        text = entry.get("annotated_thinking", "")
        for m in tag_re.finditer(text):
            act = float(m.group(1))
            cat_id = int(m.group(2))
            seg_start = m.end()
            end_m = end_re.search(text, seg_start)
            seg = text[seg_start:end_m.start()].strip() if end_m else ""
            if len(seg) >= 15:
                segments_by_cat[cat_id].append((act, seg))

    pools: dict[int, list[str]] = {}
    for cat_id, segs in segments_by_cat.items():
        acts = sorted(s[0] for s in segs)
        q75 = acts[3 * len(acts) // 4]
        top = [s for a, s in segs if a >= q75]
        pools[cat_id] = top

    total = sum(len(v) for v in pools.values())
    print(f"  Loaded {total} top-quartile segments across "
          f"{len(pools)} categories from annotations")
    return pools


def load_sae_categories(thinking_model: str, layer: int,
                        n_clusters: int) -> dict[int, dict]:
    """Load category titles, descriptions, and top-activating examples from
    SAE clustering results + annotated rollouts."""
    model_id = thinking_model.split("/")[-1].lower()
    sae_dir = os.path.join(
        os.path.dirname(__file__), "..", "train-saes", "results", "vars"
    )

    ann_pools = _load_annotation_pools(thinking_model, layer, n_clusters)

    candidates = list({model_id, model_id.replace("-", "_"),
                       model_id.replace("_", "-")})
    for mid in candidates:
        for try_layer in [layer, layer - 2, layer + 2, layer - 4, layer + 4]:
            fn = os.path.join(sae_dir,
                              f"sae_topk_results_{mid}_layer{try_layer}.json")
            if not os.path.exists(fn):
                continue
            with open(fn) as f:
                data = json.load(f)
            key = str(n_clusters)
            if key not in data.get("results_by_cluster_size", {}):
                continue
            results = data["results_by_cluster_size"][key]
            cats_raw = results["all_results"][0]["categories"]
            categories = {}
            for cat in cats_raw:
                idx = int(cat[0])
                categories[idx] = {
                    "title": cat[1],
                    "description": cat[2],
                    "top_examples": ann_pools.get(idx, []),
                }
            print(f"Loaded {len(categories)} categories from {fn} "
                  f"(n_clusters={n_clusters})")
            return categories
    print(f"WARNING: No SAE results for {thinking_model} "
          f"layer {layer} n_clusters {n_clusters}")
    return {}


# ---------------------------------------------------------------------------
# Prompts
# ---------------------------------------------------------------------------

BASELINE_TEMPLATE = (
    "Answer the question below. Explain your reasoning step by step.\n\n"
    "Question:\n{question}\n\n"
    "Step by step answer:\n"
)

EDIT_PROMPT_TEMPLATE = """\
You are an expert at editing reasoning traces to create contrastive training data.

## Purpose

We are training steering vectors via diff-of-means on (baseline, edited) response pairs. At the edit site, the baseline performs a different behaviour than the target while the edited version performs the target behaviour instead. The stronger this local behavioural contrast, the cleaner the resulting steering vector.

## Target behaviour

**Title:** {title}
**Description:** {description}
{extra_metadata}

## Instructions

1. Read through the response and identify what each segment is doing.
2. Select a segment whose reasoning behaviour is **maximally different** from the target behaviour, then replace it with new text that naturally demonstrates the target behaviour. The replacement must read as a coherent continuation of the surrounding text — not a meta-comment about the behaviour.

## Format

- Keep all text outside the edit site **identical** to the original.
- Wrap the replacement in `[CATEGORY_START]` and `[CATEGORY_END]` markers.
- Return only the full edited response. No preamble or commentary.
- You may stop 1-2 sentences after `[CATEGORY_END]`.

## Original question

{question}

## Original response

{response}

## Edited response
"""


def build_edit_prompt(question: str, response: str, cat: dict) -> str:
    extra_parts = []
    if cat.get("includes"):
        extra_parts.append(f"**Includes:** {cat['includes']}")
    if cat.get("excludes"):
        extra_parts.append(f"**Excludes:** {cat['excludes']}")
    if cat.get("examples"):
        extra_parts.append(f"**Example sentences:** {cat['examples']}")
    top_examples = cat.get("top_examples", [])
    if top_examples:
        sampled = random.sample(top_examples, min(10, len(top_examples)))
        examples_str = "\n".join(f"  - {ex[:300]}" for ex in sampled)
        extra_parts.append(
            f"**Real examples of this behaviour from reasoning model traces "
            f"(use these as style reference):**\n{examples_str}"
        )
    return EDIT_PROMPT_TEMPLATE.format(
        title=cat["title"],
        description=cat["description"],
        extra_metadata="\n".join(extra_parts),
        question=question,
        response=response,
    )


# ---------------------------------------------------------------------------
# Question loading
# ---------------------------------------------------------------------------

def load_questions(dataset: str, n_questions: int, seed: int,
                   thinking_model: str = "",
                   question_offset: int = 0) -> list[dict]:
    """Load questions from MMLU-Pro, preferring existing responses file."""
    rng = random.Random(seed)

    if dataset == "TIGER-Lab/MMLU-Pro":
        thinking_short = (thinking_model.split("/")[-1].lower()
                          if thinking_model else "")
        responses_path = os.path.join(
            os.path.dirname(__file__), "..",
            "generate-responses", "results", "vars",
            f"responses_{thinking_short}.json",
        )
        if os.path.exists(responses_path):
            print(f"Loading questions from {responses_path}")
            with open(responses_path) as f:
                data = json.load(f)
            rng.shuffle(data)
            selected = data[question_offset:question_offset + n_questions]
            return [{"question": r["original_message"]["content"],
                     "question_id": r["question_id"],
                     "category": r.get("category", "")} for r in selected]

        print("Responses file not found — loading MMLU-Pro from HuggingFace")
        from datasets import load_dataset as hf_load
        ds = hf_load("TIGER-Lab/MMLU-Pro")
        rows = list(ds["test"])
        rng.shuffle(rows)
        selected = rows[question_offset:question_offset + n_questions]
        return [{"question": r["question"],
                 "question_id": r["question_id"],
                 "category": r.get("category", "")} for r in selected]

    raise ValueError(f"Unknown dataset: {dataset}")


# ---------------------------------------------------------------------------
# Batched local generation
# ---------------------------------------------------------------------------

def generate_batch(model, tokenizer, prompts: list[str],
                   max_new_tokens: int, batch_size: int) -> list[str]:
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    all_responses = []
    for i in tqdm(range(0, len(prompts), batch_size), desc="Generating"):
        batch = prompts[i:i + batch_size]
        enc = tokenizer(batch, return_tensors="pt", padding=True,
                        truncation=False).to(model.device)
        input_len = enc["input_ids"].shape[1]
        with torch.inference_mode():
            out = model.generate(
                **enc,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
                do_sample=False,
            )
        for j in range(out.shape[0]):
            all_responses.append(
                tokenizer.decode(out[j][input_len:], skip_special_tokens=True))
        torch.cuda.empty_cache()
    return all_responses


# ---------------------------------------------------------------------------
# API editing (concurrent)
# ---------------------------------------------------------------------------

async def edit_responses_with_api(
    questions: list[str],
    baselines: list[str],
    categories: dict[int, dict],
    api_model: str,
    max_concurrent: int,
) -> dict[int, dict[int, str]]:
    """For every (question, baseline) × category, call API to produce edit."""
    sorted_cats = sorted(categories.items())
    prompts, keys = [], []
    for q_idx in range(len(questions)):
        for cat_idx, cat in sorted_cats:
            prompts.append(build_edit_prompt(
                questions[q_idx], baselines[q_idx], cat))
            keys.append((q_idx, cat_idx))

    print(f"\nCalling {api_model} for {len(prompts)} edits "
          f"({len(questions)} questions × {len(sorted_cats)} categories)...")

    responses = await utils.chat_batch(
        prompts, model=api_model, max_tokens=4096,
        max_concurrent_requests=max_concurrent,
    )

    results: dict[int, dict[int, str]] = {}
    for (q_idx, cat_idx), resp in zip(keys, responses):
        results.setdefault(q_idx, {})[cat_idx] = resp
    return results


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate_edit(text: str) -> dict:
    """Check for exactly one [CATEGORY_START]...[CATEGORY_END] block."""
    starts = [m.start() for m in re.finditer(r"\[CATEGORY_START\]", text)]
    ends = [m.start() for m in re.finditer(r"\[CATEGORY_END\]", text)]
    if len(starts) == 1 and len(ends) == 1 and starts[0] < ends[0]:
        segment = text[starts[0] + len("[CATEGORY_START]"):ends[0]].strip()
        return {"valid": True, "segment": segment}
    return {"valid": False, "n_starts": len(starts), "n_ends": len(ends)}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    categories = load_sae_categories(
        args.thinking_model, args.sae_layer, args.n_clusters)
    if not categories:
        print("ERROR: No categories found.")
        sys.exit(1)

    if args.categories:
        keep = {int(x.strip()) for x in args.categories.split(",")}
        categories = {k: v for k, v in categories.items() if k in keep}
        print(f"Filtered to {len(categories)} categories: {sorted(keep)}")

    print(f"\n=== {len(categories)} Categories ===")
    for idx, cat in sorted(categories.items()):
        n_ex = len(cat.get("top_examples", []))
        print(f"  [{idx}] {cat['title']}  ({n_ex} top examples)")

    # ---- Load questions ----
    questions = load_questions(
        args.dataset, args.n_questions, args.seed,
        thinking_model=args.thinking_model,
        question_offset=args.question_offset,
    )
    print(f"\nLoaded {len(questions)} questions (offset={args.question_offset})")

    # ---- Generate baselines ----
    print(f"\nLoading model {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()

    prompts = [BASELINE_TEMPLATE.format(question=q["question"])
               for q in questions]
    print(f"\n=== Generating {len(prompts)} baselines ===")
    baselines = generate_batch(model, tokenizer, prompts,
                               args.max_new_tokens, args.batch_size)

    del model
    torch.cuda.empty_cache()
    gc.collect()

    for i in range(min(3, len(baselines))):
        print(f"  Q{i}: {questions[i]['question'][:60]}...")
        print(f"       ({len(baselines[i].split())} words): {baselines[i][:120]}...")

    # ---- API edits ----
    print(f"\n=== Editing via {args.api_model} ===")
    q_texts = [q["question"] for q in questions]
    edited = asyncio.run(edit_responses_with_api(
        q_texts, baselines, categories, args.api_model, args.max_concurrent))

    # ---- Assemble and save ----
    os.makedirs(args.save_dir, exist_ok=True)
    model_short = args.model.split("/")[-1].lower()

    output = {
        "model": args.model,
        "api_model": args.api_model,
        "thinking_model": args.thinking_model,
        "sae_layer": args.sae_layer,
        "n_clusters": args.n_clusters,
        "n_questions": len(questions),
        "max_new_tokens": args.max_new_tokens,
        "categories": {
            str(k): {"title": v["title"], "description": v["description"]}
            for k, v in categories.items()
        },
        "pairs": [],
    }

    n_valid = n_total = 0
    for q_idx, q in enumerate(questions):
        cat_edits = {}
        for cat_idx, cat in sorted(categories.items()):
            resp = edited.get(q_idx, {}).get(cat_idx, "")
            v = validate_edit(resp)
            n_total += 1
            if v["valid"]:
                n_valid += 1
            cat_edits[str(cat_idx)] = {
                "category_title": cat["title"],
                "edited_response": resp,
                "validation": v,
            }
        output["pairs"].append({
            "question_id": q["question_id"],
            "question": q["question"],
            "baseline_prompt": prompts[q_idx],
            "baseline_response": baselines[q_idx],
            "category_edits": cat_edits,
        })

    suffix = args.output_suffix or ""
    save_path = os.path.join(
        args.save_dir,
        f"synthetic_pairs_{model_short}_{args.n_clusters}clusters{suffix}.json",
    )
    with open(save_path, "w") as f:
        json.dump(output, f, indent=2)

    print(f"\n{'='*60}")
    print(f"Valid edits: {n_valid}/{n_total} ({100*n_valid/max(n_total,1):.0f}%)")
    print(f"Saved to {save_path}")


if __name__ == "__main__":
    main()
