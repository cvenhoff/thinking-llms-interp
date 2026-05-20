#!/usr/bin/env python3
# %%
"""Train all category steering vectors simultaneously in one forward/backward loop.

This is mathematically equivalent to training each vector in isolation because
each vector's gradient comes only from examples of its own SAE category.
Running them in one pass shares the model's forward+backward cost, giving an
N_cats × speedup over the sequential approach.

Usage (called from the train-vectors/ directory):
    python optimize_cat_vectors_multi.py \
        --model Qwen/Qwen2.5-1.5B \
        --layer 10 \
        --max_iters 50 \
        --n_training_examples 2048 \
        --use_activation_perplexity_selection \
        --save_path results/vars/optimized_vectors
"""

import argparse
import dotenv
dotenv.load_dotenv("../.env")

import os, sys, random, json, pickle
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

import utils
from utils import steering_opt
from utils.responses import extract_thinking_process
from optimize_steering_vectors import (
    CATEGORY_PATTERN,
    get_sorted_categories,
    get_label_positions,
    extract_examples_for_category,
)

# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------
parser = argparse.ArgumentParser(description="Train all cat vectors simultaneously")
parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-1.5B")
parser.add_argument("--thinking_model", type=str, default=None)
parser.add_argument("--layer", type=int, required=True)
parser.add_argument("--max_iters", type=int, required=True)
parser.add_argument("--n_training_examples", type=int, default=2048)
parser.add_argument("--n_eval_examples", type=int, default=0)
parser.add_argument("--optim_minibatch_size", type=int, default=8,
                    help="Per-GPU batch size (mixed across cats). Each cat gets ~minibatch_size/n_cats examples per step.")
parser.add_argument("--lr", type=str, default="1e-2")
parser.add_argument("--min_lr", type=float, default=0.0)
parser.add_argument("--warmup_iters", type=int, default=0)
parser.add_argument("--save_path", type=str, default="results/vars/optimized_vectors")
parser.add_argument("--use_activation_perplexity_selection", action="store_true", default=False)
parser.add_argument("--load_in_8bit", action="store_true", default=False)
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--coldness", type=float, default=0.7)
parser.add_argument("--grad_clip", type=float, default=None)
parser.add_argument("--starting_norm", type=float, default=1.0)
args, _ = parser.parse_known_args()

random.seed(args.seed)
torch.manual_seed(args.seed)

# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
model_name_short = args.model.split("/")[-1].lower()
if args.thinking_model is None:
    thinking_model_name = utils.model_mapping[args.model]
    thinking_model_short = thinking_model_name.split("/")[-1].lower()
else:
    thinking_model_name = args.thinking_model
    thinking_model_short = thinking_model_name.split("/")[-1].lower()
    model_name_short = f"{model_name_short}-on-{thinking_model_short}"

print(f"Base model:     {args.model}  ({model_name_short})")
print(f"Thinking model: {thinking_model_name}")

device = "cuda" if torch.cuda.is_available() else "cpu"

if args.load_in_8bit:
    model = utils.load_model_nnsight(True, args.model, device)
    tokenizer = AutoTokenizer.from_pretrained(args.model)
else:
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model, dtype=torch.bfloat16, device_map=device)

model.eval()
for p in model.parameters():
    p.requires_grad_(False)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# ---------------------------------------------------------------------------
# Load annotated responses
# ---------------------------------------------------------------------------
responses_json_path = f"../generate-responses/results/vars/responses_{thinking_model_short}.json"
annotated_json_path = f"../generate-responses/results/vars/annotated_responses_{thinking_model_short}.json"

print(f"Loading responses from {annotated_json_path}")
with open(responses_json_path) as f:
    raw_responses = json.load(f)
with open(annotated_json_path) as f:
    annotated_responses = json.load(f)

# Merge: join on question_id
ann_by_qid: dict = {}
for r in annotated_responses:
    qid = r.get("question_id")
    if qid is not None:
        ann_by_qid[qid] = r.get("annotated_thinking", "")

valid_responses = []
for r in raw_responses:
    qid = r.get("question_id")
    if qid is not None and qid in ann_by_qid:
        r_copy = dict(r)
        r_copy["annotated_thinking"] = ann_by_qid[qid]
        valid_responses.append(r_copy)

print(f"Loaded {len(valid_responses)} annotated responses")

# ---------------------------------------------------------------------------
# Category discovery
# ---------------------------------------------------------------------------
all_categories = get_sorted_categories(valid_responses)
n_cats = len(all_categories)
print(f"Found {n_cats} categories: {all_categories}")

# ---------------------------------------------------------------------------
# Load bias vector (to apply as static vector during cat training)
# ---------------------------------------------------------------------------
bias_vector: torch.Tensor | None = None
bias_path = os.path.join(args.save_path, f"{model_name_short}_bias_linear.pt")
if os.path.exists(bias_path):
    try:
        bias_dict = torch.load(bias_path, map_location=model.device, weights_only=False)
        if isinstance(bias_dict, dict) and "bias" in bias_dict:
            bias_vector = bias_dict["bias"].to(model.device)
        elif isinstance(bias_dict, torch.Tensor):
            bias_vector = bias_dict.to(model.device)
        if bias_vector is not None:
            print(f"Loaded bias vector: shape={bias_vector.shape}, norm={bias_vector.norm().item():.3f}")
    except Exception as e:
        print(f"Warning: could not load bias vector from {bias_path}: {e}")
else:
    print(f"No bias vector found at {bias_path} — training cats without bias static vector")

# ---------------------------------------------------------------------------
# Extract examples per category
# ---------------------------------------------------------------------------
# Monkey-patch args so extract_examples_for_category can read it
import builtins
_orig_args = getattr(builtins, "_multi_args", None)

# The function uses a module-level `args` from optimize_steering_vectors; we
# work around this by manually reimplementing the selection logic inline.

def select_examples_for_category(cat_name, n_train, n_eval, use_ppl_selection):
    """Extract and select training examples for a single category."""
    examples_for_category = []

    for resp in tqdm(valid_responses, desc=f"  Extracting {cat_name}", leave=False):
        if not resp.get("annotated_thinking"):
            continue
        if cat_name not in resp["annotated_thinking"]:
            continue
        thinking_process = extract_thinking_process(resp["full_response"])
        full_text = (
            f"Task: Answer the question below. Explain your reasoning step by step.\n\n\n\n"
            f"Question:\n{resp['original_message']['content']}\n\nStep by step answer:\n{thinking_process}"
        )
        label_positions = get_label_positions(resp["annotated_thinking"], full_text, tokenizer, 0)
        if cat_name not in label_positions:
            continue
        for start, end, text, activation, text_pos in label_positions[cat_name]:
            context = full_text[:text_pos]
            if not context:
                continue
            if context[-1] not in ".?!;\n" and len(context) > 1 and context[-2] not in ".?!;\n":
                stripped = context.strip()
                if stripped and stripped[-1] not in ".?!;\n":
                    continue
            if len(text.strip().split()) < 7:
                continue
            examples_for_category.append({
                "prompt": context,
                "target_completion": text,
                "activation": activation,
            })

    if not examples_for_category:
        print(f"  WARNING: no examples found for {cat_name}")
        return [], []

    total_needed = n_train + n_eval

    if not use_ppl_selection:
        pool = examples_for_category[:]
        random.shuffle(pool)
        train_ex = pool[:min(n_train, len(pool))]
        eval_ex = pool[len(train_ex):len(train_ex) + n_eval]
        return train_ex, eval_ex

    # Activation pre-filter → perplexity selection
    examples_for_category_sorted = sorted(examples_for_category, key=lambda x: x["activation"], reverse=True)
    sample_size = min(len(examples_for_category_sorted), max(1, total_needed) * 4)
    sampled = examples_for_category_sorted[:sample_size]

    cache_dir = "results/vars/perplexity"
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"perplexities_{model_name_short}_{cat_name}.pkl")

    examples_with_metrics: list = []
    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            examples_with_metrics = pickle.load(f)
        if not (isinstance(examples_with_metrics, list) and
                all("perplexity" in e for e in examples_with_metrics) and
                len(examples_with_metrics) >= sample_size):
            print(f"  Cache invalid for {cat_name}, recomputing...")
            examples_with_metrics = []
        else:
            print(f"  Loaded {len(examples_with_metrics)} cached perplexities for {cat_name}")

    if not examples_with_metrics:
        print(f"  Computing perplexity for {len(sampled)} examples of {cat_name}...")
        for ex in tqdm(sampled, desc=f"  ppl {cat_name}", leave=False):
            try:
                full_text = ex["prompt"] + ex["target_completion"]
                inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
                with torch.no_grad():
                    out = model(**inputs)
                prompt_ids = tokenizer(ex["prompt"], return_tensors="pt")["input_ids"][0]
                pl = len(prompt_ids)
                sl = out.logits[0, :-1]
                ll = inputs["input_ids"][0][1:]
                ts, te = pl - 1, len(ll)
                if ts < 0 or ts >= te:
                    continue
                loss = torch.nn.functional.cross_entropy(sl[ts:te], ll[ts:te].to(sl.device))
                ppl = torch.exp(loss).item()
                examples_with_metrics.append({**ex, "perplexity": ppl})
            except Exception as e:
                continue
        with open(cache_path, "wb") as f:
            pickle.dump(examples_with_metrics, f)
        print(f"  Saved {len(examples_with_metrics)} perplexities for {cat_name}")

    if not examples_with_metrics:
        pool = examples_for_category[:]
        random.shuffle(pool)
        return pool[:min(n_train, len(pool))], []

    final = sorted(examples_with_metrics, key=lambda x: x["perplexity"], reverse=True)[:total_needed]
    train_ex = final[:n_train]
    eval_ex = final[n_train:n_train + n_eval]
    for ex in train_ex + eval_ex:
        ex.pop("perplexity", None)
        ex.pop("activation", None)
    return train_ex, eval_ex


# ---------------------------------------------------------------------------
# Collect training examples for all categories
# ---------------------------------------------------------------------------
per_cat_examples = []  # list of (prompts, targets) per category
cat_sizes = []

for k, cat in enumerate(all_categories):
    print(f"\n[{k+1}/{n_cats}] Collecting examples for {cat}...")
    train_ex, _ = select_examples_for_category(
        cat,
        args.n_training_examples,
        args.n_eval_examples,
        args.use_activation_perplexity_selection,
    )
    prompts = [e["prompt"] for e in train_ex]
    targets = [e["target_completion"] for e in train_ex]
    per_cat_examples.append((prompts, targets))
    cat_sizes.append(len(train_ex))
    print(f"  {cat}: {len(train_ex)} training examples")

if not any(cat_sizes):
    raise RuntimeError("No training examples found for any category. Aborting.")

# ---------------------------------------------------------------------------
# Multi-vector training
# ---------------------------------------------------------------------------
# Parse learning rates (may be comma-separated for sweep; take first)
lrs = [float(x.strip()) for x in args.lr.split(",")]
lr = lrs[0]
print(f"\nTraining {n_cats} cat vectors simultaneously on layer {args.layer}, lr={lr}")
print(f"Total examples across all cats: {sum(cat_sizes)}")
print(f"Effective per-cat batch size: {max(1, args.optim_minibatch_size // n_cats)}")

static_vecs = [bias_vector] if bias_vector is not None else []

best_vectors, info = steering_opt.optimize_multi_vector_simple(
    model,
    tokenizer,
    per_cat_examples,
    args.layer,
    lr=lr,
    max_iters=args.max_iters,
    optim_minibatch_size=args.optim_minibatch_size,
    warmup_steps=args.warmup_iters,
    min_lr=args.min_lr,
    coldness=args.coldness,
    grad_clip=args.grad_clip,
    starting_norm=args.starting_norm,
    static_vectors=static_vecs,
    steering_token_window=None,  # all positions
    cat_names=all_categories,
)

# ---------------------------------------------------------------------------
# Save each cat vector
# ---------------------------------------------------------------------------
os.makedirs(args.save_path, exist_ok=True)
for k, cat in enumerate(all_categories):
    vec = best_vectors[k]
    vectors_path = os.path.join(args.save_path, f"{model_name_short}_{cat}_linear.pt")
    torch.save({cat: vec}, vectors_path)
    print(f"Saved {cat}: {vectors_path}  (norm={vec.norm().item():.3f})")

print(f"\nAll {n_cats} cat vectors saved to {args.save_path}")
print(f"Final loss: {info['final_loss']:.4f}")
