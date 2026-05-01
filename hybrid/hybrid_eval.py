"""Hybrid model evaluation: thinking model + base model + SAE-guided steering.

Three-phase pipeline:
  1. Standalone generation for thinking and base models (cached).
  2. KV-cached batched hybrid generation with SAE-category steering and
     token-disagreement-gated coefficient sweep (paper recipe).
  3. Concurrent LLM judging with majority vote.
"""

import dotenv
dotenv.load_dotenv("../.env")

import sys
import os
import torch
import json
import re
import gc
import time
import random
import argparse
import math
from typing import Optional
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.sae import load_sae
from utils.utils import center_and_l2_normalize_torch, chat_batch
from utils.clustering import get_latent_descriptions


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

CODING_DATASETS = {"mbpp", "livecodebench"}
MCQA_DATASETS = {"medqa", "gpqa"}
TEXT_CLASSIFICATION_DATASETS = {"legalbench"}
GPQA_LETTERS = ["A", "B", "C", "D"]

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=str, default="math500",
                   choices=["gsm8k", "math500", "aime24", "aime25", "mbpp",
                            "livecodebench", "medqa", "gpqa", "legalbench"])
    p.add_argument("--thinking_model", type=str, default="Qwen/QwQ-32B")
    p.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-32B")
    p.add_argument("--sae_layer", type=int, default=27)
    p.add_argument("--n_clusters", type=int, default=10)
    p.add_argument("--n_tasks", type=int, default=0, help="0 = all")
    p.add_argument("--batch_gen_size", type=int, default=32)
    p.add_argument("--hybrid_gen_batch_size", type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=5000)
    p.add_argument("--max_thinking_tokens", type=int, default=5000)
    p.add_argument("--eval_start_idx", type=int, default=0)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--dom_vectors_dir", type=str, required=True)
    p.add_argument("--dom_vectors_model_short", type=str, default=None)
    p.add_argument("--old_vectors_dir", type=str, default=None,
                   help="Load old-format per-index optimized vectors from this dir "
                        "(files: <model_short>_idx<N>_linear.pt, all at --old_vectors_layer)")
    p.add_argument("--old_vectors_layer", type=int, default=10,
                   help="Layer at which old-format vectors are applied")
    p.add_argument("--randomize_vectors", action="store_true",
                   help="After loading steering vectors, replace each category "
                        "vector with a Gaussian-random direction of the same "
                        "norm (control/ablation). Bias vector (if any) is kept.")
    p.add_argument("--randomize_vectors_unit_norm", action="store_true",
                   help="Ablation (reproduces bug in collaborator's hybrid_token.py "
                        "'random_vectors' path): replace each per-category vector "
                        "with a UNIT-NORM Gaussian-random direction (norm=1) "
                        "instead of preserving the learned-vector norm. Bias (if "
                        "any) is kept untouched.  Since learned-vector norms are "
                        "~40-60 and bias has comparable norm, this effectively "
                        "collapses the random-vectors ablation toward bias-only.")
    p.add_argument("--random_seed", type=int, default=0,
                   help="Seed for --randomize_vectors / --random_firing / "
                        "--random_guardrail.")
    p.add_argument("--bias_vector_path", type=str, default=None,
                   help="Path to a .pt bias vector (a dict {'bias': tensor} or raw "
                        "tensor) to add on top of the per-category steering vector, "
                        "matching the OLD/paper pipeline's global bias term.")
    p.add_argument("--bias_only", action="store_true",
                   help="Ablation: zero out every per-category steering vector "
                        "AFTER loading, so only the --bias_vector_path remains. "
                        "Requires --bias_vector_path.")
    p.add_argument("--bias_layer", type=int, default=None,
                   help="When --bias_only: register the bias at this layer for "
                        "every category key (overrides per-category layer_map). "
                        "If omitted, tries to read <bias_vector_path>'s sibling "
                        "bias_layer.json, else falls back to --old_vectors_layer.")
    p.add_argument("--random_firing", action="store_true",
                   help="Ablation: at each decode step, ignore the SAE argmax "
                        "and pick a uniform-random category key per batch row "
                        "(simulates random latent firing).")
    p.add_argument("--random_guardrail", action="store_true",
                   help="Ablation: replace the thinking-model perplexity "
                        "guardrail with a uniform random choice among the "
                        "coefficient sweep candidates.")
    p.add_argument("--coef_sweep", type=str, default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
                   help="Comma-separated coefficient sweep used by the "
                        "guardrail. Default is the paper's 10-point grid "
                        "[0.1..1.0]. Collaborator's ablation report used "
                        "[0.5,0.6,0.7,0.8,0.9,1.0] (floor 0.5) which produces "
                        "a much stronger bias perturbation.")
    p.add_argument("--coef_select", type=str, default="pg",
                   choices=["pg", "kl_top3", "kl_topk",
                            "think_top1", "think_top1_match"],
                   help="Coefficient-selection rule used together with "
                        "--coef_sweep on disagreement positions. "
                        "'pg' (default, legacy): pick coef whose steered-"
                        "base argmax token has the highest log-prob under "
                        "the thinking model's distribution -- the paper's "
                        "perplexity guardrail. "
                        "'kl_top3' / 'kl_topk': pick coef that minimises "
                        "the cross-entropy of the steered-base log-prob "
                        "over the thinking model's top-K tokens, "
                        "-sum_k p_t(k) * log p_b(k).  This matches the "
                        "objective the correction vectors are trained on. "
                        "K is set by --kl_topk (default 3). "
                        "'think_top1': oracle/ceiling -- pick coef that "
                        "MAXIMISES the steered-base log-prob at the "
                        "thinking model's argmax token T at this position. "
                        "Tells us how high gap recovery can go if we "
                        "always pick the coef best aligned with thinking's "
                        "top-1.  Output token is still the steered-base "
                        "argmax at the winning coef.  Requires base and "
                        "thinking tokenizers to share a 1:1 vocabulary; "
                        "verified at startup. "
                        "'think_top1_match': stricter ceiling -- among "
                        "coefs whose steered-base argmax already EQUALS "
                        "thinking's top-1 token T, pick the smallest one "
                        "(most conservative); if no coef in the sweep "
                        "produces a match, leave the row UNSTEERED "
                        "(output unsteered base argmax, coef=0).  Real "
                        "upper bound on what the learned vectors can "
                        "achieve at this position with the given sweep.")
    p.add_argument("--kl_topk", type=int, default=3,
                   help="K for --coef_select=kl_topk (and kl_top3 alias).")
    p.add_argument("--steer_all_positions_full", action="store_true",
                   help="Faithful reproduction of hybrid_token.py semantics: "
                        "during the coefficient sweep, drop the base-model KV "
                        "cache and run a FULL forward on the entire sequence "
                        "(input + generated so far) with the hook applying "
                        "c*(bias+vec) to ALL positions of layer `steering_layer`. "
                        "Each coef gets its own full forward. Expensive "
                        "(O(N^2) per task) but matches `--token_windows [1]` "
                        "exactly. After the sweep, the winning token is "
                        "committed and the base model's KV cache advances "
                        "normally (unsteered). Mutually exclusive with "
                        "--steer_all_positions.")
    p.add_argument("--steer_all_positions", action="store_true",
                   help="Reproduce collaborator's hybrid_token.py default "
                        "'--token_windows [1]' (internally window_size=0 -> "
                        "'all tokens') semantics. Instead of reverting the "
                        "steering shift from the KV cache after each "
                        "disagreement step, the winning-coefficient shift is "
                        "persisted: we re-run the current position's forward "
                        "with the best per-row coef so layers > sae_layer "
                        "have steered K/V for that position. Across many "
                        "disagreement steps this accumulates so attention at "
                        "subsequent steps sees shifted K/V for all past "
                        "steered positions, approximating their 'apply "
                        "c*(bias+vec) to every position's layer-24 output on "
                        "every forward pass' behaviour.")
    p.add_argument("--judge_model", type=str, default="openai/gpt-5.2")
    p.add_argument("--judge_repetitions", type=int, default=1)
    p.add_argument("--max_concurrent", type=int, default=40)
    p.add_argument("--results_dir", type=str, default="results")
    p.add_argument("--results_suffix", type=str, default="")
    p.add_argument("--no_response_cache", action="store_true")
    p.add_argument("--disable_sae_mean", action="store_true")
    return p.parse_args()


def _result_suffix(args):
    s = ""
    if args.results_suffix:
        s += ("_" + args.results_suffix) if not args.results_suffix.startswith("_") else args.results_suffix
    return s


def _dataset_type(args):
    if args.dataset in CODING_DATASETS:
        return "coding"
    if args.dataset in MCQA_DATASETS:
        return "mcqa"
    if args.dataset in TEXT_CLASSIFICATION_DATASETS:
        return "classification"
    return "math"


# ---------------------------------------------------------------------------
# Dataset / prompt helpers
# ---------------------------------------------------------------------------

CODING_TASK_PREFIX = "Task: Write a single Python function for the following problem. Do not include tests or examples in your output."
CODING_BASE_SUFFIX = "Algorithmic steps to solve this problem, followed by the Python function:\n"


def _format_gpqa_item(item, index):
    correct = item["Correct Answer"]
    distractors = [item["Incorrect Answer 1"], item["Incorrect Answer 2"],
                   item["Incorrect Answer 3"]]
    all_ans = [correct] + distractors
    random.Random(42 + index).shuffle(all_ans)
    letter = GPQA_LETTERS[all_ans.index(correct)]
    opts = "\n".join(f"{l}. {a}" for l, a in zip(GPQA_LETTERS, all_ans))
    return f"{item['Question']}\n\nOptions:\n{opts}", letter


def _build_task_prompts(item, i, args):
    test_list, starter_code = None, ""
    if args.dataset == "gsm8k":
        q, a = item["question"], item["answer"]
    elif args.dataset in ("aime24", "aime25"):
        q, a = item["problem"], item["answer"]
    elif args.dataset == "math500":
        q, a = item["problem"], item["answer"]
    elif args.dataset == "mbpp":
        q, a = item["text"], item["code"]
        test_list = item["test_list"]
    elif args.dataset == "livecodebench":
        q, a = item["question_content"], ""
        raw = item.get("public_test_cases", "[]")
        pts = json.loads(raw) if raw else []
        test_list = [f"# Test {ti+1}:\n- Input:\n{t['input']}\n- Output:\n{t['output']}"
                     for ti, t in enumerate(pts)] if pts else []
        starter_code = item.get("starter_code", "")
    elif args.dataset == "medqa":
        opts = "\n".join(f"{k}. {v}" for k, v in item["options"].items())
        q = f"{item['question']}\n\nOptions:\n{opts}"
        a = item["answer_idx"]
    elif args.dataset == "gpqa":
        q, a = _format_gpqa_item(item, i)
    elif args.dataset == "legalbench":
        q, a = item.get("text", str(item)), str(item.get("answer", ""))
    else:
        q, a = str(item), ""

    if args.dataset == "mbpp":
        tc = "\n".join(test_list) if test_list else ""
        ts = f"\n\nPublic Tests:\n{tc}" if tc else ""
        tp = f"{CODING_TASK_PREFIX}\n\nProblem: {q}{ts}"
        bp = f"{tp}\n\n{CODING_BASE_SUFFIX}"
    elif args.dataset == "livecodebench":
        tc = "\n\n".join(test_list) if test_list else ""
        ts = f"\n\nPublic Tests:\n\n{tc}" if tc else ""
        sh = f"\n\nStarter code:\n```python\n{starter_code}\n```" if starter_code else ""
        tp = f"{CODING_TASK_PREFIX}\n\nProblem: {q}{sh}{ts}"
        bp = f"{tp}\n\n{CODING_BASE_SUFFIX}"
    elif args.dataset in ("medqa", "gpqa"):
        tp = f"{q}\n\nPlease select the correct answer (A, B, C, or D) and explain your reasoning."
        bp = (f"Task: Answer the following multiple choice question by selecting the "
              f"correct option (A, B, C, or D). Explain your reasoning step by step."
              f"\n\n{q}\n\nStep by step answer:\n")
    else:
        tp = q
        bp = (f"Answer the question below. Explain your reasoning step by step."
              f"\n\nQuestion:\n{q}\n\nStep by step answer:\n")

    return {"question": q, "correct_answer": a,
            "thinking_prompt": tp, "base_prompt": bp, "test_list": test_list}


# ---------------------------------------------------------------------------
# Standalone batch generation
# ---------------------------------------------------------------------------

def _batch_generate(model, tokenizer, prompts, max_new_tokens, batch_size,
                    *, use_chat_template=False, temperature=0.0,
                    on_batch_done=None, tag=""):
    """Generate completions in batches.

    If ``on_batch_done`` is provided it is invoked after each batch as
    ``on_batch_done(batch_start_idx, batch_results)`` (results use global
    prompt indices).  Use this for incremental cache flushing so an
    interrupted run does not lose earlier batches.
    """
    results = [None] * len(prompts)
    eos_id = int(tokenizer.eos_token_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = eos_id

    n_batches = (len(prompts) + batch_size - 1) // batch_size
    for bi, bs in enumerate(range(0, len(prompts), batch_size)):
        batch = prompts[bs:bs + batch_size]
        print(f"  [{tag}] batch {bi + 1}/{n_batches}  "
              f"({bs + 1}..{bs + len(batch)}/{len(prompts)})  generating...",
              flush=True)
        t0 = time.time()
        if use_chat_template:
            texts = [tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                add_generation_prompt=True, tokenize=False) for p in batch]
            enc = tokenizer(texts, return_tensors="pt", padding=True,
                            truncation=False).to(model.device)
        else:
            enc = tokenizer(batch, return_tensors="pt", padding=True,
                            truncation=False).to(model.device)

        gkw = dict(**enc, max_new_tokens=max_new_tokens,
                   pad_token_id=eos_id, eos_token_id=eos_id,
                   do_sample=(temperature > 0))
        if temperature > 0:
            gkw["temperature"] = temperature
        with torch.inference_mode():
            out = model.generate(**gkw)

        pl = enc["input_ids"].shape[1]
        batch_results = []
        for j in range(len(batch)):
            gen = out[j, pl:]
            nt = int(gen.shape[0])
            eos = bool(gen[-1].item() == eos_id) if nt > 0 else True
            rec = {
                "response": tokenizer.decode(gen, skip_special_tokens=True),
                "n_tokens": nt, "eos": eos}
            results[bs + j] = rec
            batch_results.append(rec)
        dt = time.time() - t0
        total_new = sum(r["n_tokens"] for r in batch_results)
        print(f"  [{tag}] batch {bi + 1}/{n_batches} done in {dt:.1f}s "
              f"({total_new} new tokens, "
              f"{total_new / max(dt, 1e-6):.1f} tok/s)", flush=True)
        if on_batch_done is not None:
            try:
                on_batch_done(bs, batch_results)
            except Exception as e:
                print(f"  [{tag}] WARN on_batch_done failed: {e}",
                      flush=True)
        del out, enc
        torch.cuda.empty_cache()
    return results


# ---------------------------------------------------------------------------
# DOM vector loading
# ---------------------------------------------------------------------------

def load_dom_vectors(dom_dir, model_short, descriptions):
    """Load steering vectors + per-category best layer/coeff. No bias vector."""
    vp = os.path.join(dom_dir, f"dom_vectors_multilayer_{model_short}.pt")
    mp = os.path.join(dom_dir, f"dom_metadata_multilayer_{model_short}.json")
    blp = os.path.join(dom_dir, f"dom_best_layers_{model_short}.json")
    bcp = os.path.join(dom_dir, f"dom_best_coeffs_{model_short}.json")

    vecs = torch.load(vp, map_location="cpu")
    with open(mp) as f:
        meta = json.load(f)
    with open(blp) as f:
        best_layers = json.load(f)
    best_coeffs = {}
    if os.path.exists(bcp):
        with open(bcp) as f:
            raw = json.load(f)
        best_coeffs = {k: float(v["best_coeff"]) for k, v in raw.items()}
        print(f"  Per-category best coefficients: {best_coeffs}")

    steering_vectors, layer_map = {}, {}
    for cat_id, lv in vecs.items():
        m = meta[cat_id]
        bl = int(best_layers[cat_id]["best_layer"])
        raw_vec = lv[str(bl)]
        cc = best_coeffs.get(cat_id, 1.0)
        final = raw_vec * cc
        key = f"idx{cat_id}"
        steering_vectors[key] = final.to(torch.float32)
        layer_map[key] = bl
        print(f"  Cat {cat_id} ({m['title']}): key={key}, layer={bl}, "
              f"coeff={cc}, norm={final.norm().item():.2f}")
    return steering_vectors, layer_map


# ---------------------------------------------------------------------------
# KV-cached batched hybrid generation
# ---------------------------------------------------------------------------

def _truncate_kv(kv, n=1):
    """Remove the last *n* positions from a DynamicCache in-place."""
    if n <= 0:
        return
    kv.crop(-n)


def hybrid_generate_batched(
    thinking_model, base_model, base_tokenizer,
    thinking_prompts, base_prompts, max_new_tokens,
    sae_layer, sae, steering_vectors, latent_descriptions,
    steering_layer_map, *,
    thinking_tokenizer=None,
    disable_sae_mean=False,
    show_progress=False, collect_details=True,
    random_firing=False, random_guardrail=False, random_seed=0,
    coef_sweep=None, steer_all_positions=False,
    steer_all_positions_full=False,
    coef_select="pg", kl_topk=3,
):
    """Batched KV-cached hybrid generation (paper recipe).

    For every decode step: both models produce a candidate next token.
    On token disagreement, sweep 10 coefficients 0.1..1.0 on the
    base-model steering vector (last-position, layer `old_vectors_layer`)
    and always pick the steered candidate with the highest thinking-model
    log-prob.  Steering is reverted from the cached K/V after the step
    so it only acts as a per-step logit nudge (matches the old non-KV
    pipeline's semantics).
    """
    B = len(thinking_prompts)
    assert len(base_prompts) == B and B > 0

    device = next(base_model.parameters()).device
    dtype = next(base_model.parameters()).dtype
    hidden_size = base_model.config.hidden_size
    act_mean = (sae.activation_mean if hasattr(sae, "activation_mean")
                and not disable_sae_mean else None)
    eos_id = int(base_tokenizer.eos_token_id)

    default_layer = min(steering_layer_map.values()) if steering_layer_map else 0
    all_steer_layers = sorted(set(steering_layer_map.values()))

    # Pre-compute per-key vector norms + "is this a real (non-zero) vector"
    # flag once, used in per-token debug records to track `shift_norm` and
    # `n_no_vector` (e.g. when bias_only zeroed out category vectors, or when
    # the trainer dropped a category for having too few disagreements).
    vec_norms: Dict[str, float] = {}
    has_vec_map: Dict[str, bool] = {}
    for k, v in steering_vectors.items():
        n = float(v.float().norm().item())
        vec_norms[k] = n
        has_vec_map[k] = n > 1e-8

    # Ablation RNGs. Seeded Python RNGs so runs are reproducible.
    _firing_rng = random.Random(random_seed + 1)
    _guard_rng = random.Random(random_seed + 2)
    _firing_keys = [k for k in steering_vectors.keys() if k in steering_layer_map]

    generated_ids = [[] for _ in range(B)]
    token_infos = [[] for _ in range(B)] if collect_details else None
    steer_sels = [[] for _ in range(B)]
    coeff_sels = [[] for _ in range(B)]
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    # ---- hooks: thinking model SAE layer ----
    captured = {}

    def _sae_hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captured["sae_act"] = h[:, -1, :].detach().float()

    handle_sae = thinking_model.model.layers[sae_layer].register_forward_hook(
        _sae_hook)

    # ---- hooks: steering on base model ----
    # steer_s["all_positions"]: when True and the hook sees a multi-token
    # forward (prefill / full-seq rerun during --steer_all_positions_full),
    # the shift c*v is applied to EVERY position's layer-`steering_layer`
    # output, matching hybrid_token.py's `token_windows=0` (all-positions)
    # semantics. When False (default), only the last position is shifted.
    steer_s = {"vecs": None, "layer_masks": {}, "coef": 1.0,
               "all_positions": False}
    steer_handles = []
    for li in all_steer_layers:
        def _mk(layer_i):
            def hook(mod, inp, out):
                mask = steer_s["layer_masks"].get(layer_i)
                if mask is None or not mask.any():
                    return out
                v = steer_s["vecs"]
                if v is None:
                    return out
                h = out[0] if isinstance(out, tuple) else out
                h = h.clone()
                coef = steer_s["coef"]
                if isinstance(coef, torch.Tensor):
                    # Per-row coefficient (used when committing the
                    # winning coef back into the KV cache under
                    # --steer_all_positions).
                    delta = (coef[mask].view(-1, 1, 1)
                             * v[mask].unsqueeze(1))
                else:
                    delta = (coef * v[mask]).unsqueeze(1)
                if h.shape[1] > 1 and steer_s["all_positions"]:
                    # Full-seq forward with all-positions steering
                    # (matches hybrid_token.py's token_windows=0):
                    # add the per-row shift to EVERY position.
                    h[mask, :, :] = h[mask, :, :] + delta
                elif h.shape[1] > 1:
                    # Match OLD `--token_windows -1`: only steer the
                    # last position of a full-seq forward.
                    h[mask, -1:, :] += delta
                else:
                    h[mask] += delta
                return (h,) + out[1:] if isinstance(out, tuple) else h
            return hook
        steer_handles.append(
            base_model.model.layers[li].register_forward_hook(_mk(li)))

    pbar = (tqdm(total=max_new_tokens, desc=f"Hybrid B={B}", leave=False)
            if (show_progress and tqdm) else None)

    # ---- SAE helpers ----
    def _classify(sae_act_batch):
        """Classify + set up steering vectors/masks in steer_s."""
        if act_mean is not None:
            x = sae_act_batch - act_mean.to(device)
            x = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
        else:
            x = sae_act_batch
        la = sae.encoder(x - sae.b_dec)
        ids = la.argmax(dim=-1)
        vals = la[torch.arange(B, device=device), ids]

        vecs = torch.zeros(B, hidden_size, device=device, dtype=dtype)
        assigns = [default_layer] * B
        keys, titles = [], []
        for b in range(B):
            lid = ids[b].item()
            k = latent_descriptions[lid]["key"]
            if random_firing and _firing_keys:
                # Ablation: override SAE-picked key with a uniform random
                # category key (same pool the SAE oracle selects from).
                k = _firing_rng.choice(_firing_keys)
            keys.append(k)
            titles.append(latent_descriptions[lid]["title"])
            sv = steering_vectors.get(k)
            if sv is not None:
                vecs[b] = sv
            if k in steering_layer_map:
                assigns[b] = steering_layer_map[k]
        steer_s["vecs"] = vecs
        steer_s["assigns"] = assigns
        for l in all_steer_layers:
            steer_s["layer_masks"][l] = torch.tensor(
                [a == l for a in assigns], dtype=torch.bool, device=device)
        return ids, vals, keys, titles

    def _clear_steering():
        for l in all_steer_layers:
            steer_s["layer_masks"][l] = torch.zeros(
                B, dtype=torch.bool, device=device)

    _clear_steering()  # start with no steering active

    try:
        # ---- Tokenize ----
        think_tok = thinking_tokenizer if thinking_tokenizer is not None else base_tokenizer
        think_tok.padding_side = "left"
        think_texts = [think_tok.apply_chat_template(
            [{"role": "user", "content": p}],
            add_generation_prompt=True, tokenize=False) for p in thinking_prompts]
        t_enc = think_tok(think_texts, return_tensors="pt",
                          padding=True, truncation=False).to(device)
        t_ids = t_enc["input_ids"]
        t_mask = t_enc["attention_mask"]
        t_pos = t_mask.long().cumsum(-1) - 1
        t_pos.masked_fill_(t_mask == 0, 0)
        t_lens = t_mask.sum(dim=1)

        base_tokenizer.padding_side = "left"
        b_enc = base_tokenizer(base_prompts, return_tensors="pt",
                               padding=True, truncation=False).to(device)
        b_ids = b_enc["input_ids"]
        b_mask = b_enc["attention_mask"]
        b_pos = b_mask.long().cumsum(-1) - 1
        b_pos.masked_fill_(b_mask == 0, 0)
        b_lens = b_mask.sum(dim=1)

        # ---- Prefill: thinking model ----
        # logits_to_keep=1 restricts the final lm_head projection to the
        # last position only, which matters when the fp32 lm_head wrapper is
        # active (otherwise we'd allocate a (B, L, vocab) fp32 tensor during
        # prefill -> OOM at large batch sizes).
        with torch.inference_mode():
            think_out = thinking_model(input_ids=t_ids, attention_mask=t_mask,
                                       position_ids=t_pos, use_cache=True,
                                       logits_to_keep=1)
        think_kv = think_out.past_key_values
        think_logits = think_out.logits[:, -1, :]
        del think_out

        lat_ids, act_vals, lat_keys, lat_titles = _classify(captured["sae_act"])
        _clear_steering()

        # ---- Prefill: base model (unsteered) ----
        with torch.inference_mode():
            base_out = base_model(input_ids=b_ids, attention_mask=b_mask,
                                  position_ids=b_pos, use_cache=True,
                                  logits_to_keep=1)
        base_kv = base_out.past_key_values
        base_logits = base_out.logits[:, -1, :]
        del base_out
        torch.cuda.empty_cache()

        # Token fed to base model to produce current base_logits (for re-run)
        prev_base_input = b_ids[:, -1].clone()

        # For --steer_all_positions_full: maintain the growing full input
        # sequence (prompt + tokens generated so far) and its attention
        # mask, so the coef sweep can do fresh full forwards that match
        # hybrid_token.py's `base_model.trace(base_output_ids)` exactly.
        base_ids_full = b_ids.clone()
        base_mask_full = b_mask.clone()

        think_pos = t_lens.clone()
        base_pos = b_lens.clone()
        n_gen = 0

        # Coefficient sweep. Default is paper's 10-point grid [0.1..1.0];
        # overridable via --coef_sweep (e.g. [0.5..1.0] to match the
        # collaborator's ablation-study report).
        _SWEEP = list(coef_sweep) if coef_sweep else [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        assert len(_SWEEP) > 0 and all(0.0 <= float(c) <= 10.0 for c in _SWEEP)

        while n_gen < max_new_tokens:
            # ---- 1. Candidate tokens from each model ----
            base_next_toks = torch.argmax(base_logits, dim=-1)
            think_next_toks = torch.argmax(think_logits, dim=-1)
            token_agree = (think_next_toks == base_next_toks) | finished

            best_coeff = torch.zeros(B, device=device)
            did_steer = torch.zeros(B, dtype=torch.bool, device=device)
            output_toks = base_next_toks

            if not token_agree.all():
                # ---- 2. Coefficient sweep on disagreeing rows ----
                disagree_mask = ~token_agree
                think_lp = torch.log_softmax(think_logits, dim=-1)
                arange_B = torch.arange(B, device=device)

                # KL-top-K mode: pre-compute thinking model's top-K
                # logprobs/ids and probs ONCE per step.  Used to score
                # each candidate base distribution by the same form as
                # the training objective:
                #   score_sc = -sum_k p_t(k) * log p_b_steered(k)
                # (lower = better fit to thinking's top-K target).
                if coef_select in ("kl_top3", "kl_topk"):
                    K = int(kl_topk) if coef_select == "kl_topk" else 3
                    t_topk_lp, t_topk_ix = think_lp.topk(K, dim=-1)
                    t_topk_p = t_topk_lp.exp()  # (B, K)

                # Old pipeline: on disagreement always pick best steered.
                # For PG: best_lp tracks max thinking logp of base argmax.
                # For KL-topK: best_score tracks min(-CE) i.e. max(-CE).
                if coef_select == "pg":
                    best_lp = torch.full_like(
                        think_lp[arange_B, base_next_toks], float("-inf"))
                else:
                    # store the LARGER-is-better quantity: -CE (so we can
                    # reuse the same `better = c_lp > best_lp` logic).
                    best_lp = torch.full(
                        (B,), float("-inf"),
                        device=device, dtype=think_lp.dtype)
                best_tok = base_next_toks.clone()
                # think_top1_match: track whether any coef in the sweep
                # produced an argmax matching thinking's top-1 for each
                # disagreement row.  Rows that never match stay UNSTEERED.
                if coef_select == "think_top1_match":
                    matched_row = torch.zeros(B, dtype=torch.bool, device=device)
                raw_vecs = steer_s["vecs"]

                # Random-guardrail ablation: pre-sample a random coefficient
                # per row; we'll commit whatever the sweep produces at that
                # coefficient instead of picking by thinking-model logp.
                if random_guardrail:
                    chosen_coef = torch.tensor(
                        [_guard_rng.choice(_SWEEP) for _ in range(B)],
                        device=device, dtype=torch.float32)

                # Position_ids for full-seq forward (only used under
                # --steer_all_positions_full).  Built from the growing
                # attention mask so left-padded prompts get correct RoPE
                # positions.
                if steer_all_positions_full:
                    full_pos = base_mask_full.long().cumsum(-1) - 1
                    full_pos.masked_fill_(base_mask_full == 0, 0)

                # For think_top1_match we want the SMALLEST coef that
                # produces an argmax==thinking top-1, so iterate sorted.
                _sweep_iter = (sorted(_SWEEP)
                               if coef_select == "think_top1_match"
                               else _SWEEP)
                for sc in _sweep_iter:
                    steer_s["coef"] = sc
                    for li in all_steer_layers:
                        steer_s["layer_masks"][li] = torch.tensor(
                            [steer_s["assigns"][b] == li for b in range(B)],
                            dtype=torch.bool, device=device) & disagree_mask

                    if steer_all_positions_full:
                        # Faithful reproduction of hybrid_token.py:
                        # fresh full-sequence forward (no KV cache),
                        # hook applies c*v to ALL positions of layer
                        # `steering_layer`. Logits at last position
                        # are the candidate next-token distribution.
                        steer_s["all_positions"] = True
                        with torch.inference_mode():
                            out = base_model(
                                input_ids=base_ids_full,
                                attention_mask=base_mask_full,
                                position_ids=full_pos,
                                use_cache=False,
                                logits_to_keep=1)
                        steer_s["all_positions"] = False
                        last_logits = out.logits[:, -1, :]
                        cand = torch.argmax(last_logits, dim=-1)
                        del out
                    else:
                        _truncate_kv(base_kv)
                        with torch.inference_mode():
                            out = base_model(
                                input_ids=prev_base_input.unsqueeze(1),
                                attention_mask=b_mask,
                                position_ids=(base_pos - 1).unsqueeze(1),
                                past_key_values=base_kv, use_cache=True)
                        base_kv = out.past_key_values
                        last_logits = out.logits[:, -1, :]
                        cand = torch.argmax(last_logits, dim=-1)
                        del out

                    if random_guardrail:
                        # Commit this candidate for rows whose random draw == sc.
                        picked = (torch.isclose(
                            chosen_coef, torch.tensor(sc, device=device))
                            & disagree_mask)
                        if picked.any():
                            best_coeff[picked] = sc
                            best_tok[picked] = cand[picked]
                    elif coef_select == "think_top1_match":
                        # Strict ceiling: lock in the SMALLEST coef whose
                        # steered argmax already equals thinking's top-1.
                        # Rows that already matched are skipped; rows that
                        # never match stay unsteered (best_coeff stays 0,
                        # best_tok stays base_next_toks).
                        new_match = (
                            (cand == think_next_toks)
                            & disagree_mask
                            & ~matched_row)
                        if new_match.any():
                            matched_row[new_match] = True
                            best_coeff[new_match] = sc
                            best_tok[new_match] = cand[new_match]
                    else:
                        if coef_select == "pg":
                            c_lp = think_lp[arange_B, cand]
                        elif coef_select == "think_top1":
                            # Oracle / ceiling: pick coef that maximises
                            # the steered-base log-prob at the thinking
                            # model's argmax token T.  Tokenizer alignment
                            # (base vocab == think vocab, 1:1) is verified
                            # at startup so think_next_toks indexes base
                            # logits correctly.
                            base_lp = torch.log_softmax(
                                last_logits.float(), dim=-1)
                            c_lp = base_lp[arange_B, think_next_toks]
                            c_lp = c_lp.to(best_lp.dtype)
                        else:
                            # KL-top-K score: -CE between thinking top-K
                            # (as soft target) and steered base log-probs.
                            # Larger (closer to 0) = better fit.
                            base_lp = torch.log_softmax(
                                last_logits.float(), dim=-1)
                            base_lp_at_topk = base_lp.gather(
                                -1, t_topk_ix)  # (B, K)
                            c_lp = (t_topk_p * base_lp_at_topk).sum(dim=-1)
                            c_lp = c_lp.to(best_lp.dtype)
                        better = (c_lp > best_lp) & disagree_mask
                        if better.any():
                            best_lp[better] = c_lp[better]
                            best_coeff[better] = sc
                            best_tok[better] = cand[better]

                if coef_select == "think_top1_match":
                    # Only count rows that actually matched thinking's top-1
                    # at some coef as "steered"; the rest fall through to
                    # the unsteered base argmax.
                    need_steer = matched_row & disagree_mask
                else:
                    need_steer = disagree_mask
                output_toks = torch.where(need_steer, best_tok, base_next_toks)
                did_steer = need_steer

                if steer_all_positions_full:
                    # Faithful mode: the sweep never touched base_kv
                    # (use_cache=False), so nothing to revert. The KV
                    # cache still reflects the unsteered processing up
                    # to prev_base_input.
                    _clear_steering()
                elif steer_all_positions:
                    # Reproduce collaborator's hybrid_token.py semantics:
                    # persist the winning-coef shift at this position in
                    # the K/V cache for all layers > sae_layer, rather
                    # than reverting to unsteered. Across decode steps
                    # this accumulates so subsequent attention sees
                    # steered K/V for every past disagreement position,
                    # approximating their 'shift every position's layer-
                    # sae_layer output on every forward pass' behaviour.
                    for li in all_steer_layers:
                        steer_s["layer_masks"][li] = torch.tensor(
                            [steer_s["assigns"][b] == li for b in range(B)],
                            dtype=torch.bool, device=device) & disagree_mask
                    steer_s["coef"] = best_coeff  # per-row winner
                    _truncate_kv(base_kv)
                    with torch.inference_mode():
                        commit = base_model(
                            input_ids=prev_base_input.unsqueeze(1),
                            attention_mask=b_mask,
                            position_ids=(base_pos - 1).unsqueeze(1),
                            past_key_values=base_kv, use_cache=True)
                    base_kv = commit.past_key_values
                    del commit
                    steer_s["coef"] = 1.0  # reset
                else:
                    # Revert the K/V at prev_base_input to unsteered —
                    # steering should act as a per-step logit nudge only,
                    # matching the old non-KV pipeline's semantics.
                    _clear_steering()
                    _truncate_kv(base_kv)
                    with torch.inference_mode():
                        revert = base_model(
                            input_ids=prev_base_input.unsqueeze(1),
                            attention_mask=b_mask,
                            position_ids=(base_pos - 1).unsqueeze(1),
                            past_key_values=base_kv, use_cache=True)
                    base_kv = revert.past_key_values
                    del revert

            # ---- 3. Record output ----
            # Pre-compute per-row debug tensors so the Python loop below
            # just reads scalars.
            base_top_ids = base_next_toks.detach().cpu().tolist()
            think_top_ids = think_next_toks.detach().cpu().tolist()
            disagree_row = (~token_agree).detach().cpu().tolist()
            steer_match_think = (
                did_steer
                & (output_toks == think_next_toks)
            ).detach().cpu().tolist()
            did_steer_row = did_steer.detach().cpu().tolist()
            best_coeff_row = best_coeff.detach().cpu().tolist()
            assigns_snapshot = list(steer_s.get("assigns", [default_layer] * B))

            for b in range(B):
                if finished[b]:
                    continue
                tid = output_toks[b].item()
                generated_ids[b].append(tid)
                sel = "steered" if did_steer_row[b] else "base"
                steer_sels[b].append(sel)
                cc = round(best_coeff_row[b], 1) if did_steer_row[b] else 0.0
                coeff_sels[b].append(cc)
                if token_infos is not None:
                    k = lat_keys[b]
                    v_norm = vec_norms.get(k, 0.0)
                    s_layer = assigns_snapshot[b] if assigns_snapshot else default_layer
                    token_infos[b].append({
                        "token": base_tokenizer.decode(tid),
                        "latent_id": lat_ids[b].item(),
                        "latent_title": lat_titles[b],
                        "latent_key": k,
                        "activation_value": act_vals[b].item(),
                        "coefficient": cc,
                        "selection": sel,
                        "base_tok_id": int(base_top_ids[b]),
                        "think_tok_id": int(think_top_ids[b]),
                        "disagreed": bool(disagree_row[b]),
                        "has_vector": bool(has_vec_map.get(k, False)),
                        "steer_layer": int(s_layer),
                        "shift_norm": round(cc * v_norm, 3),
                        "steered_matches_think": bool(steer_match_think[b]),
                    })
                if tid == eos_id:
                    finished[b] = True

            n_gen += 1
            if pbar:
                pbar.update(1)
            if finished.all():
                break

            # ---- 4. Advance both models with the chosen output token ----
            t_mask = torch.cat([t_mask, torch.ones(B, 1, device=device, dtype=t_mask.dtype)], dim=1)
            b_mask = torch.cat([b_mask, torch.ones(B, 1, device=device, dtype=b_mask.dtype)], dim=1)

            # Grow the full base sequence for --steer_all_positions_full.
            base_ids_full = torch.cat(
                [base_ids_full, output_toks.unsqueeze(1)], dim=1)
            base_mask_full = torch.cat(
                [base_mask_full,
                 torch.ones(B, 1, device=device, dtype=base_mask_full.dtype)],
                dim=1)

            with torch.inference_mode():
                think_out = thinking_model(
                    input_ids=output_toks.unsqueeze(1),
                    attention_mask=t_mask,
                    position_ids=think_pos.unsqueeze(1),
                    past_key_values=think_kv, use_cache=True)
            think_kv = think_out.past_key_values
            think_logits = think_out.logits[:, -1, :]
            del think_out
            think_pos += 1

            lat_ids, act_vals, lat_keys, lat_titles = _classify(captured["sae_act"])
            _clear_steering()

            with torch.inference_mode():
                base_out = base_model(
                    input_ids=output_toks.unsqueeze(1),
                    attention_mask=b_mask,
                    position_ids=base_pos.unsqueeze(1),
                    past_key_values=base_kv, use_cache=True)
            base_kv = base_out.past_key_values
            base_logits = base_out.logits[:, -1, :]
            del base_out
            base_pos += 1

            prev_base_input = output_toks.clone()

            if n_gen % 32 == 0:
                torch.cuda.empty_cache()

    finally:
        handle_sae.remove()
        for h in steer_handles:
            h.remove()
        if pbar:
            pbar.close()
        gc.collect()
        torch.cuda.empty_cache()

    out_list = []
    for b in range(B):
        n_tot = len(generated_ids[b])
        n_st = sum(1 for s in steer_sels[b] if s == "steered")
        cc_hist = {}
        for c in coeff_sels[b]:
            if c > 0:
                k = str(round(c, 1))
                cc_hist[k] = cc_hist.get(k, 0) + 1
        # Richer per-example debug aggregates (derived from token_infos).
        infos_b = token_infos[b] if token_infos else []
        n_dis = sum(1 for ti in infos_b if ti.get("disagreed"))
        n_no_vec = sum(1 for ti in infos_b
                       if ti.get("disagreed") and not ti.get("has_vector", True))
        n_match_think = sum(1 for ti in infos_b
                            if ti.get("steered_matches_think"))
        per_cat_count: Dict[str, int] = {}
        per_cat_coef_sum: Dict[str, float] = {}
        per_layer_count: Dict[int, int] = {}
        for ti in infos_b:
            if ti.get("selection") != "steered":
                continue
            k = ti.get("latent_key", "?")
            per_cat_count[k] = per_cat_count.get(k, 0) + 1
            per_cat_coef_sum[k] = per_cat_coef_sum.get(k, 0.0) + float(
                ti.get("coefficient", 0.0))
            s_layer = int(ti.get("steer_layer", -1))
            per_layer_count[s_layer] = per_layer_count.get(s_layer, 0) + 1
        per_cat_mean_coef = {k: round(per_cat_coef_sum[k] / per_cat_count[k], 3)
                             for k in per_cat_count}
        top5_cats = sorted(per_cat_count.items(),
                           key=lambda kv: -kv[1])[:5]
        out_list.append({
            "generated_ids": generated_ids[b],
            "n_generated": n_tot,
            "token_latent_info": (token_infos[b] if token_infos else []),
            "steering_selection": steer_sels[b],
            "ended_by_eos": bool(finished[b]),
            "steering_stats": {
                "n_steered": n_st,
                "n_total": n_tot,
                "frac_steered": round(n_st / max(n_tot, 1), 4),
                "coeff_distribution": cc_hist,
                # --- new debug aggregates ---
                "n_disagree": n_dis,
                "n_no_vector": n_no_vec,
                "n_steered_matches_think": n_match_think,
                "frac_steered_matches_think": round(
                    n_match_think / max(n_st, 1), 4),
                "per_category_counts": per_cat_count,
                "per_category_mean_coef": per_cat_mean_coef,
                "per_layer_counts": {str(k): v
                                     for k, v in per_layer_count.items()},
                "top5_categories": [[k, v] for k, v in top5_cats],
            },
        })
    return out_list


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------

def _safe_chat_batch(prompts, model_name, max_tokens=2000, **kwargs):
    import asyncio
    import concurrent.futures
    async def _run():
        return await chat_batch(prompts, model=model_name,
                                max_tokens=max_tokens, **kwargs)
    try:
        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor() as ex:
            return ex.submit(lambda: asyncio.run(_run())).result()
    except RuntimeError:
        return asyncio.run(_run())


def _build_judge_prompt(answer, gold, question, ds_type="math", test_list=None):
    if ds_type == "coding":
        tc = "\n\n".join(test_list) if test_list else "No test cases"
        ref = f"\n\nReference:\n```python\n{gold}\n```" if gold else ""
        return (f"Evaluate correctness.\n\nProblem: {question}\n\n"
                f"Response:\n{answer}\n{ref}\n\nTests:\n{tc}\n\n"
                f"Answer YES or NO.")
    elif ds_type == "mcqa":
        return (f"Question: {question}\nCorrect: {gold}\n\n"
                f"Response: {answer}\n\nDid they pick {gold}? YES or NO.")
    elif ds_type == "classification":
        return (f"Question: {question}\nCorrect: {gold}\n\n"
                f"Response: {answer}\n\nCorrect classification? YES or NO.")
    return (f"Question: {question}\nCorrect answer: {gold}\n\n"
            f"Response: {answer}\n\n"
            f"Does the response contain the correct answer? YES or NO.")


def judge_batch(items, judge_model, n_reps=1, max_concurrent=40):
    """Judge items concurrently. Returns list of {correct, raw, reps}."""
    prompts = []
    for it in items:
        p = _build_judge_prompt(it["answer"], it["gold"], it["question"],
                                ds_type=it.get("ds_type", "math"),
                                test_list=it.get("test_list"))
        prompts.extend([p] * max(1, n_reps))

    responses = _safe_chat_batch(prompts, judge_model, max_tokens=100,
                                 max_concurrent_requests=max_concurrent)

    out = []
    idx = 0
    for it in items:
        reps = []
        for _ in range(max(1, n_reps)):
            r = responses[idx] if idx < len(responses) else ""
            if not isinstance(r, str):
                r = ""
            reps.append({"correct": "yes" in r.lower(), "raw": r})
            idx += 1
        n_yes = sum(r["correct"] for r in reps)
        correct = n_yes > len(reps) / 2
        label = it.get("label", "")
        verdicts = ["Y" if r["correct"] else "N" for r in reps]
        print(f"  {label} -> {verdicts}" if n_reps > 1
              else f"  {label} -> {reps[0]['raw'][:40]}")
        out.append({"correct": correct, "raw": reps[0]["raw"], "repetitions": reps})
    return out


# ---------------------------------------------------------------------------
# Response cache
# ---------------------------------------------------------------------------

def _cache_path(results_dir, role, model_id, dataset, temp, max_tok):
    os.makedirs(f"{results_dir}/response_cache", exist_ok=True)
    ts = f"{temp:.2f}".rstrip("0").rstrip(".")
    return f"{results_dir}/response_cache/{role}_{model_id}_{dataset}_temp{ts}_max{max_tok}.jsonl"


def _load_cache(path):
    cache = {}
    if not os.path.exists(path):
        return cache
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            cache[e["dataset_idx"]] = {
                "response": e["response"],
                "n_tokens": e["n_tokens"], "eos": e["eos"]}
    return cache


def _append_cache(path, entries):
    if not entries:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Rolling results
# ---------------------------------------------------------------------------

ROLLING_MAX = 90 * 1024 * 1024


def _rolling_prefix(args, base_id, think_id):
    os.makedirs(f"{args.results_dir}/rolling", exist_ok=True)
    s = _result_suffix(args)
    return f"{args.results_dir}/rolling/rolling_{base_id}_{args.dataset}{s}"


def _count_completed(args, base_id, think_id):
    prefix = _rolling_prefix(args, base_id, think_id)
    total = 0
    for path in _list_rolling(prefix):
        with open(path) as f:
            for _ in f:
                total += 1
    return total


def _list_rolling(prefix):
    d = os.path.dirname(prefix)
    b = os.path.basename(prefix)
    files = []
    legacy = os.path.join(d, b + ".jsonl")
    if os.path.exists(legacy):
        files.append(legacy)
    try:
        for fn in sorted(os.listdir(d)):
            if fn.startswith(b + "_") and fn.endswith(".jsonl"):
                files.append(os.path.join(d, fn))
    except FileNotFoundError:
        pass
    return files


def _load_prev_counts(args, base_id, think_id):
    files = _list_rolling(_rolling_prefix(args, base_id, think_id))
    n = 0
    counts = {"thinking": 0, "base": 0, "hybrid": 0}
    for path in files:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                for k in counts:
                    j = rec.get("judges", {}).get(k, {})
                    if isinstance(j, dict) and j.get("correct"):
                        counts[k] += 1
                n += 1
    return n, counts


def append_rolling(record, args, base_id, think_id):
    prefix = _rolling_prefix(args, base_id, think_id)
    files = _list_rolling(prefix)
    data = json.dumps(record) + "\n"
    if files:
        target = files[-1]
        if os.path.getsize(target) + len(data.encode()) > ROLLING_MAX:
            idx = len([f for f in files if "_" in os.path.basename(f)])
            target = f"{prefix}_{idx}.jsonl"
    else:
        target = prefix + ".jsonl"
    with open(target, "a", encoding="utf-8") as f:
        f.write(data)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    if args.steer_all_positions and args.steer_all_positions_full:
        raise SystemExit(
            "--steer_all_positions and --steer_all_positions_full are "
            "mutually exclusive (both re-interpret the sweep semantics).")
    os.makedirs(args.results_dir, exist_ok=True)
    ds_type = _dataset_type(args)

    # ---- Load dataset ----
    print(f"Loading {args.dataset}...")
    if args.dataset == "gsm8k":
        dataset = load_dataset("openai/gsm8k", "main")["test"]
    elif args.dataset == "math500":
        dataset = load_dataset("HuggingFaceH4/MATH-500")["test"]
    elif args.dataset == "aime24":
        dataset = load_dataset("HuggingFaceH4/aime_2024")["train"]
    elif args.dataset == "aime25":
        dataset = load_dataset("yentinglin/aime_2025")["train"]
    elif args.dataset == "mbpp":
        dataset = load_dataset("google-research-datasets/mbpp", "full")["test"]
    elif args.dataset == "livecodebench":
        dataset = load_dataset("bzantium/livecodebench", "release_v5")["test"]
    elif args.dataset == "medqa":
        dataset = load_dataset("GBaker/MedQA-USMLE-4-options")["test"].select(range(500))
    elif args.dataset == "gpqa":
        dataset = load_dataset("Idavidrein/gpqa", "gpqa_main", split="train")
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # ---- Quick judge test ----
    try:
        r = _safe_chat_batch(["Reply YES."], args.judge_model, max_tokens=5)
        print(f"Judge API test: {'OK' if r and isinstance(r[0], str) else 'FAIL'}")
    except Exception as e:
        print(f"Judge API test: FAIL ({e})")

    # ---- Load models ----
    think_id = args.thinking_model.split("/")[-1].lower()
    base_id = args.base_model.split("/")[-1].lower()
    dom_model_short = args.dom_vectors_model_short or base_id

    print(f"\nLoading thinking model {args.thinking_model}...")
    think_tok = AutoTokenizer.from_pretrained(args.thinking_model)
    if think_tok.pad_token is None:
        think_tok.pad_token = think_tok.eos_token
    think_model = AutoModelForCausalLM.from_pretrained(
        args.thinking_model, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="sdpa")
    think_model.eval()

    print(f"Loading base model {args.base_model}...")
    base_tok = AutoTokenizer.from_pretrained(args.base_model)
    if base_tok.pad_token is None:
        base_tok.pad_token = base_tok.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map="auto",
        attn_implementation="sdpa")
    base_model.eval()

    # ---- Tokenizer alignment check ----
    # Several --coef_select modes (pg, kl_top*, think_top1) index base-model
    # logits with thinking-model token ids and vice versa.  This only works
    # if the two tokenizers share the same vocab 1:1.  Verify on a probe set
    # (cheaper than a full vocab scan, catches realistic divergence).
    assert think_model.config.vocab_size == base_model.config.vocab_size, (
        f"vocab_size mismatch: think={think_model.config.vocab_size} "
        f"base={base_model.config.vocab_size} -- coef_select needs aligned "
        f"vocabularies.")
    _probes = [
        " the quick brown fox jumps over the lazy dog.",
        "\n\nLet me think step by step.\n\n",
        "Wait, lets backtrack.",
        "Therefore, the answer is \\boxed{42}.",
        "\\frac{1}{2}",
        "<|im_start|>user\nHello<|im_end|>",
    ]
    for _s in _probes:
        _b = base_tok(_s, add_special_tokens=False)["input_ids"]
        _t = think_tok(_s, add_special_tokens=False)["input_ids"]
        assert _b == _t, (
            f"Tokenizer mismatch on probe {_s!r}: base={_b} think={_t}. "
            f"coef_select modes that mix indices across models would be "
            f"incorrect; refusing to run.")
    print("Tokenizer alignment: OK (base/thinking vocabs match 1:1 on probes).")

    # Up-cast the final lm_head weight to fp32 on both models.  Hidden
    # states stay bf16; we cast them to fp32 just before the matmul via a
    # wrapper.  Kills most bf16 argmax flips on near-tied logits and is
    # essentially free at runtime.
    def _wrap_lm_head(model):
        lm = model.lm_head
        # For models with tied word embeddings (e.g. Qwen2.5-0.5B,
        # Qwen2.5-1.5B, small Llamas), lm.weight is the SAME tensor as
        # model.model.embed_tokens.weight.  Casting lm to fp32 in-place
        # would also cast the embedding weight, causing the transformer
        # residual stream to go fp32 while decoder-layer weights remain
        # bf16 -> mat1/mat2 dtype mismatch.  Break the tie first by
        # cloning the weight, then cast the copy to fp32.
        try:
            tied = getattr(getattr(model, "config", None),
                           "tie_word_embeddings", False)
        except Exception:
            tied = False
        if tied:
            new_w = torch.nn.Parameter(
                lm.weight.detach().clone().to(torch.float32))
            lm.weight = new_w
        else:
            lm.to(torch.float32)
        orig_forward = lm.forward
        def _fwd(x):
            return orig_forward(x.to(torch.float32))
        lm.forward = _fwd
    _wrap_lm_head(think_model)
    _wrap_lm_head(base_model)

    # ---- Load SAE ----
    print(f"Loading SAE (layer={args.sae_layer}, clusters={args.n_clusters})...")
    sae, _ = load_sae(think_id, args.sae_layer, args.n_clusters,
                      require_activation_mean=False)
    ckpt_path = os.path.join(
        os.path.dirname(__file__),
        f"../train-saes/results/vars/saes/sae_{think_id}_layer{args.sae_layer}"
        f"_clusters{args.n_clusters}.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "activation_mean" in ckpt:
            sae.activation_mean = ckpt["activation_mean"]
            print("  Loaded activation_mean from checkpoint")
        del ckpt
    if not hasattr(sae, "activation_mean") and not args.disable_sae_mean:
        args.disable_sae_mean = True
        print("  Auto-set disable_sae_mean (no activation_mean)")
    sae = sae.to(next(think_model.parameters()).device)
    descriptions = get_latent_descriptions(think_id, args.sae_layer, args.n_clusters)

    # ---- Load steering vectors ----
    base_dev = next(base_model.parameters()).device
    base_dt = next(base_model.parameters()).dtype

    if args.old_vectors_dir:
        print(f"Loading OLD optimized vectors from {args.old_vectors_dir}...")
        old_layer = args.old_vectors_layer
        steering_vectors, layer_map = {}, {}
        # Optional per-category layer map: written by the layer-sweep
        # trainer in train-vectors/optimize_correction_vectors.py.  If
        # present, it overrides --old_vectors_layer on a per-key basis.
        per_key_layers = {}
        lm_path = os.path.join(args.old_vectors_dir, "layer_map.json")
        if os.path.exists(lm_path):
            with open(lm_path) as f:
                per_key_layers = {k: int(v) for k, v in json.load(f).items()}
            print(f"  Using per-category layer_map.json: {per_key_layers}")
        for cat_id in range(args.n_clusters):
            key = f"idx{cat_id}"
            fpath = os.path.join(args.old_vectors_dir,
                                 f"{dom_model_short}_idx{cat_id}_linear.pt")
            if not os.path.exists(fpath):
                print(f"  WARNING: {fpath} not found, skipping {key}")
                continue
            ckpt = torch.load(fpath, map_location="cpu", weights_only=False)
            vec = ckpt[key]
            steering_vectors[key] = vec.to(torch.float32)
            layer_map[key] = per_key_layers.get(key, old_layer)
            print(f"  {key}: layer={layer_map[key]}, "
                  f"norm={vec.norm().item():.2f}")
    else:
        print(f"Loading DOM vectors from {args.dom_vectors_dir}...")
        steering_vectors, layer_map = load_dom_vectors(
            args.dom_vectors_dir, dom_model_short, descriptions)

    for k, v in steering_vectors.items():
        steering_vectors[k] = v.to(device=base_dev, dtype=base_dt)

    # ---- Optional: replace each category vector with a random direction of
    # the same norm.  Runs BEFORE bias is folded in, so the bias (if any) is
    # left untouched -- this isolates the contribution of the *category-
    # specific direction* vs norm + bias.
    assert not (args.randomize_vectors and args.randomize_vectors_unit_norm), (
        "--randomize_vectors and --randomize_vectors_unit_norm are mutually "
        "exclusive")
    if args.randomize_vectors:
        print(f"Randomizing steering vectors (seed={args.random_seed}, "
              f"norm-preserving)...")
        g = torch.Generator(device="cpu").manual_seed(int(args.random_seed))
        for k in list(steering_vectors.keys()):
            v = steering_vectors[k]
            orig_norm = v.float().norm().item()
            rnd = torch.randn(v.shape, generator=g, dtype=torch.float32)
            rnd = rnd * (orig_norm / (rnd.norm().item() + 1e-12))
            steering_vectors[k] = rnd.to(device=v.device, dtype=v.dtype)
            print(f"  {k}: orig_norm={orig_norm:.2f}, "
                  f"rand_norm={steering_vectors[k].float().norm().item():.2f}")
    if args.randomize_vectors_unit_norm:
        print(f"Randomizing steering vectors (seed={args.random_seed}, "
              f"UNIT-NORM -- reproducing collaborator bug)...")
        g = torch.Generator(device="cpu").manual_seed(int(args.random_seed))
        for k in list(steering_vectors.keys()):
            v = steering_vectors[k]
            orig_norm = v.float().norm().item()
            rnd = torch.randn(v.shape, generator=g, dtype=torch.float32)
            rnd = rnd / (rnd.norm().item() + 1e-12)
            steering_vectors[k] = rnd.to(device=v.device, dtype=v.dtype)
            print(f"  {k}: orig_norm={orig_norm:.2f}, "
                  f"rand_norm={steering_vectors[k].float().norm().item():.4f}")

    # ---- Optional: zero out per-category vectors (bias-only ablation) ----
    if args.bias_only:
        assert args.bias_vector_path, "--bias_only requires --bias_vector_path"
        print("Bias-only ablation: zeroing all per-category steering vectors "
              "(only the global bias will remain).")
        for k in list(steering_vectors.keys()):
            steering_vectors[k] = torch.zeros_like(steering_vectors[k])

    # ---- Optional: add a global bias vector on top of each category vector ----
    # The OLD (paper) pipeline applies  coef * (steer_vec + bias_vec)  at the
    # steering layer for each (coef, window) candidate.  Since we always apply
    # the category vector and the bias at the same positions with the same
    # coefficient, folding bias into each category vector at load time is
    # mathematically equivalent and keeps the hot loop untouched.
    if args.bias_vector_path:
        print(f"Loading bias vector from {args.bias_vector_path}...")
        bias_obj = torch.load(args.bias_vector_path, map_location="cpu",
                              weights_only=False)
        if isinstance(bias_obj, dict):
            bias_vec = bias_obj.get("bias", next(iter(bias_obj.values())))
        else:
            bias_vec = bias_obj
        bias_vec = bias_vec.to(device=base_dev, dtype=base_dt)
        print(f"  bias: shape={tuple(bias_vec.shape)}, "
              f"norm={bias_vec.float().norm().item():.2f}")
        hidden = getattr(base_model.config, "hidden_size", 0)
        assert bias_vec.shape[-1] == hidden, (
            f"bias_vec shape {tuple(bias_vec.shape)} incompatible with base model")
        for k in list(steering_vectors.keys()):
            steering_vectors[k] = steering_vectors[k] + bias_vec

        # When running the BIAS-ONLY ablation, we want every key to steer at
        # the bias's own chosen layer, not whatever layer each category was
        # individually selected for.  Resolve bias_layer priority:
        #   1. explicit --bias_layer CLI
        #   2. bias_layer.json sibling of --bias_vector_path (written by
        #      optimize_correction_vectors.py when --train_global_bias)
        #   3. --old_vectors_layer (legacy fallback)
        if args.bias_only:
            bias_layer = args.bias_layer
            if bias_layer is None:
                sib = os.path.join(os.path.dirname(args.bias_vector_path),
                                   "bias_layer.json")
                if os.path.exists(sib):
                    with open(sib) as f:
                        bias_layer = int(json.load(f)["layer"])
                    print(f"  Using bias_layer={bias_layer} from {sib}")
            if bias_layer is None:
                bias_layer = args.old_vectors_layer
                print(f"  Falling back to --old_vectors_layer={bias_layer} "
                      f"for bias-only steering layer")
            for k in list(layer_map.keys()):
                layer_map[k] = bias_layer
            print(f"  [bias_only] layer_map overridden to "
                  f"{bias_layer} for all keys")

    # ---- One-time debug metadata dump (config + loaded vectors) ----
    # This lets us reconstruct AFTER THE FACT exactly what was steered,
    # from which files, at which layers, with which norms.  Cheap (few
    # KB) and written once per run, right before task loop starts.
    try:
        dbg_meta = {
            "base_model": args.base_model,
            "thinking_model": args.thinking_model,
            "base_id": base_id,
            "thinking_id": think_id,
            "dataset": args.dataset,
            "sae_layer": args.sae_layer,
            "n_clusters": args.n_clusters,
            "sae_path": getattr(args, "sae_path", None),
            "coef_sweep": [float(x) for x in args.coef_sweep.split(",")],
            "coef_select": args.coef_select,
            "kl_topk": int(args.kl_topk),
            "bias_only": bool(args.bias_only),
            "bias_vector_path": args.bias_vector_path,
            "bias_layer": (
                int(locals()["bias_layer"])
                if args.bias_only and "bias_layer" in locals()
                and locals()["bias_layer"] is not None else None),
            "random_firing": bool(args.random_firing),
            "random_guardrail": bool(args.random_guardrail),
            "random_seed": int(args.random_seed),
            "steer_all_positions": bool(args.steer_all_positions),
            "steer_all_positions_full": bool(args.steer_all_positions_full),
            "disable_sae_mean": bool(args.disable_sae_mean),
            "old_vectors_dir": getattr(args, "old_vectors_dir", None),
            "old_vectors_layer": getattr(args, "old_vectors_layer", None),
            "n_latent_descriptions": len(descriptions),
            "layer_map": {k: int(v) for k, v in layer_map.items()},
            "steering_vectors": {
                k: {"shape": list(v.shape),
                    "norm": float(v.float().norm().item()),
                    "layer": int(layer_map.get(k, -1))}
                for k, v in steering_vectors.items()},
            "suffix": _result_suffix(args),
        }
        dbg_dir = args.results_dir
        os.makedirs(dbg_dir, exist_ok=True)
        dbg_path = os.path.join(
            dbg_dir,
            f"debug_meta_{base_id}_{think_id}_{args.dataset}"
            f"{_result_suffix(args)}.json")
        with open(dbg_path, "w") as f:
            json.dump(dbg_meta, f, indent=2)
        print(f"[debug] wrote run metadata -> {dbg_path}")
    except Exception as e:
        print(f"[debug] WARNING: could not write debug_meta: {e}")

    # ---- Prepare tasks ----
    if args.n_tasks <= 0:
        args.n_tasks = len(dataset) - args.eval_start_idx

    completed = _count_completed(args, base_id, think_id)
    if completed > 0:
        if completed >= args.n_tasks:
            print(f"Already completed {completed} tasks. Nothing to do.")
            return
        print(f"Resuming: {completed} done, {args.n_tasks - completed} remaining")
        args.eval_start_idx = completed
        args.n_tasks -= completed

    tasks = []
    for i, item in enumerate(dataset):
        if i < args.eval_start_idx:
            continue
        if len(tasks) >= args.n_tasks:
            break
        t = _build_task_prompts(item, i, args)
        t["dataset_idx"] = i
        tasks.append(t)
    n_tasks = len(tasks)
    print(f"\n=== {n_tasks} tasks ===")

    # ---- Phase 2: Standalone responses ----
    use_cache = not args.no_response_cache
    tc_path = _cache_path(args.results_dir, "thinking", think_id,
                          args.dataset, args.temperature, args.max_thinking_tokens)
    bc_path = _cache_path(args.results_dir, "base", base_id,
                          args.dataset, args.temperature, args.max_new_tokens)
    tc = _load_cache(tc_path) if use_cache else {}
    bc = _load_cache(bc_path) if use_cache else {}
    print(f"  Cache: thinking={len(tc)}, base={len(bc)}")

    uncached_t = [(i, t) for i, t in enumerate(tasks) if t["dataset_idx"] not in tc]
    print(f"\n=== Thinking: {n_tasks - len(uncached_t)} cached, {len(uncached_t)} to generate ===")
    if uncached_t:
        def _flush_t(bstart, brecs):
            if not use_cache:
                return
            entries = []
            for k, r in enumerate(brecs):
                oi, t = uncached_t[bstart + k]
                entries.append(dict(dataset_idx=t["dataset_idx"], **r))
            _append_cache(tc_path, entries)
        res = _batch_generate(think_model, think_tok,
                              [t["thinking_prompt"] for _, t in uncached_t],
                              args.max_thinking_tokens, args.batch_gen_size,
                              use_chat_template=True, temperature=args.temperature,
                              on_batch_done=_flush_t, tag="thinking")
        for (oi, t), r in zip(uncached_t, res):
            t.update(thinking_response=r["response"],
                     thinking_n_tokens=r["n_tokens"], thinking_eos=r["eos"])
        del res; torch.cuda.empty_cache()

    for t in tasks:
        if "thinking_response" not in t:
            c = tc[t["dataset_idx"]]
            t.update(thinking_response=c["response"],
                     thinking_n_tokens=c["n_tokens"], thinking_eos=c["eos"])

    uncached_b = [(i, t) for i, t in enumerate(tasks) if t["dataset_idx"] not in bc]
    print(f"\n=== Base: {n_tasks - len(uncached_b)} cached, {len(uncached_b)} to generate ===")
    if uncached_b:
        def _flush_b(bstart, brecs):
            if not use_cache:
                return
            entries = []
            for k, r in enumerate(brecs):
                oi, t = uncached_b[bstart + k]
                entries.append(dict(dataset_idx=t["dataset_idx"], **r))
            _append_cache(bc_path, entries)
        res = _batch_generate(base_model, base_tok,
                              [t["base_prompt"] for _, t in uncached_b],
                              args.max_new_tokens, args.batch_gen_size,
                              temperature=args.temperature,
                              on_batch_done=_flush_b, tag="base")
        for (oi, t), r in zip(uncached_b, res):
            t.update(base_response=r["response"],
                     base_n_tokens=r["n_tokens"], base_eos=r["eos"])
        del res; torch.cuda.empty_cache()

    for t in tasks:
        if "base_response" not in t:
            c = bc[t["dataset_idx"]]
            t.update(base_response=c["response"],
                     base_n_tokens=c["n_tokens"], base_eos=c["eos"])

    # ---- Phase 3: Hybrid + judge ----
    hbs = args.hybrid_gen_batch_size
    print(f"\n=== Hybrid (B={hbs}, KV-cached, coeff-sweep) + judge ===")

    prev_n, prev_counts = _load_prev_counts(args, base_id, think_id)
    results = {"thinking_correct": 0, "base_correct": 0, "hybrid_correct": 0,
               "thinking_eos": [], "base_eos": [], "hybrid_eos": [],
               "thinking_lengths": [], "base_lengths": [], "hybrid_lengths": []}

    for batch_start in range(0, n_tasks, hbs):
        batch = tasks[batch_start:batch_start + hbs]
        B_ = len(batch)
        print(f"\n--- Batch {batch_start//hbs+1} "
              f"(tasks {batch_start+1}-{batch_start+B_}/{n_tasks}) ---")

        torch.cuda.empty_cache()
        hr = hybrid_generate_batched(
            think_model, base_model, base_tok,
            [t["thinking_prompt"] for t in batch],
            [t["base_prompt"] for t in batch],
            args.max_new_tokens, args.sae_layer, sae,
            steering_vectors, descriptions, layer_map,
            thinking_tokenizer=think_tok,
            disable_sae_mean=args.disable_sae_mean,
            show_progress=True, collect_details=True,
            random_firing=args.random_firing,
            random_guardrail=args.random_guardrail,
            random_seed=args.random_seed,
            coef_sweep=[float(x) for x in args.coef_sweep.split(",")],
            steer_all_positions=args.steer_all_positions,
            steer_all_positions_full=args.steer_all_positions_full,
            coef_select=args.coef_select,
            kl_topk=args.kl_topk)

        judge_items = []
        batch_meta = []
        for j, (t, h) in enumerate(zip(batch, hr)):
            hybrid_resp = base_tok.decode(h["generated_ids"], skip_special_tokens=True)
            q, gold, tl = t["question"], t["correct_answer"], t["test_list"]
            common = dict(gold=gold, question=q, ds_type=ds_type, test_list=tl)
            ti = batch_start + j
            judge_items.append(dict(answer=re.sub(r'\s+', ' ', t["thinking_response"]).strip(),
                                    label=f"T{ti+1} Think", **common))
            judge_items.append(dict(answer=re.sub(r'\s+', ' ', t["base_response"]).strip(),
                                    label=f"T{ti+1} Base", **common))
            judge_items.append(dict(answer=re.sub(r'\s+', ' ', hybrid_resp).strip(),
                                    label=f"T{ti+1} Hybrid", **common))
            batch_meta.append(dict(
                task_idx=ti, question=q, gold=gold, test_list=tl,
                think_resp=t["thinking_response"], base_resp=t["base_response"],
                hybrid_resp=hybrid_resp, hybrid_eos=h["ended_by_eos"],
                hybrid_toks=h["n_generated"],
                think_eos=t["thinking_eos"], base_eos=t["base_eos"],
                think_toks=t["thinking_n_tokens"], base_toks=t["base_n_tokens"],
                steering_stats=h.get("steering_stats")))

        if any(m.get("steering_stats") for m in batch_meta):
            ss_all = [m["steering_stats"] for m in batch_meta if m.get("steering_stats")]
            tot_st = sum(s["n_steered"] for s in ss_all)
            tot_tok = sum(s["n_total"] for s in ss_all)
            tot_dis = sum(s.get("n_disagree", 0) for s in ss_all)
            tot_no_vec = sum(s.get("n_no_vector", 0) for s in ss_all)
            tot_match = sum(s.get("n_steered_matches_think", 0) for s in ss_all)
            agg_cc = {}
            agg_cat = {}
            agg_layer = {}
            for s in ss_all:
                for k, v in s.get("coeff_distribution", {}).items():
                    agg_cc[k] = agg_cc.get(k, 0) + v
                for k, v in s.get("per_category_counts", {}).items():
                    agg_cat[k] = agg_cat.get(k, 0) + v
                for k, v in s.get("per_layer_counts", {}).items():
                    agg_layer[k] = agg_layer.get(k, 0) + v
            top5 = sorted(agg_cat.items(), key=lambda kv: -kv[1])[:5]
            match_pct = tot_match / max(tot_st, 1) * 100
            print(f"  Steering: {tot_st}/{tot_tok} tokens "
                  f"({tot_st/max(tot_tok,1)*100:.1f}%)  "
                  f"disagree={tot_dis}  no_vec={tot_no_vec}  "
                  f"match_think={tot_match} ({match_pct:.1f}%)")
            print(f"    coeff dist: {dict(sorted(agg_cc.items()))}")
            print(f"    top cats:   {top5}")
            if len(agg_layer) > 1:
                print(f"    per-layer:  {dict(sorted(agg_layer.items()))}")

        print(f"\n  Judging {len(judge_items)} items...")
        jr = judge_batch(judge_items, args.judge_model,
                         n_reps=args.judge_repetitions,
                         max_concurrent=args.max_concurrent)

        for j, m in enumerate(batch_meta):
            te, be, he = jr[j*3], jr[j*3+1], jr[j*3+2]
            if te["correct"]: results["thinking_correct"] += 1
            if be["correct"]: results["base_correct"] += 1
            if he["correct"]: results["hybrid_correct"] += 1
            results["thinking_eos"].append(m["think_eos"])
            results["base_eos"].append(m["base_eos"])
            results["hybrid_eos"].append(m["hybrid_eos"])
            results["thinking_lengths"].append(m["think_toks"])
            results["base_lengths"].append(m["base_toks"])
            results["hybrid_lengths"].append(m["hybrid_toks"])

            append_rolling({
                "ts": time.time(), "dataset": args.dataset,
                "question": m["question"], "gold_answer": m["gold"],
                "answers": {"thinking": m["think_resp"], "base": m["base_resp"],
                            "hybrid": m["hybrid_resp"]},
                "judges": {"thinking": te, "base": be, "hybrid": he},
                "eos": {"thinking": m["think_eos"], "base": m["base_eos"],
                        "hybrid": m["hybrid_eos"]},
                "n_tokens": {"thinking": m["think_toks"], "base": m["base_toks"],
                             "hybrid": m["hybrid_toks"]},
                "steering_stats": m.get("steering_stats"),
            }, args, base_id, think_id)

            done = m["task_idx"] + 1 + prev_n
            ct = prev_counts["thinking"] + results["thinking_correct"]
            cb = prev_counts["base"] + results["base_correct"]
            ch = prev_counts["hybrid"] + results["hybrid_correct"]
            tp, bp, hp = ct/done*100, cb/done*100, ch/done*100
            gap = abs(tp - bp)
            gs = f"  Gap: {max(0,(hp-min(tp,bp))/gap*100):.1f}%" if gap > 0 else ""
            print(f"  [{done}] T={ct}/{done}({tp:.1f}%) B={cb}/{done}({bp:.1f}%) "
                  f"H={ch}/{done}({hp:.1f}%){gs}")

        del hr; gc.collect(); torch.cuda.empty_cache()

    # ---- Final summary ----
    total = prev_n + len(tasks)
    ct = prev_counts["thinking"] + results["thinking_correct"]
    cb = prev_counts["base"] + results["base_correct"]
    ch = prev_counts["hybrid"] + results["hybrid_correct"]
    print(f"\n===== Final =====")
    print(f"Thinking: {ct}/{total} ({ct/total*100:.1f}%)")
    print(f"Base:     {cb}/{total} ({cb/total*100:.1f}%)")
    print(f"Hybrid:   {ch}/{total} ({ch/total*100:.1f}%)")
    gap = abs(ct/total - cb/total) * 100
    if gap > 0:
        rec = (ch/total*100 - min(ct/total, cb/total)*100) / gap
        print(f"Gap recovered: {max(0, rec)*100:.1f}%")

    plt.figure(figsize=(10, 6))
    accs = [cb/total*100, ct/total*100, ch/total*100]
    plt.bar(["Base", "Thinking", "Hybrid"], accs,
            color=["#3498db", "#e74c3c", "#2ecc71"])
    plt.title(f"Accuracy on {total} {args.dataset} Tasks")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    for i, a in enumerate(accs):
        plt.text(i, a + 2, f"{a:.1f}%", ha="center")
    plt.tight_layout()
    s = _result_suffix(args)
    plt.savefig(f"{args.results_dir}/accuracy_{base_id}_{args.dataset}{s}.png")
    plt.close()
    print("Done.")


if __name__ == "__main__":
    main()
