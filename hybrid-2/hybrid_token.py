# %%
import dotenv
dotenv.load_dotenv("../.env")

import sys
import torch
import json
from utils.sae import load_sae
from utils.utils import load_model
from utils.clustering import get_latent_descriptions
from utils.utils import chat, chat_batch
from utils.utils import load_steering_vectors as _load_all_steering_vectors
import os
import time
import gc
import colorsys
import math
import matplotlib.pyplot as plt
import re
import argparse
from typing import List, Optional
from collections import Counter
from matplotlib.patches import Rectangle
from fractions import Fraction
try:
    from tqdm.auto import tqdm  # progress bar
except Exception:
    tqdm = None


from datasets import load_dataset

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate hybrid model on datasets (token-level steering)')
    parser.add_argument('--dataset', type=str, choices=['gsm8k', 'math500', "aime"], default='aime',
                      help='Dataset to evaluate on (gsm8k or math500)')
    parser.add_argument('--thinking_model', type=str, default='Qwen/QwQ-32B', choices=['deepseek-ai/DeepSeek-R1-Distill-Llama-8B', 'Qwen/QwQ-32B'],
                      help='Model for thinking/perplexity')
    parser.add_argument('--base_model', type=str, default='Qwen/Qwen2.5-32B', choices=['meta-llama/Llama-3.1-8B', 'Qwen/Qwen2.5-32B'],
                      help='Model for base generation')
    parser.add_argument('--steering_layer', type=int, default=24,
                      help='Layer to steer in the base model')
    parser.add_argument('--sae_layer', type=int, default=27,
                      help='Layer to read from in the thinking model for SAE projection')
    parser.add_argument('--n_clusters', type=int, default=10,
                      help='Number of clusters for SAE')
    parser.add_argument('--n_tasks', type=int, default=500,
                      help='Number of tasks to evaluate')
    parser.add_argument('--max_new_tokens', type=int, default=5000,
                      help='Maximum number of tokens to generate')
    parser.add_argument('--max_thinking_tokens', type=int, default=5000,
                      help='Maximum number of tokens for the thinking model only')
    parser.add_argument('--eval_start_idx', type=int, default=0,
                      help='Starting index in the dataset')
    parser.add_argument('--temperature', type=float, default=0.0,
                      help='Temperature for sampling')
    parser.add_argument('--coefficients', type=float, nargs='+', default=[0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
                        help='List of steering coefficients to evaluate per-token under the guardrail')
    parser.add_argument('--token_windows', type=int, nargs='+', default=[1],
                        help='List of token windows (negative = last N tokens) to apply steering to; 0 or [1] means all tokens; e.g., -1 applies only to the last token')
    parser.add_argument('--n_cold_start_tokens', type=int, default=0,
                      help='Number of initial tokens from the thinking model to prepend as a cold-start prefix')
    parser.add_argument('--results_dir', type=str, default='results',
                      help='Directory to save results')
    parser.add_argument('--example_idx', type=int, default=0,
                      help='Index of example to run')
    parser.add_argument('--use_perplexity_guardrail', action='store_true', default=True,
                      help='If set, select among steered candidates based on thinking-model perplexity')
    parser.add_argument('--run_example', action='store_true', default=False,
                      help='Run a single example before evaluation')
    parser.add_argument('--show_progress', action='store_true', default=True,
                      help='Show tqdm progress during hybrid token generation')
    parser.add_argument('--disable_disagreement_only', action='store_true', default=False,
                      help='Disable optimization that only steers when base vs steered disagree')
    parser.add_argument('--store_per_token_details', action='store_true', default=True,
                      help='Keep per-token arrays in RAM during eval (uses more memory)')
    args = parser.parse_known_args()[0]
    # Special handling: if [1] is provided, treat as "all tokens"
    if isinstance(args.token_windows, list) and len(args.token_windows) == 1 and int(args.token_windows[0]) == 1:
        args.token_windows = [0]
    return args

def get_next_token(logits, temperature, model, input_ids=None):
    """Get next token from logits using temperature sampling or greedy decoding (repetition penalty removed)"""
    if isinstance(logits, torch.Tensor):
        logits = logits.to(dtype=torch.float32)
    if temperature > 0:
        logits = logits / temperature
        probs = torch.softmax(logits, dim=-1)
        token = torch.multinomial(probs, num_samples=1).item()
        del logits, probs
        return token
    else:
        token = torch.argmax(logits).item()
        del logits
        return token

def get_token_and_string(logits, temperature, tokenizer, input_ids=None):
    """Get token ID and string from logits"""
    token = get_next_token(logits, temperature, tokenizer, input_ids)
    token_string = tokenizer.decode(token)
    return token, token_string

def get_perplexity(token_string, logits, model):
    """Calculate perplexity of a token string under the given logits.

    Accepts either full logits [batch, seq, vocab], [seq, vocab], or last-step [vocab].
    Works on CPU tensors to reduce GPU memory pressure.
    """
    token_id = model.tokenizer.encode(token_string, return_tensors="pt", add_special_tokens=False).to(torch.long)
    # Normalize to last step logits vector
    if isinstance(logits, torch.Tensor):
        if logits.dim() == 1:
            last_logits = logits
        elif logits.dim() == 2:
            last_logits = logits[-1]
        elif logits.dim() == 3:
            last_logits = logits[0, -1]
        else:
            del token_id
            return float('inf')
    else:
        del token_id
        return float('inf')

    if token_id.shape[1] == 0:
        del token_id
        return float('inf')
    idx = int(token_id[0, 0].item())
    if idx < 0 or idx >= last_logits.shape[-1]:
        del token_id
        return float('inf')
    log_prob = torch.log_softmax(last_logits, dim=-1)[idx].item()
    perplexity = math.exp(-log_prob)
    del token_id
    return perplexity

# NEW: Helper to identify sentence boundaries (unused but handy for analysis)
def is_sentence_end(token_str: str) -> bool:
    stripped = token_str.strip()
    return bool(re.search(r'[.!?]$', stripped)) or stripped == "\n"

# NEW: Prepare cold-start inputs using the already generated thinking tokens
def prepare_cold_start(
    thinking_outputs: torch.Tensor,
    thinking_input_ids: torch.Tensor,
    base_input_ids: torch.Tensor,
    *,
    thinking_tokenizer,
    base_tokenizer,
    n_cold_start_tokens: int,
):
    if n_cold_start_tokens <= 0:
        return base_input_ids, thinking_input_ids, ""

    gen_slice = thinking_outputs[
        :,
        thinking_input_ids.shape[1] : thinking_input_ids.shape[1] + n_cold_start_tokens,
    ]

    cold_start_text: str = thinking_tokenizer.decode(gen_slice[0], skip_special_tokens=True)

    thinking_with_cold = torch.cat([thinking_input_ids, gen_slice.to(torch.long)], dim=1)

    base_cold_ids = (
        base_tokenizer.encode(cold_start_text, return_tensors="pt")
        .to(base_input_ids.device)
        .to(torch.long)
    )
    base_with_cold = torch.cat([base_input_ids, base_cold_ids], dim=1)

    del gen_slice, base_cold_ids
    torch.cuda.empty_cache()

    return base_with_cold, thinking_with_cold, cold_start_text

# ---------------------------------------------------------------------------------
# Token-level hybrid generation
# ---------------------------------------------------------------------------------

def hybrid_generate_token(
    thinking_model,
    base_model,
    base_tokenizer,
    thinking_input_ids,
    base_input_ids,
    max_new_tokens,
    steering_layer,
    sae_layer,
    sae,
    steering_vectors,
    latent_descriptions,
    *,
    coefficient: float = 1.0,
    temperature: float = 1.0,
    verbose: bool = False,
    use_perplexity_guardrail: bool = False,
    coefficients: Optional[List[float]] = None,
    token_windows: Optional[List[int]] = None,
    show_progress: bool = False,
    disagreement_only: bool = True,
    collect_details: bool = True,
):
    """Per-token variant of hybrid generation.

    For each output token:
      1) Use thinking model's last-token hidden state at `sae_layer` to choose the dominant latent and its steering vector.
      2) For each (coefficient, token_window) candidate, compute steered logits on base model.
      3) Select the next token by minimum perplexity under the thinking model (guardrail). If guardrail disabled, use the first provided candidate.
    """

    # Normalize special-case: [1] means all tokens
    if token_windows is not None and isinstance(token_windows, list) and len(token_windows) == 1:
        try:
            if int(token_windows[0]) == 1:
                token_windows = [0]
        except Exception:
            pass

    # Clone inputs so we do not modify the originals in-place
    base_output_ids = base_input_ids.clone()
    thinking_output_ids = thinking_input_ids.clone()
    del base_input_ids, thinking_input_ids
    torch.cuda.empty_cache()

    token_latent_info = [] if collect_details else None
    per_token_perplexity = [] if collect_details else None
    token_position = [] if collect_details else None
    steering_selection = []

    generated_tokens = 0
    ended_by_eos = False

    # Access bias vector if present
    bias_vector = steering_vectors.get("bias", None)

    pbar = None
    if show_progress and tqdm is not None:
        pbar = tqdm(total=max_new_tokens, desc="Hybrid tokens", leave=False)

    while generated_tokens < max_new_tokens:
        # 1) THINKING MODEL — derive steering vector from current position
        with torch.inference_mode():
            with thinking_model.trace(thinking_output_ids) as tracer:
                activation_curr = thinking_model.model.layers[sae_layer].output[0][0, -1, :].save()

        activation_curr = activation_curr.detach().clone()
        latent_acts = sae.encoder(activation_curr.to(torch.float32) - sae.b_dec)
        del activation_curr
        torch.cuda.empty_cache()

        latent_id = torch.argmax(latent_acts).item()
        latent_title = latent_descriptions[latent_id]["title"]
        latent_key = latent_descriptions[latent_id]["key"]
        activation_value = latent_acts[latent_id].item()
        steering_vector = steering_vectors[latent_key]
        del latent_acts
        torch.cuda.empty_cache()

        if verbose and (generated_tokens % 20 == 0):
            print(f"Token {generated_tokens}: latent={latent_title} (value={activation_value:.3f})")

        # 2) BASE MODEL — build candidate tokens across (coef, window)
        candidate_tokens = []
        # Precompute shapes and vectors outside tracing context to avoid Proxy conditionals
        hidden_size_expected = int(getattr(base_model.config, "hidden_size", 0))
        bias_vec = (
            bias_vector
            if (
                bias_vector is not None
                and hasattr(bias_vector, "shape")
                and bias_vector.shape[-1] == hidden_size_expected
            )
            else None
        )
        steer_vec = (
            steering_vector
            if (
                hasattr(steering_vector, "shape")
                and steering_vector.shape[-1] == hidden_size_expected
            )
            else None
        )
        # First compute the unsteered base token (save only last-step logits, then move to CPU)
        with torch.inference_mode():
            with base_model.trace(base_output_ids) as tracer:
                _last_logits_unsteered = base_model.lm_head.output[0, -1].save()
        _last_logits_unsteered = _last_logits_unsteered.detach().to("cpu")
        # Greedy token for disagreement gating (deterministic)
        base_pred_tok = int(torch.argmax(_last_logits_unsteered).item())
        # Actual next-token candidate according to chosen temperature
        tok_unsteered, tok_unsteered_str = get_token_and_string(
            _last_logits_unsteered,
            temperature,
            base_tokenizer,
            base_output_ids,
        )
        del _last_logits_unsteered

        # Compute thinking model logits at current position (for gating and guardrail)
        with torch.inference_mode():
            with thinking_model.trace(thinking_output_ids) as tracer:
                last_logits_thinking = thinking_model.lm_head.output[0, -1].save()
        last_logits_thinking = last_logits_thinking.detach().to("cpu")
        thinking_pred_tok = int(torch.argmax(last_logits_thinking).item())

        # Decide whether to perform full steering based on base vs thinking disagreement
        perform_steering = True
        if disagreement_only and thinking_pred_tok == base_pred_tok:
            perform_steering = False

        if not perform_steering:
            # Choose unsteered token and compute its perplexity for logging
            p = get_perplexity(tok_unsteered_str, last_logits_thinking, thinking_model)
            next_tok = tok_unsteered
            next_tok_str = tok_unsteered_str
            token_perpl = p
            chosen = "unsteered"
            chosen_coef = None
            chosen_window = None
            del candidate_tokens
            del last_logits_thinking
        else:
            # Build candidate tokens; compute initial steered candidate after gating
            if use_perplexity_guardrail:
                coef_list = coefficients if (coefficients is not None and len(coefficients) > 0) else [coefficient]
                win_list = token_windows if (token_windows is not None and len(token_windows) > 0) else [-1]
                c0 = float(coef_list[0])
                w0 = int(win_list[0])
            else:
                c0 = float((coefficients[0] if (coefficients is not None and len(coefficients) > 0) else coefficient))
                w0 = int((token_windows[0] if (token_windows is not None and len(token_windows) > 0) else -1))
            window_size0 = abs(int(w0)) if int(w0) != 0 else 0
            with torch.inference_mode():
                with base_model.trace(base_output_ids) as tracer:
                    layer_out0 = base_model.model.layers[steering_layer].output[0]
                    if window_size0 > 0:
                        if bias_vec is not None:
                            layer_out0[:, -window_size0:, :] += c0 * bias_vec
                        if steer_vec is not None:
                            layer_out0[:, -window_size0:, :] += c0 * steer_vec
                    else:
                        if bias_vec is not None:
                            layer_out0[:, :, :] += c0 * bias_vec
                        if steer_vec is not None:
                            layer_out0[:, :, :] += c0 * steer_vec
                    _last_logits_steered0 = base_model.lm_head.output[0, -1].save()
            # Initial steered candidate
            _last_logits_steered0 = _last_logits_steered0.detach().to("cpu")
            tok_steered0, tok_steered0_str = get_token_and_string(
                _last_logits_steered0,
                temperature,
                base_tokenizer,
                base_output_ids,
            )
            del _last_logits_steered0

            # Seed candidate list
            candidate_tokens.append({
                "type": "steered",
                "coef": float(c0),
                "window": int(w0),
                "tok": tok_steered0,
                "tok_str": tok_steered0_str,
            })
            if use_perplexity_guardrail:
                for coef in coef_list:
                    for win in win_list:
                        if float(coef) == c0 and int(win) == w0:
                            continue
                        window_size = abs(int(win)) if int(win) != 0 else 0
                        with torch.inference_mode():
                            with base_model.trace(base_output_ids) as tracer:
                                layer_out = base_model.model.layers[steering_layer].output[0]
                                if window_size > 0:
                                    if bias_vec is not None:
                                        layer_out[:, -window_size:, :] += coef * bias_vec
                                    if steer_vec is not None:
                                        layer_out[:, -window_size:, :] += coef * steer_vec
                                else:
                                    if bias_vec is not None:
                                        layer_out[:, :, :] += coef * bias_vec
                                    if steer_vec is not None:
                                        layer_out[:, :, :] += coef * steer_vec
                                _last_logits_steered_c = base_model.lm_head.output[0, -1].save()
                        _last_logits_steered_c = _last_logits_steered_c.detach().to("cpu")
                        tok_c, tok_c_str = get_token_and_string(
                            _last_logits_steered_c,
                            temperature,
                            base_tokenizer,
                            base_output_ids,
                        )
                        del _last_logits_steered_c
                        candidate_tokens.append({
                            "type": "steered",
                            "coef": float(coef),
                            "window": int(win),
                            "tok": tok_c,
                            "tok_str": tok_c_str,
                        })
            else:
                # No guardrail; use the initial steered candidate directly
                p = get_perplexity(tok_steered0_str, last_logits_thinking, thinking_model)
                next_tok = tok_steered0
                next_tok_str = tok_steered0_str
                token_perpl = p
                chosen = "steered"
                chosen_coef = float(c0)
                chosen_window = int(w0)
                del candidate_tokens
                del last_logits_thinking

        # 3) Evaluate perplexity of each candidate under thinking model and pick best (if steering performed and guardrail enabled)
        if perform_steering and use_perplexity_guardrail:
            best = None  # tuple(perplexity, index)
            for idx, cand in enumerate(candidate_tokens):
                p = get_perplexity(cand["tok_str"], last_logits_thinking, thinking_model)
                cand["perplexity"] = p
                if best is None or p < best[0]:
                    best = (p, idx)
            assert best is not None
            chosen_cand = candidate_tokens[best[1]]
            next_tok = chosen_cand["tok"]
            next_tok_str = chosen_cand["tok_str"]
            token_perpl = chosen_cand["perplexity"]
            chosen = chosen_cand["type"]
            chosen_coef = chosen_cand.get("coef", None)
            chosen_window = chosen_cand.get("window", None)
            del candidate_tokens
            del last_logits_thinking
            torch.cuda.empty_cache()


        # 4) Append chosen token to both sequences
        base_tok_ids = base_tokenizer.encode(
            next_tok_str, return_tensors="pt", add_special_tokens=False
        ).to(base_model.device).to(torch.long)
        thinking_tok_ids = thinking_model.tokenizer.encode(
            next_tok_str, return_tensors="pt", add_special_tokens=False
        ).to(thinking_model.device).to(torch.long)

        base_output_ids = torch.cat([base_output_ids, base_tok_ids], dim=1)
        thinking_output_ids = torch.cat([thinking_output_ids, thinking_tok_ids], dim=1)
        del base_tok_ids, thinking_tok_ids
        torch.cuda.empty_cache()

        # 5) Book-keeping
        steering_selection.append(chosen)
        if collect_details:
            per_token_perplexity.append(token_perpl)
            token_position.append(len(token_latent_info))
            token_latent_info.append(
                {
                    "token": next_tok_str,
                    "latent_id": latent_id if chosen == "steered" else None,
                    "latent_title": latent_title if chosen == "steered" else "No Steering",
                    "latent_key": latent_key if chosen == "steered" else None,
                    "activation_value": activation_value if chosen == "steered" else 0.0,
                    "perplexity": token_perpl,
                    "coefficient": chosen_coef if chosen == "steered" else None,
                    "window": chosen_window if chosen == "steered" else None,
                    "selection": chosen,
                    "future_token": None,
                }
            )

        generated_tokens += 1

        if pbar is not None:
            pbar.update(1)

        # Periodic cleanup to reduce fragmentation
        if (generated_tokens % 8) == 0:
            gc.collect()
            torch.cuda.empty_cache()
        # Drop temporary references
        try:
            del bias_vec
            del steer_vec
        except Exception:
            pass

        # Stop if EOS
        if next_tok == base_tokenizer.eos_token_id:
            ended_by_eos = True
            break

    # Final cleanup
    if pbar is not None:
        pbar.close()
    try:
        del steering_vector
    except Exception:
        pass
    try:
        del bias_vector
    except Exception:
        pass
    gc.collect()
    torch.cuda.empty_cache()

    return (
        base_output_ids,
        (token_latent_info if collect_details else []),
        (per_token_perplexity if collect_details else []),
        (token_position if collect_details else []),
        steering_selection,
        ended_by_eos,
    )

def load_steering_vectors(model_id, thinking_model_id, sae_layer, n_clusters):
    """Load steering vectors for the specific base model from train_vectors output.

    This filters vector files by the provided base-model `model_id` (e.g., "qwen2.5-32b"),
    avoiding collisions with vectors trained for other architectures (e.g., Llama),
    which can cause hidden-size mismatches.
    """
    hyperparams_dir_abs = os.path.join(os.path.dirname(__file__), "../train-vectors/results/vars/hyperparams")
    vectors_dir_abs = os.path.join(os.path.dirname(__file__), "../train-vectors/results/vars/optimized_vectors")

    # Build a mapping {category (e.g., "idx7" or "bias"): tensor} only for this model_id
    model_specific_vectors = {}
    try:
        for fname in os.listdir(hyperparams_dir_abs):
            if not fname.startswith(f"steering_vector_hyperparams_{model_id}_"):
                continue
            hp_path = os.path.join(hyperparams_dir_abs, fname)
            try:
                with open(hp_path, "r") as f:
                    hp = json.load(f)
                category = hp.get("category")  # "idxN" or "bias"
                if not category:
                    continue
                # Derive the matching vector file name
                idx_stub = fname.split(f"steering_vector_hyperparams_{model_id}_", 1)[1].rsplit(".json", 1)[0]
                vec_path = os.path.join(vectors_dir_abs, f"{model_id}_{idx_stub}.pt")
                if not os.path.exists(vec_path):
                    continue
                vec_obj = torch.load(vec_path, map_location="cpu")
                if isinstance(vec_obj, dict):
                    vector_tensor = vec_obj.get(category)
                    if vector_tensor is None and len(vec_obj) == 1:
                        # Fallback for older files: single-entry dict
                        vector_tensor = next(iter(vec_obj.values()))
                else:
                    vector_tensor = vec_obj
                if vector_tensor is None:
                    continue
                model_specific_vectors[category] = vector_tensor
            except Exception:
                continue
    except FileNotFoundError:
        pass

    # Map SAE latent keys ("idxN") to the corresponding vectors
    descriptions = get_latent_descriptions(thinking_model_id, sae_layer, n_clusters)
    steering_vectors = {}
    for _, desc in descriptions.items():
        latent_key = desc.get("key", "")  # e.g., "idx7"
        if not latent_key:
            continue
        key = latent_key.lower().replace(" ", "-")
        if key in model_specific_vectors:
            steering_vectors[key] = model_specific_vectors[key]

    # Include general bias vector (if present for this model)
    if "bias" in model_specific_vectors and "bias" not in steering_vectors:
        steering_vectors["bias"] = model_specific_vectors["bias"]
    else:
        print("No bias vector found")

    print(steering_vectors.keys())

    return steering_vectors

def generate_latent_colors(latent_descriptions):
    colors = {}
    unique_latents = set([desc["title"] for desc in latent_descriptions.values()])
    num_colors = len(unique_latents)
    for i, latent_title in enumerate(unique_latents):
        hue = i / num_colors
        saturation = 0.7
        value = 0.9
        r, g, b = colorsys.hsv_to_rgb(hue, saturation, value)
        hex_color = "#{:02x}{:02x}{:02x}".format(int(r*255), int(g*255), int(b*255))
        colors[latent_title] = hex_color
    colors["Cold Start"] = "#808080"
    colors["Fallback"] = "#CCCCCC"
    colors["No Steering"] = "#111111"
    return colors

def visualize_generation_results(token_latent_info, steering_selection, per_token_perplexity, token_position, latent_colors):
    tokens = [info["token"] for info in token_latent_info]
    fig = plt.figure(figsize=(14, 10))
    plt.subplot(4, 1, 1)
    plt.axis('off')
    for i, info in enumerate(token_latent_info):
        token = info["token"].replace('$', '\\$')
        latent_title = info["latent_title"]
        color = latent_colors[latent_title]
        plt.text(i, 0, token, color=color, fontsize=10, ha='center')
    plt.xlim(-1, len(tokens))
    plt.title("Generated Text (Colored by Latent)")
    plt.subplot(4, 1, 2)
    plt.plot(token_position, per_token_perplexity, marker='o', linestyle='-', color='blue', alpha=0.7)
    plt.yscale('log')
    plt.title("Perplexity by Token Position")
    plt.subplot(4, 1, 3)
    choices = []
    for choice in steering_selection:
        if choice == "steered":
            choices.append(1)
        elif choice == "unsteered":
            choices.append(0)
        else:
            choices.append(-1)
    plt.imshow([choices], cmap="RdYlGn", aspect="auto", vmin=-1, vmax=1)
    escaped_tokens = [t.replace('$', '\\$') for t in tokens]
    plt.xticks(range(len(tokens)), escaped_tokens, rotation=90, fontsize=8)
    plt.yticks([])
    cbar = plt.colorbar(orientation="horizontal", pad=0.1, ticks=[-1, 0, 1])
    cbar.set_ticklabels(["None", "No Steering", "Steering Used"])
    plt.title("Steering Selection")
    plt.subplot(4, 1, 4)
    latent_titles = [info.get("latent_title", "None") for info in token_latent_info]
    unique_latents = sorted(set(latent_titles))
    latent_map = {latent: i for i, latent in enumerate(unique_latents)}
    latent_values = [latent_map[latent] for latent in latent_titles]
    plt.imshow([latent_values], cmap="tab20", aspect="auto")
    plt.xticks(range(len(tokens)), escaped_tokens, rotation=90, fontsize=8)
    plt.yticks([])
    handles = []
    for latent in unique_latents:
        color = latent_colors.get(latent, "#000000")
        patch = Rectangle((0, 0), 1, 1, fc=color)
        handles.append(patch)
    plt.legend(handles, unique_latents, loc='upper center', bbox_to_anchor=(0.5, -0.15), 
               ncol=min(5, len(unique_latents)), frameon=False)
    plt.title("Latent Features")
    plt.tight_layout()
    plt.show()

def load_models_and_sae(args):
    thinking_model_id = args.thinking_model.split('/')[-1].lower()
    base_model_id = args.base_model.split('/')[-1].lower()
    print(f"Loading models {args.thinking_model} and {args.base_model}...")
    thinking_model, thinking_tokenizer = load_model(model_name=args.thinking_model)
    thinking_model.tokenizer = thinking_tokenizer
    if args.temperature > 0:
        thinking_model.generation_config.do_sample = True
    base_model, base_tokenizer = load_model(model_name=args.base_model)
    if args.temperature > 0:
        base_model.generation_config.do_sample = True
    print(f"Loading SAE for model {thinking_model_id}, layer {args.sae_layer}...")
    sae, _ = load_sae(thinking_model_id, args.sae_layer, args.n_clusters)
    sae = sae.to(thinking_model.device)
    print(f"Loading steering vectors and layer effects...")
    descriptions = get_latent_descriptions(thinking_model_id, args.sae_layer, args.n_clusters)
    steering_vectors = load_steering_vectors(base_model_id, thinking_model_id, args.sae_layer, args.n_clusters)
    # Move steering vectors to base model device and dtype
    base_device = base_model.device
    base_dtype = next(base_model.parameters()).dtype if hasattr(base_model, "parameters") else torch.float32
    for k, v in list(steering_vectors.items()):
        if isinstance(v, torch.Tensor):
            steering_vectors[k] = v.to(device=base_device, dtype=base_dtype, non_blocking=True)
    return thinking_model, thinking_tokenizer, base_model, base_tokenizer, sae, steering_vectors, descriptions, thinking_model_id, base_model_id

def run_example(thinking_model, thinking_tokenizer, base_model, base_tokenizer, 
               sae, steering_vectors, descriptions, args, dataset):
    sample_idx = args.example_idx
    for i, item in enumerate(dataset):
        if i == sample_idx:
            if args.dataset == "gsm8k":
                example = {
                    "question": item["question"],
                    "answer": item["answer"]
                }
            elif args.dataset == "math500":
                example = {
                    "question": item["problem"],
                    "answer": item["answer"]
                }
            elif args.dataset == "aime":
                example = {
                    "question": item["problem"],
                    "answer": item["answer"]
                }
            break

    question = example["question"]
    answer = example["answer"]

    print("\n===== Example =====")
    print(f"Question: {question}")

    thinking_input_ids = thinking_tokenizer.apply_chat_template(
        [{"role": "user", "content": question}], 
        add_generation_prompt=True, 
        return_tensors="pt"
    ).to(thinking_model.device).to(torch.long)

    base_prompt = f"Task: Answer the question below. Explain your reasoning step by step.\n\n\n\nQuestion:\n{question}\n\nStep by step answer:\n"
    base_input_ids = base_tokenizer.encode(
        base_prompt,
        return_tensors="pt"
    ).to(base_model.device).to(torch.long)
    
    # Generate with thinking model
    print("\n===== Generating with Thinking Model =====")
    with thinking_model.generate(thinking_input_ids, max_new_tokens=args.max_thinking_tokens, temperature=args.temperature, pad_token_id=thinking_tokenizer.eos_token_id) as gen:
        thinking_outputs = thinking_model.generator.output.save()
    thinking_response = thinking_tokenizer.decode(thinking_outputs[0][len(thinking_input_ids[0]):], skip_special_tokens=True)
    print(thinking_response)
    
    # Generate with base model
    print("\n===== Generating with Base Model =====")
    base_input_with_cold_start, _, cold_start_text = prepare_cold_start(
        thinking_outputs,
        thinking_input_ids,
        base_input_ids,
        thinking_tokenizer=thinking_tokenizer,
        base_tokenizer=base_tokenizer,
        n_cold_start_tokens=args.n_cold_start_tokens,
    )
    
    with base_model.generate(base_input_with_cold_start, max_new_tokens=100, temperature=args.temperature, pad_token_id=base_tokenizer.eos_token_id) as gen:
        base_outputs = base_model.generator.output.save()
    base_response = f"{cold_start_text}{base_tokenizer.decode(base_outputs[0][len(base_input_with_cold_start[0]):], skip_special_tokens=True)}"
    print(base_response)
    
    # Clean up base model outputs
    del base_outputs
    torch.cuda.empty_cache()
    
    # Generate with hybrid approach
    print("\n===== Generating with Hybrid Approach (Token-Level) =====")
    base_input_with_cold_start, thinking_input_with_cold_start, cold_start_text = prepare_cold_start(
        thinking_outputs,
        thinking_input_ids,
        base_input_ids,
        thinking_tokenizer=thinking_tokenizer,
        base_tokenizer=base_tokenizer,
        n_cold_start_tokens=args.n_cold_start_tokens,
    )
    
    hybrid_output_ids, token_latent_info, per_token_perplexity, token_position, steering_selection, _ = hybrid_generate_token(
        thinking_model=thinking_model,
        base_model=base_model,
        base_tokenizer=base_tokenizer,
        thinking_input_ids=thinking_input_with_cold_start,
        base_input_ids=base_input_with_cold_start,
        max_new_tokens=args.max_new_tokens,
        steering_layer=args.steering_layer,
        sae_layer=args.sae_layer,
        sae=sae,
        steering_vectors=steering_vectors,
        latent_descriptions=descriptions,
        temperature=args.temperature,
        coefficient=(args.coefficients[0] if args.coefficients else 0.3),
        coefficients=args.coefficients,
        token_windows=args.token_windows,
        verbose=False,
        use_perplexity_guardrail=args.use_perplexity_guardrail,
        show_progress=args.show_progress,
        disagreement_only=(not args.disable_disagreement_only),
        collect_details=True,
    )
    hybrid_response = f"{cold_start_text}{base_tokenizer.decode(hybrid_output_ids[0][len(base_input_with_cold_start[0]):], skip_special_tokens=True)}"
    print(hybrid_response)
    
    # Clean up hybrid outputs
    del hybrid_output_ids
    torch.cuda.empty_cache()
    
    # Print correct answer for reference
    print("\n===== Correct Answer =====")
    print(answer)
    
    # Visualize results
    latent_colors = generate_latent_colors(descriptions)
    visualize_generation_results(token_latent_info, steering_selection, per_token_perplexity, token_position, latent_colors)
    
    # Clean up example-specific variables
    del latent_colors, per_token_perplexity, token_position
    
    return thinking_response, base_response, hybrid_response, token_latent_info, steering_selection

def clean_answer(text):
    return re.sub(r'\s+', ' ', text).strip()

def safe_chat_batch(prompts, model_name: str = "openai/gpt-4.1", max_tokens: int = 1024, **kwargs):
    import asyncio
    import concurrent.futures
    async def _run():
        return await chat_batch(
            prompts,
            model=model_name,
            max_tokens=max_tokens,
            **kwargs,
        )
    try:
        loop = asyncio.get_running_loop()
        def _thread_runner():
            return asyncio.run(_run())
        with concurrent.futures.ThreadPoolExecutor() as ex:
            fut = ex.submit(_thread_runner)
            return fut.result()
    except RuntimeError:
        return asyncio.run(_run())

def quick_judge_api_test():
    """Run a fast connectivity test against the judge chat API and print status.

    Uses a tiny prompt and small max_tokens to keep cost minimal. This does not
    affect program flow; it only reports readiness early.
    """
    test_prompt = "Reply with YES."
    try:
        responses = safe_chat_batch([test_prompt], model_name="openai/gpt-4.1", max_tokens=5)
        ok = isinstance(responses, (list, tuple)) and len(responses) > 0 and isinstance(responses[0], str)
        if ok:
            print("Judge API test: OK")
        else:
            print("Judge API test: FAILED (no response). Check OPENAI_API_KEY/OPENAI_PROJECT.")
    except Exception as e:
        print(f"Judge API test: FAILED ({e}). Check OPENAI_API_KEY/OPENAI_PROJECT.")

def _extract_final_numeric_value(text: str):
    """Extract the final numeric value from an answer string.

    Heuristics:
    - Prefer a '#### <answer>' marker if present (GSM8K style)
    - Otherwise, take the last occurrence of a number or fraction in the text
    Returns a Fraction if parseable, else None.
    """
    if not text:
        return None
    # Prefer GSM8K final marker
    m = re.search(r"####\s*([^\n#]+)", text)
    candidate = None
    if m:
        candidate = m.group(1).strip()
    else:
        # Find last fraction like a/b or last number (int or decimal)
        frac_matches = list(re.finditer(r"-?\d[\d,]*\s*/\s*-?\d[\d,]*", text))
        if frac_matches:
            candidate = frac_matches[-1].group(0)
        else:
            num_matches = list(re.finditer(r"-?\d[\d,]*(?:\.\d+)?", text))
            if num_matches:
                candidate = num_matches[-1].group(0)
    if candidate is None:
        return None
    candidate = candidate.strip()
    # Strip units/words after number for simple cases (e.g., "7 dozen")
    candidate = re.match(r"(-?\d[\d,]*(?:\.\d+)?(?:\s*/\s*-?\d[\d,]*)?)", candidate).group(1) if re.match(r"(-?\d[\d,]*(?:\.\d+)?(?:\s*/\s*-?\d[\d,]*)?)", candidate) else candidate
    # Remove commas
    candidate = candidate.replace(",", "")
    try:
        if "/" in candidate:
            a, b = [p.strip() for p in candidate.split("/", 1)]
            return Fraction(a) / Fraction(b)
        return Fraction(candidate)
    except Exception:
        return None

def _local_numeric_compare(model_answer: str, correct_answer: str):
    """Deterministically compare final numeric answers without external APIs."""
    gold = _extract_final_numeric_value(correct_answer)
    pred = _extract_final_numeric_value(model_answer)
    if gold is None or pred is None:
        return False
    return gold == pred

def evaluate_answer(model_answer, correct_answer, question, model_name):
    prompt = f"""Please evaluate whether the following answer to a math problem is correct.

Question: {question}

Correct answer: {correct_answer}

Model's answer: {model_answer}

First, extract the final numerical answer from both the correct answer and model's answer. 
Then determine if the model's final numerical answer is equivalent to the correct final numerical answer.
Just answer YES if the model's answer is correct, or NO if it's incorrect. Nothing else.
"""
    
    # Try remote judge first
    try:
        response_list = safe_chat_batch([prompt], model_name="openai/gpt-4.1", max_tokens=100)
        if isinstance(response_list, (list, tuple)) and len(response_list) > 0 and isinstance(response_list[0], str):
            response = response_list[0]
            is_correct = "yes" in response.lower()
            print(f"{model_name} evaluated as: {response}")
            return is_correct, response
        else:
            print("Judge API returned no response; falling back to local numeric comparison.")
    except Exception as e:
        print(f"Judge API failed: {e}. Falling back to local numeric comparison.")

    # Fallback: local numeric comparison
    local_ok = _local_numeric_compare(model_answer, correct_answer)
    response = "YES" if local_ok else "NO"
    print(f"{model_name} evaluated locally as: {response}")
    return local_ok, response

def append_rolling_result(record: dict, args, base_model_id: str):
    os.makedirs(f"{args.results_dir}/rolling", exist_ok=True)
    path = f"{args.results_dir}/rolling/rolling_{base_model_id}_{args.dataset}.jsonl"
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")

def analyze_hybrid_stats(token_latent_info, steering_selection):
    steered_count = steering_selection.count("steered")
    unsteered_count = steering_selection.count("unsteered")
    total = steered_count + unsteered_count
    no_steering_fraction = unsteered_count / total if total > 0 else 0
    latent_counts = Counter()
    for info in token_latent_info:
        if info["latent_title"] != "No Steering":
            latent_counts[info["latent_title"]] += 1
    latent_percentages = {}
    if steered_count > 0:
        for latent, count in latent_counts.items():
            latent_percentages[latent] = (count / steered_count) * 100
    steering_stats = {
        "steered_count": steered_count,
        "unsteered_count": unsteered_count,
        "total_tokens": total,
        "steering_fraction": steered_count / total if total > 0 else 0,
        "no_steering_fraction": no_steering_fraction
    }
    return no_steering_fraction, latent_counts, latent_percentages, steering_stats

def save_detailed_results(results, args, thinking_model_id, base_model_id):
    os.makedirs(f"{args.results_dir}/detailed", exist_ok=True)
    filename = f"{args.results_dir}/detailed/hybrid_stats_{base_model_id}_{args.dataset}.json"
    avg_steering_stats = {
        "steered_count": sum(stat["steered_count"] for stat in results["steering_stats"]) / len(results["steering_stats"]),
        "unsteered_count": sum(stat["unsteered_count"] for stat in results["steering_stats"]) / len(results["steering_stats"]),
        "total_tokens": sum(stat["total_tokens"] for stat in results["steering_stats"]) / len(results["steering_stats"]),
        "steering_fraction": sum(stat["steering_fraction"] for stat in results["steering_stats"]) / len(results["steering_stats"]),
        "no_steering_fraction": sum(stat["no_steering_fraction"] for stat in results["steering_stats"]) / len(results["steering_stats"])
    }
    detailed_data = {
        "metadata": {
            "base_model": args.base_model,
            "thinking_model": args.thinking_model,
            "dataset": args.dataset,
            "temperature": args.temperature,
            "coefficients": args.coefficients,
            "n_tasks": len(results["questions"])
        },
        "answer_lengths": {
            "base_model": results["base_lengths"],
            "thinking_model": results["thinking_lengths"],
            "hybrid_model": results["hybrid_lengths"],
            "avg_base": sum(results["base_lengths"]) / len(results["base_lengths"]) if results["base_lengths"] else 0,
            "avg_thinking": sum(results["thinking_lengths"]) / len(results["thinking_lengths"]) if results["thinking_lengths"] else 0,
            "avg_hybrid": sum(results["hybrid_lengths"]) / len(results["hybrid_lengths"]) if results["hybrid_lengths"] else 0
        },
        "steering_stats": {
            "no_steering_fractions": results["no_steering_fractions"],
            "avg_no_steering": sum(results["no_steering_fractions"]) / len(results["no_steering_fractions"]) if results["no_steering_fractions"] else 0,
            "detailed_stats": results["steering_stats"],
            "average_stats": avg_steering_stats
        },
        "latent_usage": results["latent_usage"],
        "accuracies": {
            "base_model": results["base_correct"] / len(results["questions"]) * 100,
            "thinking_model": results["thinking_correct"] / len(results["questions"]) * 100,
            "hybrid_model": results["hybrid_correct"] / len(results["questions"]) * 100
        }
    }
    with open(filename, 'w') as f:
        json.dump(detailed_data, f, indent=2)
    print(f"Detailed results saved to {filename}")
    return detailed_data

def run_evaluation(thinking_model, thinking_tokenizer, base_model, base_tokenizer, 
                  sae, steering_vectors, descriptions, args, dataset, thinking_model_id, base_model_id):

    def clear_gpu_memory():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    results = {
        "base_correct": 0,
        "thinking_correct": 0,
        "hybrid_correct": 0,
        "base_answers": [],
        "thinking_answers": [],
        "hybrid_answers": [],
        "questions": [],
        "correct_answers": [],
        "base_lengths": [],
        "thinking_lengths": [],
        "hybrid_lengths": [],
        "thinking_eos": [],
        "base_eos": [],
        "hybrid_eos": [],
        "no_steering_fractions": [],
        "latent_usage": [],
        "steering_stats": [],
        "token_latent_info": [],
        "steering_selection": []
    }
    
    task_counter = 0
    for i, item in enumerate(dataset):
        if i < args.eval_start_idx:
            continue
        if task_counter >= args.n_tasks:
            break
        task_counter += 1
        print(f"\n===== Processing Task {task_counter}/{args.n_tasks} =====")
        
        if args.dataset == "gsm8k":
            question = item["question"]
            correct_answer = item["answer"]
        elif args.dataset == "aime":
            question = item["problem"]
            correct_answer = item["answer"]
        elif args.dataset == "math500":
            question = item["problem"]
            correct_answer = item["answer"]
        
        print(f"Question: {question[:100]}...")
        print(f"Correct answer: {correct_answer}")
        results["questions"].append(question)
        results["correct_answers"].append(correct_answer)
        
        thinking_input_ids = thinking_tokenizer.apply_chat_template(
            [{"role": "user", "content": question}], 
            add_generation_prompt=True, 
            return_tensors="pt"
        ).to(thinking_model.device).to(torch.long)

        base_prompt = f"Task: Answer the question below. Explain your reasoning step by step.\n\n\n\nQuestion:\n{question}\n\nStep by step answer:\n"
        base_input_ids = base_tokenizer.encode(
            base_prompt,
            return_tensors="pt"
        ).to(base_model.device).to(torch.long)
        
        # Thinking model
        print("Generating with Thinking Model...")
        clear_gpu_memory()
        with thinking_model.generate(thinking_input_ids, max_new_tokens=args.max_thinking_tokens, temperature=args.temperature, pad_token_id=thinking_tokenizer.eos_token_id) as gen:
            thinking_outputs = thinking_model.generator.output.save()
        thinking_response = thinking_tokenizer.decode(thinking_outputs[0][len(thinking_input_ids[0]):], skip_special_tokens=True)
        results["thinking_answers"].append(thinking_response)
        results["thinking_lengths"].append(len(thinking_response.split()))
        # Track EOS termination
        try:
            thinking_eos_end = bool(int(thinking_outputs[0, -1].item()) == int(thinking_tokenizer.eos_token_id))
        except Exception:
            thinking_eos_end = False
        results["thinking_eos"].append(thinking_eos_end)

        # Base model
        print("Generating with Base Model...")
        clear_gpu_memory()
        base_input_with_cold_start, _, cold_start_text = prepare_cold_start(
            thinking_outputs,
            thinking_input_ids,
            base_input_ids,
            thinking_tokenizer=thinking_tokenizer,
            base_tokenizer=base_tokenizer,
            n_cold_start_tokens=args.n_cold_start_tokens,
        )
        with base_model.generate(base_input_with_cold_start, max_new_tokens=args.max_new_tokens, temperature=args.temperature, pad_token_id=base_tokenizer.eos_token_id) as gen:
            base_outputs = base_model.generator.output.save()
        # Track EOS termination
        try:
            base_eos_end = bool(int(base_outputs[0, -1].item()) == int(base_tokenizer.eos_token_id))
        except Exception:
            base_eos_end = False
        base_response = f"{cold_start_text}{base_tokenizer.decode(base_outputs[0][len(base_input_with_cold_start[0]):], skip_special_tokens=True)}"
        del base_outputs, base_input_with_cold_start
        clear_gpu_memory()
        results["base_answers"].append(base_response)
        results["base_lengths"].append(len(base_response.split()))
        results["base_eos"].append(base_eos_end)
        
        # Hybrid token-level
        print("Generating with Hybrid Approach (Token-Level)...")
        clear_gpu_memory()
        base_input_with_cold_start, thinking_input_with_cold_start, cold_start_text = prepare_cold_start(
            thinking_outputs,
            thinking_input_ids,
            base_input_ids,
            thinking_tokenizer=thinking_tokenizer,
            base_tokenizer=base_tokenizer,
            n_cold_start_tokens=args.n_cold_start_tokens,
        )
        hybrid_output_ids, token_latent_info, per_token_perplexity, token_position, steering_selection, hybrid_eos_end = hybrid_generate_token(
            thinking_model=thinking_model,
            base_model=base_model,
            base_tokenizer=base_tokenizer,
            thinking_input_ids=thinking_input_with_cold_start,
            base_input_ids=base_input_with_cold_start,
            max_new_tokens=args.max_new_tokens,
            steering_layer=args.steering_layer,
            sae_layer=args.sae_layer,
            sae=sae,
            steering_vectors=steering_vectors,
            latent_descriptions=descriptions,
            temperature=args.temperature,
            coefficient=(args.coefficients[0] if args.coefficients else 0.3),
            coefficients=args.coefficients,
            token_windows=args.token_windows,
            verbose=False,
            use_perplexity_guardrail=args.use_perplexity_guardrail,
            show_progress=args.show_progress,
            disagreement_only=(not args.disable_disagreement_only),
            collect_details=bool(args.store_per_token_details),
        )
        hybrid_response = f"{cold_start_text}{base_tokenizer.decode(hybrid_output_ids[0][len(base_input_with_cold_start[0]):], skip_special_tokens=True)}"
        del hybrid_output_ids, base_input_with_cold_start, thinking_input_with_cold_start
        clear_gpu_memory()
        print(hybrid_response)
        results["hybrid_answers"].append(hybrid_response)
        results["hybrid_lengths"].append(len(hybrid_response.split()))
        results["hybrid_eos"].append(bool(hybrid_eos_end))
        
        # Store token latent info and steering selection (optional to reduce RAM)
        if args.store_per_token_details:
            results["token_latent_info"].append(token_latent_info)
            results["steering_selection"].append(steering_selection)
        
        # Analyze and store steering statistics
        no_steering_fraction, latent_counts, latent_percentages, steering_stats = analyze_hybrid_stats(token_latent_info, steering_selection)
        results["no_steering_fractions"].append(no_steering_fraction)
        results["latent_usage"].append(latent_percentages)
        results["steering_stats"].append(steering_stats)
        
        # Clean and evaluate answers
        clean_thinking_answer = clean_answer(thinking_response)
        clean_base_answer = clean_answer(base_response)
        clean_hybrid_answer = clean_answer(hybrid_response)
        
        # Evaluate answers
        print("\nEvaluating answers...")
        thinking_correct, thinking_judge_raw = evaluate_answer(clean_thinking_answer, correct_answer, question, "Thinking Model")
        base_correct, base_judge_raw = evaluate_answer(clean_base_answer, correct_answer, question, "Base Model")
        hybrid_correct, hybrid_judge_raw = evaluate_answer(clean_hybrid_answer, correct_answer, question, "Hybrid Model")
        
        if thinking_correct:
            results["thinking_correct"] += 1
        if base_correct:
            results["base_correct"] += 1
        if hybrid_correct:
            results["hybrid_correct"] += 1
        
        # Rolling save for this task
        rolling_record = {
            "ts": time.time(),
            "dataset": args.dataset,
            "question": question,
            "gold_answer": correct_answer,
            "answers": {
                "thinking": thinking_response,
                "base": base_response,
                "hybrid": hybrid_response,
            },
            "judges": {
                "thinking": {"correct": bool(thinking_correct), "raw": thinking_judge_raw},
                "base": {"correct": bool(base_correct), "raw": base_judge_raw},
                "hybrid": {"correct": bool(hybrid_correct), "raw": hybrid_judge_raw},
            },
            "hybrid_details": {
                "per_token": token_latent_info,
                "steering_selection": steering_selection,
                "coefficients": args.coefficients,
                "token_windows": args.token_windows,
                "sae_layer": args.sae_layer,
                "steering_layer": args.steering_layer,
            },
        }
        append_rolling_result(rolling_record, args, base_model_id)

        # Print current results
        print(f"\nCurrent Results after {task_counter} tasks:")
        print(f"Thinking Model: {results['thinking_correct']}/{task_counter} correct ({results['thinking_correct']/task_counter*100:.1f}%)")
        print(f"Base Model: {results['base_correct']}/{task_counter} correct ({results['base_correct']/task_counter*100:.1f}%)")
        print(f"Hybrid Model: {results['hybrid_correct']}/{task_counter} correct ({results['hybrid_correct']/task_counter*100:.1f}%)")
        # Concise gap recovery and EOS summary so far
        so_far = task_counter
        base_acc_now = results['base_correct'] / so_far * 100
        thinking_acc_now = results['thinking_correct'] / so_far * 100
        hybrid_acc_now = results['hybrid_correct'] / so_far * 100
        gap_now = abs(thinking_acc_now - base_acc_now)
        if gap_now > 0:
            recovered_now = (hybrid_acc_now - min(base_acc_now, thinking_acc_now)) / gap_now
            print(f"Gap recovered by hybrid: {max(0.0, recovered_now)*100:.1f}% of |Thinking-Base|")
        else:
            print("Gap recovered by hybrid: n/a")
        base_eos_pct_now = (sum(results['base_eos']) / so_far) * 100 if so_far > 0 else 0.0
        thinking_eos_pct_now = (sum(results['thinking_eos']) / so_far) * 100 if so_far > 0 else 0.0
        hybrid_eos_pct_now = (sum(results['hybrid_eos']) / so_far) * 100 if so_far > 0 else 0.0
        print(f"EOS endings (%): base {base_eos_pct_now:.1f}, thinking {thinking_eos_pct_now:.1f}, hybrid {hybrid_eos_pct_now:.1f}")
        
        # Clean up to prevent memory leaks
        del thinking_input_ids, base_input_ids, thinking_outputs
        try:
            del token_latent_info, per_token_perplexity, token_position
        except Exception:
            pass
        del steering_selection
        del thinking_response, base_response, hybrid_response
        del clean_thinking_answer, clean_base_answer, clean_hybrid_answer
        del latent_counts, latent_percentages, steering_stats
        del cold_start_text
        torch.cuda.empty_cache()
        gc.collect()

    thinking_accuracy = results["thinking_correct"] / task_counter * 100
    base_accuracy = results["base_correct"] / task_counter * 100
    hybrid_accuracy = results["hybrid_correct"] / task_counter * 100

    print("\n===== Final Results =====")
    print(f"Thinking Model: {results['thinking_correct']}/{task_counter} correct ({thinking_accuracy:.1f}%)")
    print(f"Base Model: {results['base_correct']}/{task_counter} correct ({base_accuracy:.1f}%)")
    print(f"Hybrid Model: {results['hybrid_correct']}/{task_counter} correct ({hybrid_accuracy:.1f}%)")
    # Concise end-of-run gap and EOS summary
    gap_final = abs(thinking_accuracy - base_accuracy)
    if gap_final > 0:
        recovered_final = (hybrid_accuracy - min(base_accuracy, thinking_accuracy)) / gap_final
        print(f"Gap recovered by hybrid: {max(0.0, recovered_final)*100:.1f}% of |Thinking-Base|")
    else:
        print("Gap recovered by hybrid: n/a")
    base_eos_pct = (sum(results['base_eos']) / task_counter) * 100 if task_counter > 0 else 0.0
    thinking_eos_pct = (sum(results['thinking_eos']) / task_counter) * 100 if task_counter > 0 else 0.0
    hybrid_eos_pct = (sum(results['hybrid_eos']) / task_counter) * 100 if task_counter > 0 else 0.0
    print(f"EOS endings (%): base {base_eos_pct:.1f}, thinking {thinking_eos_pct:.1f}, hybrid {hybrid_eos_pct:.1f}")

    plt.figure(figsize=(10, 6))
    model_names = ["Base", "Thinking", "Hybrid"]
    accuracies = [base_accuracy, thinking_accuracy, hybrid_accuracy]
    colors = ["#3498db", "#e74c3c", "#2ecc71"]
    plt.bar(model_names, accuracies, color=colors)
    plt.title(f"Model Accuracy on {task_counter} {args.dataset} Tasks")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    for i, accuracy in enumerate(accuracies):
        plt.text(i, accuracy + 2, f"{accuracy:.1f}%", ha='center')
    plt.tight_layout()
    plt.savefig(f"{args.results_dir}/accuracy_{base_model_id}_{args.dataset}.png")
    plt.show()

    benchmark_data = {
        "metadata": {
            "base_model": args.base_model,
            "thinking_model": args.thinking_model,
            "n_tasks": task_counter,
        },
        "results": {
            "accuracy": {
                "base_model": base_accuracy,
                "thinking_model": thinking_accuracy,
                "hybrid_model": hybrid_accuracy
            },
            "correct_count": {
                "base_model": results["base_correct"],
                "thinking_model": results["thinking_correct"],
                "hybrid_model": results["hybrid_correct"]
            }
        },
        "tasks": []
    }
    for i in range(task_counter):
        task_data = {
            "question": results["questions"][i],
            "correct_answer": results["correct_answers"][i],
            "model_answers": {
                "base_model": results["base_answers"][i],
                "thinking_model": results["thinking_answers"][i],
                "hybrid_model": results["hybrid_answers"][i]
            }
        }
        benchmark_data["tasks"].append(task_data)
    json_path = f"{args.results_dir}/benchmark_results_{base_model_id}_{args.dataset}.json"
    with open(json_path, 'w') as f:
        json.dump(benchmark_data, f, indent=2)
    print(f"Benchmark results saved to {json_path}")
    return results

# Get command line arguments
args = parse_args()

# Create results directory if it doesn't exist
os.makedirs(args.results_dir, exist_ok=True)
os.makedirs(f"{args.results_dir}/vars", exist_ok=True)

# Quick judge API connectivity test
quick_judge_api_test()

# %% Load dataset
print(f"Loading {args.dataset} dataset...")
if args.dataset == 'gsm8k':
    dataset = load_dataset("openai/gsm8k", "main")["test"]  # type: ignore
elif args.dataset == "aime":
    dataset = load_dataset("HuggingFaceH4/aime_2024")["train"]  # type: ignore
elif args.dataset == "math500":
    dataset = load_dataset("HuggingFaceH4/MATH-500")["test"]  # type: ignore

# %% Load models and SAE
thinking_model, thinking_tokenizer, base_model, base_tokenizer, sae, steering_vectors, descriptions, thinking_model_id, base_model_id = load_models_and_sae(args)

# %% Run an example (optional)
if args.run_example:
    print("\n===== Running Example =====")
    thinking_response, base_response, hybrid_response, token_latent_info, steering_selection = run_example(
        thinking_model, thinking_tokenizer, base_model, base_tokenizer, 
        sae, steering_vectors, descriptions, args, dataset
    )

    # Analyze example stats
    no_steering_fraction, latent_counts, latent_percentages, steering_stats = analyze_hybrid_stats(token_latent_info, steering_selection)

    print("\n===== Example Statistics =====")
    print(f"No-steering fraction: {no_steering_fraction:.2f}")
    print("Latent usage (top 5):")
    for latent, count in latent_counts.most_common(5):
        print(f"  {latent}: {count} tokens ({latent_percentages[latent]:.1f}%)")

# %% Run evaluation
print("\n===== Running Evaluation =====")
results = run_evaluation(
    thinking_model, thinking_tokenizer, base_model, base_tokenizer,
    sae, steering_vectors, descriptions, args, dataset, thinking_model_id, base_model_id
)

# Save detailed results
detailed_data = save_detailed_results(results, args, thinking_model_id, base_model_id)

# Plot additional statistics
if results["no_steering_fractions"]:
    plt.figure(figsize=(10, 6))
    plt.hist(results["no_steering_fractions"], bins=10, color="#2ecc71", alpha=0.7)
    plt.title("Distribution of No-Steering Fraction")
    plt.xlabel("Fraction of Tokens Using No Steering")
    plt.ylabel("Number of Tasks")
    plt.axvline(x=detailed_data["steering_stats"]["avg_no_steering"], color='red', linestyle='--', 
                label=f"Average: {detailed_data['steering_stats']['avg_no_steering']:.2f}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"{args.results_dir}/no_steering_distribution_{base_model_id}_{args.dataset}.png")
    plt.show()

# Clean up
print("Evaluation complete. Cleaning up...")
del thinking_model, base_model, sae
torch.cuda.empty_cache()
gc.collect()
