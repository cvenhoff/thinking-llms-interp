"""Compute diff-of-means steering vectors, layer attribution, and coefficient selection.

Pipeline:
  1. Load contrastive pairs from generate_pairs.py output.
  2. Compute steering vectors at ALL decoder layers (D+ vs D- via streaming means).
  3. Gradient-based layer attribution to find the best layer per category.
  4. Rescale vectors to mean-activation norm.
  5. Batched multi-steer coefficient sweep with auto-grading.
  6. Select best coefficient per category.
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
from typing import Optional

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
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
    p.add_argument("--pairs_file", type=str, required=True,
                   help="Training pairs JSON from generate_pairs.py")
    p.add_argument("--eval_pairs_file", type=str, default="",
                   help="Separate eval pairs JSON (unseen data)")
    p.add_argument("--save_dir", type=str, default="results/diff_of_means")
    p.add_argument("--n_eval_questions", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=8,
                   help="Batch size for activation forward passes")
    p.add_argument("--attrib_batch_size", type=int, default=0,
                   help="Attribution batch size (0 = batch_size // 4)")
    p.add_argument("--gen_batch_size", type=int, default=16)
    p.add_argument("--n_attribution_examples", type=int, default=30)
    p.add_argument("--steer_coeffs", type=str, default="0.5,1.0,1.5,2.0")
    p.add_argument("--eval_max_tokens", type=int, default=32)
    p.add_argument("--api_model", type=str, default="gpt-4.1")
    p.add_argument("--max_concurrent", type=int, default=40)
    p.add_argument("--min_layer_frac", type=float, default=0.0,
                   help="Exclude first X%% of layers for best-layer selection")
    p.add_argument("--skip_vectors", action="store_true")
    p.add_argument("--skip_attribution", action="store_true")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

CATEGORY_START_TAG = "[CATEGORY_START]"
CATEGORY_END_TAG = "[CATEGORY_END]"


def strip_markers(text: str) -> tuple[str, Optional[str], Optional[int]]:
    """Remove markers, return (cleaned_text, segment_content, seg_char_start)."""
    s = re.search(re.escape(CATEGORY_START_TAG), text)
    e = re.search(re.escape(CATEGORY_END_TAG), text)
    if s is None or e is None:
        return text, None, None
    seg = text[s.end():e.start()]
    cleaned = text[:s.start()] + seg + text[e.end():]
    return cleaned, seg, s.start()


def find_segment_token_positions(
    cleaned_text: str, seg_char_start: int, seg_char_end: int, tokenizer,
) -> Optional[dict]:
    """Map segment char offsets to token positions in marker-free text."""
    if seg_char_start is None or seg_char_start <= 0:
        return None
    char_to_tok = utils.get_char_to_token_map(cleaned_text, tokenizer)
    start_tok = char_to_tok.get(seg_char_start)
    if start_tok is None:
        return None
    end_tok = None
    if seg_char_end is not None and seg_char_end > seg_char_start:
        last_tok = char_to_tok.get(seg_char_end - 1)
        end_tok = (last_tok + 1) if last_tok is not None else None
    if end_tok is None:
        end_tok = len(tokenizer.encode(cleaned_text))
    return {"segment_start": start_tok, "segment_end": end_tok}


def _response_start_tok(prompt: str, tokenizer) -> int:
    return len(tokenizer.encode(prompt))


# ---------------------------------------------------------------------------
# Diff-of-means computation (streaming running averages, all layers)
# ---------------------------------------------------------------------------

def _hf_forward_layers(model, input_ids, attention_mask,
                       layers: list[int]) -> dict[int, torch.Tensor]:
    """Plain HF forward pass → hidden states for requested layers."""
    with torch.no_grad():
        out = model(input_ids=input_ids, attention_mask=attention_mask,
                    output_hidden_states=True)
    hidden = {l: out.hidden_states[l + 1] for l in layers}
    del out
    return hidden


def compute_steering_vectors(
    model, tokenizer, pairs_data: dict, layers: list[int], batch_size: int,
) -> dict:
    """Compute diff-of-means steering vectors for each category at each layer.

    D+ = mean hidden state at edited segment token positions.
    D- = mean hidden state at the same char span in the baseline.
    """
    pairs = pairs_data["pairs"]
    categories = pairs_data["categories"]
    d_model = model.config.hidden_size
    device = next(model.parameters()).device

    _EDITED, _BASELINE = "edited", "baseline"
    items: list[dict] = []
    for pair in pairs:
        prompt = pair["baseline_prompt"]
        resp_start = _response_start_tok(prompt, tokenizer)
        baseline_text = prompt + pair["baseline_response"]
        for cat_id, ce in pair.get("category_edits", {}).items():
            if cat_id not in categories:
                continue
            if not ce.get("validation", {}).get("valid", False):
                continue
            raw_edited = prompt + ce["edited_response"]
            cleaned, seg, seg_cs = strip_markers(raw_edited)
            if seg_cs is None or seg is None:
                continue
            seg_ce = seg_cs + len(seg)

            pos = find_segment_token_positions(cleaned, seg_cs, seg_ce, tokenizer)
            if pos is None:
                continue
            items.append({"text": cleaned, "kind": _EDITED, "cat_id": cat_id,
                          "pos": pos})

            bl_cs = min(seg_cs, len(baseline_text))
            bl_ce = min(seg_ce, len(baseline_text))
            if bl_cs >= bl_ce:
                items.pop()
                continue
            bl_pos = find_segment_token_positions(
                baseline_text, bl_cs, bl_ce, tokenizer)
            if bl_pos is None:
                items.pop()
                continue
            items.append({"text": baseline_text, "kind": _BASELINE,
                          "cat_id": cat_id, "pos": bl_pos})

    n = len(items)
    print(f"\nD+/D- collection: {n} texts, batch_size={batch_size}, "
          f"{len(layers)} layers")

    sorted_idx = sorted(range(n), key=lambda i: len(items[i]["text"]))

    plus_mean = {c: {l: torch.zeros(d_model, dtype=torch.float64)
                     for l in layers} for c in categories}
    minus_mean = {c: {l: torch.zeros(d_model, dtype=torch.float64)
                      for l in layers} for c in categories}
    plus_count: dict[str, int] = {c: 0 for c in categories}
    minus_count: dict[str, int] = {c: 0 for c in categories}

    for bs in tqdm(range(0, n, batch_size), desc="Forward passes"):
        bi = sorted_idx[bs:bs + batch_size]
        batch = [items[i] for i in bi]
        enc = tokenizer([it["text"] for it in batch],
                        return_tensors="pt", padding=True).to(device)
        hidden = _hf_forward_layers(model, enc["input_ids"],
                                    enc["attention_mask"], layers)
        for j, it in enumerate(batch):
            seq_len = int(enc["attention_mask"][j].sum().item())
            pos = it["pos"]
            sel = list(range(pos["segment_start"], pos["segment_end"]))
            for l in layers:
                acts = hidden[l][j, -seq_len:, :]
                max_p = acts.shape[0] - 1
                valid = [p for p in sel if 0 <= p <= max_p]
                if not valid:
                    continue
                vec = acts[valid].mean(dim=0).cpu().to(torch.float64)
                cid = it["cat_id"]
                if it["kind"] == _EDITED:
                    if l == layers[0]:
                        plus_count[cid] += 1
                    plus_mean[cid][l] += (vec - plus_mean[cid][l]) / plus_count[cid]
                else:
                    if l == layers[0]:
                        minus_count[cid] += 1
                    minus_mean[cid][l] += (vec - minus_mean[cid][l]) / minus_count[cid]

        del hidden, enc
        torch.cuda.empty_cache()

    result = {}
    for cat_id, info in sorted(categories.items()):
        np_, nm_ = plus_count[cat_id], minus_count[cat_id]
        if np_ == 0 or nm_ == 0:
            print(f"  Cat {cat_id} ({info['title']}): n+={np_}, n-={nm_} — skip")
            continue
        vectors, mean_norms, raw_norms = {}, {}, {}
        for l in layers:
            u = plus_mean[cat_id][l] - minus_mean[cat_id][l]
            vectors[l] = u.to(torch.float32)
            raw_norms[l] = u.norm().item()
            mean_norms[l] = minus_mean[cat_id][l].norm().item()
        result[cat_id] = {
            "vectors": vectors, "title": info["title"],
            "n_positive": np_,
            "overall_mean_norms": mean_norms, "raw_norms": raw_norms,
        }
        print(f"  Cat {cat_id} ({info['title']}): {np_} D+ / {nm_} D-")
    return result


# ---------------------------------------------------------------------------
# Gradient-based layer attribution
# ---------------------------------------------------------------------------

def compute_layer_attribution(
    model, tokenizer, pairs_data: dict, steering_vectors: dict,
    layers: list[int], n_examples_per_cat: int, batch_size: int,
) -> dict:
    """Gradient · normalized_sv at the decision point (segment_start - 1)."""
    pairs = pairs_data["pairs"]
    categories = pairs_data["categories"]
    device = next(model.parameters()).device

    cat_items: dict[str, list[dict]] = {c: [] for c in categories}
    for pair in pairs:
        for cat_id, ce in pair.get("category_edits", {}).items():
            if cat_id not in categories or cat_id not in steering_vectors:
                continue
            if not ce.get("validation", {}).get("valid", False):
                continue
            raw = pair["baseline_prompt"] + ce["edited_response"]
            cleaned, seg, seg_cs = strip_markers(raw)
            if seg_cs is None or seg is None:
                continue
            pos = find_segment_token_positions(
                cleaned, seg_cs, seg_cs + len(seg), tokenizer)
            if pos is None:
                continue
            cat_items[cat_id].append({"text": cleaned, "pos": pos})

    param_grad = {}
    for name, p in model.named_parameters():
        param_grad[name] = p.requires_grad
        p.requires_grad_(False)
    embed_param = None
    for p in model.model.embed_tokens.parameters():
        embed_param = p
        p.requires_grad_(True)
        break

    sv_normed_all: dict[str, dict[int, torch.Tensor]] = {}
    for cat_id in cat_items:
        if cat_id not in steering_vectors:
            continue
        sv_normed_all[cat_id] = {}
        for l in layers:
            sv = steering_vectors[cat_id]["vectors"][l]
            sv_normed_all[cat_id][l] = (sv / (sv.norm() + 1e-12)).cpu().float()

    all_items: list[dict] = []
    for cat_id, items in sorted(cat_items.items()):
        if not items or cat_id not in sv_normed_all:
            continue
        for it in items[:n_examples_per_cat]:
            all_items.append({**it, "cat_id": cat_id})
    all_items.sort(key=lambda x: len(x["text"]))

    cat_counts = {}
    for it in all_items:
        cat_counts[it["cat_id"]] = cat_counts.get(it["cat_id"], 0) + 1
    for cat_id, cnt in sorted(cat_counts.items()):
        title = steering_vectors[cat_id]["title"]
        print(f"  Attribution cat {cat_id} ({title}): {cnt} examples")
    print(f"  Total pooled: {len(all_items)} examples across {len(cat_counts)} categories")

    attribution: dict[str, dict[int, list[float]]] = {
        c: {l: [] for l in layers} for c in cat_counts}

    for bs in tqdm(range(0, len(all_items), batch_size),
                   desc="Attribution", leave=False):
        batch = all_items[bs:bs + batch_size]
        texts = [it["text"] for it in batch]
        dps = [max(0, it["pos"]["segment_start"] - 1) for it in batch]
        batch_cats = [it["cat_id"] for it in batch]

        enc = tokenizer(texts, return_tensors="pt", padding=True).to(device)
        ids = enc["input_ids"]
        mask = enc["attention_mask"]
        B, S = ids.shape
        seq_lens = mask.sum(dim=1).tolist()

        adj_dps, valid = [], []
        for j in range(B):
            pad = S - int(seq_lens[j])
            adp = dps[j] + pad
            ok = adp < S
            adj_dps.append(adp if ok else 0)
            valid.append(ok)

        batch_scores: dict[int, list[tuple[str, float]]] = {l: [] for l in layers}
        handles = []

        def make_hook(li, _adps, _valid, _B, _bcats, _svn_all):
            def fn(mod, gi, go):
                g = go[0]
                if g is None:
                    return
                for j in range(_B):
                    if not _valid[j]:
                        continue
                    gv = g[j, _adps[j], :].detach().cpu().float()
                    svn = _svn_all[_bcats[j]][li]
                    batch_scores[li].append(
                        (_bcats[j], torch.dot(svn, gv).abs().item()))
            return fn

        for l in layers:
            h = model.model.layers[l].register_full_backward_hook(
                make_hook(l, adj_dps, valid, B, batch_cats, sv_normed_all))
            handles.append(h)

        try:
            if embed_param is not None:
                embed_param.grad = None
            out = model(input_ids=ids, attention_mask=mask)
            logits = out.logits

            total_loss = torch.tensor(0.0, device=device, requires_grad=True)
            n_valid = 0
            for j in range(B):
                if not valid[j]:
                    continue
                dp = adj_dps[j]
                top_tok = logits[j, dp].argmax()
                nll = -F.log_softmax(logits[j, dp], dim=-1)[top_tok]
                total_loss = total_loss + nll
                n_valid += 1
            if n_valid > 0:
                total_loss.backward()
                for l in layers:
                    for cat_id, score in batch_scores[l]:
                        attribution[cat_id][l].append(score)
        except Exception as e:
            print(f"    [WARN] Attribution batch failed: {e}")
        finally:
            for h in handles:
                h.remove()
            if embed_param is not None:
                embed_param.grad = None
            del ids, mask, enc
            gc.collect()
            torch.cuda.empty_cache()

    for name, p in model.named_parameters():
        if name in param_grad:
            p.requires_grad_(param_grad[name])
    return attribution


def find_best_layers(attribution: dict, layers: list[int],
                     min_layer_frac: float = 0.0) -> dict:
    n = len(layers)
    candidates = layers[int(n * min_layer_frac):] or layers
    best = {}
    for cat_id, ls in attribution.items():
        means = {l: (np.mean(s) if s else 0.0) for l, s in ls.items()}
        bl = max(candidates, key=lambda l: means.get(l, 0.0))
        best[cat_id] = {"best_layer": bl, "mean_score": means[bl],
                        "all_means": means}
    return best


def plot_attribution(attribution: dict, sv: dict, layers: list[int],
                     best_layers: dict, path: str):
    n_cats = len(attribution)
    if n_cats == 0:
        return
    cols = min(4, n_cats)
    rows = (n_cats + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows),
                             squeeze=False)
    colors = plt.cm.tab10.colors
    for i, (cid, ls) in enumerate(sorted(attribution.items())):
        ax = axes[i // cols][i % cols]
        means = [np.mean(ls.get(l, [])) if ls.get(l) else 0 for l in layers]
        stds = [np.std(ls.get(l, [])) if ls.get(l) else 0 for l in layers]
        c = colors[i % len(colors)]
        ax.fill_between(layers, np.array(means) - np.array(stds),
                        np.array(means) + np.array(stds), alpha=0.2, color=c)
        ax.plot(layers, means, color=c, linewidth=2, marker="o", markersize=3)
        ax.axvline(best_layers[cid]["best_layer"], color="red",
                   linestyle="--", linewidth=1, alpha=0.7)
        ax.set_title(f"[{cid}] {sv[cid]['title']}", fontsize=10)
        ax.set_xlabel("Layer")
        ax.set_ylabel("Attribution")
        ax.grid(True, alpha=0.3)
    for j in range(n_cats, rows * cols):
        axes[j // cols][j % cols].set_visible(False)
    fig.suptitle("Layer Attribution", fontsize=14, weight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved attribution plot to {path}")


# ---------------------------------------------------------------------------
# Batched multi-steer generation
# ---------------------------------------------------------------------------

def generate_batch_multi_steer(
    model, tokenizer, jobs: list[dict], max_new_tokens: int, batch_size: int,
) -> list[str]:
    """Batched generation with per-sample steering.

    Each job: {prompt, sv (Tensor or None), layer, coeff}.
    """
    device = next(model.parameters()).device
    dtype = next(model.parameters()).dtype
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    results: list[Optional[str]] = [None] * len(jobs)

    for bs in tqdm(range(0, len(jobs), batch_size),
                   desc="Generating (multi-steer)"):
        batch = jobs[bs:bs + batch_size]
        B = len(batch)
        enc = tokenizer([j["prompt"] for j in batch],
                        return_tensors="pt", padding=True).to(device)

        layer_svs: dict[int, torch.Tensor] = {}
        for i, j in enumerate(batch):
            if j.get("sv") is None:
                continue
            L = j["layer"]
            if L not in layer_svs:
                D = j["sv"].shape[-1]
                layer_svs[L] = torch.zeros(B, D, device=device, dtype=dtype)
            layer_svs[L][i] = (j["sv"] * j["coeff"]).to(dtype)

        handles = []
        for li, svs in layer_svs.items():
            def _hook(svs_ref):
                def fn(mod, inp, out):
                    hs = out[0] if isinstance(out, tuple) else out
                    b = hs.shape[0]
                    sv = svs_ref[:b]
                    if hs.shape[1] > 1:
                        hs[:, 1:, :] += sv.unsqueeze(1)
                    else:
                        hs += sv.unsqueeze(1)
                    return out
                return fn
            handles.append(
                model.model.layers[li].register_forward_hook(_hook(svs)))

        try:
            with torch.no_grad():
                gen = model.generate(
                    **enc, max_new_tokens=max_new_tokens,
                    pad_token_id=tokenizer.eos_token_id, do_sample=False)
        finally:
            for h in handles:
                h.remove()

        pl = enc["input_ids"].shape[1]
        for j in range(B):
            results[bs + j] = tokenizer.decode(gen[j, pl:],
                                                skip_special_tokens=True)
        del enc, gen, layer_svs
        torch.cuda.empty_cache()
    return results


# ---------------------------------------------------------------------------
# Auto-grading
# ---------------------------------------------------------------------------

GRADE_PROMPT = """\
You are a strict evaluator of reasoning behaviour in language model outputs.

**Question:** {question}

**Short continuation snippet (~1-2 sentences) to evaluate:**
{response}

**Target reasoning behaviour:**
Title: {cat_title}
Description: {cat_description}

**Examples of this behaviour (from other responses):**
{examples}

Evaluate on three axes:

1. **answer_quality** (1-5): Is the text coherent and non-degraded?
   - 5 = Normal fluent reasoning text, logically connected to context
   - 4 = Minor awkwardness but fully readable
   - 3 = Somewhat off but readable
   - 2 = Noticeably degraded — broken logic, non-sequiturs, partial repetition
   - 1 = Gibberish, repetition loops, or incoherent fragments

2. **behaviour_presence** (1-5): Does this snippet DIRECTLY and PRECISELY perform the described behaviour, similar to the examples? The snippet's primary purpose must match. Merely being on-topic or vaguely related is NOT enough.
   - 5 = Clearly and precisely performs this specific behaviour
   - 4 = Performs this behaviour but with minor imprecision
   - 3 = Partially present — related reasoning but not a clean match
   - 2 = Only superficially related
   - 1 = Absent — does not perform this behaviour at all

3. **reasoning_quality** (1-5): Is the reasoning in this snippet substantive and correct, regardless of category match?
   - 5 = Sound reasoning — logically valid steps, correct inferences
   - 4 = Mostly correct with minor issues
   - 3 = Mediocre — some valid reasoning mixed with errors
   - 2 = Poor — significant logical errors or meaningless filler
   - 1 = No meaningful reasoning content

Return ONLY a JSON object: {{"answer_quality": <int>, "behaviour_presence": <int>, "reasoning_quality": <int>}}
"""


def _extract_behaviour_examples(pairs_data: dict,
                                n_examples: int = 3) -> dict[str, str]:
    segs: dict[str, list[str]] = defaultdict(list)
    for pair in pairs_data.get("pairs", []):
        for cid, ce in pair.get("category_edits", {}).items():
            if not ce.get("validation", {}).get("valid", False):
                continue
            ed = ce.get("edited_response", "")
            s = ed.find("[CATEGORY_START]")
            e = ed.find("[CATEGORY_END]")
            if s >= 0 and e > s:
                seg = ed[s + len("[CATEGORY_START]"):e].strip()
                if 20 < len(seg) < 500:
                    segs[cid].append(seg)
    return {cid: "\n".join(f'  - "{s[:200]}"' for s in ss[:n_examples])
            or "(no examples)" for cid, ss in segs.items()}


def _parse_grade(text: str) -> dict:
    try:
        m = re.search(r'\{[^}]+\}', text)
        if m:
            p = json.loads(m.group())
            return {k: int(p.get(k, 0))
                    for k in ("answer_quality", "behaviour_presence",
                              "reasoning_quality")}
    except Exception:
        pass
    return {"answer_quality": 0, "behaviour_presence": 0, "reasoning_quality": 0}


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    os.makedirs(args.save_dir, exist_ok=True)
    fig_dir = os.path.join(args.save_dir, "figures")
    os.makedirs(fig_dir, exist_ok=True)

    model_short = args.model.split("/")[-1].lower()
    steer_coeffs = [float(c) for c in args.steer_coeffs.split(",")]

    # ---- Load pairs ----
    with open(args.pairs_file) as f:
        pairs_data = json.load(f)
    categories = pairs_data["categories"]
    print(f"Loaded {len(pairs_data['pairs'])} train pairs, "
          f"{len(categories)} categories")

    eval_pairs_data = None
    if args.eval_pairs_file and os.path.exists(args.eval_pairs_file):
        with open(args.eval_pairs_file) as f:
            eval_pairs_data = json.load(f)
        print(f"Loaded {len(eval_pairs_data['pairs'])} eval pairs")

    # ---- Load model ----
    print(f"\nLoading {args.model}...")
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map="auto")
    model.eval()
    all_layers = list(range(model.config.num_hidden_layers))

    # ---- Compute or load steering vectors ----
    vectors_path = os.path.join(args.save_dir,
                                f"dom_vectors_multilayer_{model_short}.pt")
    meta_path = os.path.join(args.save_dir,
                             f"dom_metadata_multilayer_{model_short}.json")

    if args.skip_vectors and os.path.exists(vectors_path):
        print(f"\nLoading pre-saved vectors from {vectors_path}")
        raw = torch.load(vectors_path, map_location="cpu")
        with open(meta_path) as f:
            meta = json.load(f)
        steering_vectors = {}
        for cid, lv in raw.items():
            m = meta[cid]
            steering_vectors[cid] = {
                "vectors": {int(l): v for l, v in lv.items()},
                "title": m["title"], "n_positive": m["n_positive"],
                "overall_mean_norms": {int(k): v for k, v
                                       in m["overall_mean_norms"].items()},
                "raw_norms": {int(k): v for k, v in m["raw_norms"].items()},
            }
    else:
        steering_vectors = compute_steering_vectors(
            model, tokenizer, pairs_data, all_layers, args.batch_size)
        if not steering_vectors:
            print("No steering vectors computed — exiting.")
            return
        vs = {c: {str(l): v for l, v in sv["vectors"].items()}
              for c, sv in steering_vectors.items()}
        ms = {c: {"title": sv["title"], "n_positive": sv["n_positive"],
                   "overall_mean_norms": {str(l): v for l, v
                                          in sv["overall_mean_norms"].items()},
                   "raw_norms": {str(l): v for l, v
                                 in sv["raw_norms"].items()}}
              for c, sv in steering_vectors.items()}
        torch.save(vs, vectors_path)
        with open(meta_path, "w") as f:
            json.dump(ms, f, indent=2)
        print(f"Saved vectors to {vectors_path}")

    # ---- Layer attribution ----
    attrib_path = os.path.join(args.save_dir,
                               f"dom_attribution_{model_short}.json")
    best_layers_path = os.path.join(args.save_dir,
                                    f"dom_best_layers_{model_short}.json")

    if args.skip_attribution and os.path.exists(attrib_path):
        print(f"\nLoading attribution from {attrib_path}")
        with open(attrib_path) as f:
            attribution = {c: {int(l): s for l, s in ls.items()}
                           for c, ls in json.load(f).items()}
    else:
        abs_ = args.attrib_batch_size or max(1, args.batch_size // 4)
        print(f"\nComputing attribution (batch_size={abs_})...")
        attribution = compute_layer_attribution(
            model, tokenizer, pairs_data, steering_vectors,
            all_layers, args.n_attribution_examples, abs_)
        with open(attrib_path, "w") as f:
            json.dump({c: {str(l): s for l, s in ls.items()}
                       for c, ls in attribution.items()}, f, indent=2)

    best_layers = find_best_layers(attribution, all_layers, args.min_layer_frac)
    bl_save = {c: {"best_layer": b["best_layer"],
                    "mean_score": b["mean_score"],
                    "all_means": {str(l): v for l, v in b["all_means"].items()}}
               for c, b in best_layers.items()}
    with open(best_layers_path, "w") as f:
        json.dump(bl_save, f, indent=2)
    print(f"Saved best layers to {best_layers_path}")

    plot_attribution(attribution, steering_vectors, all_layers, best_layers,
                     os.path.join(fig_dir, f"dom_attribution_{model_short}.pdf"))

    for cid, bl in sorted(best_layers.items()):
        print(f"  Cat {cid} ({steering_vectors[cid]['title']}): "
              f"layer {bl['best_layer']} (score={bl['mean_score']:.6f})")

    print("\nSteering vector norms (raw diff-of-means, no rescaling):")
    for cid, sv in steering_vectors.items():
        bl = best_layers[cid]["best_layer"]
        print(f"  Cat {cid}: layer {bl} norm "
              f"{sv['vectors'][bl].norm().item():.2f}")

    # ---- Coefficient sweep ----
    if args.n_eval_questions <= 0:
        print("No eval requested — done.")
        return

    torch.cuda.empty_cache()
    gc.collect()

    src = eval_pairs_data if eval_pairs_data else pairs_data
    eval_pairs = src["pairs"][:args.n_eval_questions]
    print(f"\n=== Coefficient sweep: {len(eval_pairs)} eval pairs ===")

    eval_items: list[dict] = []
    for pair in eval_pairs:
        prompt = pair["baseline_prompt"]
        baseline = pair.get("baseline_response", "")
        for cid, ce in pair.get("category_edits", {}).items():
            if cid not in categories or cid not in steering_vectors:
                continue
            if not ce.get("validation", {}).get("valid", False):
                continue
            ed = ce["edited_response"]
            mp = ed.find(CATEGORY_START_TAG)
            if mp < 0:
                continue
            prefix = prompt + ed[:mp]
            raw_cont = baseline[len(ed[:mp]):]
            eval_items.append({
                "question_id": pair["question_id"],
                "question": pair["question"], "cat_id": cid,
                "prefix": prefix, "raw_cont": raw_cont,
            })

    cat_groups: dict[str, list[int]] = defaultdict(list)
    for idx, ei in enumerate(eval_items):
        cat_groups[ei["cat_id"]].append(idx)

    tok_limit = args.eval_max_tokens
    for ei in eval_items:
        toks = tokenizer.encode(ei["raw_cont"], add_special_tokens=False)
        ei["raw_cont"] = tokenizer.decode(toks[:tok_limit],
                                          skip_special_tokens=True)

    jobs: list[dict] = []
    job_map: list[tuple[float, int]] = []
    for cid, indices in sorted(cat_groups.items()):
        bl = best_layers[cid]["best_layer"]
        vec = steering_vectors[cid]["vectors"][bl]
        for coeff in steer_coeffs:
            for gi in indices:
                jobs.append({"prompt": eval_items[gi]["prefix"],
                             "sv": vec, "layer": bl, "coeff": coeff})
                job_map.append((coeff, gi))

    print(f"Generating {len(jobs)} steered continuations "
          f"(batch_size={args.gen_batch_size})...")
    gen_results = generate_batch_multi_steer(
        model, tokenizer, jobs, tok_limit, args.gen_batch_size)

    steered: dict[float, list] = {c: [None] * len(eval_items)
                                  for c in steer_coeffs}
    for ri, (coeff, gi) in enumerate(job_map):
        steered[coeff][gi] = gen_results[ri]

    eval_results = []
    for idx, ei in enumerate(eval_items):
        entry = {
            "question_id": ei["question_id"], "question": ei["question"],
            "cat_id": ei["cat_id"],
            "category_title": steering_vectors[ei["cat_id"]]["title"],
            "steer_layer": best_layers[ei["cat_id"]]["best_layer"],
            "raw_continuation": ei["raw_cont"],
        }
        for c in steer_coeffs:
            entry[f"steered_{c}"] = steered[c][idx]
        eval_results.append(entry)

    # ---- Auto-grade ----
    print("\n=== Auto-grading ===")
    beh_examples = _extract_behaviour_examples(pairs_data)
    grade_prompts, grade_keys = [], []
    for idx, er in enumerate(eval_results):
        ci = categories.get(er["cat_id"], {})
        title = ci.get("title", er.get("category_title", ""))
        desc = ci.get("description", "")
        exs = beh_examples.get(er["cat_id"], "(no examples)")
        grade_prompts.append(GRADE_PROMPT.format(
            question=er["question"], response=er["raw_continuation"],
            cat_title=title, cat_description=desc, examples=exs))
        grade_keys.append((idx, "raw"))
        for c in steer_coeffs:
            grade_prompts.append(GRADE_PROMPT.format(
                question=er["question"], response=er.get(f"steered_{c}", ""),
                cat_title=title, cat_description=desc, examples=exs))
            grade_keys.append((idx, f"steered_{c}"))

    print(f"Grading {len(grade_prompts)} continuations...")
    try:
        api_resp = asyncio.run(utils.chat_batch(
            grade_prompts, model=args.api_model, max_tokens=512,
            max_concurrent_requests=args.max_concurrent))
        for key, resp in zip(grade_keys, api_resp):
            idx, kn = key
            eval_results[idx][f"{kn}_grade"] = _parse_grade(resp)
    except Exception as e:
        print(f"[WARN] Grading failed: {e}")

    # ---- Select best coefficient ----
    print("\n=== Best coefficient per category ===")
    best_coeffs: dict[str, dict] = {}
    for cid in sorted(cat_groups.keys()):
        indices = cat_groups[cid]
        raw_beh = [eval_results[i].get("raw_grade", {}).get(
            "behaviour_presence", 0) for i in indices]
        raw_qual = [eval_results[i].get("raw_grade", {}).get(
            "answer_quality", 0) for i in indices]
        raw_rq = [eval_results[i].get("raw_grade", {}).get(
            "reasoning_quality", 0) for i in indices]
        mr_beh = np.mean(raw_beh) if raw_beh else 0
        mr_qual = np.mean(raw_qual) if raw_qual else 0
        mr_rq = np.mean(raw_rq) if raw_rq else 0

        scores = {}
        for c in steer_coeffs:
            sb = [eval_results[i].get(f"steered_{c}_grade", {}).get(
                "behaviour_presence", 0) for i in indices]
            sq = [eval_results[i].get(f"steered_{c}_grade", {}).get(
                "answer_quality", 0) for i in indices]
            sr = [eval_results[i].get(f"steered_{c}_grade", {}).get(
                "reasoning_quality", 0) for i in indices]
            msb = np.mean(sb) if sb else 0
            msq = np.mean(sq) if sq else 0
            msr = np.mean(sr) if sr else 0
            rep = (sum(1 for q in sq if q == 1) / len(sq) * 100) if sq else 0
            scores[c] = {
                "beh_delta": float(msb - mr_beh),
                "qual_delta": float(msq - mr_qual),
                "rq_delta": float(msr - mr_rq),
                "mean_beh": float(msb), "mean_qual": float(msq),
                "mean_rq": float(msr), "rep_pct": float(rep),
            }

        def composite(c):
            s = scores[c]
            return (s["beh_delta"] + 0.5 * s["qual_delta"]
                    + 0.5 * s["rq_delta"] - 0.1 * s["rep_pct"])

        best_c = max(steer_coeffs, key=composite)
        best_coeffs[cid] = {
            "best_coeff": best_c, "scores_by_coeff": scores,
            "raw_beh": float(mr_beh), "raw_qual": float(mr_qual),
            "raw_rq": float(mr_rq),
        }
        title = steering_vectors[cid]["title"]
        print(f"  Cat {cid} ({title}): best_coeff={best_c}")
        for c in steer_coeffs:
            s = scores[c]
            sel = ">" if c == best_c else " "
            print(f"    {sel} {c}: beh={s['beh_delta']:+.2f} "
                  f"qual={s['qual_delta']:+.2f} rq={s['rq_delta']:+.2f} "
                  f"rep={s['rep_pct']:.0f}% comp={composite(c):+.3f}")

    # ---- Save results ----
    eval_path = os.path.join(args.save_dir, f"dom_eval_{model_short}.json")
    coeffs_path = os.path.join(args.save_dir,
                               f"dom_best_coeffs_{model_short}.json")
    with open(eval_path, "w") as f:
        json.dump({"model": args.model, "steer_coeffs": steer_coeffs,
                    "best_coeffs": {c: bc["best_coeff"]
                                    for c, bc in best_coeffs.items()},
                    "n_eval_items": len(eval_results),
                    "results": eval_results}, f, indent=2)
    with open(coeffs_path, "w") as f:
        json.dump(best_coeffs, f, indent=2)
    print(f"\nSaved eval to {eval_path}")
    print(f"Saved coefficients to {coeffs_path}")

    # ---- Plots ----
    n_cats = len(cat_groups)
    if n_cats > 0:
        fig, axes = plt.subplots(n_cats, 1, figsize=(8, 3 * n_cats),
                                 sharex=True)
        if n_cats == 1:
            axes = [axes]
        for ai, cid in enumerate(sorted(cat_groups.keys())):
            ax = axes[ai]
            bc = best_coeffs[cid]
            best_c = bc["best_coeff"]
            bv = [bc["scores_by_coeff"][c]["beh_delta"] for c in steer_coeffs]
            qv = [bc["scores_by_coeff"][c]["qual_delta"] for c in steer_coeffs]
            x = np.arange(len(steer_coeffs))
            w = 0.35
            bars_b = ax.bar(x - w/2, bv, w, label="Behaviour Δ", color="#2196F3")
            bars_q = ax.bar(x + w/2, qv, w, label="Quality Δ", color="#FF9800")
            ax.axhline(0, color="black", linewidth=0.5)
            best_idx = steer_coeffs.index(best_c) if best_c in steer_coeffs else None
            if best_idx is not None:
                bars_b[best_idx].set_edgecolor("red")
                bars_b[best_idx].set_linewidth(2.5)
                bars_q[best_idx].set_edgecolor("red")
                bars_q[best_idx].set_linewidth(2.5)
                ax.annotate(f"★ {best_c}", xy=(best_idx, max(bv[best_idx], qv[best_idx])),
                            xytext=(0, 8), textcoords="offset points",
                            ha="center", fontsize=8, fontweight="bold", color="red")
            ax.set_title(f"Cat {cid}: {steering_vectors[cid]['title']}",
                         fontsize=10, fontweight="bold")
            ax.set_ylabel("Δ score")
            ax.set_xticks(x)
            ax.set_xticklabels([str(c) for c in steer_coeffs])
            if ai == 0:
                ax.legend(loc="upper left", fontsize=8)
        axes[-1].set_xlabel("Coefficient")
        plt.tight_layout()
        plt.savefig(os.path.join(fig_dir,
                    f"dom_coeff_sweep_{model_short}.pdf"), bbox_inches="tight")
        plt.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
