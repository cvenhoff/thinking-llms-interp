# %%
import argparse
import gc
import sys
import json
import torch
from tqdm import tqdm
import dotenv

dotenv.load_dotenv("../.env")

sys.path.append('..')
from utils.utils import load_model, get_char_to_token_map, center_and_l2_normalize_torch
from utils.sae import load_sae
from utils.responses import extract_thinking_process
from utils import utils

parser = argparse.ArgumentParser(description="Annotate thinking processes in generated responses")
parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                    help="Model used to generate responses")
parser.add_argument("--layer", type=int, default=6,
                    help="Layer to analyze")
parser.add_argument("--n_clusters", type=int, default=15,
                    help="Number of clusters in the SAE")
parser.add_argument("--max_batch_tokens", type=int, default=32_000,
                    help="Token-budget cap per batch (tune for your GPU)")
parser.add_argument("--max_batch_size", type=int, default=16,
                    help="Maximum number of responses per batch")
args, _ = parser.parse_known_args()


def split_into_sentences(text):
    """Delegates to the canonical implementation in utils.utils
    to stay consistent with SAE training."""
    return utils.split_into_sentences(text)


def _build_batches(items, tokenizer, max_batch_tokens, max_batch_size):
    """Group pre-tokenised items into batches respecting a token budget."""
    sorted_items = sorted(items, key=lambda x: x["n_tokens"])
    batches = []
    cur = []
    cur_max_len = 0
    for item in sorted_items:
        new_max = max(cur_max_len, item["n_tokens"])
        new_total = new_max * (len(cur) + 1)
        if cur and (new_total > max_batch_tokens or len(cur) >= max_batch_size):
            batches.append(cur)
            cur = [item]
            cur_max_len = item["n_tokens"]
        else:
            cur.append(item)
            cur_max_len = new_max
    if cur:
        batches.append(cur)
    return batches


def process_responses(responses_file, model, tokenizer, sae, layer, output_file,
                      max_batch_tokens, max_batch_size):
    """Annotate thinking processes using batched hook-based forward passes."""
    with open(responses_file, 'r') as f:
        responses_data = json.load(f)

    raw_model = model._model if hasattr(model, '_model') else model
    device = next(raw_model.parameters()).device
    sae_device = next(sae.parameters()).device

    print(f"Processing {len(responses_data)} responses (layer {layer})...")

    # Pre-process: extract thinking, tokenise, compute char-to-token maps
    items = []
    for idx, response_item in enumerate(responses_data):
        full_response = response_item['full_response']
        thinking_process = extract_thinking_process(full_response)
        sentences = split_into_sentences(thinking_process)
        if not sentences:
            items.append({
                "idx": idx, "response_item": response_item,
                "full_response": full_response, "sentences": [],
                "n_tokens": 0,
            })
            continue
        n_tokens = len(tokenizer.encode(full_response))
        items.append({
            "idx": idx, "response_item": response_item,
            "full_response": full_response, "sentences": sentences,
            "n_tokens": n_tokens,
        })

    # Separate items with sentences (need forward pass) from empty ones
    items_with_text = [it for it in items if it["sentences"]]
    items_empty = [it for it in items if not it["sentences"]]

    batches = _build_batches(items_with_text, tokenizer, max_batch_tokens, max_batch_size)
    print(f"  {len(items_with_text)} items with text → {len(batches)} batches, "
          f"{len(items_empty)} items empty")

    results_by_idx = {}

    # Process empty items (no thinking process)
    for it in items_empty:
        ri = it["response_item"]
        results_by_idx[it["idx"]] = {
            'question_id': ri['question_id'],
            'category': ri['category'],
            'dataset_name': ri['dataset_name'],
            'annotated_thinking': '',
        }

    # Pre-compute char-to-token maps for all items (avoids re-tokenizing per batch)
    print("  Pre-computing char-to-token maps...")
    for it in tqdm(items_with_text, desc="Char→token maps", leave=False):
        it["_c2t"] = get_char_to_token_map(it["full_response"], tokenizer)

    act_mean_cpu = sae.activation_mean.cpu()
    b_dec_sae = sae.b_dec.detach()

    # Process batches
    for batch in tqdm(batches, desc="Annotating batches"):
        batch_texts = [it["full_response"] for it in batch]
        encodings = tokenizer(
            batch_texts, return_tensors="pt", padding=True, truncation=False,
        ).to(device)
        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]

        layer_output = {}

        def hook_fn(module, inp, out):
            hidden = out[0] if isinstance(out, tuple) else out
            layer_output["acts"] = hidden.detach().cpu().to(torch.float32)

        handle = raw_model.model.layers[layer].register_forward_hook(hook_fn)
        with torch.no_grad():
            raw_model(input_ids=input_ids, attention_mask=attention_mask)
        handle.remove()
        acts_cpu = layer_output["acts"]  # (B, S, d_model)
        del input_ids, attention_mask, encodings, layer_output
        torch.cuda.empty_cache()

        B, S, _ = acts_cpu.shape

        # Gather all sentence mean activations across the batch for batched SAE encoding
        all_avg_acts = []
        # (batch_item_idx, sentence_idx, sentence_text, sentence_pos)
        all_sent_meta: list[tuple[int, int, str]] = []

        for j, it in enumerate(batch):
            full_response = it["full_response"]
            sentences = it["sentences"]
            seq_len = it["n_tokens"]
            pad_offset = S - seq_len
            char_to_token = it["_c2t"]

            for s_idx, sentence in enumerate(sentences):
                sentence_pos = full_response.find(sentence)
                if sentence_pos < 0:
                    continue
                token_start = char_to_token.get(sentence_pos)
                token_end = char_to_token.get(sentence_pos + len(sentence))
                if (token_start is None or token_end is None
                        or token_start >= token_end
                        or token_start >= seq_len or token_end > seq_len):
                    continue

                adj_start = pad_offset + token_start - 1
                adj_end = pad_offset + token_end
                if adj_start < 0:
                    adj_start = 0
                avg_act = acts_cpu[j, adj_start:adj_end, :].mean(dim=0)
                all_avg_acts.append(avg_act)
                all_sent_meta.append((j, s_idx, sentence))

        del acts_cpu

        if not all_avg_acts:
            for j, it in enumerate(batch):
                ri = it["response_item"]
                results_by_idx[it["idx"]] = {
                    'question_id': ri['question_id'], 'category': ri['category'],
                    'dataset_name': ri['dataset_name'], 'annotated_thinking': '',
                }
            continue

        # Batched center + L2-normalise + SAE encode
        stacked = torch.stack(all_avg_acts)  # (N_sentences, d_model)
        centered = stacked - act_mean_cpu.unsqueeze(0)
        norms = centered.norm(dim=1, keepdim=True).clamp(min=1e-12)
        x_norm = centered / norms

        with torch.no_grad():
            latents = sae.encoder((x_norm.to(sae_device) - b_dec_sae))  # (N, n_clusters)
        top_vals, top_idxs = latents.max(dim=1)  # (N,), (N,)
        top_vals = top_vals.cpu()
        top_idxs = top_idxs.cpu()

        # Reassemble annotations per item
        per_item_annotations: dict[int, list[tuple[int, str, int, float]]] = {}
        for k, (j, s_idx, sentence) in enumerate(all_sent_meta):
            if j not in per_item_annotations:
                per_item_annotations[j] = []
            per_item_annotations[j].append(
                (s_idx, sentence, int(top_idxs[k].item()), round(float(top_vals[k].item()), 2)))

        for j, it in enumerate(batch):
            ri = it["response_item"]
            annots = per_item_annotations.get(j, [])
            annots.sort(key=lambda x: x[0])
            annotated_thinking = ""
            for _, sentence, tidx, tact in annots:
                annotated_thinking += f'["{tact}:idx{tidx}"]{sentence}["end-section"]'
            results_by_idx[it["idx"]] = {
                'question_id': ri['question_id'], 'category': ri['category'],
                'dataset_name': ri['dataset_name'],
                'annotated_thinking': annotated_thinking.strip(),
            }

        del stacked, centered, x_norm, latents, top_vals, top_idxs
        gc.collect()

    # Reassemble in original order
    annotated_responses = [results_by_idx[i] for i in range(len(responses_data))]

    with open(output_file, 'w') as f:
        json.dump(annotated_responses, f, indent=2)

    return annotated_responses


# %% Get model ID from model name
model_name = args.model
model_id = model_name.split('/')[-1].lower()

# %%  Cluster count in filename so different configs don't overwrite each other
responses_file = f"results/vars/responses_{model_id}.json"
output_file = f"results/vars/annotated_responses_{model_id}_{args.n_clusters}clusters_layer{args.layer}.json"

# Load model and tokenizer
print(f"Loading model {model_name}...")
model, tokenizer = load_model(model_name=model_name)

# %% Load SAE (skip the file-vs-checkpoint cross-check since we only need the
#    checkpoint's embedded mean for annotation)
print(f"Loading SAE for model {model_id}, layer {args.layer}, clusters {args.n_clusters}...")
sae, _ = load_sae(model_id, args.layer, args.n_clusters, require_activation_mean=False)
sae_path = f'../train-saes/results/vars/saes/sae_{model_id}_layer{args.layer}_clusters{args.n_clusters}.pt'
ckpt = torch.load(sae_path, weights_only=False)
assert "activation_mean" in ckpt, "SAE checkpoint must contain activation_mean"
sae.activation_mean.copy_(ckpt["activation_mean"].to(torch.float32))
del ckpt
raw_model = model._model if hasattr(model, '_model') else model
sae = sae.to(next(raw_model.parameters()).device)

# %% Process responses
processed_data = process_responses(
    responses_file,
    model,
    tokenizer,
    sae,
    args.layer,
    output_file,
    max_batch_tokens=args.max_batch_tokens,
    max_batch_size=args.max_batch_size,
)

print(f"Annotation complete. {len(processed_data)} responses saved to {output_file}")

# %%
