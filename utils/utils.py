import dotenv
dotenv.load_dotenv("../.env")

import gc
import torch
from nnsight import LanguageModel
import time
import anthropic
from openai import OpenAI
import json
import re
import numpy as np
import sys
import os
import random
import pickle
from tqdm import tqdm
from chat_limiter import (
    ChatLimiter,
    process_chat_completion_batch,
    create_chat_completion_requests,
    BatchConfig
)
from utils.responses import extract_thinking_process

def print_and_flush(message):
    """Prints a message and flushes stdout."""
    print(message)
    sys.stdout.flush()

def chat(prompt, model="gpt-4.1", max_tokens=28000):

    model_provider = ""

    if model in ["gpt-4o", "gpt-4.1"]:
        model_provider = "openai"
        client = OpenAI()
    elif model in ["claude-3-opus", "claude-3-7-sonnet", "claude-3-5-haiku"]:
        model_provider = "anthropic"
        client = anthropic.Anthropic()
    elif model in ["deepseek-v3", "gemini-2-0-think", "gemini-2-0-flash", "deepseek-r1"]:
        model_provider = "openrouter"
        client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=os.getenv("OPENROUTER_API_KEY"),
        )

    # try 3 times with 3 second sleep between attempts
    for _ in range(3):
        try:
            if model_provider == "openai":
                client = OpenAI()
                response = client.chat.completions.create(
                    model=model,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "text",
                                    "text": prompt,
                                },
                            ],
                        }
                    ],
                    max_completion_tokens=max_tokens,
                    temperature=1e-19,
                )
                return response.choices[0].message.content
            elif model_provider == "anthropic":
                model_mapping = {
                    "claude-3-opus": "claude-3-opus-latest",
                    "claude-3-7-sonnet": "claude-3-7-sonnet-latest",
                    "claude-3-5-haiku": "claude-3-5-haiku-latest"
                }

                if model == "claude-3-7-sonnet":
                    response = client.messages.create(
                        model=model_mapping[model],
                        temperature=1,
                        messages=[
                            {
                                "role": "user", 
                                "content": [
                                    {
                                        "type": "text",
                                        "text": prompt
                                    }
                                ]
                            }
                        ],
                        thinking = {
                            "type": "enabled",
                            "budget_tokens": max_tokens
                        },
                        max_tokens=max_tokens+1
                    )

                    thinking_response = response.content[0].thinking
                    answer_response = response.content[1].text

                    return f"<think>{thinking_response}\n</think>\n{answer_response}"

                else:
                    response = client.messages.create(
                        model=model_mapping[model],
                        temperature=1e-19,
                        messages=[
                            {
                                "role": "user", 
                                "content": [
                                    {
                                        "type": "text",
                                        "text": prompt
                                    }
                                ]
                            }
                        ],
                        max_tokens=max_tokens
                    )

                    return response.content[0].text
            elif model_provider == "openrouter":
                # Map model names to OpenRouter model IDs
                model_mapping = {
                    "deepseek-r1": "deepseek/deepseek-r1",
                    "deepseek-v3": "deepseek/deepseek-chat",
                    "gemini-2-0-think": "google/gemini-2.0-flash-thinking-exp:free",
                    "gemini-2-0-flash": "google/gemini-2.0-flash-001"
                }
                
                response = client.chat.completions.create(
                    model=model_mapping[model],
                    extra_body={},
                    messages=[
                        {
                            "role": "user",
                            "content": prompt
                        }
                    ],
                    temperature=1e-19,
                    max_tokens=max_tokens
                )

                if hasattr(response.choices[0].message, "reasoning"):
                    thinking_response = response.choices[0].message.reasoning
                    answer_response = response.choices[0].message.content

                    return f"<think>{thinking_response}\n</think>\n{answer_response}"
                else:
                    return response.choices[0].message.content
            
        except Exception as e:
            print(f"Error: {e}")
            time.sleep(20)

    return None

async def chat_batch(prompts, model="gpt-4.1", max_tokens=28000, max_concurrent_requests=100, max_retries_per_item=3, json_mode=False):
    """
    Process a batch of prompts using the chat_limiter library for parallel processing.
    
    Args:
        prompts (list): List of prompts to process
        model (str): Model to use for the chat
        max_tokens (int): Maximum number of tokens per response
        max_concurrent_requests (int): Maximum number of concurrent requests
        max_retries_per_item (int): Maximum number of retries per item
        
    Returns:
        list: List of responses corresponding to the prompts
    """
    temperature = 1e-19
    if model.startswith("o3") or model.startswith("o4"):
        temperature = 1

    # Create chat completion requests
    # Newer OpenAI models (gpt-5.x, o3, o4) require max_completion_tokens instead of max_tokens
    # Handle both "gpt-5.2" and "openai/gpt-5.2" formats
    model_name_only = model.split("/")[-1] if "/" in model else model
    use_max_completion_tokens = model_name_only.startswith("gpt-5") or model_name_only.startswith("o3") or model_name_only.startswith("o4")
    if use_max_completion_tokens:
        requests = create_chat_completion_requests(
            model=model,
            prompts=prompts,
            max_completion_tokens=max_tokens,
            temperature=temperature,
        )
    else:
        requests = create_chat_completion_requests(
            model=model,
            prompts=prompts,
            max_tokens=max_tokens,
            temperature=temperature,
        )
    
    # Instantiate BatchConfig, accounting for versions without a `json_mode` parameter
    try:
        config = BatchConfig(
            max_concurrent_requests=max_concurrent_requests,
            max_retries_per_item=max_retries_per_item,
            group_by_model=True,
            json_mode=json_mode,
            # print_request_initiation=True,
        )
    except TypeError:
        # Fallback for older/newer versions of chat_limiter without `json_mode`
        config = BatchConfig(
            max_concurrent_requests=max_concurrent_requests,
            max_retries_per_item=max_retries_per_item,
            group_by_model=True,
            # print_request_initiation=True,
        )
    
    # Process batch with increased timeout for reliability
    async with ChatLimiter.for_model(model, timeout=240.0) as limiter:
        results = await process_chat_completion_batch(limiter, requests, config)
    
    # Extract responses and handle errors
    responses = []
    for i, result in enumerate(results):
        # print(f"Batch request {i} result: {result}")
        if result.success:
            # Handle different response formats based on model
            response = result.result
            if hasattr(response, 'choices') and response.choices:
                content = response.choices[0].message.content
                # Handle thinking models that might have reasoning
                if hasattr(response.choices[0].message, 'reasoning') and response.choices[0].message.reasoning:
                    thinking_response = response.choices[0].message.reasoning
                    responses.append(f"<think>{thinking_response}\n</think>\n{content}")
                else:
                    responses.append(content)
            else:
                responses.append(str(response))
        else:
            print(f"Batch request {i} failed: {result.error_message}")
    
    return responses

def get_char_to_token_map(text, tokenizer):
    """Create a mapping from character positions to token positions"""
    token_offsets = tokenizer.encode_plus(text, return_offsets_mapping=True)['offset_mapping']
    
    # Create mapping from character position to token index
    char_to_token = {}
    for token_idx, (start, end) in enumerate(token_offsets):
        for char_pos in range(start, end):
            char_to_token[char_pos] = token_idx
            
    return char_to_token


def _activation_cache_paths(model_id: str, n_examples: int, layer: int) -> tuple[str, str]:
    """
    Returns (activations_pkl_path, mean_pkl_path) for the given cache key.

    NOTE: These paths intentionally mirror `process_saved_responses`'s cache layout.
    """
    activations_pkl = f"../generate-responses/results/vars/activations_{model_id}_{n_examples}_{layer}.pkl"
    mean_pkl = f"../generate-responses/results/vars/activations_{model_id}_{n_examples}_{layer}_mean.pkl"
    return activations_pkl, mean_pkl


def load_activation_mean(model_id: str, n_examples: int, layer: int) -> np.ndarray:
    """
    Load the centering mean vector used to build the cached activations for (model_id, n_examples, layer).

    Returns:
        mean: np.ndarray shape (d_model,) dtype float32
    """
    _acts_pkl, mean_pkl = _activation_cache_paths(model_id=model_id, n_examples=n_examples, layer=layer)
    assert os.path.exists(mean_pkl), (
        f"Missing activation mean file: {mean_pkl}\n"
        "Expected this to be produced by `utils.utils.process_saved_responses` and uploaded alongside the "
        "cached activations."
    )
    with open(mean_pkl, "rb") as f:
        payload = pickle.load(f)
    assert isinstance(payload, dict), f"Bad mean payload type: {type(payload)}"
    assert payload.get("model_id") == model_id, f"Mean model_id mismatch: {payload.get('model_id')} vs {model_id}"
    assert int(payload.get("layer")) == int(layer), f"Mean layer mismatch: {payload.get('layer')} vs {layer}"
    assert int(payload.get("n_examples")) == int(n_examples), f"Mean n_examples mismatch: {payload.get('n_examples')} vs {n_examples}"
    mean = np.asarray(payload.get("activation_mean"))
    assert mean.ndim == 1, f"Expected 1D mean, got shape {mean.shape}"
    mean = mean.astype(np.float32, copy=False)
    assert np.isfinite(mean).all(), "Non-finite values in activation mean"
    return mean


def center_and_l2_normalize_torch(x: torch.Tensor, mean: torch.Tensor) -> torch.Tensor:
    """
    Center and L2-normalize activations using a precomputed mean.

    Supported shapes:
      - x: (d_model,)
      - x: (..., d_model)
    mean:
      - (d_model,)
    Returns:
      - same shape as x, float32
    """
    assert isinstance(x, torch.Tensor), f"x must be torch.Tensor, got {type(x)}"
    assert isinstance(mean, torch.Tensor), f"mean must be torch.Tensor, got {type(mean)}"
    assert mean.ndim == 1, f"Expected mean.ndim==1, got {mean.ndim}"
    assert x.ndim >= 1, f"Expected x.ndim>=1, got {x.ndim}"
    assert x.shape[-1] == mean.shape[0], f"Bad shapes: x.shape={tuple(x.shape)} mean.shape={tuple(mean.shape)}"

    x_f32 = x.to(dtype=torch.float32)
    mean_f32 = mean.to(device=x_f32.device, dtype=torch.float32)
    centered = x_f32 - mean_f32

    denom = torch.norm(centered, dim=-1, keepdim=True)
    assert torch.isfinite(denom).all(), "Non-finite norm in center_and_l2_normalize_torch"
    assert torch.all(denom > 0), "Zero-norm encountered in center_and_l2_normalize_torch"
    return centered / denom

def center_and_normalize_activations(all_activations, overall_mean):
    """Centers and normalizes activations."""
    
    print_and_flush(f"Centering activations...")
    start_time = time.time()
    all_activations = [x - overall_mean for x in all_activations]
    all_activations = np.stack([a.reshape(-1) for a in all_activations])
    norms = np.linalg.norm(all_activations, axis=1, keepdims=True)
    all_activations = all_activations / norms
    end_time = time.time()
    print(f"Centered and normalized activations in {end_time - start_time} seconds")

    return all_activations

def process_saved_responses(model_name, n_examples, model, tokenizer, layer_or_layers):
    """Load and process saved responses to get activations"""

    # Ensure layer_or_layers is a list
    if isinstance(layer_or_layers, (int, str)):
        layers_to_process = [int(layer_or_layers)]
    else:
        layers_to_process = [int(l) for l in layer_or_layers]

    model_id = model_name.split('/')[-1].lower()
    
    # Dictionary to store results for each layer
    results_by_layer = {}
    
    # Check for cached files for each layer
    uncached_layers = []
    for layer in layers_to_process:
        pickle_filename, _mean_filename = _activation_cache_paths(model_id=model_id, n_examples=n_examples, layer=layer)
        if os.path.exists(pickle_filename):
            print(f"Loading cached activations for layer {layer} from {pickle_filename}...")
            with open(pickle_filename, 'rb') as f:
                results_by_layer[layer] = pickle.load(f)
        else:
            uncached_layers.append(layer)

    if not uncached_layers:
        print("All requested layers were loaded from cache.")
        # If only one layer was requested, return in the old format for backward compatibility
        if len(layers_to_process) == 1:
            return results_by_layer[layers_to_process[0]]
        return results_by_layer

    print(f"Processing saved responses for layers: {uncached_layers}...")
    
    # Load responses if there are any uncached layers
    responses_json_path = f"../generate-responses/results/vars/responses_{model_id}.json"
    print(f"Loading responses from {responses_json_path}...")
    with open(responses_json_path, 'r') as f:
        responses_data = json.load(f)
    
    # Limit to n_examples
    random.shuffle(responses_data)
    responses_data = responses_data[:n_examples]
    
    # Initialize data structures for uncached layers
    activations_by_layer = {layer: [] for layer in uncached_layers}
    texts_by_layer = {layer: [] for layer in uncached_layers}
    mean_by_layer = {layer: torch.zeros(1, model.config.hidden_size) for layer in uncached_layers}
    count_by_layer = {layer: 0 for layer in uncached_layers}

    def clear_gpu_memory():
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    clear_gpu_memory()

    # Pre-filter responses with valid thinking processes and precompute metadata
    valid_items = []
    for response_data in responses_data:
        thinking_process = extract_thinking_process(response_data["full_response"])
        if not thinking_process:
            continue
        full_response = response_data["full_response"]
        sentences = split_into_sentences(thinking_process)
        valid_items.append({
            "full_response": full_response,
            "sentences": sentences,
        })

    # Sort by token length for efficient batching (less padding waste)
    for item in valid_items:
        item["_tok_len"] = len(tokenizer.encode(item["full_response"]))
    valid_items.sort(key=lambda x: x["_tok_len"])

    # Dynamic batching: cap total tokens per batch to avoid OOM from nnsight trace overhead
    MAX_BATCH_TOKENS = 16_000
    MAX_BATCH_SIZE = 16
    batches = []
    i = 0
    while i < len(valid_items):
        batch = [valid_items[i]]
        max_len = valid_items[i]["_tok_len"]
        i += 1
        while i < len(valid_items) and len(batch) < MAX_BATCH_SIZE:
            candidate_max = max(max_len, valid_items[i]["_tok_len"])
            if candidate_max * (len(batch) + 1) > MAX_BATCH_TOKENS:
                break
            batch.append(valid_items[i])
            max_len = candidate_max
            i += 1
        batches.append(batch)

    print(f"Extracting activations for {len(valid_items)} valid responses "
          f"({len(batches)} batches, max_tokens={MAX_BATCH_TOKENS}) "
          f"across layers {uncached_layers}...")

    # Use forward hooks instead of nnsight for batched extraction (avoids memory leaks)
    raw_model = model._model if hasattr(model, '_model') else model

    for batch_items in tqdm(batches):
        batch_texts = [item["full_response"] for item in batch_items]

        encodings = tokenizer(
            batch_texts, return_tensors="pt", padding=True, truncation=False,
        ).to(raw_model.device)
        input_ids = encodings["input_ids"]
        attention_mask = encodings["attention_mask"]

        item_seq_info = []
        for b_idx in range(len(batch_items)):
            seq_len = int(attention_mask[b_idx].sum().item())
            max_seq_len = int(attention_mask.shape[1])
            item_seq_info.append((seq_len, max_seq_len))

        layer_outputs = {}
        hooks = []
        for layer in uncached_layers:
            def _make_hook(l):
                def hook_fn(module, inp, out):
                    hidden = out[0] if isinstance(out, tuple) else out
                    layer_outputs[l] = hidden.detach().cpu().to(torch.float32)
                return hook_fn
            h = raw_model.model.layers[layer].register_forward_hook(_make_hook(layer))
            hooks.append(h)

        with torch.no_grad():
            raw_model(input_ids=input_ids, attention_mask=attention_mask)

        for h in hooks:
            h.remove()
        del input_ids, attention_mask, encodings, hooks
        clear_gpu_memory()

        # Process each item in the batch (all data is on CPU now)
        for b_idx, item in enumerate(batch_items):
            full_response = item["full_response"]
            sentences = item["sentences"]
            seq_len, max_seq_len = item_seq_info[b_idx]
            pad_offset = max_seq_len - seq_len

            char_to_token = get_char_to_token_map(full_response, tokenizer)

            for layer in uncached_layers:
                layer_output = layer_outputs[layer][b_idx:b_idx+1, pad_offset:, :]
                min_token_start = float('inf')
                max_token_end = -float('inf')

                for sentence in sentences:
                    text_pos = full_response.find(sentence)
                    if text_pos >= 0:
                        token_start = char_to_token.get(text_pos, None)
                        token_end = char_to_token.get(text_pos + len(sentence), None)

                        if token_start is not None and token_end is not None and token_start < token_end:
                            if token_start < min_token_start:
                                min_token_start = token_start
                            if token_end > max_token_end:
                                max_token_end = token_end

                            segment = layer_output[:, token_start - 1:token_end, :]
                            assert segment.shape[1] > 0, (
                                f"Empty token slice at layer {layer}: token_start={token_start}, "
                                f"token_end={token_end}, sentence='{sentence[:80]}'"
                            )
                            segment_activations = segment.mean(dim=1).numpy()
                            assert np.isfinite(segment_activations).all(), (
                                f"Layer {layer}: non-finite values after numpy conversion"
                            )

                            activations_by_layer[layer].append(segment_activations)
                            texts_by_layer[layer].append(sentence)

                if min_token_start < layer_output.shape[1] and max_token_end > 0:
                    vector = layer_output[:, min_token_start:max_token_end, :].mean(dim=1).cpu()
                    mean_by_layer[layer] = mean_by_layer[layer] + (vector - mean_by_layer[layer]) / (count_by_layer[layer] + 1)
                    count_by_layer[layer] += 1

        del layer_outputs

    # Save results for each newly processed layer
    for layer in uncached_layers:
        print(f"Found {len(activations_by_layer[layer])} sentences with activations for layer {layer} across {count_by_layer[layer]} examples")
        if len(activations_by_layer[layer]) == 0:
            raise ValueError(f"No activations found for layer {layer} across {count_by_layer[layer]} examples")

        overall_running_mean = mean_by_layer[layer].cpu().numpy()
        overall_running_mean = np.asarray(overall_running_mean).reshape(-1).astype(np.float32, copy=False)
        assert overall_running_mean.shape == (model.config.hidden_size,), (
            f"Bad mean shape: {overall_running_mean.shape} vs expected {(model.config.hidden_size,)}"
        )

        # Persist the mean used for centering (required for downstream SAE usage parity).
        _pickle_filename, mean_filename = _activation_cache_paths(model_id=model_id, n_examples=n_examples, layer=layer)
        with open(mean_filename, "wb") as f:
            pickle.dump(
                {
                    "model_id": model_id,
                    "layer": int(layer),
                    "n_examples": int(n_examples),
                    "count_vectors": int(count_by_layer[layer]),
                    "activation_mean": overall_running_mean,
                },
                f,
            )
        print(f"Saved activation mean for layer {layer} to {mean_filename}")

        # Center and normalize activations
        activations_by_layer[layer] = center_and_normalize_activations(activations_by_layer[layer], overall_running_mean)
        
        result = (activations_by_layer[layer], texts_by_layer[layer])
        results_by_layer[layer] = result
        
        pickle_filename, _mean_filename = _activation_cache_paths(model_id=model_id, n_examples=n_examples, layer=layer)
        with open(pickle_filename, 'wb') as f:
            pickle.dump(result, f)
        print(f"Saved activations for layer {layer} to {pickle_filename}")

    # If only one layer was requested, return in the old format
    if len(layers_to_process) == 1:
        return results_by_layer[layers_to_process[0]]
        
    return results_by_layer


def load_model(device="auto", load_in_8bit=False, model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"):
    """
    Load model, tokenizer and mean vectors. Optionally compute feature vectors.
    
    Args:
        load_in_8bit (bool): If True, load the model in 8-bit mode
        model_name (str): Name/path of the model to load
    """
    model = LanguageModel(model_name, dispatch=True, load_in_8bit=load_in_8bit, device_map=device, dtype=torch.bfloat16)
    
    model.generation_config.temperature=None
    model.generation_config.top_p=None
    model.generation_config.top_k=None
    model.generation_config.do_sample=False
    
    tokenizer = model.tokenizer

    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    return model, tokenizer

def custom_generate_with_steering(model, tokenizer, input_ids, max_new_tokens, steering_vector=None, layer=None, normalize=False, coefficient=1.0):
    """
    Generate text while steering with a specific feature vector.
    
    Args:
        model: The model to use for generation
        tokenizer: The tokenizer
        input_ids: Input token ids
        max_new_tokens: Maximum number of tokens to generate
        steering_vector: Vector to use for steering (should match model hidden size)
        layer: Layer index to apply steering to
        coefficient: Strength of steering (higher values = stronger effect)
        sae: Sparse Autoencoder model to use for activation-based steering
    """
    model_layers = model.model.layers

    with model.generate(
        {
            "input_ids": input_ids, 
            "attention_mask": (input_ids != tokenizer.pad_token_id).long()
        },
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.pad_token_id,
    ) as tracer:
        # Apply .all() to model to ensure interventions work across all generations
        model_layers.all()

        if steering_vector is not None and layer is not None:
            # Convert steering vector to correct device and dtype if needed
            steering_vector = steering_vector.to(model.device).to(model.dtype)
            avg_norm = model.model.layers[layer].output[0][:, 1:, :].norm(dim=-1).mean(dim=1)
            if normalize:
                steering_vector = steering_vector.unsqueeze(0).unsqueeze(0) * avg_norm
            model.model.layers[layer].output[0][:, 1:, :] += coefficient * steering_vector
        
        outputs = model.generator.output.save()
                    
    return outputs

def get_random_distinct_colors(labels):
    """
    Generate random distinct ANSI colors for each label.
    
    Args:
        labels: List of label names
        
    Returns:
        Dictionary mapping labels to ANSI color codes
    """
    # List of distinct ANSI colors (excluding black, white, and hard-to-see colors)
    # Format is "\033[COLORm" where COLOR is a number between 31-96
    distinct_colors = [
        "\033[31m",  # Red
        "\033[32m",  # Green
        "\033[33m",  # Yellow
        "\033[34m",  # Blue
        "\033[35m",  # Magenta
        "\033[36m",  # Cyan
        "\033[91m",  # Bright Red
        "\033[92m",  # Bright Green
        "\033[93m",  # Bright Yellow
        "\033[94m",  # Bright Blue
        "\033[95m",  # Bright Magenta
        "\033[96m",  # Bright Cyan
    ]
    
    # Shuffle the colors to randomize them
    random.shuffle(distinct_colors)
    
    # Ensure we have enough colors
    if len(labels) > len(distinct_colors):
        # If we need more colors, create additional ones with random RGB values
        additional_needed = len(labels) - len(distinct_colors)
        for _ in range(additional_needed):
            # Generate random RGB foreground color (38;2;r;g;b)
            r, g, b = random.randint(50, 255), random.randint(50, 255), random.randint(50, 255)
            # Ensure colors are distinct by checking minimum distance from existing colors
            # (simplified approach)
            distinct_colors.append(f"\033[38;2;{r};{g};{b}m")
    
    # Assign colors to labels
    label_colors = {}
    for i, label in enumerate(labels):
        label_colors[label] = distinct_colors[i % len(distinct_colors)]
    
    return label_colors

# Create NumpyEncoder for JSON serialization
class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        if isinstance(obj, np.floating):
            return float(obj)
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.bool_):
            return bool(obj)
        return super(NumpyEncoder, self).default(obj)


# Function to convert numpy types to Python native types
def convert_numpy_types(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.integer, np.int64, np.int32, np.int16, np.int8)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32, np.float16)):
        return float(obj)
    elif isinstance(obj, np.bool_):
        return bool(obj)
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list) or isinstance(obj, tuple):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj


def process_batch_annotations(thinking_processes):
    """Annotate a batch of reasoning chains using the 7-category reasoning framework."""
    annotated_responses = []
    for thinking in thinking_processes:
        annotated_response = chat(f"""
Please annotate the following reasoning trace by marking segments with categories from the reasoning framework below. Use this format: ["<category>"] ... ["<end-section>"]. A sentence can be split into multiple segments if it exhibits different behaviors. Only use the categories provided below.

Reasoning Framework Categories:

1. problem-identification-framing – Problem Identification and Framing
   *Description:* This reflects the model's initial orientation toward the problem—an explicit commitment to focus attention on a particular question or task. It's not solving yet; it's mentally staking out the terrain and clarifying the goal.
   *Includes:* Explicit declarations of the question or topic to be addressed; clarifying scope or rephrasing the goal of the reasoning.
   *Excludes:* Any move toward analysis, solution generation, or speculation.
   *Examples:* "Okay, so I'm trying to figure out how pressure affects the boiling point of water.", "Okay, so I'm trying to figure out the ripple effects of making college education free."

2. metacognitive-setup – Metacognitive Setup and Decomposition Initiation
   *Description:* This captures the model's pre-analytic cognitive preparation—noticing uncertainty or complexity and deciding to plan, organize, or scaffold the reasoning process before diving in.
   *Includes:* Metacognitive statements about strategy or planning; moves to mentally break a problem into manageable parts.
   *Excludes:* Execution of any actual reasoning steps or guesses.
   *Examples:* "Hmm, let me think about this step by step.", "Let me try to visualize this.", "I'm not entirely sure where to start, but I think it's important to break it down step by step."

3. stepwise-calculation – Stepwise Calculation / Enumeration / Local Inference
   *Description:* This cluster captures the model's mechanistic reasoning—applying rules, performing arithmetic, listing possibilities. It's executing a mental algorithm.
   *Includes:* Arithmetic, combinatorics, enumeration of cases; explicit inferences from rules or facts.
   *Excludes:* High-level summaries or contextual reasoning.
   *Examples:* "3 times 7 is 21, and 21 times 11 is 231.", "So, the probability of drawing a red on the first draw is 4 out of 7, which is 4/7.", "Each face is a base for one pyramid, so 6 pyramids."

4. generating-alternatives – Generating Alternatives / Hypotheses
   *Description:* This cluster reflects the model's attempt to expand the hypothesis space. It's not committing to an answer—it's surfacing possible explanations, mechanisms, or paths forward.
   *Includes:* Generative thinking under uncertainty; multiple speculative branches or mechanisms.
   *Excludes:* Final answers or rule-based deductions.
   *Examples:* "Or maybe it's about controlling invasive species.", "Maybe it's just the body's way of fighting off the infection.", "I should also consider different scenarios."

5. information-seeking – Information-Seeking and Epistemic Uncertainty
   *Description:* The model confronts a knowledge gap and initiates action to resolve it. This is a pivot away from internal reasoning toward acquiring more information.
   *Includes:* Statements of uncertainty paired with information-seeking intent; declarations that external info is needed.
   *Excludes:* Internal speculation without intent to learn more; passive confusion without action.
   *Examples:* "I should probably look up some information to get a better understanding.", "Maybe I should ask someone or look it up to find out more information.", "I think I'll just have to check online or maybe ask a friend."

6. consequence-projection – Consequence Projection / Scenario Elaboration
   *Description:* This is forward simulation. The model is running a mental model of the world to ask: "What would happen if...?"
   *Includes:* Counterfactuals, conditionals, and policy simulation; exploration of second- or third-order effects.
   *Excludes:* Simple cause-effect or binary conclusions.
   *Examples:* "Also, with more free time, people might pursue further education.", "If the species affects farming, there might be compensation programs.", "Cities might save money on road repairs due to AVs."

7. conclusion-articulation – (Sub)-Conclusion Articulation
   *Description:* This is the "wrap up this step" reflex. It's when the model finishes part of the reasoning and states a result—before continuing onward.
   *Includes:* (Partial) conclusions or intermediate inferences; logic checkpoints or sanity checks.
   *Excludes:* Problem framing or speculative reasoning.
   *Examples:* "So, each face is a base for one pyramid, so 6 pyramids.", "So, the next month is December, which is D.", "So, if the surgeon is the mother, then yes, the patient is her son."

Reasoning trace to annotate:
{thinking}

Only return the annotated text using the specified format. Do not include any explanation or commentary outside the annotations.
If the last sentence is not finished, do not include it in the annotations.
""")
        annotated_responses.append(annotated_response)
    
    return annotated_responses


model_mapping = {
    "meta-llama/Llama-3.1-8B":"deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
    "Qwen/Qwen2.5-0.5B":"Open-Reasoner-Zero/Open-Reasoner-Zero-0.5B",
    "Qwen/Qwen2.5-1.5B":"Open-Reasoner-Zero/Open-Reasoner-Zero-1.5B",
    "Qwen/Qwen2.5-Math-1.5B":"deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
    "Qwen/Qwen2.5-7B":"Open-Reasoner-Zero/Open-Reasoner-Zero-7B",
    "Qwen/Qwen2.5-14B":"deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    "Qwen/Qwen2.5-32B":"Qwen/QwQ-32B",
    "meta-llama/Llama-3.3-70B-Instruct": "deepseek-ai/DeepSeek-R1-Distill-Llama-70B",
}

def split_into_sentences(text, min_words=3):
    """
    Split text into sentences and filter based on quality criteria.
    
    Args:
        text (str): The input text to split into sentences
        
    Returns:
        list: List of cleaned sentences with at least 3 words each
    """
    # Split after sentence-ending punctuation and newlines while keeping delimiters
    # Use positive lookbehind to split after delimiters, but with edge case handling
    
    # First handle edge cases by temporarily replacing problematic patterns
    # Protect decimal numbers like "3.14" and single letter abbreviations like "E. coli"
    protected_text = text
    replacements = []
    
    # Protect decimal numbers
    for match in re.finditer(r'\d+\.\d+', text):
        placeholder = f"__DECIMAL_{len(replacements)}__"
        replacements.append((placeholder, match.group()))
        protected_text = protected_text.replace(match.group(), placeholder)
    
    # Protect single letter abbreviations (letter followed by period and space/word)
    for match in re.finditer(r'\b[A-Za-z]\.\s+[A-Za-z]', text):
        placeholder = f"__ABBREV_{len(replacements)}__"
        replacements.append((placeholder, match.group()))
        protected_text = protected_text.replace(match.group(), placeholder)
    
    # Protect mathematical expressions like "k!" (letter followed by exclamation)
    for match in re.finditer(r'\b[A-Za-z]!', text):
        placeholder = f"__MATH_{len(replacements)}__"
        replacements.append((placeholder, match.group()))
        protected_text = protected_text.replace(match.group(), placeholder)
    
    # Handle consecutive punctuation by normalizing it first
    # Replace consecutive punctuation with single punctuation for splitting
    consecutive_punct_pattern = r'([.!?;])\1+'
    consecutive_matches = []
    for match in re.finditer(consecutive_punct_pattern, protected_text):
        consecutive_matches.append((match.start(), match.end(), match.group()))
    
    # Split using simple lookbehind after normalizing consecutive punctuation
    normalized_text = re.sub(consecutive_punct_pattern, r'\1', protected_text)
    sentences = re.split(r'(?<=[.!?;\n])', normalized_text)
    
    # Restore consecutive punctuation in the sentences
    if consecutive_matches:
        # Map back to original positions
        for start, end, original in consecutive_matches:
            # Find which sentence contains this punctuation and restore it
            for i, sentence in enumerate(sentences):
                if sentence and start < len(protected_text):
                    # This is a simplified restoration - may need refinement for complex cases
                    if original[0] in sentence and len(original) > 1:
                        sentences[i] = sentence.replace(original[0], original, 1)
    
    # Restore protected patterns
    for placeholder, original in replacements:
        sentences = [s.replace(placeholder, original) for s in sentences]
    
    # Clean up sentences
    sentences = [s.strip() for s in sentences if s.strip()]
    sentences = [s for s in sentences if len(s.split()) >= min_words]
    
    # Post-processing: Handle sentences that start with quotes after period splits
    # If a sentence starts with a quote, move it to the end of the previous sentence
    processed_sentences = []
    for i, sentence in enumerate(sentences):
        if i > 0 and sentence.startswith('"') and processed_sentences:
            # Move the quote to the previous sentence and remove it from current
            processed_sentences[-1] += '"'
            current_sentence = sentence[1:].strip()  # Remove quote and leading space
            if current_sentence and len(current_sentence.split()) >= min_words:
                processed_sentences.append(current_sentence)
        else:
            processed_sentences.append(sentence)
    
    return processed_sentences


def load_steering_vectors(device: str = "cpu", hyperparams_dir: str | None = None, vectors_dir: str | None = None, verbose: bool = False):
    """Load all optimized steering vectors and return a mapping of category -> vector.

    The optimizer (see `train-vectors/optimize_steering_vectors.py`) saves:
      1. Hyperparameter files:  train-vectors/results/vars/hyperparams/steering_vector_hyperparams_{model}_{idx}.json
         Each JSON has a top-level key ``category`` indicating which reasoning category the vector targets.
         Also loads steering_vector_hyperparams_{model}_bias.json for the general bias vector.
      2. Vector files:          train-vectors/results/vars/optimized_vectors/{model}_idx{idx}.pt
         Each ``.pt`` file stores a ``dict`` mapping that same category name to a ``torch.Tensor``.
         Also loads {model}_bias.pt for the general bias vector.

    This helper searches those directories, pairs the JSONs and ``.pt`` files by *model* and *idx*,
    then builds a single dictionary ``{category_name: tensor_on_requested_device}``.

    Parameters
    ----------
    device : str, default "cpu"
        Device onto which the vectors should be loaded (e.g. "cuda", "cuda:0").
    hyperparams_dir : str | None
        Path to directory containing the hyperparameter JSONs.  If ``None`` we use the
        default ``cot-interp/train-vectors/results/vars/hyperparams``.
    vectors_dir : str | None
        Path to directory containing the ``.pt`` vector files.  If ``None`` we use the
        default ``cot-interp/train-vectors/results/vars/optimized_vectors``.
    verbose : bool, default False
        If True, print information about loaded vectors and any files that could not be matched.

    Returns
    -------
    dict[str, torch.Tensor]
        Mapping from reasoning *category* to the corresponding steering *vector* tensor.
        Also includes a "bias" key for the general bias vector if available.
    """

    import glob  # Local import to avoid slowing start-up when not needed

    # Resolve default directories relative to this utils.py file
    base_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "train-vectors"))

    if hyperparams_dir is None:
        hyperparams_dir = os.path.join(base_dir, "results", "vars", "hyperparams")
    if vectors_dir is None:
        vectors_dir = os.path.join(base_dir, "results", "vars", "optimized_vectors")

    if verbose:
        print_and_flush(f"Loading steering vectors from:\n  Hyperparams: {hyperparams_dir}\n  Vectors:     {vectors_dir}")

    # Pattern to extract {model_name_short} and {idx} from filenames
    hp_pattern = re.compile(r"steering_vector_hyperparams_(.+?)_(idx\d+|bias)\.json")

    category_to_vector: dict[str, torch.Tensor] = {}

    for hp_path in glob.glob(os.path.join(hyperparams_dir, "steering_vector_hyperparams_*.json")):
        hp_file = os.path.basename(hp_path)
        match = hp_pattern.match(hp_file)
        if match is None:
            if verbose:
                print_and_flush(f"[load_steering_vectors] Skipping unrecognised file name: {hp_file}")
            continue

        model_name_short, idx_str = match.groups()
        # Handle both numbered indices and "bias"
        vector_path = os.path.join(vectors_dir, f"{model_name_short}_{'idx' + idx_str if idx_str.isdigit() else idx_str}.pt")

        # Load hyperparameters JSON to get the category name
        try:
            with open(hp_path, "r") as f:
                hp_data = json.load(f)
            category = hp_data.get("category")
        except Exception as e:
            if verbose:
                print_and_flush(f"[load_steering_vectors] Failed to read {hp_file}: {e}")
            continue

        if category is None:
            if verbose:
                print_and_flush(f"[load_steering_vectors] No 'category' field in {hp_file}. Skipping.")
            continue

        # Ensure the vector file exists
        if not os.path.exists(vector_path):
            if verbose:
                print_and_flush(f"[load_steering_vectors] Vector file not found for {category}: {vector_path}")
            continue

        try:
            vec_dict = torch.load(vector_path, map_location=device)
        except Exception as e:
            if verbose:
                print_and_flush(f"[load_steering_vectors] Could not load tensor from {vector_path}: {e}")
            continue

        # The saved dict is {category_name: tensor}
        if category not in vec_dict:
            # Some older runs may save just the tensor; handle that case.
            if isinstance(vec_dict, torch.Tensor):
                vector_tensor = vec_dict
            else:
                if verbose:
                    print_and_flush(f"[load_steering_vectors] Category '{category}' not in vector file {vector_path}. Keys: {list(vec_dict.keys())}")
                continue
        else:
            vector_tensor = vec_dict[category]

        # Move tensor to desired device and ensure float32/float16 is preserved
        vector_tensor = vector_tensor.to(device)
        # For bias vector, store under "bias" key regardless of what's in the JSON
        if idx_str == "bias":
            category_to_vector["bias"] = vector_tensor
        else:
            category_to_vector[category] = vector_tensor

        if verbose:
            print_and_flush(f"[load_steering_vectors] Loaded vector for '{category}' from {idx_str} (model {model_name_short})")

    if verbose:
        print_and_flush(f"[load_steering_vectors] Loaded {len(category_to_vector)} vectors.")

    return category_to_vector


# ---------------------------------------------------------------------------
# Annotation-based per-question per-category character-count rankings
# ---------------------------------------------------------------------------

def load_annotation_rankings(
    annotations_path: str,
    n_clusters: int | None = None,
) -> dict[int, dict[str, int]]:
    """Parse an annotated-responses JSON and compute per-question, per-category
    character counts.

    Returns:
        {question_id: {cat_id_str: total_chars_in_that_category, ...}, ...}
    """
    with open(annotations_path) as f:
        data = json.load(f)

    pat = re.compile(r'\["[\d.]+:idx(\d+)"\](.*?)\["end-section"\]', re.DOTALL)
    rankings: dict[int, dict[str, int]] = {}
    for entry in data:
        qid = entry["question_id"]
        counts: dict[str, int] = {}
        for m in pat.finditer(entry.get("annotated_thinking", "")):
            idx = m.group(1)
            if n_clusters is not None and int(idx) >= n_clusters:
                continue
            counts[idx] = counts.get(idx, 0) + len(m.group(2))
        rankings[qid] = counts
    return rankings


def select_top_questions_per_category(
    rankings: dict[int, dict[str, int]],
    n_per_category: int,
    n_categories: int,
) -> tuple[dict[str, list[int]], set[int]]:
    """For each category, return question_ids ranked by character count (desc).

    Returns:
        per_cat_top: {cat_id_str: [qid, qid, ...]}  (top n_per_category)
        union_qids:  set of all question_ids in the union of per-cat tops
    """
    per_cat_top: dict[str, list[int]] = {}
    for cat_id in range(n_categories):
        cid = str(cat_id)
        scored = [
            (qid, counts.get(cid, 0))
            for qid, counts in rankings.items()
        ]
        scored.sort(key=lambda x: -x[1])
        per_cat_top[cid] = [qid for qid, _ in scored[:n_per_category]]
    union_qids = set()
    for qids in per_cat_top.values():
        union_qids.update(qids)
    return per_cat_top, union_qids


def select_balanced_bias_questions(
    rankings: dict[int, dict[str, int]],
    n_total: int,
    n_categories: int,
    exclude_qids: set[int] | None = None,
) -> list[int]:
    """Select a balanced set of question_ids for bias diff-of-means.

    Takes the top examples from each category in round-robin fashion,
    skipping duplicates, until n_total unique questions are collected.
    """
    exclude = exclude_qids or set()
    per_cat_ranked: dict[str, list[int]] = {}
    for cat_id in range(n_categories):
        cid = str(cat_id)
        scored = [
            (qid, counts.get(cid, 0))
            for qid, counts in rankings.items()
            if qid not in exclude
        ]
        scored.sort(key=lambda x: -x[1])
        per_cat_ranked[cid] = [qid for qid, sc in scored if sc > 0]

    selected: list[int] = []
    seen: set[int] = set()
    cat_ptrs = {cid: 0 for cid in per_cat_ranked}
    while len(selected) < n_total:
        added_any = False
        for cid in sorted(per_cat_ranked.keys(), key=int):
            if len(selected) >= n_total:
                break
            ranked = per_cat_ranked[cid]
            while cat_ptrs[cid] < len(ranked):
                qid = ranked[cat_ptrs[cid]]
                cat_ptrs[cid] += 1
                if qid not in seen:
                    selected.append(qid)
                    seen.add(qid)
                    added_any = True
                    break
        if not added_any:
            break
    return selected
