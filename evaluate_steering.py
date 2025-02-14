# %%
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from nnsight import NNsight
from deepseek_steering.utils import chat, chat
import re
import json
import random
from deepseek_steering.messages import eval_messages
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
from collections import defaultdict
import gc
import os
import deepseek_steering.utils as utils
import asyncio

# %%
def load_model_and_vectors(model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"):
    model, tokenizer, feature_vectors = utils.load_model_and_vectors(compute_features=True, model_name=model_name)
    return model, tokenizer, feature_vectors


def generate_and_analyze(model, tokenizer, message, feature_vectors, label, layer_effects, steer_mode="none"):
    input_ids = tokenizer.apply_chat_template([message], add_generation_prompt=True, return_text=True)
    
    steer_positive = True if steer_mode == "positive" else False

    layers = []
    effects = torch.stack([torch.tensor(x) for x in layer_effects[label]], dim=0).mean(0)

    positive_indices = [i for i, effect in enumerate(effects) if effect > 0]
    # Sort indices by effect size and select layers until sum reaches threshold
    sorted_indices = sorted(positive_indices, key=lambda i: effects[i], reverse=True)
    selected_effects_sum = 0
    target_sum = 0.01
    
    for idx in sorted_indices:
        layers.append(idx)
        selected_effects_sum += effects[idx]
        if selected_effects_sum >= target_sum:
            break
    
    print(f"Selected effects sum: {selected_effects_sum}")
    print("Layers: ", layers)

    output_ids = utils.custom_generate_with_projection_removal(
        model,
        tokenizer,
        input_ids,
        max_new_tokens=500,
        label=label if steer_mode != "none" else "none",
        feature_vectors=feature_vectors,
        layers=layers,
        coefficient=1,
        steer_positive=steer_positive,
        show_progress=False
    )
    
    response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
    
    # Extract thinking process
    think_start = response.index("<think>") + len("<think>")
    try:
        think_end = response.index("</think>")
    except ValueError:
        think_end = len(response)
    thinking_process = response[think_start:think_end].strip()
    
    return thinking_process


def plot_label_statistics(results, model_name):
    # Create figures directory if it doesn't exist
    os.makedirs('figures', exist_ok=True)
    
    # Get model identifier for file naming
    model_id = model_name.split('/')[-1].lower()
    
    # Use white background
    plt.style.use('seaborn-v0_8-white')
    
    fig, ax = plt.subplots(figsize=(12, 7))
    labels_list = list(results.keys())
    x = np.arange(len(labels_list))
    width = 0.25
    
    # Calculate means as before
    original_means = []
    positive_means = []
    negative_means = []
    
    for label in labels_list:
        orig_fracs = [ex["original"]["label_fractions"].get(label, 0) for ex in results[label]]
        pos_fracs = [ex["positive"]["label_fractions"].get(label, 0) for ex in results[label]]
        neg_fracs = [ex["negative"]["label_fractions"].get(label, 0) for ex in results[label]]
        
        original_means.append(np.mean(orig_fracs))
        positive_means.append(np.mean(pos_fracs))
        negative_means.append(np.mean(neg_fracs))
    
    # Plot bars with black edges
    ax.bar(x - width, original_means, width, label='Original', color='#2E86C1', alpha=0.8, edgecolor='black', linewidth=1)
    ax.bar(x, positive_means, width, label='Positive Steering', color='#27AE60', alpha=0.8, edgecolor='black', linewidth=1)
    ax.bar(x + width, negative_means, width, label='Negative Steering', color='#E74C3C', alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add percentage labels on top of bars
    def add_labels(positions, values):
        for pos, val in zip(positions, values):
            ax.text(pos, val, f'{val*100:.1f}%', ha='center', va='bottom', fontsize=14)
    
    add_labels(x - width, original_means)
    add_labels(x, positive_means)
    add_labels(x + width, negative_means)
    
    # Improve styling with larger font sizes and bold title
    ax.set_ylabel('Average Token Fraction (%)', fontsize=24, labelpad=10)
    ax.set_title('DeepSeek-R1-Distill-Llama-8B', fontsize=24, pad=20, weight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([label.replace('-', '\n') for label in labels_list], rotation=0, fontsize=24)
    ax.tick_params(axis='y', labelsize=16)
    
    # Convert y-axis to percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0f}%'.format(y * 100)))
    
    # Add grid for better readability
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    
    # Customize legend with larger font
    ax.legend(frameon=True, fancybox=True, shadow=True, fontsize=20)
    
    # Show all spines (lines around the plot)
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    
    plt.tight_layout()
    plt.savefig(f'figures/steering_results_{model_id}.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

# %% Parameters
n_examples = 1
random.seed(42)

# Create data directory if it doesn't exist
os.makedirs('data', exist_ok=True)

# Load model and vectors
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"  # Can be changed to use different models
model_id = model_name.split('/')[-1].lower()

layer_effects = json.load(open(f'data/layer_effects_{model_name.split("/")[-1].lower()}.json', 'r'))

# %%
model, tokenizer, feature_vectors = load_model_and_vectors(model_name)

# %% Randomly sample evaluation examples
eval_indices = random.sample(range(len(eval_messages)), n_examples)

# Store results
labels = ['adding-knowledge', 'uncertainty-estimation', 'example-testing', 'backtracking']
results = {label: [] for label in labels}

# First phase: Generate all responses
all_generations = {}  # Will store all generated responses
for label in labels:
    all_generations[label] = []
    for idx in tqdm(eval_indices, desc=f"Generating responses for {label}"):
        message = eval_messages[idx]
        
        # Generate all versions for this example
        example_generations = {
            "original": generate_and_analyze(model, tokenizer, message, feature_vectors, label, layer_effects, "none"),
            "positive": generate_and_analyze(model, tokenizer, message, feature_vectors, label, layer_effects, "positive"),
            "negative": generate_and_analyze(model, tokenizer, message, feature_vectors, label, layer_effects, "negative")
        }
        all_generations[label].append(example_generations)

# Second phase: Create all chat requests for parallel processing
all_chat_requests = []
request_mapping = []  # Keep track of which request belongs to which label/index/mode

for label in labels:
    for idx, example_generations in enumerate(all_generations[label]):
        for mode, thinking_process in example_generations.items():
            prompt = f"""
            Please split the following reasoning chain of an LLM into annotated parts using labels and the following format ["label"]...["end-section"]. A sentence should be split into multiple parts if it incorporates multiple behaviours indicated by the labels.

            Available labels:
            0. initializing -> The model is rephrasing the given task and states initial thoughts.
            1. deduction -> The model is performing a deduction step based on its current approach and assumptions.
            2. adding-knowledge -> The model is enriching the current approach with recalled facts.
            3. example-testing -> The model generates examples to test its current approach.
            4. uncertainty-estimation -> The model is stating its own uncertainty.
            5. backtracking -> The model decides to change its approach.

            The reasoning chain to analyze:
            {thinking_process}

            Answer only with the annotated text. Only use the labels outlined above. If there is a tail that has no annotation leave it out.
            """
            all_chat_requests.append(prompt)
            request_mapping.append((label, idx, mode))

# %% Process all chat requests in parallel
all_annotated_responses = []
for i in tqdm(range(len(all_chat_requests)), desc="Processing chat requests"):
    all_annotated_responses.append(chat(all_chat_requests[i], max_tokens=1000))

# %% Third phase: Process responses and organize results
results = {label: [] for label in labels}

current_label = None
current_example = None
temp_results = {}

def get_label_counts(thinking_process, labels):
    # Initialize sentence counts for each label
    label_sentence_counts = {label: 0 for label in labels}
    
    # Find all annotated sections
    pattern = r'\["([\w-]+)"\]([^\[]+)'
    matches = re.finditer(pattern, thinking_process)
    
    # Count total annotated sentences
    total_sentences = 0
    
    # Count sentences for each label
    for match in matches:
        label = match.group(1)
        text = match.group(2).strip()
        if label != "end-section" and label in labels:
            # Count sentences in this section (split by periods followed by space or newline)
            sentences = len(re.split(r'[.!?]+[\s\n]+', text.strip()))
            if sentences == 0:  # Handle case where there's just one sentence without ending punctuation
                sentences = 1
            label_sentence_counts[label] += sentences
            total_sentences += sentences
    
    # Convert to fractions
    label_fractions = {
        label: count / total_sentences if total_sentences > 0 else 0 
        for label, count in label_sentence_counts.items()
    }
            
    return label_fractions

# %%
for (label, idx, mode), annotated_response in zip(request_mapping, all_annotated_responses):
    thinking_process = all_generations[label][idx][mode]
    
    # Get label fractions using sentence counting
    label_fractions = get_label_counts(annotated_response, labels)
    
    # Store results
    if current_label != label or current_example != idx:
        if current_label is not None:
            results[current_label].append(temp_results)
        temp_results = {}
        current_label = label
        current_example = idx
    
    temp_results[mode] = {
        "thinking_process": thinking_process,
        "label_fractions": label_fractions,
        "annotated_response": annotated_response
    }

# Add the last example
if current_label is not None:
    results[current_label].append(temp_results)

# Save results
with open(f'data/steering_evaluation_results_{model_id}.json', 'w') as f:
    json.dump(results, f, indent=2)

# %% Plot statistics
results = json.load(open(f'data/steering_evaluation_results_{model_id}.json'))
plot_label_statistics(results, model_name)

# %%
