# %%
import torch
import json
from transformers import AutoTokenizer, AutoModelForCausalLM
from nnsight import NNsight
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import torch.nn.functional as F
import deepseek_steering.utils as utils

# %%
def find_label_positions(annotated_response, original_text, tokenizer, label):
    """Parse annotations and find token positions for each label"""
    label_positions = []
    pattern = f'\\["{label}"\\]([^\\[]+?)(?=\\[|$)'
    matches = re.finditer(pattern, annotated_response)
    thinking_tokens = tokenizer.encode(original_text)[1:]
    
    for match in matches:

        text = match.group(1).strip()
        text_tokens = tokenizer.encode(text)[1:]
        
        for j in range(len(thinking_tokens) - len(text_tokens) + 1):
            if thinking_tokens[j:j + len(text_tokens)] == text_tokens:
                token_start = j
                token_end = j + len(text_tokens)
                label_positions.append((token_start, token_end))
                continue
    
    return label_positions

def compute_cross_entropy_metric(logits):
    """Compute cross entropy between predicted distribution and detached version"""
    probs = F.softmax(logits, dim=-1)
    detached_probs = F.softmax(logits.detach(), dim=-1)
    return F.cross_entropy(logits, detached_probs.argmax(dim=-1))

def analyze_layer_effects(model, tokenizer, text, label, mean_vectors_dict, label_positions):
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to("cuda")
    
    patching_effects = [0 for _ in range(model.config.num_hidden_layers)]

    if len(label_positions) == 0:
        return None

    for pos in label_positions:
        layer_activations = []
        layer_gradients = []
        
        start, end = pos

        with model.trace() as tracer:
            with tracer.invoke(input_ids[:, :end]) as invoker:
                # Collect activations from each layer
                for layer_idx in range(model.config.num_hidden_layers):
                    layer_activations.append(model.model.layers[layer_idx].output[0].save())
                    layer_gradients.append(model.model.layers[layer_idx].output[0].grad.save())
                
                # Get logits for the endpoints
                logits = model.lm_head.output.save()
                
                # Compute cross entropy metric for each labeled section
                value = compute_cross_entropy_metric(logits[0, start])

                # Backward pass
                value.backward()
    
        layer_activations = [layer_activations[i].value for i in range(model.config.num_hidden_layers)]
        layer_gradients = [layer_gradients[i].value for i in range(model.config.num_hidden_layers)]

        feature_activation = mean_vectors_dict[label]['mean'].to(torch.bfloat16).to("cuda") - mean_vectors_dict['overall']['mean'].to(torch.bfloat16).to("cuda")

        for layer_idx in range(model.config.num_hidden_layers):
            # Get activations and gradients for the entire labeled section
            activations = layer_activations[layer_idx][0, start-1:start]
            gradients = layer_gradients[layer_idx][0, start-1:start]
            
            effect = torch.einsum('d,sd->s', -feature_activation[layer_idx], gradients).mean()
            
            patching_effects[layer_idx] += effect.cpu().item()

        del activations, gradients, effect, feature_activation, layer_activations, layer_gradients
        torch.cuda.empty_cache()
        
    patching_effects = [effect / len(label_positions) for effect in patching_effects]

    return patching_effects

def analyze_layer_effects_fixed_vector(model, tokenizer, text, label, mean_vectors_dict, label_positions, fixed_layer_idx):
    """Analyze effects when using a fixed steering vector from one layer applied to all layers"""
    input_ids = tokenizer(text, return_tensors="pt").input_ids.to("cuda")
    
    patching_effects = [0 for _ in range(model.config.num_hidden_layers)]

    if len(label_positions) == 0:
        return None

    # Get the fixed steering vector from the specified layer
    feature_activation = mean_vectors_dict[label]['mean'].to(torch.bfloat16).to("cuda") - mean_vectors_dict['overall']['mean'].to(torch.bfloat16).to("cuda")
    fixed_steering_vector = feature_activation[fixed_layer_idx]

    for pos in label_positions:
        layer_activations = []
        layer_gradients = []
        
        start, end = pos

        with model.trace() as tracer:
            with tracer.invoke(input_ids[:, :end]) as invoker:
                # Collect activations from each layer
                for layer_idx in range(model.config.num_hidden_layers):
                    layer_activations.append(model.model.layers[layer_idx].output[0].save())
                    layer_gradients.append(model.model.layers[layer_idx].output[0].grad.save())
                
                logits = model.lm_head.output.save()
                value = compute_cross_entropy_metric(logits[0, start])
                value.backward()
    
        layer_activations = [layer_activations[i].value for i in range(model.config.num_hidden_layers)]
        layer_gradients = [layer_gradients[i].value for i in range(model.config.num_hidden_layers)]

        for layer_idx in range(model.config.num_hidden_layers):
            gradients = layer_gradients[layer_idx][0, start-1:start]
            effect = torch.einsum('d,sd->s', -fixed_steering_vector, gradients).mean()
            patching_effects[layer_idx] += effect.cpu().item()
        
    patching_effects = [effect / len(label_positions) for effect in patching_effects]
    return patching_effects

def plot_layer_effects(layer_effects, model_name):
    # Set white background
    plt.figure(figsize=(12, 8), facecolor='white')
    ax = plt.gca()
    ax.set_facecolor('white')
    
    # Remove ggplot style to keep default lines
    # plt.style.use('ggplot')  # Removing this line
    
    # Color scheme
    colors = ['#2E86C1', '#E67E22', '#27AE60', '#C0392B']
    
    for (label, effects), color in zip(layer_effects.items(), colors):
        if not effects:  # Skip if no effects for this label
            continue
            
        effects_array = np.array(effects)
        mean_effects = np.mean(effects_array, axis=0)
        
        # Apply smoothing using convolution
        window_size = 2  # Increase coarseness by reducing window size
        kernel = np.ones(window_size) / window_size
        smoothed_effects = np.convolve(mean_effects, kernel, mode='valid')
        
        x = range(len(smoothed_effects))

        std_effects = np.std(effects_array, axis=0)
        std_smoothed = np.convolve(std_effects, kernel, mode='valid')
        
        plt.fill_between(x, 
                        smoothed_effects - std_smoothed,
                        smoothed_effects + std_smoothed,
                        alpha=0.2, 
                        color=color)
        
        plt.plot(x, smoothed_effects, 
                label="{}".format(label.replace('-', '\n').title()),
                color=color,
                linewidth=2.5,
                marker='o',
                markersize=4)
    
    plt.xlabel('Layer', fontsize=24, labelpad=10, color='black')  # Set font color to black
    plt.ylabel('Mean Cross Entropy', fontsize=24, labelpad=10, color='black')  # Set font color to black
    plt.title('DeepSeek-R1-Distill-Llama-8B', fontsize=24, pad=20, color='black')  # Set font color to black
    plt.xticks(fontsize=24, color='black')  # Set font color to black
    plt.yticks(fontsize=24, color='black')  # Set font color to black
    
    # Remove offset on x-axis
    ax.margins(x=0)

    # Add box and grid with stronger visibility
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)  # Make the box lines thicker
        spine.set_color('black')  # Set explicit color
    
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    
    # Enhanced grid settings
    plt.grid(True, 
             linestyle='--',      # Dashed lines
             alpha=0.4,           # More opaque
             color='gray',        # Gray color
             which='major')       # Show major grid lines
    
    plt.legend(bbox_to_anchor=(1, 1), 
              loc='upper right', 
              borderaxespad=0.,
              frameon=True,
              fontsize=22,
              facecolor='#f5f5f5')  # Set light gray background for the axes
    
    plt.tight_layout()
    
    model_id = model_name.split('/')[-1].lower()
    
    plt.savefig(f'figures/layer_effects_{model_id}.pdf', 
                dpi=300, 
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none')
    plt.show()
    plt.close()

def plot_fixed_vector_effects(fixed_vector_effects, model_name):
    plt.figure(figsize=(12, 8), facecolor='white')
    ax = plt.gca()
    ax.set_facecolor('white')
    
    colors = ['#2E86C1', '#E67E22', '#27AE60', '#C0392B']
    
    for (label, effects_dict), color in zip(fixed_vector_effects.items(), colors):
        if not effects_dict:  # Skip if no effects for this label
            continue
            
        # Convert dictionary to arrays for plotting
        layers = sorted(list(map(int, effects_dict.keys())))
        total_effects = [effects_dict[str(layer)] for layer in layers]
        
        plt.plot(layers, total_effects,
                label=f"{label.replace('-', ' ').title()}",
                color=color,
                linewidth=2.5,
                marker='o',
                markersize=4)
    
    plt.xlabel('Source Layer of Steering Vector', fontsize=24, labelpad=10, color='black')
    plt.ylabel('Total Cross Entropy Effect', fontsize=24, labelpad=10, color='black')
    plt.title('DeepSeek-R1-Distill-Llama-8B\nFixed Vector Analysis', fontsize=24, pad=20, color='black')
    plt.xticks(fontsize=24, color='black')
    plt.yticks(fontsize=24, color='black')
    
    ax.margins(x=0)
    
    for spine in ax.spines.values():
        spine.set_linewidth(1.5)
        spine.set_color('black')
    
    ax.spines['top'].set_visible(True)
    ax.spines['right'].set_visible(True)
    ax.spines['bottom'].set_visible(True)
    ax.spines['left'].set_visible(True)
    
    plt.grid(True, linestyle='--', alpha=0.4, color='gray', which='major')
    
    plt.legend(bbox_to_anchor=(1, 1),
              loc='upper right',
              borderaxespad=0.,
              frameon=True,
              fontsize=22,
              facecolor='#f5f5f5')
    
    plt.tight_layout()
    
    model_id = model_name.split('/')[-1].lower()
    plt.savefig(f'figures/fixed_vector_effects_{model_id}.pdf',
                dpi=300,
                bbox_inches='tight',
                facecolor='white',
                edgecolor='none')
    plt.show()
    plt.close()

# %%
# Load model and data
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
model, tokenizer, mean_vectors_dict = utils.load_model_and_vectors(compute_features=False, model_name=model_name)

max_length = 500

# %%
with open(f'data/base_responses_{model_name.split("/")[-1].lower()}.json', 'r') as f:
    base_results = json.load(f)["responses"]

with open(f'data/annotated_responses_{model_name.split("/")[-1].lower()}.json', 'r') as f:
    annotated_results = json.load(f)["responses"]

# %%
labels = ['uncertainty-estimation','adding-knowledge', 'example-testing', 'backtracking']
target_positions = 1  # Number of positions to analyze per label

# %%
fixed_vector_effects = {label: {} for label in labels}

for label in labels:
    print(f"Analyzing label: {label}")
    
    # For each layer as source of steering vector
    for fixed_layer_idx in range(model.config.num_hidden_layers):
        print(f"Using steering vector from layer {fixed_layer_idx}")
        layer_total_effects = []
        processed_positions = 0
        
        for annotated_example in tqdm(annotated_results):
            if processed_positions >= target_positions:
                break
                
            annotated_text = annotated_example['annotated_response']
            
            example = next((x for x in base_results if x['response_uuid'] == annotated_example['response_uuid']), None)
            original_text = example['response_str']
            
            label_positions = find_label_positions(annotated_text, original_text, tokenizer, label)
            label_positions = [x for x in label_positions if x[1] < max_length]
            
            if label_positions:
                # Only process up to remaining needed positions
                positions_to_process = min(len(label_positions), target_positions - processed_positions)
                label_positions = label_positions[:positions_to_process]
                processed_positions += positions_to_process

                effects = analyze_layer_effects_fixed_vector(
                    model,
                    tokenizer,
                    original_text,
                    label,
                    mean_vectors_dict,
                    label_positions,
                    fixed_layer_idx
                )
                
                if effects:
                    layer_total_effects.append(sum(effects))
        
        if layer_total_effects:
            fixed_vector_effects[label][str(fixed_layer_idx)] = np.mean(layer_total_effects)

json.dump(fixed_vector_effects, open(f'data/fixed_vector_effects_{model_name.split("/")[-1].lower()}.json', 'w'))

# Plot the results
plot_fixed_vector_effects(fixed_vector_effects, model_name)

# %%
# Store results
layer_effects = {label: [] for label in labels}

# Analyze each label
for label in labels:
    print(f"Analyzing label: {label}")
    processed_positions = 0
    
    for annotated_example in tqdm(annotated_results):
        if processed_positions >= target_positions:
            break
            
        annotated_text = annotated_example['annotated_response']
        
        example = next((x for x in base_results if x['response_uuid'] == annotated_example['response_uuid']), None)
        original_text = example['response_str']

        label_positions = find_label_positions(annotated_text, original_text, tokenizer, label)
        label_positions = [x for x in label_positions if x[1] < max_length]

        if label_positions:
            # Only process up to remaining needed positions
            positions_to_process = min(len(label_positions), target_positions - processed_positions)
            label_positions = label_positions[:positions_to_process]
            processed_positions += positions_to_process

            effects = analyze_layer_effects(
                model,
                tokenizer,
                original_text,
                label,
                mean_vectors_dict,
                label_positions
            )

            if effects:
                layer_effects[label].append(effects)

json.dump(layer_effects, open(f'data/layer_effects_{model_name.split("/")[-1].lower()}.json', 'w'))

# %% Plot results
layer_effects = json.load(open(f'data/layer_effects_{model_name.split("/")[-1].lower()}.json', 'r'))
plot_layer_effects(layer_effects, model_name)
# %%
