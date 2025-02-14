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
from deepseek_steering.utils import prepare_model_input

# %%
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
model, tokenizer, mean_vectors_dict = utils.load_model_and_vectors(compute_features=False, model_name=model_name)

max_length = 500

# %%
with open(f'data/base_responses_{model_name.split("/")[-1].lower()}.json', 'r') as f:
    base_results = json.load(f)["responses"]

with open(f'data/annotated_responses_{model_name.split("/")[-1].lower()}.json', 'r') as f:
    annotated_results = json.load(f)["responses"]

with open(f'data/tasks.json', 'r') as f:
    tasks_data = json.load(f)

# %%
labels = ['uncertainty-estimation','adding-knowledge', 'example-testing', 'backtracking']
target_positions = 1  # Number of positions to analyze per label

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

def collect_gradients_and_compute_effects(model, input_ids, start, end, steering_vector):
    """Helper function to collect gradients and compute effects for a position
    
    Args:
        model: The model to analyze
        input_ids: Input token ids
        start: Start position to analyze
        end: End position for context
        steering_vector: Vector to use for computing effects
    """
    layer_activations = []
    layer_gradients = []
    
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
    
    patching_effects = []

    for layer_idx in range(model.config.num_hidden_layers):
        gradients = layer_gradients[layer_idx][0, start-1:start]
        normed_steering_vector = steering_vector / steering_vector.norm()
        vector = normed_steering_vector * layer_activations[layer_idx][0, start-1].norm()

        effect = torch.einsum('d,sd->s', -vector, gradients).mean()
        patching_effects.append(effect.cpu().item())

    del layer_activations, layer_gradients
    torch.cuda.empty_cache()
    
    return patching_effects

def analyze_layer_effects(model, input_ids, label, feature_vectors, label_positions):

    patching_effects = [0 for _ in range(model.config.num_hidden_layers)]

    if len(label_positions) == 0:
        return None

    for pos in label_positions:
        start, end = pos
        effects = collect_gradients_and_compute_effects(
            model, input_ids, start, end, feature_vectors[label]
        )
        patching_effects = [p + e for p, e in zip(patching_effects, effects)]
        
    patching_effects = [effect / len(label_positions) for effect in patching_effects]
    return patching_effects

def analyze_layer_effects_fixed_vector(model, input_ids, label, mean_vectors_dict, label_positions, fixed_layer_idx):
    
    patching_effects = [0 for _ in range(model.config.num_hidden_layers)]

    if len(label_positions) == 0:
        return None

    # Get the fixed steering vector from the specified layer
    feature_activation = mean_vectors_dict[label]['mean'].to(torch.bfloat16).to("cuda") - mean_vectors_dict['overall']['mean'].to(torch.bfloat16).to("cuda")
    steering_vector = feature_activation[fixed_layer_idx]

    for pos in label_positions:
        start, end = pos
        effects = collect_gradients_and_compute_effects(
            model, input_ids, start, end, steering_vector
        )
        patching_effects = [p + e for p, e in zip(patching_effects, effects)]
        
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
      
            input_ids = prepare_model_input(
                response_uuid=example["response_uuid"],
                annotated_responses_data=annotated_results,
                tasks_data=tasks_data,
                original_messages_data=base_results,
                tokenizer=tokenizer
            )['prompt_and_response_ids']

            original_full_text = tokenizer.decode(input_ids[0])

            label_positions = find_label_positions(annotated_text, original_full_text, tokenizer, label)
            label_positions = [x for x in label_positions if x[1] < max_length]
                    
            if label_positions:
                # Only process up to remaining needed positions
                positions_to_process = min(len(label_positions), target_positions - processed_positions)
                label_positions = label_positions[:positions_to_process]
                processed_positions += positions_to_process

                effects = analyze_layer_effects_fixed_vector(
                    model,
                    input_ids,
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
# print layer with max effect for each label
steering_vectors = {}
for label in labels:
    max_effect = max(fixed_vector_effects[label].values())
    max_layer = max(fixed_vector_effects[label], key=fixed_vector_effects[label].get)
    print(f"Label: {label}, Max Effect: {max_effect}, Max Layer: {max_layer}")
    feature_vector = mean_vectors_dict[label]['mean'][int(max_layer)] - mean_vectors_dict['overall']['mean'][int(max_layer)]
    steering_vectors[label] = feature_vector.to(torch.bfloat16).to("cuda")

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
                        
        input_ids = prepare_model_input(
            response_uuid=example["response_uuid"],
            annotated_responses_data=annotated_results,
            tasks_data=tasks_data,
            original_messages_data=base_results,
            tokenizer=tokenizer
        )['prompt_and_response_ids']

        original_full_text = tokenizer.decode(input_ids[0])

        label_positions = find_label_positions(annotated_text, original_full_text, tokenizer, label)
        label_positions = [x for x in label_positions if x[1] < max_length]
            

        if label_positions:
            # Only process up to remaining needed positions
            positions_to_process = min(len(label_positions), target_positions - processed_positions)
            label_positions = label_positions[:positions_to_process]
            processed_positions += positions_to_process

            effects = analyze_layer_effects(
                model,
                input_ids,
                label,
                steering_vectors,
                label_positions
            )

            if effects:
                layer_effects[label].append(effects)

json.dump(layer_effects, open(f'data/layer_effects_{model_name.split("/")[-1].lower()}.json', 'w'))

torch.save(steering_vectors, f'data/steering_vectors_{model_name.split("/")[-1].lower()}.pt')
layer_effects = json.load(open(f'data/layer_effects_{model_name.split("/")[-1].lower()}.json', 'r'))
plot_layer_effects(layer_effects, model_name)

# %%
for layer in range(model.config.num_hidden_layers):
    print("-"*100)
    print(f"Layer: {layer}")
    unembed_weights = model.lm_head.weight / model.lm_head.weight.norm()
    feature_vectors = {}
    for label in labels:
        feature_vector = mean_vectors_dict[label]['mean'][14] - mean_vectors_dict['overall']['mean'][14]
        feature_vectors[label] = feature_vector / feature_vector.norm()

    for label, feature_vector in feature_vectors.items():
        dot_product = unembed_weights @ feature_vector.T.to(torch.bfloat16).to("cuda")
        max_index = torch.argmax(dot_product)
        max_index_value = dot_product.max()
        print(f"Label: {label}, Max Token: {tokenizer.decode([max_index])}, Max Value: {max_index_value}")
# %%
