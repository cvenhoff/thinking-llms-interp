# %%
import dotenv
dotenv.load_dotenv(".env")

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
from nnsight import NNsight
from collections import defaultdict
from tqdm import tqdm
import random
import json
import os
import time  # Add this import at the top
import gc

# %% Load model

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"  # Can be changed to use different models

num_rollouts = -1
max_examples_per_first_token = 100

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.bfloat16)
model = NNsight(model).to("cuda")

model.generation_config.temperature=None
model.generation_config.top_p=None

model.eval()  # Ensure model is in eval mode
torch.set_grad_enabled(False)  # Disable gradient computation

mean_vectors = defaultdict(lambda: {
    'mean': torch.zeros(model.config.num_hidden_layers, model.config.hidden_size),
    'count': 0
})

# %% Define functions

def process_model_output(prompt_and_model_response_input_ids, model):
    """Get model output and layer activations"""
    start_time = time.time()
    layer_outputs = []
    
    with model.trace(prompt_and_model_response_input_ids):
        for layer_idx in range(model.config.num_hidden_layers):
            layer_outputs.append(model.model.layers[layer_idx].output[0].save())
    
    # Stack tensors and move to CPU immediately
    layer_outputs = torch.cat([x.value.cpu().detach().to(torch.float32) for x in layer_outputs], dim=0)
    
    # Clear CUDA cache
    torch.cuda.empty_cache()
    gc.collect()
    
    elapsed = time.time() - start_time
    return layer_outputs

def get_label_positions(annotated_response: str, prompt_and_model_response_input_ids: list[int], tokenizer: AutoTokenizer):
    """Parse annotations and find token positions for each label"""
    start_time = time.time()
    label_positions = {}
    pattern = r'\["([\w-]+)"\]([^\[]+)'
    matches = re.finditer(pattern, annotated_response)
    
    for match in matches:
        labels = [label.strip() for label in match.group(1).strip('"').split(',')]
        if "end-section" in labels:
            continue

        # Get the text between the label and the next label or end-section
        text = match.group(2).strip()

        # Encode the text and remove the BOS token
        text_tokens: list[int] = tokenizer.encode(text)[1:]
        
        # Find the position of the text in the thinking tokens
        # Once found, we save the positions for each label
        for j in range(len(prompt_and_model_response_input_ids) - len(text_tokens) + 1):
            fragment = prompt_and_model_response_input_ids[j:j + len(text_tokens)]
            if fragment == text_tokens:
                for label in labels:
                    if label not in label_positions:
                        label_positions[label] = []
                    token_start = j
                    token_end = j + len(text_tokens)
                    label_positions[label].append((token_start, token_end))
                break
    
    elapsed = time.time() - start_time
    return label_positions

def should_skip_position(label, text, used_counts, max_examples):
    """Determine if we should skip this specific position based on frequency caps"""
    return used_counts[label][text] >= max_examples

def update_mean_vectors(mean_vectors, layer_outputs, label_positions, used_counts, index, max_examples=10):
    """Update mean vectors for overall and individual labels, skipping overused positions"""

    # Process all labels
    for label, positions in label_positions.items():
        valid_positions = []
        for start, end in positions:
            text = tokenizer.decode(prompt_and_model_response_input_ids[0][start:start+1])
            if not should_skip_position(label, text, used_counts, max_examples):
                valid_positions.append((start, end))
                used_counts[label][text] += 1
                
        if not valid_positions:
            continue
        
        vectors = []
        for start, end in valid_positions:
            vectors.append(layer_outputs[:, start-1:start+1])
        vectors = torch.cat(vectors, dim=1)
        mean_vector = vectors.mean(dim=1)
        
        # Update mean for this label
        current_count = mean_vectors[label]['count']
        current_mean = mean_vectors[label]['mean']
        mean_vectors[label]['mean'] = current_mean + (mean_vector - current_mean) / (current_count + 1)
        mean_vectors[label]['count'] += 1

        current_count = mean_vectors['overall']['count']
        current_mean = mean_vectors['overall']['mean']
        mean_vectors['overall']['mean'] = current_mean + (mean_vector - current_mean) / (current_count + 1)
        mean_vectors['overall']['count'] += 1

def calculate_next_token_frequencies(responses_data, tokenizer):
    """Calculate frequencies of next tokens for each label"""
    label_token_frequencies = defaultdict(lambda: defaultdict(int))
    
    for response in responses_data:
        annotated_text = response["annotated_response"]
        pattern = r'\["([\w-]+)"\](.*?)\["end-section"\]'
        matches = re.finditer(pattern, annotated_text, re.DOTALL)
        
        for match in matches:
            label = match.group(1)
            text = match.group(2).strip()
            # Get first token after label
            tokens = tokenizer.encode(text)[1:2]  # Just get the first token
            if tokens:
                next_token_text = tokenizer.decode(tokens)
                label_token_frequencies[label][next_token_text] += 1
    
    return label_token_frequencies

def should_skip_example(label, next_token, used_counts, max_examples=50):
    """Determine if we should skip this example based on frequency caps"""
    if used_counts[label][next_token] >= max_examples:
        return True
    return False

# %% Load data

save_every = 10
save_path = f"data/mean_vectors_{model_name.split('/')[-1].lower()}.pt"

annotated_responses_json_path = f"data/annotated_responses_{model_name.split('/')[-1].lower()}.json"
original_messages_json_path = f"data/base_responses_{model_name.split('/')[-1].lower()}.json"

tasks_json_path = "data/tasks.json"

if not os.path.exists(annotated_responses_json_path):
    raise FileNotFoundError(f"Annotated responses file not found at {annotated_responses_json_path}")
if not os.path.exists(original_messages_json_path):
    raise FileNotFoundError(f"Original messages file not found at {original_messages_json_path}")
if not os.path.exists(tasks_json_path):
    raise FileNotFoundError(f"Tasks file not found at {tasks_json_path}")

print(f"Loading existing annotated responses from {annotated_responses_json_path}")
with open(annotated_responses_json_path, 'r') as f:
    if num_rollouts == -1:
        annotated_responses_data = json.load(f)["responses"]
        random.shuffle(annotated_responses_data)
    else:
        annotated_responses_data = random.sample(json.load(f)["responses"], k=num_rollouts)

print(f"Loading existing original messages from {original_messages_json_path}")
with open(original_messages_json_path, 'r') as f:
    original_messages_data = json.load(f)["responses"]
random.shuffle(original_messages_data)

print(f"Loading existing tasks from {tasks_json_path}")
with open(tasks_json_path, 'r') as f:
    tasks_data = json.load(f)

# %% Calculate token frequencies for each label
label_token_frequencies = calculate_next_token_frequencies(annotated_responses_data, tokenizer)

# %%

# Track how many times we've used each token for each label
used_counts = defaultdict(lambda: defaultdict(int))


for i, annotated_response_data in tqdm(enumerate(annotated_responses_data), total=num_rollouts, desc="Processing annotated responses"):
    iter_start_time = time.time()
    response_uuid = annotated_response_data["response_uuid"]

    # Fetch the task and base response data
    task_data = next((task for task in tasks_data if task["task_uuid"] == annotated_response_data["task_uuid"]), None)
    base_response_data = next((msg for msg in original_messages_data if msg["response_uuid"] == response_uuid), None)

    # Build prompt message, appending the task prompt and the original response
    prompt_message = [task_data["prompt_message"]]
    prompt_message_input_ids = tokenizer.apply_chat_template(prompt_message, add_generation_prompt=True, return_tensors="pt").to("cuda")
    base_response_str = base_response_data["response_str"]
    if base_response_str.startswith("<think>"):
        # Remove the <think> tag prefix, since we already added it to the prompt_message_input_ids
        base_response_str = base_response_str[len("<think>"):]
    base_response_input_ids = tokenizer.encode(base_response_str, add_special_tokens=False, return_tensors="pt").to("cuda")

    prompt_and_model_response_input_ids = torch.cat([prompt_message_input_ids, base_response_input_ids], dim=1)

    # Move tensors to CPU after use
    prompt_message_input_ids = prompt_message_input_ids.cpu()
    base_response_input_ids = base_response_input_ids.cpu()
    prompt_and_model_response_input_ids = prompt_and_model_response_input_ids.cpu()

    # Get the positions for each label in the combined tokenized prompt and model response
    label_positions = get_label_positions(annotated_response_data["annotated_response"], prompt_and_model_response_input_ids[0].tolist(), tokenizer)
    
    # Get activations and update mean vectors, skipping only overused positions
    layer_outputs = process_model_output(prompt_and_model_response_input_ids, model)
    update_mean_vectors(mean_vectors, layer_outputs, label_positions, used_counts, i, max_examples_per_first_token)
    
    if i % save_every == 0:
        save_dict = {k: {'mean': v['mean'], 'count': v['count']} for k, v in mean_vectors.items()}
        torch.save(save_dict, save_path)
        print(f"Current mean_vectors: {mean_vectors.keys()}. Saved...")
        print("Token usage statistics:")
        for label in used_counts:
            print(f"{label}: {dict(used_counts[label])}")
        iter_elapsed = time.time() - iter_start_time
        # print(f"GPU Memory allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        # print(f"GPU Memory cached: {torch.cuda.memory_reserved() / 1e9:.2f} GB")

    # Clear memory
    del layer_outputs

# Save final results
save_dict = {k: {'mean': v['mean'], 'count': v['count']} for k, v in mean_vectors.items()}
torch.save(save_dict, save_path)
print("Saved final mean vectors")

# %%