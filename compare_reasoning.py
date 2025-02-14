# %%
from deepseek_steering.utils import chat
import json
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
from transformers import AutoTokenizer
import re

# %%
def get_label_counts(thinking_process, tokenizer, labels, annotate_response=True):
    if annotate_response:
        # Get annotated version using chat function
        annotated_response = chat(f"""
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
        Ensure that the annotated sentences are exactly the same as before, including linebreaks and spacing.
        """,
        max_tokens=1000
        )
    else:
        annotated_response = thinking_process
    
    # Initialize sentence counts for each label
    label_sentence_counts = {label: 0 for label in labels}
    
    # Find all annotated sections
    pattern = r'\["([\w-]+)"\]([^\[]+)'
    matches = re.finditer(pattern, annotated_response)
    
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
            
    return label_fractions, annotated_response

def process_chat_response(message, tokenizer, labels):
    """Process a single message through chat function"""
    response = chat(f"""
    Please answer the following question:
    
    Question:
    {message}
    
    Please format your response like this:
    <think>
    ...
    </think>
    [Your answer here]
    """)
    
    # Extract thinking process
    think_start = response.index("<think>") + len("<think>")
    try:
        think_end = response.index("</think>")
    except ValueError:
        think_end = len(response)
    thinking_process = response[think_start:think_end].strip()
    
    label_fractions, annotated_response = get_label_counts(thinking_process, tokenizer, labels)
    
    return {
        "annotated_response": thinking_process,
        "label_fractions": label_fractions,
        "annotated_response": annotated_response
    }

def plot_comparison(chat_results, deepseek_results, labels, model_name):
    """Plot comparison between chat and deepseek results"""
    os.makedirs('figures', exist_ok=True)
    model_id = model_name.split('/')[-1].lower()
    
    # Calculate mean fractions for each label
    chat_means = []
    deepseek_means = []
    
    for label in labels:
        chat_fracs = [ex["label_fractions"].get(label, 0) for ex in chat_results]
        deepseek_fracs = [ex["label_fractions"].get(label, 0) for ex in deepseek_results]
        
        chat_means.append(np.mean(chat_fracs))
        deepseek_means.append(np.mean(deepseek_fracs))
    
    # Create bar plot with wider aspect ratio
    plt.style.use('seaborn-v0_8-paper')  # Use a clean scientific style
    fig, ax = plt.subplots(figsize=(14, 6))
    x = np.arange(len(labels))
    width = 0.35
    
    # Enhanced black box around the plot with slightly thicker lines
    for spine in ax.spines.values():
        spine.set_color('black')
        spine.set_linewidth(1.5)
    
    # Use more professional colors and add slight transparency
    bars1 = ax.bar(x - width/2, chat_means, width, label='GPT-4o', 
                   color='#2E86C1', alpha=0.85, edgecolor='black', linewidth=1)
    bars2 = ax.bar(x + width/2, deepseek_means, width, label='DeepSeek-R1-Distill-Qwen-14B', 
                   color='#E67E22', alpha=0.85, edgecolor='black', linewidth=1)
    
    # Improve grid and ticks
    ax.yaxis.grid(True, linestyle='--', alpha=0.7, zorder=0)
    ax.set_axisbelow(True)  # Put grid below bars
    
    # Set y-axis limit with more headroom and add label
    ymax = max(max(chat_means), max(deepseek_means))
    ax.set_ylim(0, ymax * 1.15)  # Add 15% headroom
    ax.set_ylabel('Sentence Fraction', fontsize=16)  # Updated y-axis label
    
    # Convert y-axis to percentage
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0f}%'.format(y * 100)))
    
    # Format x-axis labels more professionally
    ax.set_xticks(x)
    formatted_labels = [label.replace('-', ' ').title() for label in labels]
    formatted_labels = [label.replace(' ', '\n') for label in formatted_labels]
    ax.set_xticklabels(formatted_labels, rotation=0, ha='center', fontsize=16)
    ax.tick_params(axis='y', labelsize=16)
    
    # Add value labels on top of bars with more vertical offset and percentage format
    def autolabel(rects):
        for rect in rects:
            height = rect.get_height()
            ax.annotate(f'{height*100:.1f}%',
                       xy=(rect.get_x() + rect.get_width() / 2, height),
                       xytext=(0, 8),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=12)
    
    autolabel(bars1)
    autolabel(bars2)
    
    # Enhance legend
    ax.legend(fontsize=16, frameon=True, framealpha=1, 
             edgecolor='black', bbox_to_anchor=(1, 1.02), 
             loc='upper right', ncol=2)
    
    # Adjust layout and save with high quality
    plt.tight_layout()
    plt.savefig(f'figures/reasoning_comparison_{model_id}.pdf', 
                dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    plt.show()
    plt.close()

# %% Parameters
n_examples = 100
random.seed(42)
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
model_id = model_name.split('/')[-1].lower()
compute_from_json = False

labels = ['initializing', 'deduction', 'adding-knowledge', 'example-testing', 
          'uncertainty-estimation', 'backtracking']

# Create directories
os.makedirs('data', exist_ok=True)

# %% Load model and existing responses
tokenizer = AutoTokenizer.from_pretrained(model_name)

if compute_from_json:
    # Load existing results and recompute scores
    print("Loading existing results and recomputing scores...")
    results = json.load(open(f'data/reasoning_comparison_{model_id}.json'))
    
    # Recompute chat results
    chat_results = []
    for result in tqdm(results["chat_results"], desc="Recomputing chat scores"):
        label_fractions, annotated_response = get_label_counts(
            result["annotated_response"], 
            tokenizer, 
            labels,
            annotate_response=False
        )
        chat_results.append({
            "annotated_response": result["annotated_response"],
            "label_fractions": label_fractions,
            "annotated_response": annotated_response
        })
    
    # Recompute deepseek results
    deepseek_results = []
    for result in tqdm(results["deepseek_results"], desc="Recomputing deepseek scores"):
        label_fractions, annotated_response = get_label_counts(
            result["annotated_response"], 
            tokenizer, 
            labels,
            annotate_response=False
        )
        deepseek_results.append({
            "annotated_response": result["annotated_response"],
            "label_fractions": label_fractions,
            "annotated_response": annotated_response
        })
else:
    # Load existing DeepSeek responses
    with open(f'data/annotated_responses_{model_id}.json', 'r') as f:
        annotated_data = random.sample(json.load(f)["responses"], n_examples)

    with open(f'data/base_responses_{model_id}.json', 'r') as f:
        base_results = json.load(f)["responses"]

    selected_messages = []
    for annotated_example in annotated_data:
        original_example = next((x for x in base_results if x['response_uuid'] == annotated_example['response_uuid']), None)
        original_text = original_example['response_str']
        selected_messages.append(original_text)

    selected_deepseek_responses = []
    for deepseek_example in annotated_data:
        selected_deepseek_responses.append(deepseek_example["annotated_response"])
    
    # Process chat responses
    chat_results = []
    deepseek_results = []

    for message, deepseek_response in tqdm(zip(selected_messages, selected_deepseek_responses), desc="Processing examples"):
        # Process chat response
        chat_result = process_chat_response(message, tokenizer, labels)
        chat_results.append(chat_result)
        
        # Process existing DeepSeek response
        label_fractions, annotated_response = get_label_counts(deepseek_response, tokenizer, labels)
        deepseek_results.append({
            "annotated_response": deepseek_response,
            "label_fractions": label_fractions,
            "annotated_response": annotated_response
        })

# Save results
results = {
    "chat_results": chat_results,
    "deepseek_results": deepseek_results
}

with open(f'data/reasoning_comparison_{model_id}.json', 'w') as f:
    json.dump(results, f, indent=2)

# %% Load and plot results
results = json.load(open(f'data/reasoning_comparison_{model_id}.json'))
chat_results = results["chat_results"]
deepseek_results = results["deepseek_results"]

# %%
plot_comparison(chat_results, deepseek_results, labels, model_name)
