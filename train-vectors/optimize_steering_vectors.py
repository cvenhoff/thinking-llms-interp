# %%
import argparse
import dotenv
dotenv.load_dotenv("../.env")

from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
import re
import os
import random
import json
import sys
import os
# Add the parent directory to the path to ensure we import from the correct utils
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import utils
import gc
from utils import steering_opt
from tqdm import tqdm
import traceback
import wandb
from utils.responses import extract_thinking_process

# %% Parse arguments
parser = argparse.ArgumentParser(description="Optimize steering vectors from annotated responses")
parser.add_argument("--model", type=str, default="meta-llama/Llama-3.1-8B",
                    help="Model to train steering vectors for")
parser.add_argument("--n_training_examples", type=int, default=8,
                    help="Number of training examples to use per category")
parser.add_argument("--save_path", type=str, default="results/vars/optimized_vectors",
                    help="Path to save optimized vectors")
parser.add_argument("--layer", type=int, default=6,
                    help="Layer to optimise the steering vector for")
parser.add_argument("--max_iters", type=int, required=True,
                    help="Maximum optimization iterations")
parser.add_argument("--lr", type=str, default="1e-1",
                    help="Learning rate(s) for optimization. Can be a single value or comma-separated list")
parser.add_argument("--min_lr", type=float, default=0,
                    help="Minimum learning rate for optimization")
parser.add_argument("--warmup_iters", type=int, default=0,
                    help="Number of warmup iterations")
parser.add_argument("--context_sentences", type=int, default=0,
                    help="Number of additional sentences to include after target completion")
parser.add_argument("--test_max_tokens", type=int, default=100,
                    help="Maximum tokens to generate when testing")
parser.add_argument("--load_in_8bit", action="store_true", default=False,
                    help="Load model in 8-bit mode")
parser.add_argument("--optim_minibatch_size", type=int, default=6,
                    help="Size of minibatches for optimization loop")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed")
parser.add_argument("--steering_vector_idx", type=int, default=0,
                    help="Index of the specific steering vector to optimize")
parser.add_argument("--grad_clip", type=float, default=None,
                    help="Maximum L2 norm of gradients for gradient clipping. None means no clipping.")
parser.add_argument("--steering_token_window", type=int, default=None,
                    help="Number of previous tokens in the target completion to apply the steering vector to (None means all)")
parser.add_argument("--use_wandb", action="store_true", default=False,
                    help="Use wandb for logging")
parser.add_argument("--wandb_project", type=str, default="optimize-steering-vectors",
                    help="Wandb project name")
parser.add_argument("--use_activation_perplexity_selection", action="store_true", default=False,
                    help="If set, use activation/perplexity-based selection. If not set, use random sampling over the full set.")
args, _ = parser.parse_known_args()

# At module level
ANNOTATION_PATTERN = re.compile(r'\["([\d.]+):(\S+?)"\](.*?)\["end-section"\]', re.DOTALL)
CATEGORY_PATTERN = re.compile(r'\["[\d.]+:(\S+?)"\]')

def get_label_positions(annotated_thinking, response_text, tokenizer, context_sentences=0):
    """Parse SAE annotations and find token positions for each label"""
    label_positions = {}
    
    # Use a pattern that captures labeled segments in the format [activation:category-name] text [end-section]
    # Now supporting activation strength values in the format [56.86:category-name]
    matches = list(ANNOTATION_PATTERN.finditer(annotated_thinking))
    
    # Create character to token mapping once
    char_to_token = utils.get_char_to_token_map(response_text, tokenizer)
    
    # Split response into sentences for context
    sentences = utils.split_into_sentences(response_text, min_words=0)
    
    for match in matches[:-1]:
        activation_str = match.group(1).strip()
        label = match.group(2).strip()
        text = match.group(3).strip()
        
        try:
            activation = float(activation_str)
        except ValueError:
            print(f"Warning: Could not parse activation value '{activation_str}' for category '{label}'")
            continue
            
        if not text:  # Skip empty text
            continue
            
        # Find this text in the original response
        pattern = r'(?:[.?!;\n]|\n\n)\s*(' + re.escape(text) + ')'
        match = re.search(pattern, response_text)
        text_pos = match.start(1) if match else -1
        if text_pos >= 0:
            # Get start and end token positions
            token_start = char_to_token.get(text_pos, None)
            token_end = char_to_token.get(text_pos + len(text) - 1, None)
            
            # Adjust token_end to include the entire token
            if token_end is not None:
                token_end += 1

            if token_start is None or token_end is None or token_start >= token_end:
                continue
            
            # Find the sentence containing our target text
            target_sentence_idx = -1
            for i, sentence in enumerate(sentences):
                if text in sentence:
                    target_sentence_idx = i
                    break
            
            if target_sentence_idx == -1:
                continue
                
            # Get additional context sentences if requested
            additional_context = ""
            if context_sentences > 0 and target_sentence_idx < len(sentences) - 1:
                end_idx = min(target_sentence_idx + context_sentences + 1, len(sentences))
                additional_sentences = sentences[target_sentence_idx + 1:end_idx]
                
                # Find the original whitespace between sentences
                if additional_sentences:
                    # Get the text up to the end of our target text
                    text_end_pos = text_pos + len(text)
                    # Get the text up to the start of the next sentence
                    next_sentence_start = response_text.find(additional_sentences[0], text_end_pos)
                    if next_sentence_start > text_end_pos:
                        # Extract the original whitespace
                        original_whitespace = response_text[text_end_pos:next_sentence_start]
                        # Use the original whitespace to join sentences
                        additional_context = original_whitespace + original_whitespace.join(additional_sentences)
                    else:
                        # Fallback to space if we can't find the original whitespace
                        additional_context = " " + " ".join(additional_sentences)
                
                # Update token_end to include additional context
                if additional_context:
                    context_end_pos = text_pos + len(text) + len(additional_context)
                    context_token_end = char_to_token.get(context_end_pos - 1, None)
                    if context_token_end is not None:
                        token_end = context_token_end + 1
            
            # If we found valid token positions
            if label not in label_positions:
                label_positions[label] = []
            label_positions[label].append((token_start, token_end, text + additional_context, activation, text_pos))
    
    return label_positions

def extract_examples_for_category(responses_data, category_name, tokenizer, n_training_examples, model):
    """Extract *training* examples for `category_name`.

    Returns (training_examples, max_activation).
    If `--use_activation_perplexity_selection` is off, we random-sample up to the requested
    number; otherwise we use activation + perplexity ranking as before but without keeping a
    separate test split.
    """
    examples_for_category = []
    
    # Process each response to extract labeled segments for the specified category
    n_annotated_thinking_containing_category = 0
    for resp in tqdm(responses_data, desc="Extracting examples for category"):
        if not resp.get('annotated_thinking'):
            continue

        thinking_process = extract_thinking_process(resp["full_response"])

        full_text = f"Task: Answer the question below. Explain your reasoning step by step.\n\n\n\nQuestion:\n{resp['original_message']['content']}\n\nStep by step answer:\n{thinking_process}"
        
        # Look for the specific category in the annotated thinking
        if category_name not in resp['annotated_thinking']:
            continue
            
        n_annotated_thinking_containing_category += 1
        label_positions = get_label_positions(resp['annotated_thinking'], full_text, tokenizer, args.context_sentences)
        
        if category_name in label_positions:
            for start, end, text, activation, text_pos in label_positions[category_name]:
                # Get the text up to this point using the saved text_pos
                context = full_text[:text_pos]

                # Check if context ends properly
                if context[-1] not in ['.', '?', '!', ';', '\n', '\n\n'] and context[-2] not in ['.', '?', '!', ';', '\n', '\n\n'] and context.strip()[-1] not in ['.', '?', '!', ';', '\n', '\n\n']:
                    continue
                
                # Filter out target sequences with fewer than 3 words
                word_count = len(text.strip().split())
                if word_count < 10:
                    continue
                
                examples_for_category.append({
                    'prompt': context,
                    'target_completion': text,
                    'original_question': resp['original_message']['content'],
                    'full_thinking': extract_thinking_process(resp["full_response"]),
                    'activation': activation
                })
    
    if not examples_for_category:
        print(f"No valid examples found for category {category_name}. Exiting.")
        return [], 0.0

    print(f"Found {n_annotated_thinking_containing_category} annotated thinking containing category {category_name}")
    print(f"Found {len(examples_for_category)} examples for category {category_name}")

    # Calculate how many total examples we need to get the desired number of training and test
    # Now: only training examples are returned.
    total_examples_needed = n_training_examples

    if not args.use_activation_perplexity_selection:
        # Use random sampling over the full set
        if len(examples_for_category) > total_examples_needed:
            final_examples = random.sample(examples_for_category, total_examples_needed)
        else:
            final_examples = examples_for_category.copy()
        random.shuffle(final_examples)
        training_examples = final_examples[:total_examples_needed]
        print(f"Final selection (random): {len(training_examples)} training examples")
        return training_examples, 1.0

    # --- CORRECTED LOGIC: Activation pre-filter, then select by perplexity ---
    # 1. Sort all examples by activation (descending)
    examples_for_category_sorted = sorted(examples_for_category, key=lambda x: x['activation'], reverse=True)
    # 2. Take the top 4x the data needed
    sample_size = min(len(examples_for_category_sorted), total_examples_needed * 2)
    sampled_examples = examples_for_category_sorted[:sample_size]

    print(f"Top {sample_size} examples by activation selected for perplexity ranking")

    # 3. Calculate perplexity for each of these examples
    print("Calculating perplexity for sampled examples...")
    examples_with_metrics = []
    for example in tqdm(sampled_examples, desc="Computing perplexity"):
        try:
            # Tokenize the full sequence (prompt + target completion)
            full_text = example['prompt'] + example['target_completion']
            inputs = tokenizer(full_text, return_tensors='pt')
            with torch.no_grad():
                outputs = model(**inputs)
                logits = outputs.logits[0]  # Remove batch dimension
                prompt_tokens = tokenizer(example['prompt'], return_tensors='pt')['input_ids'][0]
                prompt_len = len(prompt_tokens)
                shift_logits = logits[:-1, :].contiguous()
                shift_labels = inputs['input_ids'][0][1:].contiguous()
                target_start = prompt_len - 1
                target_end = len(shift_labels)
                if target_start >= target_end or target_start < 0:
                    continue
                target_logits = shift_logits[target_start:target_end, :]
                target_labels = shift_labels[target_start:target_end]
                if target_logits.size(0) == 0:
                    continue
                loss = torch.nn.functional.cross_entropy(target_logits, target_labels, reduction='mean')
                perplexity = torch.exp(loss).item()
                examples_with_metrics.append({
                    **example,
                    'perplexity': perplexity,
                    'activation': example['activation']
                })
        except Exception as e:
            print(f"Error calculating perplexity for example: {e}")
            continue

    if not examples_with_metrics:
        print("No valid examples with perplexity found. Falling back to random sampling.")
        # Fall back to random sampling if perplexity calculation fails
        max_examples = min(len(examples_for_category), total_examples_needed)
        selected_examples = random.sample(examples_for_category, max_examples)
        # Split based on test percentage
        training_examples = selected_examples[:total_examples_needed]
        return training_examples, 1.0

    # 4. From the top N (train + test) by perplexity (descending), split into train and test
    final_examples = sorted(examples_with_metrics, key=lambda x: x['perplexity'], reverse=True)[:total_examples_needed]

    # 5. Split into train and test
    training_examples = final_examples[:total_examples_needed]

    print(f"Final selection (perplexity): {len(training_examples)} training examples")

    # Calculate max activation before removing the field
    max_activation = max(ex['activation'] for ex in final_examples) if final_examples else 1.0

    # Remove perplexity/activation from the final examples
    for ex in final_examples:
        ex.pop('perplexity', None)
        ex.pop('activation', None)

    return training_examples, max_activation

def generate_bias_examples(responses_data, tokenizer, model, n_training_examples):
    """Generate examples for the *bias* vector.
    Each example uses the *entire* thinking process as the target completion, with the task + question
    (everything *before* the reasoning) as the prompt.

    Selection logic mirrors `extract_examples_for_category` except that we have no activation scores,
    so we optionally rank purely by perplexity.
    """
    all_examples = []

    for resp in tqdm(responses_data, desc="Preparing bias examples"):
        if not resp.get('thinking_process'):
            continue

        # Build prompt exactly as in other parts of the script
        prompt = (
            f"Task: Answer the question below. Explain your reasoning step by step.\n\n\n\nQuestion:\n{resp['original_message']['content']}\n\nStep by step answer:\n"
        )
        target_completion = resp['thinking_process'].strip()

        # Simple quality filter – skip very short reasoning chains
        if len(target_completion.split()) < 20:
            continue

        all_examples.append({
            'prompt': prompt,
            'target_completion': target_completion,
            'original_question': resp['original_message']['content'],
        })

    if not all_examples:
        print("No valid bias examples found. Exiting.")
        return []

    print(f"Collected {len(all_examples)} candidate bias examples")

    total_examples_needed = n_training_examples

    # If the user disabled perplexity selection, just random-sample
    if not args.use_activation_perplexity_selection:
        if len(all_examples) > total_examples_needed:
            selected = random.sample(all_examples, total_examples_needed)
        else:
            selected = all_examples.copy()
        random.shuffle(selected)
        print(f"Selected {len(selected)} random training examples (no perplexity ranking)")
        return selected

    # Otherwise: random pre-sample 2×, then rank by perplexity
    sample_size = min(len(all_examples), total_examples_needed * 2)
    presampled = random.sample(all_examples, sample_size)

    print(f"Presampled {sample_size} examples for perplexity ranking; computing perplexities…")
    examples_with_metrics = []

    for ex in tqdm(presampled, desc="Computing perplexity"):
        try:
            full_txt = ex['prompt'] + ex['target_completion']
            inputs = tokenizer(full_txt, return_tensors='pt')
            with torch.no_grad():
                outputs = model(**inputs.to(model.device))
                logits = outputs.logits[0]
                prompt_len = len(tokenizer(ex['prompt'], return_tensors='pt')['input_ids'][0])
                shift_logits = logits[:-1, :].contiguous()
                shift_labels = inputs['input_ids'][0][1:].contiguous()
                tgt_start = prompt_len - 1
                tgt_end = len(shift_labels)
                if tgt_start >= tgt_end or tgt_start < 0:
                    continue
                tgt_logits = shift_logits[tgt_start:tgt_end, :]
                tgt_labels = shift_labels[tgt_start:tgt_end]
                if tgt_logits.size(0) == 0:
                    continue
                loss = torch.nn.functional.cross_entropy(tgt_logits, tgt_labels, reduction='mean')
                ppl = torch.exp(loss).item()
                examples_with_metrics.append({**ex, 'perplexity': ppl})
        except Exception as e:
            print(f"Error computing perplexity: {e}")
            continue

    if not examples_with_metrics:
        print("Perplexity computation failed for all examples; falling back to random selection.")
        return random.sample(all_examples, total_examples_needed)

    # Sort by perplexity descending and take top N
    final_examples = sorted(examples_with_metrics, key=lambda x: x['perplexity'], reverse=True)[:total_examples_needed]
    for ex in final_examples:
        ex.pop('perplexity', None)
    print(f"Selected {len(final_examples)} training examples after perplexity ranking")
    return final_examples

def get_sorted_categories(responses_data):
    """Extract all unique categories from responses data and return them in order by idx number"""
    categories = set()
    
    for resp in responses_data:
        if not resp.get('annotated_thinking'):
            continue
            
        # Extract category names from annotated thinking (now idx<number> format)
        matches = CATEGORY_PATTERN.finditer(resp['annotated_thinking'])
        for match in matches:
            category = match.group(1).strip()
            categories.add(category)
    
    # Sort by numerical index extracted from idx<number> format
    def extract_idx(category_name):
        if category_name.startswith('idx'):
            try:
                return int(category_name[3:])  # Extract number after 'idx'
            except ValueError:
                return float('inf')  # Put invalid formats at the end
        return float('inf')  # Put non-idx formats at the end
    
    return sorted(list(categories), key=extract_idx)

def test_on_example(model, tokenizer, vector, layer, test_example, max_new_tokens=50, steering_token_window=None, additional_vectors=None):
    """Test the optimized vector on an example.

    Parameters
    ----------
    vector : torch.Tensor
        The *trainable* category vector we just optimised.
    additional_vectors : list[torch.Tensor] | None
        Extra *static* vectors (e.g. the bias vector) to attach alongside the main one.
    """

    if additional_vectors is None:
        additional_vectors = []

    prompt = test_example['prompt']
    target_completion = test_example['target_completion']
    
    print("\n--- Testing on example ---")
    print(f"Prompt: ...{prompt[-100:]}")
    print("\n--- Generated Completions ---")
    
    # Generate without steering
    generated_tokens = model.generate(
        **tokenizer(prompt, return_tensors='pt').to(model.device), 
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
        suppress_tokens=[tokenizer.eos_token_id]
    )
    unsteered_completion = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
    unsteered_generated_part = unsteered_completion[len(prompt):]
    print("UNSTEERED:")
    print(unsteered_generated_part)
    print()
    
    # Figure out the correct slice for steering
    prompt_tok = tokenizer(prompt, return_tensors='pt')['input_ids'][0]
    target_tok = tokenizer(target_completion, return_tensors='pt')['input_ids'][0]
    prompt_len = len(prompt_tok)
    target_len = len(target_tok)
    if steering_token_window is None:
        steering_start = prompt_len
    else:
        steering_start = prompt_len + max(0, target_len - steering_token_window)
    steering_token_slice = slice(steering_start, None)
    
    # Generate with steering

    hooks = [(layer, steering_opt.make_steering_hook_hf(vector, token=steering_token_slice))]
    for extra_v in additional_vectors:
        if extra_v is not None:
            hooks.append((layer, steering_opt.make_steering_hook_hf(extra_v, token=steering_token_slice)))

    with steering_opt.hf_hooks_contextmanager(model, hooks): 
        generated_tokens = model.generate(
            **tokenizer(prompt, return_tensors='pt').to(model.device), 
            max_new_tokens=max_new_tokens,
            pad_token_id=tokenizer.eos_token_id
        )
        steered_completion = tokenizer.batch_decode(generated_tokens, skip_special_tokens=True)[0]
        steered_generated_part = steered_completion[len(prompt):]
        print("STEERED:")
        print(steered_generated_part)
    
    print("\nTARGET:")
    print(test_example['target_completion'])
    print("\n" + "-"*50 + "\n")
    
    return unsteered_completion, steered_completion

def save_hyperparameters(hyperparams, model_name_short, steering_vector_idx, category):
    """Save hyperparameters for a specific steering vector for reproducibility"""
    hp_dir = "results/vars/hyperparams"
    os.makedirs(hp_dir, exist_ok=True)
    
    hp_file = f"{hp_dir}/steering_vector_hyperparams_{model_name_short}_{steering_vector_idx}.json"

    result = {
        "category": category,
        "hyperparameters": hyperparams
    }
    with open(hp_file, 'w') as f:
        json.dump(result, f, indent=2)
    
    print(f"Saved hyperparameters to {hp_file}")

def main():
    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Parse learning rates
    try:
        learning_rates = [float(lr.strip()) for lr in args.lr.split(',')]
    except ValueError:
        raise ValueError("Learning rates must be comma-separated numbers")
    
    # Create directories
    os.makedirs(os.path.dirname(args.save_path), exist_ok=True)
    
    tokenizer = AutoTokenizer.from_pretrained(args.model)
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model, 
        device_map="auto",
        load_in_8bit=args.load_in_8bit,
        dtype=torch.bfloat16
    )

    torch.set_default_device(model.device)

    for param in model.parameters():
        param.requires_grad = False

    # Default responses path
    model_name_short = args.model.split('/')[-1].lower()
    thinking_model_name = utils.model_mapping.get(args.model, model_name_short)
    if thinking_model_name is None:
        thinking_model_name = model_name_short
    thinking_model_short = thinking_model_name.split('/')[-1].lower()
    responses_json_path = f"../generate-responses/results/vars/responses_{thinking_model_short}.json"
    annotated_responses_json_path = f"../generate-responses/results/vars/annotated_responses_{thinking_model_short}.json"
    
    if not os.path.exists(responses_json_path):
        raise FileNotFoundError(f"Annotated responses file not found at {responses_json_path}. Please annotate responses first.")
    
    # Load responses
    print(f"Loading annotated responses from {responses_json_path}")
    with open(responses_json_path, 'r') as f:
        responses_data = json.load(f)

    # Load annotated responses
    print(f"Loading annotated responses from {annotated_responses_json_path}")
    with open(annotated_responses_json_path, 'r') as f:
        annotated_responses_data = json.load(f)
    
    # Match responses with their annotations by index and validate
    valid_responses = []
    for i, resp in enumerate(responses_data):
        if i < len(annotated_responses_data):
            annotated_resp = annotated_responses_data[i]
            # Verify that the responses match by question_id and dataset_name
            if (resp['question_id'] == annotated_resp['question_id'] and 
                resp['dataset_name'] == annotated_resp['dataset_name'] and
                annotated_resp.get('annotated_thinking')):
                # Create merged response with annotated_thinking
                merged_resp = resp.copy()
                merged_resp['annotated_thinking'] = annotated_resp['annotated_thinking']
                merged_resp['thinking_process'] = extract_thinking_process(resp["full_response"])
                valid_responses.append(merged_resp)
            else:
                print(f"Warning: Mismatch at index {i} - response question_id: {resp['question_id']}, annotated question_id: {annotated_resp.get('question_id')}")
    
    print(f"Found {len(valid_responses)} responses with annotations out of {len(responses_data)} total responses")
    
    # Get all available categories sorted alphabetically
    all_categories = get_sorted_categories(valid_responses)
    
    # Print all available categories with their indices
    print("\nAvailable categories with indices:")
    for idx, category in enumerate(all_categories):
        print(f"  [{idx}] {category}")
    
    # Check if steering_vector_idx is valid
    if args.steering_vector_idx < -1 or args.steering_vector_idx >= len(all_categories):
        raise ValueError(f"Invalid steering_vector_idx: {args.steering_vector_idx}. Must be between -1 and {len(all_categories)-1}")
    
    bias_vector = None  # may be populated later

    # Handle general bias vector case
    if args.steering_vector_idx == -1:
        target_category = "bias"
        print("\nOptimizing general bias vector (full rollouts)")
        training_examples = generate_bias_examples(
            valid_responses,
            tokenizer,
            model,
            args.n_training_examples
        )
        max_activation = 1.0  # Not used for bias vector
    else:
        # Get the target category name
        target_category = all_categories[args.steering_vector_idx]
        print(f"\nOptimizing vector for category: {target_category} (index {args.steering_vector_idx})")
        
        # Extract examples only for the target category
        print(f"Extracting examples for category {target_category}...")
        training_examples, max_activation = extract_examples_for_category(
            valid_responses, 
            target_category, 
            tokenizer,
            args.n_training_examples,
            model
        )

        # --- Load bias vector (if available) to attach during optimisation ---
        bias_vector = None
        bias_path = os.path.join(args.save_path, f"{model_name_short}_bias.pt")
        if os.path.exists(bias_path):
            try:
                bias_dict = torch.load(bias_path, map_location=model.device)
                if isinstance(bias_dict, dict):
                    bias_vector = bias_dict.get("bias", None)
                elif isinstance(bias_dict, torch.Tensor):
                    bias_vector = bias_dict
            except Exception as e:
                print(f"Warning: failed to load bias vector from {bias_path}: {e}")

    # Proceed even if no separate test examples are available.
        
    print(f"Found {len(training_examples)} training examples")
    
    # Extract prompts and target completions for training
    train_prompts = [ex['prompt'] for ex in training_examples]
    train_target_completions = [ex['target_completion'] for ex in training_examples]
    
    # Evaluation dataset removed – optimisation now relies solely on training loss.
    eval_prompts = None
    eval_target_completions = None
    
    # Store results for each learning rate
    all_results = {}
    
    # Run optimization for each learning rate
    for lr in tqdm(learning_rates, desc="Optimizing with learning rates"):
        print(f"\nOptimizing with learning rate: {lr}")
        try:
            vector, loss_info = steering_opt.optimize_vector_simple(
                model, 
                tokenizer,
                train_prompts, 
                train_target_completions, 
                args.layer,
                lr=lr,
                max_iters=args.max_iters,
                optim_minibatch_size=args.optim_minibatch_size,
                warmup_steps=args.warmup_iters,
                min_lr=args.min_lr,
                starting_norm=1,
                max_norm=None,
                grad_clip=args.grad_clip,
                early_stopping_patience=5,
                early_stopping_min_delta=0.01,
                return_info=True,
                return_loss_history=True,
                steering_token_window=args.steering_token_window,
                wandb_run=None,
                static_vectors=[bias_vector] if bias_vector is not None else None
            )
            
            # Store results
            if lr not in all_results:
                all_results[lr] = []
            
            all_results[lr].append({
                'vector': vector.detach().cpu(),
                'loss_info': loss_info,
                'final_loss': loss_info['final_loss']
            })
            
        except Exception as e:
            print(f"Error optimizing vector for category {target_category} with lr {lr}: {e}")
            traceback.print_exc()
            continue
    
    if not all_results:
        print("No successful optimization runs. Exiting.")
        return
    
    # Find best learning rate and run based on final loss
    best_lr = None
    best_result = None
    best_loss = float('inf')
    
    for lr, results in all_results.items():
        for result in results:
            # Selection based solely on training loss
            final_loss = result['final_loss']
            if final_loss < best_loss:
                best_loss = final_loss
                best_lr = lr
                best_result = result
    
    if best_result is None:
        print("No valid optimization results found. Exiting.")
        return
    
    print(f"\nBest learning rate: {best_lr} (final loss: {best_result['final_loss']:.4f})")
    # Evaluation loss removed – only training loss is reported.
    
    # Initialize wandb
    wandb_run = None
    lrs_str = "-".join([str(lr) for lr in learning_rates])
    # Use "bias" instead of -1 in run names and file paths
    vector_id = "bias" if args.steering_vector_idx == -1 else f"idx{args.steering_vector_idx}"
    run_name = f"{model_name_short}_{vector_id}_lr{lrs_str}_n{args.n_training_examples}"
    if args.use_wandb:
        if wandb is None:
            raise ImportError("wandb is not installed. Please install it with `pip install wandb`")
        
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args)
        )

    # Save hyperparameters for this vector
    hyperparams = {
        "layer": args.layer,
        "max_iters": args.max_iters,
        "learning_rates": learning_rates,
        "best_lr": best_lr,
        "context_sentences": args.context_sentences,
        "seed": args.seed,
        "n_training_examples": len(training_examples),
        "vector_norm": best_result['vector'].norm().item(),
        "grad_clip": args.grad_clip,
        "optim_minibatch_size": args.optim_minibatch_size,
        "steering_token_window": args.steering_token_window
    }
    # Use "bias" instead of steering_vector_idx in hyperparams filename
    save_hyperparameters(hyperparams, model_name_short, vector_id, target_category)
    
    # Log best hyperparameters and summary to wandb
    if wandb_run:
        wandb_run.config.update(hyperparams, allow_val_change=True)
        wandb_run.summary['best_lr'] = best_lr
        wandb_run.summary['final_loss'] = best_result['final_loss']
        # Evaluation loss removed – only training loss is reported.

    # Save training losses for best learning rate in the format expected by visualization
    losses_dir = f"results/vars/losses"
    os.makedirs(losses_dir, exist_ok=True)
    # Use "bias" instead of steering_vector_idx in losses filename
    losses_path = f"{losses_dir}/losses_{model_name_short}_{vector_id}.pt"
    
    # Prepare loss data in the format expected by visualize_vector_losses.py
    loss_data = {
        best_lr: [{
            'train_losses': best_result['loss_info']['loss_history'],
            'final_loss': best_result['final_loss']
        }]
    }
    
    torch.save(loss_data, losses_path)
    print(f"\nSaved training losses for best learning rate {best_lr} to {losses_path}")
    
    # Show steering results on a random *training* example to visualise effect
    if training_examples:
        print(f"\nTesting {target_category} vector on a random TRAIN example (best lr: {best_lr}):")
        train_example = random.choice(training_examples)
        test_on_example(
            model, 
            tokenizer, 
            best_result['vector'], 
            args.layer, 
            train_example,
            max_new_tokens=args.test_max_tokens,
            steering_token_window=args.steering_token_window,
            additional_vectors=[bias_vector] if bias_vector is not None else None
        )
    
    # Free memory
    gc.collect()
    torch.cuda.empty_cache()
    
    # Save best vector
    # TODO: make this dynamic
    os.makedirs(args.save_path, exist_ok=True)
    # Use "bias" instead of steering_vector_idx in vectors filename
    vectors_path = f"{args.save_path}/{model_name_short}_{vector_id}.pt"
    optimized_vectors = {}
    optimized_vectors[target_category] = best_result['vector']
    torch.save(optimized_vectors, vectors_path)
    print(f"\nSaved optimized vectors to {vectors_path}")

    if wandb_run:
        artifact = wandb.Artifact(run_name, type='model')
        artifact.add_file(vectors_path)
        wandb_run.log_artifact(artifact)
        wandb_run.finish()

if __name__ == "__main__":
    main() 
# %%
