# %%
import argparse
import dotenv
dotenv.load_dotenv("../.env")

import torch
from datasets import load_dataset
import json
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
import utils
from collections import defaultdict
import gc
import torch.multiprocessing as mp
from functools import partial

# Parse arguments
parser = argparse.ArgumentParser(description="Evaluate hybrid model performance on math problems")
parser.add_argument("--n_batches", type=int, default=60,
                    help="Number of batches to evaluate")
parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                    help="Thinking model to use")
parser.add_argument("--base_model", type=str, default="meta-llama/Llama-3.1-8B",
                    help="Base model to use")
parser.add_argument("--probe_layer", type=int, default=20,
                    help="Layer to use for probe")
parser.add_argument("--load_in_8bit", type=bool, default=False,
                    help="Load in 8bit")
parser.add_argument("--max_new_tokens", type=int, default=2000,
                    help="Max new tokens")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed")
parser.add_argument("--n_gpus", type=int, default=4,
                    help="Number of GPUs to use")
parser.add_argument("--batch_size", type=int, default=2,
                    help="Batch size for parallel generation")
args, _ = parser.parse_known_args()

def extract_answer(response):
    """Extract the final answer from the model's response."""
    try:
        # Look for the answer after ####
        answer = response.split("</think>")
        if len(answer) > 1:
            answer = answer[-1].strip()
            if answer == "":
                answer = None
        else:
            answer = None
        # Try to convert to float
        return answer
    except:
        return None

def get_batched_question_ids(tokenizer, questions):
    max_token_length = max([len(tokenizer.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, return_tensors="pt")[0]) for q in questions])
    input_ids = torch.cat([tokenizer.apply_chat_template([{"role": "user", "content": q}], add_generation_prompt=True, padding="max_length", max_length=max_token_length, return_tensors="pt").to("cuda") for q in questions])
    return input_ids

def evaluate_answer(question, model_answer, correct_answer):

    final_answer = model_answer["response"] if model_answer["extracted_answer"] is None else model_answer["extracted_answer"]

    """Use chat API to evaluate if the answer is correct."""
    evaluation_prompt = f"""
    Consider the following question with the given correct answer:
    Question: {question}
    Correct final answer: {correct_answer}

    Is the following written out response to the question arriving at the correct final answer?
    Response: {final_answer}

    Respond with only "correct" if the Response derives the final answer or "incorrect" if the Response provides a wrong- or no answer.
    """
    
    response = utils.chat(evaluation_prompt)
    return response.strip().lower() == "correct"

def generate_and_evaluate_base(model, tokenizer, questions, answers):
    """Generate and evaluate using base model in batches."""
    results = []
    
    # Process questions in batches
    input_ids = get_batched_question_ids(tokenizer, questions)

    with torch.no_grad():
        with model.generate(
            {
                "input_ids": input_ids, 
                "attention_mask": (input_ids != tokenizer.pad_token_id).long()
            },
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id
        ) as tracer:
            outputs = model.generator.output.save()
    
    # Process each response in the batch
    for j in range(len(questions)):
        response = tokenizer.decode(outputs[j], skip_special_tokens=True)
        extracted_answer = extract_answer(response)
        
        # Evaluate answer immediately after generation
        is_correct = evaluate_answer(questions[j], {"response": response, "extracted_answer": extracted_answer}, answers[j])
        
        results.append({
            "response": response,
            "extracted_answer": extracted_answer,
            "correct": is_correct,
            "question": questions[j],
            "answer": answers[j]
        })
    
    return results

def generate_and_evaluate_thinking(model, tokenizer, questions, answers):
    """Generate and evaluate using thinking model in batches."""
    results = []
    
    # Process questions in batches
    input_ids = get_batched_question_ids(tokenizer, questions)

    with torch.no_grad():
        with model.generate(
            {
                "input_ids": input_ids, 
                "attention_mask": (input_ids != tokenizer.pad_token_id).long()
            },
            max_new_tokens=args.max_new_tokens,
            pad_token_id=tokenizer.pad_token_id
        ) as tracer:
            outputs = model.generator.output.save()
    
    # Process each response in the batch
    for j in range(len(questions)):
        response = tokenizer.decode(outputs[j], skip_special_tokens=True)
        extracted_answer = extract_answer(response)
        
        # Evaluate answer immediately after generation
        is_correct = evaluate_answer(questions[j], {"response": response, "extracted_answer": extracted_answer}, answers[j])
        
        results.append({
            "response": response,
            "extracted_answer": extracted_answer,
            "correct": is_correct,
            "question": questions[j],
            "answer": answers[j]
        })
    
    return results

def generate_and_evaluate_hybrid(thinking_model, base_model, tokenizer, questions, answers, hybrid_config, baseline_method, warmup=7):
    """Generate and evaluate using random baseline model in batches."""
    results = []
    
    # Process questions in batches
    input_ids = get_batched_question_ids(tokenizer, questions)
    
    output_ids, forced_positions, forced_labels, forced_tokens = utils.custom_hybrid_generate(
        thinking_model,
        base_model,
        tokenizer,
        input_ids,
        max_new_tokens=args.max_new_tokens,
        baseline_method=baseline_method,
        baseline_config=hybrid_config,
        warmup=warmup,
        show_progress=False,
        color_output=False
    )
    
    # Process each response in the batch
    for j in range(len(questions)):
        response = tokenizer.decode(output_ids[j], skip_special_tokens=True)
        extracted_answer = extract_answer(response)
        
        # Evaluate answer immediately after generation
        is_correct = evaluate_answer(questions[j], {"response": response, "extracted_answer": extracted_answer}, answers[j])
        
        # Calculate the number of generated tokens (excluding input tokens)
        generated_tokens_count = len(output_ids[j]) - len(input_ids[j])
        
        results.append({
            "response": response,
            "extracted_answer": extracted_answer,
            "correct": is_correct,
            "question": questions[j],
            "answer": answers[j],
            "forced_positions": forced_positions[j],
            "forced_labels": forced_labels[j],
            "forced_tokens": forced_tokens[j],
            "generated_tokens_count": generated_tokens_count
        })
    
    return results


def calculate_thinking_length(response):
    """Calculate the length of thinking process between <think> and </think> tags."""
    start_idx = response.find("<think>")
    try:
        end_idx = response.find("</think>")
        if start_idx != -1 and end_idx != -1:
            thinking_text = response[start_idx + 7:end_idx].strip()
            return len(thinking_text.split())
    except:
        pass
    return len(response[start_idx + 7:].strip())

def calculate_text_fractions(response):
    """Calculate the fraction of text before and after the think token."""
    start_idx = response.find("<think>")
    if start_idx == -1:
        return 0.0, 1.0  # If no think token, all text is after
    before_fraction = start_idx / len(response)
    after_fraction = 1 - before_fraction
    return before_fraction, after_fraction

def plot_results(results, model_name):
    """Plot the evaluation results showing only accuracy."""
    os.makedirs('results/figures', exist_ok=True)
    model_id = model_name.split('/')[-1].lower()
    
    # Calculate accuracy for each model
    models = ['base', 'thinking', 'hybrid']
    baseline_models = ['random', 'norm_diff', 'kl_div']
    accuracies = []
    baseline_accuracies = {}
    
    for model in models:
        if model in results and results[model]:
            correct = sum(1 for r in results[model] if r["correct"])
            accuracy = correct / len(results[model])
            accuracies.append(accuracy)
        else:
            accuracies.append(0)
    
    # Calculate baseline accuracies
    for baseline in baseline_models:
        if baseline in results and results[baseline]:
            correct = sum(1 for r in results[baseline] if r["correct"])
            accuracy = correct / len(results[baseline])
            baseline_accuracies[baseline] = accuracy
    
    # Calculate forced stats for all models
    forced_stats = {}
    for model_type in models + baseline_models:
        if model_type in results and results[model_type]:
            # Calculate total attempt fractions (state 1 + state 2)
            total_attempts = []
            successful_forces = []
            
            for r in results[model_type]:
                if 'forced_states' in r:
                    # Using the new forced_states field
                    total_attempted = sum(1 for state in r['forced_states'] if state == 1 or state == 2)
                    successful_forced = sum(1 for state in r['forced_states'] if state == 2)
                elif 'forced_positions' in r:
                    # Backward compatibility with old format
                    total_attempted = sum(r['forced_positions'])
                    successful_forced = sum(r['forced_positions'])
                    
                total_attempts.append(total_attempted / max(r['generated_tokens_count'], 1))
                successful_forces.append(successful_forced / max(r['generated_tokens_count'], 1))
            
            forced_stats[model_type] = {
                'attempt_fraction': np.mean(total_attempts),
                'success_fraction': np.mean(successful_forces)
            }
    
    # Create bar plot
    plt.style.use('seaborn-v0_8-white')
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(len(models))
    width = 0.35
    
    bars = ax.bar(x, accuracies, width, alpha=0.8, edgecolor='black', linewidth=1)
    
    # Add percentage labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
                f'{height*100:.1f}%',
                ha='center', va='bottom', fontsize=10)
    
    # Add baseline lines
    colors = ['red', 'blue', 'green']
    line_styles = ['--', '-.', ':']
    
    for i, baseline in enumerate(baseline_models):
        if baseline in baseline_accuracies:
            stats = forced_stats.get(baseline, {'attempt_fraction': 0, 'success_fraction': 0})
            attempt_pct = stats['attempt_fraction'] * 100
            success_pct = stats['success_fraction'] * 100
            
            ax.axhline(
                y=baseline_accuracies[baseline], 
                color=colors[i], 
                linestyle=line_styles[i], 
                alpha=0.7, 
                label=f'{baseline.replace("_", " ").title()} Baseline (Attempted: {attempt_pct:.1f}%, Successful: {success_pct:.1f}%)'
            )
    
    # Improve styling
    ax.set_ylabel('Accuracy (%)', fontsize=16, labelpad=10)
    ax.set_title(f'Hybrid MATH Evaluation - {model_name}', fontsize=20, pad=20, weight='bold')
    ax.set_xticks(x)
    
    # Update model labels with forcing stats
    model_labels = []
    for model in models:
        if model in forced_stats:
            if model == 'base' or model == 'thinking':
                model_labels.append(model.title())
            else:
                stats = forced_stats[model]
                attempt_pct = stats['attempt_fraction'] * 100
                success_pct = stats['success_fraction'] * 100
                model_labels.append(f"{model.title()} (Attempted: {attempt_pct:.1f}%, Successful: {success_pct:.1f}%)")
        else:
            model_labels.append(model.title())
    
    ax.set_xticklabels(model_labels, fontsize=12)
    ax.tick_params(axis='y', labelsize=12)
    ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: '{:.0f}%'.format(y * 100)))
    ax.yaxis.grid(True, linestyle='--', alpha=0.7)
    ax.set_axisbelow(True)
    ax.legend()
    
    plt.tight_layout()
    plt.savefig(f'results/figures/hybrid_results_{model_id}.pdf', dpi=300, bbox_inches='tight')
    plt.show()
    plt.close()

def process_gpu_chunk(chunk_id, dataset_chunk, gpu_id, args):
    """Process a chunk of the dataset on a specific GPU."""
    print(f"Starting process for GPU {gpu_id} with {len(dataset_chunk)} examples")
    
    # Set device
    torch.cuda.set_device(gpu_id)
    
    # Set random seed for reproducibility within this process
    random.seed(args.seed + gpu_id)
    torch.manual_seed(args.seed + gpu_id)
    
    # Load models and probe for this GPU
    model, tokenizer, base_model, base_tokenizer, feature_vectors = utils.load_model(
        load_in_8bit=args.load_in_8bit,
        compute_features=True,
        normalize_features=True,
        return_steering_vector_set=True,
        model_name=args.model,
        base_model_name=args.base_model,
        device=f"cuda:{gpu_id}"
    )
    
    # Load probe
    probe_state_dict = torch.load(
        f"./results/vars/probe_layer{args.probe_layer}_{args.model.split('/')[-1].lower()}.pt", 
        weights_only=False,
        map_location=f"cuda:{gpu_id}"
    )["probe_state_dict"]
    
    label_to_idx = {
        "backtracking": 0, 
        "uncertainty-estimation": 1, 
        "adding-knowledge": 2, 
        "example-testing": 3
    }
    probe = utils.LinearProbe(hidden_size=model.config.hidden_size, num_labels=len(label_to_idx))
    probe.load_state_dict(probe_state_dict)
    probe.to(f"cuda:{gpu_id}")
    
    # Initialize results for this GPU
    gpu_results = defaultdict(list)
    
    # Process each batch in this GPU's chunk with tqdm progress bar
    pbar = tqdm(range(0, len(dataset_chunk), args.batch_size), 
                desc=f"GPU {gpu_id}", 
                position=gpu_id,
                leave=True)
    
    for i in pbar:
        batch = dataset_chunk[i:i + args.batch_size]
        if not batch:  # Skip empty batches
            continue
            
        questions = [ex["problem"] for ex in batch]
        answers = [ex["answer"] for ex in batch]
        batch_num = i//args.batch_size + 1
        total_batches = (len(dataset_chunk) + args.batch_size - 1) // args.batch_size
        
        # Update progress bar description
        pbar.set_description(f"GPU {gpu_id} [{batch_num}/{total_batches}]")

        # Thinking model evaluation
        pbar.set_postfix({"current": "thinking model"})
        thinking_result = generate_and_evaluate_thinking(model, tokenizer, questions, answers)
        gpu_results["thinking"].extend(thinking_result)
        
        # Hybrid model evaluation
        pbar.set_postfix({"current": "hybrid model"})
        hybrid_config = {
            "probe": probe,
            "label_to_idx": label_to_idx,
            "probe_layer": args.probe_layer,
            "threshold": 0.8,
            "forcing": ["backtracking", "example-testing", "uncertainty-estimation", "adding-knowledge"]
        }

        hybrid_results = generate_and_evaluate_hybrid(model, base_model, tokenizer, questions, answers, hybrid_config, "probe", warmup=7)
        gpu_results["hybrid"].extend(hybrid_results)
        
        # Base model evaluation
        pbar.set_postfix({"current": "base model"})
        base_result = generate_and_evaluate_base(base_model, tokenizer, questions, answers)
        gpu_results["base"].extend(base_result)
        
        # Random baseline evaluation
        pbar.set_postfix({"current": "random baseline"})
        random_config = {
            "forced_token_rate": 0.15,
        }
        random_result = generate_and_evaluate_hybrid(model, base_model, tokenizer, questions, answers, random_config, "random", warmup=7)
        gpu_results["random"].extend(random_result) 
        
        # Norm difference baseline evaluation
        pbar.set_postfix({"current": "norm diff baseline"})
        norm_diff_config = {
            "label_to_idx": label_to_idx,
            "threshold": 0.1,
            "target_layer": args.probe_layer
        }
        norm_diff_result = generate_and_evaluate_hybrid(model, base_model, tokenizer, questions, answers, norm_diff_config, "norm_diff", warmup=7)
        gpu_results["norm_diff"].extend(norm_diff_result)
        
        # KL divergence baseline evaluation
        pbar.set_postfix({"current": "kl div baseline"})
        kl_div_config = {
            "label_to_idx": label_to_idx,
            "threshold": 1.0,
        }
        kl_div_result = generate_and_evaluate_hybrid(model, base_model, tokenizer, questions, answers, kl_div_config, "kl_div", warmup=7)
        gpu_results["kl_div"].extend(kl_div_result)
        
        # Final update to show completion of this batch
        pbar.set_postfix({"status": "completed"})
    
    # Clear GPU memory
    del model, tokenizer, base_model, base_tokenizer, feature_vectors, probe
    torch.cuda.empty_cache()
    gc.collect()
    
    return gpu_results

def main():
    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    
    # Load dataset with progress bar
    print("Loading dataset...")
    dataset = load_dataset("HuggingFaceH4/MATH-500")["test"]
    dataset = list(dataset)[:args.n_batches * args.batch_size]
    print(f"Loaded {len(dataset)} examples")
    
    # Split dataset into chunks for each GPU
    n_gpus = min(args.n_gpus, torch.cuda.device_count())
    print(f"Using {n_gpus} GPUs out of {torch.cuda.device_count()} available")
    
    chunk_size = len(dataset) // n_gpus
    dataset_chunks = [dataset[i:i+chunk_size] for i in range(0, len(dataset), chunk_size)]
    
    # If there are more chunks than GPUs, merge the last chunks
    if len(dataset_chunks) > n_gpus:
        dataset_chunks = dataset_chunks[:n_gpus-1] + [sum(dataset_chunks[n_gpus-1:], [])]
    
    # Print chunk distribution
    for i, chunk in enumerate(dataset_chunks):
        print(f"GPU {i}: {len(chunk)} examples")
    
    # Initialize multiprocessing
    mp.set_start_method('spawn', force=True)
    
    # Create partial function with fixed arguments
    process_func = partial(process_gpu_chunk, args=args)
    
    # Run processes in parallel
    all_results = defaultdict(list)
    
    if n_gpus > 1:
        print("Starting parallel processing on multiple GPUs...")
        # Create process pool
        with mp.Pool(n_gpus) as pool:
            # Map processes to GPUs with progress tracking
            chunk_results = pool.starmap(
                process_func,
                [(chunk_id, chunk, gpu_id) for chunk_id, (chunk, gpu_id) in enumerate(zip(dataset_chunks, range(n_gpus)))]
            )
            
            # Aggregate results from all GPUs
            print("Aggregating results from all GPUs...")
            for gpu_id, gpu_results in enumerate(chunk_results):
                for model_name in ["base", "thinking", "hybrid", "random", "norm_diff", "kl_div"]:
                    if model_name in gpu_results:
                        print(f"GPU {gpu_id}, model {model_name}: {len(gpu_results[model_name])} results")
                        all_results[model_name].extend(gpu_results[model_name])
    else:
        # Single GPU mode - just process everything on one GPU
        print("Running on a single GPU...")
        gpu_results = process_gpu_chunk(0, dataset, 0, args)
        all_results = gpu_results
    
    # Calculate metrics
    print("Calculating metrics...")
    metrics = {}
    for model_name in tqdm(["base", "thinking", "hybrid", "random", "norm_diff", "kl_div"], desc="Processing metrics"):
        if model_name in all_results and all_results[model_name]:
            correct = sum(1 for r in all_results[model_name] if r["correct"])
            accuracy = correct / len(all_results[model_name])
            metrics[model_name] = {
                "accuracy": accuracy,
                "thinking_length": np.mean([calculate_thinking_length(r["response"]) for r in all_results[model_name]]).item(),
                "text_fractions": np.mean([calculate_text_fractions(r["response"]) for r in all_results[model_name]], axis=0).tolist()
            }
            print(f"{model_name}: accuracy = {accuracy:.4f}")

    # Save results
    print("Saving results...")
    os.makedirs("results/vars", exist_ok=True)
    result_file = f"results/vars/hybrid_results_{args.model.split('/')[-1].lower()}.json"
    with open(result_file, "w") as f:
        json.dump({
            "metrics": metrics,
            "results": all_results
        }, f, indent=2)
    print(f"Results saved to {result_file}")

    # Plot results
    print("Generating plots...")
    plot_results(all_results, args.model)
    print("Experiment completed successfully!")

# %%
if __name__ == "__main__":
    main()
# %%
