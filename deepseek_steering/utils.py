from openai import OpenAI
import dotenv
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from nnsight import LanguageModel
from tqdm import tqdm
import gc
import asyncio
from openai import AsyncOpenAI
from typing import List, Dict, Any

dotenv.load_dotenv(".env")

def chat(
    prompt,
    temperature=0.01,
    model="gpt-4o",
    max_tokens=5_00,
    top_p=0.90
):
    client = OpenAI(
        organization="org-E6iEJQGSfb0SNHMw6NFT1Cmi",
    )
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
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p
    )
    return response.choices[0].message.content

async def _process_chat_request(
    client: AsyncOpenAI,
    prompt: str,
    temperature: float = 0.01,
    model: str = "gpt-4o",
    max_tokens: int = 5_000,
    top_p: float = 0.90
) -> str:
    """Process a single chat request asynchronously"""
    response = await client.chat.completions.create(
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
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p
    )
    return response.choices[0].message.content

async def chat_batch(
    prompts: List[str],
    batch_size: int = 50,
    temperature: float = 0.01,
    model: str = "gpt-4o",
    max_tokens: int = 5_000,
    top_p: float = 0.90
) -> List[str]:
    """
    Process multiple chat requests in batches asynchronously.
    
    Args:
        prompts: List of prompts to process
        batch_size: Number of concurrent requests
        temperature: Temperature for generation
        model: Model to use
        max_tokens: Maximum tokens for generation
        top_p: Top p for generation
        
    Returns:
        List of responses in the same order as prompts
    """
    client = AsyncOpenAI(
        organization="org-E6iEJQGSfb0SNHMw6NFT1Cmi",
    )
    
    async def process_batch(batch_prompts: List[str]) -> List[str]:
        tasks = [
            _process_chat_request(
                client=client,
                prompt=prompt,
                temperature=temperature,
                model=model,
                max_tokens=max_tokens,
                top_p=top_p
            )
            for prompt in batch_prompts
        ]
        return await asyncio.gather(*tasks)

    results = []
    total_batches = (len(prompts) + batch_size - 1) // batch_size
    with tqdm(total=total_batches, desc="Processing batches") as pbar:
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i + batch_size]
            batch_results = await process_batch(batch)
            results.extend(batch_results)
            pbar.update(1)
    
    return results


def chat_batch_sync(
    prompts: List[str],
    batch_size: int = 50,
    **kwargs
) -> List[str]:
    """Synchronous wrapper for chat_batch"""
    return asyncio.run(chat_batch(prompts, batch_size, **kwargs))

def load_model_and_vectors(compute_features=True, model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"):
    """
    Load model, tokenizer and mean vectors. Optionally compute feature vectors.
    
    Args:
        compute_features (bool): If True, compute and return feature vectors by subtracting overall mean
        model_name (str): Name/path of the model to load
    """
    model = LanguageModel(model_name, dispatch=True, device_map="auto", torch_dtype=torch.bfloat16)
    tokenizer = model.tokenizer
    
    # Get model identifier for file naming
    model_id = model_name.split('/')[-1].lower()
    mean_vectors_dict = torch.load(f"data/mean_vectors_{model_id}.pt")
    
    if compute_features:
        # Compute feature vectors by subtracting overall mean
        overall_mean = mean_vectors_dict['overall']['mean']
        feature_vectors = {}
        
        for label in mean_vectors_dict:
            if label != 'overall':
                feature_vectors[label] = mean_vectors_dict[label]['mean'] - overall_mean
        
        return model, tokenizer, feature_vectors
    
    return model, tokenizer, mean_vectors_dict

def custom_generate_with_projection_removal(model, tokenizer, input_ids, max_new_tokens, label, feature_vectors, layers=[10], coefficient=0.1, steer_positive=False, show_progress=True):
    """
    Generate text while removing or adding projections of specific features using efficient generation.
    
    Args:
        model: The model to use for generation
        tokenizer: The tokenizer
        input_ids: Input token ids
        max_new_tokens: Maximum number of tokens to generate
        label: The label to steer towards/away from
        feature_vectors: Dictionary of feature vectors
        layers: List of layers to apply steering
        coefficient: Steering strength
        steer_positive: If True, steer towards the label, if False steer away
        show_progress: If True, show progress bar
    """
    if label in feature_vectors:
        feature_vector = feature_vectors[label].to("cuda").to(torch.bfloat16)
    else:
        print(f"Label {label} not found. No steering applied.")
        feature_vector = None

    model_layers = model.model.layers

    with model.generate(
        input_ids,
        max_new_tokens=max_new_tokens,
        pad_token_id=tokenizer.eos_token_id,
    ) as tracer:
        # Apply .all() to model to ensure interventions work across all generations
        model_layers.all()
        
        if feature_vector is not None:
            for layer_idx in layers:

                normed_feature_vector = feature_vector / feature_vector.norm()
                normed_feature_vector = normed_feature_vector * model.model.layers[layer_idx].output[0][:, 1:].norm(dim=1, keepdim=True)

                if steer_positive:
                    model.model.layers[layer_idx].output[0][:, 1:] += coefficient * normed_feature_vector
                else:
                    model.model.layers[layer_idx].output[0][:, 1:] -= coefficient * normed_feature_vector
        
        # Save the final output
        outputs = model.generator.output.save()

    return outputs

def prepare_model_input(
    response_uuid: str,
    annotated_responses_data: List[Dict[str, Any]],
    tasks_data: List[Dict[str, Any]],
    original_messages_data: List[Dict[str, Any]],
    tokenizer: AutoTokenizer
) -> Dict[str, Any]:
    """
    Prepare model input for a given response UUID.
    Returns the tokenized input ready for the model.
    Returns:
        Dict with keys:
            'prompt_and_response_ids': Tensor of shape (1, sequence_length)
            'annotated_response': str
    """
    # Fetch the relevant response data
    annotated_response_data = next((r for r in annotated_responses_data if r["response_uuid"] == response_uuid), None)
    if not annotated_response_data:
        raise ValueError(f"Could not find annotated response data for UUID {response_uuid}")
    task_data = next((t for t in tasks_data if t["task_uuid"] == annotated_response_data["task_uuid"]), None)
    if not task_data:
        raise ValueError(f"Could not find task data for UUID {annotated_response_data['task_uuid']}")
    base_response_data = next((m for m in original_messages_data if m["response_uuid"] == response_uuid), None)
    if not base_response_data:
        raise ValueError(f"Could not find base response data for UUID {response_uuid}")
    # Build prompt message
    prompt_message = [task_data["prompt_message"]]
    prompt_message_input_ids = tokenizer.apply_chat_template(
        conversation=prompt_message,
        add_generation_prompt=True,
        return_tensors="pt"
    )
    # Process base response
    base_response_str = base_response_data["response_str"]
    if base_response_str.startswith("<think>"):
        base_response_str = base_response_str[len("<think>"):]
    base_response_input_ids = tokenizer.encode(
        text=base_response_str,
        add_special_tokens=False,
        return_tensors="pt"
    )
    prompt_and_response_ids = torch.cat(
        tensors=[prompt_message_input_ids, base_response_input_ids],
        dim=1
    )
    # Find start and end positions of thinking process (-1 if not found)
    thinking_start_token_id = tokenizer.encode("<think>", add_special_tokens=False)[0]
    thinking_end_token_id = tokenizer.encode("</think>", add_special_tokens=False)[0]
    prompt_and_response_ids_list = prompt_and_response_ids.tolist()[0]
    thinking_start_token_index = next((i + 1 for i, token in enumerate(prompt_and_response_ids_list) if token == thinking_start_token_id), -1)
    thinking_end_token_index = next((i for i, token in enumerate(prompt_and_response_ids_list) if token == thinking_end_token_id), -1)
    thinking_token_ids = prompt_and_response_ids[:, thinking_start_token_index:thinking_end_token_index]
    return {
        'prompt_and_response_ids': prompt_and_response_ids,
        'annotated_response': annotated_response_data["annotated_response"],
        'thinking_start_token_index': thinking_start_token_index,
        'thinking_end_token_index': thinking_end_token_index,
        'thinking_token_ids': thinking_token_ids
    }