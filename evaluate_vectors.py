# %%
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from nnsight import NNsight
import matplotlib.pyplot as plt
from deepseek_steering.utils import chat
import re
import numpy as np
from deepseek_steering.messages import eval_messages, labels
from tqdm import tqdm
import gc
import random
import os
import deepseek_steering.utils as utils
from deepseek_steering.messages import messages

os.system('')  # Enable ANSI support on Windows

random.shuffle(messages)

# %% Evaluation examples - 3 from each category
def load_model_and_vectors(model_name="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"):
    model, tokenizer, feature_vectors = utils.load_model_and_vectors(compute_features=True, model_name=model_name)
    return model, tokenizer, feature_vectors

# %%
model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"  # Can be changed to use different models
model, tokenizer, feature_vectors = load_model_and_vectors(model_name)
steering_vectors = torch.load(f'data/steering_vectors_{model_name.split("/")[-1].lower()}.pt')

# %% Get activations and response
data_idx = 3
message = messages[data_idx]
input = tokenizer.apply_chat_template([message], add_generation_prompt=True, return_text=True)

# %%
print("Original response:")
input_ids = tokenizer.apply_chat_template([eval_messages[data_idx]], add_generation_prompt=True, return_tensors="pt").to("cuda")
output_ids = utils.custom_generate_with_projection_removal(
    model,
    tokenizer,
    input,
    max_new_tokens=500,
    label="none", 
    feature_vectors=steering_vectors,
    show_progress=True
)
response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(response)
print("\n================\n")

# %%
input_ids = tokenizer.apply_chat_template([eval_messages[data_idx]], add_generation_prompt=True, return_tensors="pt").to("cuda")
output_ids = utils.custom_generate_with_projection_removal(
    model,
    tokenizer,
    input,
    max_new_tokens=300,
    label="backtracking",
    feature_vectors=steering_vectors,
    layers=[25,28,27,30],
    coefficient=0.01,
    steer_positive=True,
    show_progress=True
)
response = tokenizer.decode(output_ids[0], skip_special_tokens=True)
print(response)
print("\n================\n")


# %%
