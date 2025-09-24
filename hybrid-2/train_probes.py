# %%
import argparse
import dotenv
dotenv.load_dotenv("../.env")

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import json
import random
import re
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import os
import utils
from utils import LinearProbe
from sklearn.metrics import f1_score, accuracy_score, precision_score, recall_score

# Parse arguments
parser = argparse.ArgumentParser(description="Train probes for hybrid models")
parser.add_argument("--model", type=str, default="deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
                    help="Model to train probes for")
parser.add_argument("--epochs", type=int, default=100,
                    help="Number of training epochs")
parser.add_argument("--batch_size", type=int, default=16,
                    help="Batch size for training")
parser.add_argument("--num_samples", type=int, default=100,
                    help="Number of samples to train on")
parser.add_argument("--lr", type=float, default=1e-4,
                    help="Learning rate")
parser.add_argument("--seed", type=int, default=42,
                    help="Random seed")
parser.add_argument("--probe_layer", type=int, default=15,
                    help="Model layer to use for probe")
parser.add_argument("--load_in_8bit", type=bool, default=False,
                    help="Load model in 8-bit mode")
args, _ = parser.parse_known_args()

# %%
def find_label_positions(annotated_response, original_text, tokenizer, label_to_idx):
    """Parse annotations and find token positions for each label more accurately"""
    label_positions = {}
    
    # Use a pattern that captures labeled segments properly
    pattern = r'\["([\w-]+)"\](.*?)(?=\["[\w-]+"\]|$)'
    matches = list(re.finditer(pattern, annotated_response, re.DOTALL))
    
    # Create character to token mapping once
    char_to_token = get_char_to_token_map(original_text, tokenizer)
    
    for match in matches:
        label = match.group(1)
        if label not in label_to_idx:
            continue
            
        text = match.group(2).replace('["<end-section>"]', "").strip()
        if not text:  # Skip empty text
            continue
            
        # Find this text in the original response
        text_pos = original_text.find(text)
        if text_pos >= 0:
            if label not in label_positions:
                label_positions[label] = []
                
            # Get start and end token positions
            token_start = char_to_token.get(text_pos, None)
            token_end = char_to_token.get(text_pos + len(text), None)
            
            # If we found valid token positions
            if token_start is not None and token_end is not None:
                label_positions[label].append((token_start, token_end))
    
    return label_positions

def get_char_to_token_map(text, tokenizer):
    """Create a mapping from character positions to token positions"""
    token_offsets = tokenizer.encode_plus(text, return_offsets_mapping=True)['offset_mapping']
    
    # Create mapping from character position to token index
    char_to_token = {}
    for token_idx, (start, end) in enumerate(token_offsets):
        for char_pos in range(start, end):
            char_to_token[char_pos] = token_idx
            
    return char_to_token

def get_activations(text, tokenizer, model):
    """Get activations for a text input"""
    input_ids = tokenizer.encode(text, return_tensors="pt").to("cuda")
    layer_outputs = []
    
    with torch.no_grad():
        with model.trace() as tracer:
            with tracer.invoke(
                {
                    "input_ids": input_ids, 
                    "attention_mask": (input_ids != tokenizer.pad_token_id).long()
                }
            ) as invoker:
                for layer_idx in range(model.config.num_hidden_layers):
                    layer_outputs.append(model.model.layers[layer_idx].output[0].save())
    
    layer_outputs = [output.value for output in layer_outputs]
    return layer_outputs, input_ids[0]

def create_training_examples(layer_output, label_positions, label_to_idx):
    """Create training examples from layer output and label positions"""
    if not label_positions:
        return None, None
    
    examples = []
    labels = []
    
    # Get all token positions
    all_positions = []
    for label, positions in label_positions.items():
        if label in label_to_idx:
            for pos in positions:
                all_positions.append((pos, label))
        
    if not all_positions:
        return None, None
    
    # Create examples and labels
    for pos, label in all_positions:
        # Get activation for this position
        start, end = pos
        example = layer_output[0, start-1:end-1].mean(dim=0).unsqueeze(0).to(torch.float32)
        if example.isnan().sum() > 0:
            print(f"NaN example at {pos}")
            print(example)
            print(layer_output.shape)
            print(start, end)
            print(label_positions)
            print(label_to_idx)
        examples.append(example)
        
        # Create one-hot encoded label
        label_tensor = torch.zeros(len(label_to_idx))
        label_tensor[label_to_idx[label]] = 1
        labels.append(label_tensor.unsqueeze(0))
    
    if not examples:
        return None, None
    
    return torch.cat(examples, dim=0), torch.cat(labels, dim=0)

def balance_examples(examples, labels, label_to_idx):
    """Balance examples by limiting deduction examples to max of other labels"""
    # Count examples per label
    label_counts = {}
    for label_idx in range(len(label_to_idx)):
        count = int(labels[:, label_idx].sum().item())
        label_counts[list(label_to_idx.keys())[label_idx]] = count
    
    # Find max count among non-deduction labels
    max_other_count = max(count for label, count in label_counts.items() if label != "deduction")
    
    # Get indices of deduction examples
    deduction_idx = label_to_idx["deduction"]
    deduction_indices = torch.where(labels[:, deduction_idx] == 1)[0]
    
    # If we have more deduction examples than max_other_count, randomly sample
    if len(deduction_indices) > max_other_count:
        keep_indices = torch.randperm(len(deduction_indices))[:max_other_count]
        deduction_indices = deduction_indices[keep_indices]
    
    # Get indices of non-deduction examples
    other_indices = []
    for label, idx in label_to_idx.items():
        if label != "deduction":
            other_indices.extend(torch.where(labels[:, idx] == 1)[0].tolist())
    
    # Combine all indices to keep
    keep_indices = torch.tensor(other_indices + deduction_indices.tolist())
    
    # Return balanced examples and labels
    return examples[keep_indices], labels[keep_indices]

def train_probe(model, tokenizer, train_responses, val_responses, labels, layer_idx=20, epochs=10, batch_size=32, lr=1e-3):
    """Train a linear probe for a specific layer"""
    print(f"Training probe for layer {layer_idx}")
    
    # Initialize the probe
    probe = LinearProbe(model.config.hidden_size, len(labels)).to("cuda")
    optimizer = optim.Adam(probe.parameters(), lr=lr)
    
    # Create label mapping
    label_to_idx = {label: i for i, label in enumerate(labels)}
    
    # Process responses to create training data
    print("Processing training examples...")
    train_examples = []
    train_labels = []
    
    for response_data in tqdm(train_responses, desc="Processing training data"):
        # Get activations for full response
        layer_outputs, _ = get_activations(response_data["full_response"], tokenizer, model)
        
        # Get label positions in full response
        label_positions = find_label_positions(
            response_data["annotated_thinking"], 
            response_data["full_response"], 
            tokenizer,
            label_to_idx
        )
        
        # Create training examples
        examples, label_tensors = create_training_examples(
            layer_outputs[layer_idx], 
            label_positions, 
            label_to_idx
        )

        if examples is not None:
            train_examples.append(examples)
            train_labels.append(label_tensors)
    
    if not train_examples:
        raise ValueError("No training examples found!")
    
    # Concatenate all examples
    train_examples = torch.cat(train_examples, dim=0).to("cuda")
    train_labels = torch.cat(train_labels, dim=0).to("cuda")

    # Balance examples
    if "deduction" in label_to_idx:
        train_examples, train_labels = balance_examples(train_examples, train_labels, label_to_idx)
    
    # print indices of nan train_examples
    nan_indices = torch.where(train_examples.isnan())[0]
    print(nan_indices)
    # print examples at nan_indices
    print(train_examples[nan_indices])
    
    # Count examples per label type
    label_counts = {}
    for label_idx in range(len(labels)):
        # Count examples where this label is 1
        count = int(train_labels[:, label_idx].sum().item())
        label_counts[labels[label_idx]] = count
    
    # Print label distribution
    print("\nTraining examples per label type (after balancing):")
    for label, count in label_counts.items():
        print(f"  {label}: {count} examples")
    print(f"  Total: {len(train_examples)} examples")
    
    # Process validation examples
    print("\nProcessing validation examples...")
    val_examples = []
    val_labels = []
    
    for response_data in tqdm(val_responses, desc="Processing validation data"):
        # Get activations for full response
        layer_outputs, _ = get_activations(response_data["full_response"], tokenizer, model)
        
        # Get label positions in full response
        label_positions = find_label_positions(
            response_data["annotated_thinking"], 
            response_data["full_response"], 
            tokenizer,
            label_to_idx
        )
        
        # Create validation examples
        examples, label_tensors = create_training_examples(
            layer_outputs[layer_idx], 
            label_positions, 
            label_to_idx
        )
                
        if examples is not None:
            val_examples.append(examples)
            val_labels.append(label_tensors)
    
    if not val_examples:
        raise ValueError("No validation examples found!")
    
    # Concatenate all examples
    val_examples = torch.cat(val_examples, dim=0).to("cuda")
    val_labels = torch.cat(val_labels, dim=0).to("cuda")
    
    # Balance validation examples
    if "deduction" in label_to_idx:
        val_examples, val_labels = balance_examples(val_examples, val_labels, label_to_idx)
    
    print(f"Training on {len(train_examples)} examples, validating on {len(val_examples)}")
    
    # Training loop
    train_losses = []
    val_losses = []
    val_f1s = []
    val_accuracies = []
    val_precisions = []
    val_recalls = []
    best_val_loss = float('inf')
    best_state_dict = None
    best_epoch = 0
    best_metrics = {}
    
    # Track best F1 score instead of loss
    best_f1 = -float('inf')
    
    for epoch in range(epochs):
        # Training
        probe.train()
        epoch_loss = 0
        
        # Shuffle indices
        indices = torch.randperm(len(train_examples))
        
        # Process batches
        for i in range(0, len(indices), batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_x = train_examples[batch_indices]
            batch_y = train_labels[batch_indices]
            
            # Forward pass
            outputs = probe(batch_x)
            loss = F.binary_cross_entropy_with_logits(outputs, batch_y)
            
            # Backward and optimize
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item() * len(batch_indices)
        
        avg_train_loss = epoch_loss / len(train_examples)
        train_losses.append(avg_train_loss)
        
        # Validation
        probe.eval()
        with torch.no_grad():
            val_outputs = probe(val_examples)
            val_loss = F.binary_cross_entropy_with_logits(val_outputs, val_labels)
            val_losses.append(val_loss.item())
            
            # Calculate metrics
            val_preds = (torch.sigmoid(val_outputs) > 0.5).float().cpu().numpy()
            val_true = val_labels.cpu().numpy()
            
            # Macro F1 score across all labels
            f1 = f1_score(val_true, val_preds, average='weighted')
            accuracy = accuracy_score(val_true.flatten(), val_preds.flatten())
            precision = precision_score(val_true, val_preds, average='weighted', zero_division=0)
            recall = recall_score(val_true, val_preds, average='weighted', zero_division=0)
            
            val_f1s.append(f1)
            val_accuracies.append(accuracy)
            val_precisions.append(precision)
            val_recalls.append(recall)
            
            # Save best model based on F1 score instead of validation loss
            if f1 > best_f1:
                best_f1 = f1
                best_state_dict = probe.state_dict()
                best_epoch = epoch
                best_metrics = {
                    'val_loss': val_loss.item(),
                    'f1': f1,
                    'accuracy': accuracy,
                    'precision': precision,
                    'recall': recall
                }
        
        print(f"Epoch {epoch+1}/{epochs} - Train loss: {avg_train_loss:.4f}, Val loss: {val_loss.item():.4f}, F1: {f1:.4f}, Accuracy: {accuracy:.4f}, Precision: {precision:.4f}, Recall: {recall:.4f}")
    
    # Load best model
    probe.load_state_dict(best_state_dict)
    
    # Print best model metrics
    print("\nBest probe saved from epoch", best_epoch + 1)
    print(f"Best validation metrics (selected by highest F1) - Loss: {best_metrics['val_loss']:.4f}, F1: {best_metrics['f1']:.4f}, Accuracy: {best_metrics['accuracy']:.4f}, Precision: {best_metrics['precision']:.4f}, Recall: {best_metrics['recall']:.4f}")
    
    # Compute and print per-label metrics
    with torch.no_grad():
        val_outputs = probe(val_examples)
        val_preds = (torch.sigmoid(val_outputs) > 0.5).float().cpu().numpy()
        val_true = val_labels.cpu().numpy()
        
        print("\nPer-label metrics:")
        for label_name, label_idx in label_to_idx.items():
            # Get metrics for this specific label
            label_f1 = f1_score(val_true[:, label_idx], val_preds[:, label_idx], average='binary', zero_division=0)
            label_accuracy = accuracy_score(val_true[:, label_idx], val_preds[:, label_idx])
            label_precision = precision_score(val_true[:, label_idx], val_preds[:, label_idx], zero_division=0)
            label_recall = recall_score(val_true[:, label_idx], val_preds[:, label_idx], zero_division=0)
            label_support = np.sum(val_true[:, label_idx])
            
            print(f"  {label_name}:")
            print(f"    F1: {label_f1:.4f}, Accuracy: {label_accuracy:.4f}, Precision: {label_precision:.4f}, Recall: {label_recall:.4f}, Support: {int(label_support)}")
    
    return probe, train_losses, val_losses, val_f1s, val_accuracies, val_precisions, val_recalls

# %% Set seed for reproducibility
random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)

# Create directories
os.makedirs('results/vars', exist_ok=True)
os.makedirs('results/figures', exist_ok=True)

#%%  Load model and responses
model_name = args.model
print(f"Loading model {model_name}...")
model, tokenizer, feature_dict = utils.load_model(compute_features=False, model_name=model_name, load_in_8bit=args.load_in_8bit)

# %% Get model identifier for file naming
model_id = model_name.split('/')[-1].lower()
responses_path = f'../generate-responses/results/vars/responses_{model_id}.json'

with open(responses_path, 'r') as f:
    responses = json.load(f)

responses = responses[:args.num_samples]
print([re.findall(r'\["(.*?)"\]', x["annotated_thinking"]) for x in responses])
random.shuffle(responses)

# %% Shuffle responses and split train/val
train_size = int(0.8 * len(responses))
train_responses = responses[:train_size]
val_responses = responses[train_size:]

print(f"Training on {len(train_responses)} examples, validating on {len(val_responses)}")

# Define label mapping (same as in other scripts)
label_to_idx = {
    label: i for i, label in enumerate([x for x in feature_dict.keys() if x != "overall"])
}

labels = list(label_to_idx.keys())
print(labels)

# %% Train the probe
layer_idx = args.probe_layer
epochs = args.epochs
batch_size = args.batch_size
lr = args.lr

probe, train_losses, val_losses, val_f1s, val_accuracies, val_precisions, val_recalls = train_probe(
    model, tokenizer, train_responses, val_responses, 
    labels, layer_idx, epochs, batch_size, lr
)

# Save the probe
probe_path = f"results/vars/probe_layer{layer_idx}_{model_id}.pt"
torch.save({
    'probe_state_dict': probe.state_dict(),
    'train_losses': train_losses,
    'val_losses': val_losses,
    'val_f1s': val_f1s,
    'val_accuracies': val_accuracies,
    'val_precisions': val_precisions,
    'val_recalls': val_recalls,
    'label_to_idx': label_to_idx
}, probe_path)
print(f"Saved probe to {probe_path}")

# Plot training curves
plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
plt.plot(train_losses, label='Train Loss')
plt.plot(val_losses, label='Val Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()

plt.subplot(1, 3, 2)
plt.plot(val_f1s, label='F1 Score')
plt.plot(val_precisions, label='Precision')
plt.plot(val_recalls, label='Recall')
plt.title('Validation Metrics')
plt.xlabel('Epoch')
plt.ylabel('Score')
plt.legend()

plt.subplot(1, 3, 3)
plt.plot(val_accuracies, label='Accuracy')
plt.title('Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')

plt.tight_layout()
plt.savefig(f'results/figures/probe_training_layer{layer_idx}_{model_id}.pdf')
plt.show()


# %%
