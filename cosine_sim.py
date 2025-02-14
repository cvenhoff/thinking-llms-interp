# %%
import torch
import matplotlib.pyplot as plt

model_name = "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
mean_vectors_dict = torch.load(f"data/mean_vectors_{model_name.split('/')[-1].lower()}.pt")
feature_vectors = {}

print(mean_vectors_dict.keys())

overall_mean = mean_vectors_dict['overall']['mean']
print(overall_mean[0].norm())
for label in mean_vectors_dict:
    if label != 'overall':
        print(mean_vectors_dict[label]['mean'][0].norm())
        feature_vectors[label] = mean_vectors_dict[label]['mean'] - overall_mean

def plot_cosine_similarity_heatmap(feature_vectors, model_id):
    labels = list(feature_vectors.keys())

    labels.remove('initializing')
    labels.remove('deduction')

    n_labels = len(labels)
    
    # Create similarity matrix
    similarity_matrix = torch.zeros((n_labels, n_labels))

    n_layers = feature_vectors[labels[0]].shape[0]

    for layer_idx in range(n_layers):
            for i, label_1 in enumerate(labels):
                for j, label_2 in enumerate(labels):
                    similarity_matrix[i, j] += torch.cosine_similarity(
                        feature_vectors[label_1][layer_idx], 
                        feature_vectors[label_2][layer_idx], 
                        dim=-1
                    )
    
    similarity_matrix /= n_layers
    
    # Create heatmap
    plt.figure(figsize=(10, 8))
    im = plt.imshow(similarity_matrix, cmap='RdBu', vmin=-1, vmax=1)
    plt.colorbar(label='Cosine Similarity')
    
    # Add labels and text annotations
    plt.xticks(range(n_labels), labels, rotation=45, ha='right')
    plt.yticks(range(n_labels), labels)
    
    # Add text annotations
    for i in range(n_labels):
        for j in range(n_labels):
            value = similarity_matrix[i, j].item()
            # Choose text color based on similarity value
            text_color = 'white' if value > 0.8 else 'black'
            plt.text(j, i, f'{value:.2f}', 
                    ha='center', va='center',
                    color=text_color)
    
    plt.title('Cosine Similarity Between Feature Vectors')
    plt.tight_layout()
    plt.savefig(f'figures/cosine_similarity_heatmap_{model_id}.png', dpi=300)
    plt.show()


# Plot the aggregated heatmap
plot_cosine_similarity_heatmap(feature_vectors, model_id=model_name.split('/')[-1].lower())

# %%
