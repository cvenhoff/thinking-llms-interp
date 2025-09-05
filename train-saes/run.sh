#!/usr/bin/env bash
set -euo pipefail

# Resolve project root (one directory up from this script) and ensure local imports
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")"/.. && pwd)"
export PYTHONPATH="${ROOT_DIR}:${PYTHONPATH-}"

# Use the project's virtual environment Python
PY="${ROOT_DIR}/.venv/bin/python"
if [ ! -x "$PY" ]; then
    echo "Expected virtualenv Python at $PY not found. Create it with: uv venv .venv && uv sync" 1>&2
    exit 1
fi

CLUSTERS="10" # 5 10 15 20 25 30 35 40 45 50
N_EXAMPLES=100000  # all responses

# CLUSTERING_METHODS="gmm pca_gmm spherical_kmeans pca_kmeans agglomerative pca_agglomerative sae_topk"
CLUSTERING_METHODS="sae_topk"

# MODELS="deepseek-ai/DeepSeek-R1-Distill-Llama-8B deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
MODELS="qwen/QwQ-32B"

REPETITIONS=5

get_layers() {
    local model=$1
    case "$model" in
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B") echo "6 10 14 18 22 26" ;; # 6 10 14 18 22 26
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B") echo "4 8 12 16 20 24" ;;
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B") echo "8 14 20 26 32 38" ;;
        "qwen/QwQ-32B") echo "9" ;; # 9 18 27 36 45 54
        *) echo "" ;;
    esac
}

# Generate activations for all models and layers
for MODEL in $MODELS; do
    LAYERS_TO_PROCESS=$(get_layers "$MODEL")
    if [ -n "$LAYERS_TO_PROCESS" ]; then
        "$PY" generate_activations.py --model "$MODEL" --layers $LAYERS_TO_PROCESS --n_examples $N_EXAMPLES
    fi
done

# huggingface-cli upload-large-folder iarcuschin/base-models-reasoning-interp --repo-type=model ../generate-responses/results/vars --include *.pkl --num-workers=16

# Train all clustering methods for all models and layers
# for MODEL in $MODELS; do
#     for LAYER in $(get_layers $MODEL); do
#         "$PY" train_clustering.py --model $MODEL --layer $LAYER --clusters $CLUSTERS --n_examples $N_EXAMPLES --clustering_methods $CLUSTERING_METHODS --sae_debug
#     done
# done

# # Generate titles for all clustering methods for all models and layers
# for MODEL in $MODELS; do
#     for LAYER in $(get_layers $MODEL); do
#         "$PY" generate_titles_trained_clustering.py --model $MODEL --layer $LAYER --clusters $CLUSTERS --n_examples $N_EXAMPLES --clustering_methods $CLUSTERING_METHODS --repetitions $REPETITIONS --command direct
#     done
# done

# # Wait for titles to be generated
# for MODEL in $MODELS; do
#     for LAYER in $(get_layers $MODEL); do
#         "$PY" generate_titles_trained_clustering.py --model $MODEL --layer $LAYER --clusters $CLUSTERS --n_examples $N_EXAMPLES --clustering_methods $CLUSTERING_METHODS --repetitions $REPETITIONS --command process --wait-batch-completion
#     done
# done

# # Evaluate all clustering methods for all models and layers
# for MODEL in $MODELS; do
#     for LAYER in $(get_layers $MODEL); do
#         # Extra flags to disable re-computing some of the evaluation metrics, use as needed: --no-accuracy --no-completeness --no-orth --no-sem-orth
#         "$PY" evaluate_trained_clustering.py --model $MODEL --layer $LAYER --clusters $CLUSTERS --n_examples $N_EXAMPLES --clustering_methods $CLUSTERING_METHODS --repetitions $REPETITIONS --command submit --accuracy_target_cluster_percentage 0.2
#     done
# done

# # Wait for evaluation to complete
# for MODEL in $MODELS; do
#     for LAYER in $(get_layers $MODEL); do
#         "$PY" evaluate_trained_clustering.py --model $MODEL --layer $LAYER --clusters $CLUSTERS --n_examples $N_EXAMPLES --clustering_methods $CLUSTERING_METHODS --repetitions $REPETITIONS --command process --wait-batch-completion
#     done
# done

# # Visualize all clustering methods for all models and layers
# for MODEL in $MODELS; do
#     for LAYER in $(get_layers $MODEL); do
#         "$PY" visualize_results.py --model $MODEL --layer $LAYER --clusters 5 10 15 20 25 30 35 40 45 50 --clustering_methods $CLUSTERING_METHODS
#         "$PY" visualize_comparison.py --model $MODEL --layer $LAYER
#     done
#     "$PY" visualize_clusters.py --model $MODEL
# done
# "$PY" visualize_clusters.py --model all