#!/usr/bin/env bash
# ============================================================
# Legacy CE-loss steering vector training: ORZ-0.5B
# Base model:     Qwen/Qwen2.5-0.5B        (layer 9)
# Thinking model: Open-Reasoner-Zero-0.5B  (inferred from model_mapping)
# SAE:            layer 8, 10 clusters
#
# Phase 1: global bias   (--steering_vector_idx -1)
# Phase 2: cat idx 0..9  (--use_activation_perplexity_selection)
#
# Usage:
#   CUDA_VISIBLE_DEVICES=1 bash run_legacy_orz_0.5b.sh [--n_train N] [--n_iters I]
# ============================================================
set -e

N_TRAIN=${N_TRAIN:-2048}
N_EVAL=${N_EVAL:-512}
MAX_ITERS=${MAX_ITERS:-50}
LAYER=9
N_CATS=10
MINI_BS=4
LR="1e-2"
MODEL="Qwen/Qwen2.5-0.5B"
SAVE_DIR="results/vars/optimized_vectors"

echo "======================================"
echo " ORZ-0.5B Legacy Training"
echo "   N_TRAIN=$N_TRAIN  N_EVAL=$N_EVAL  MAX_ITERS=$MAX_ITERS"
echo "   LAYER=$LAYER  MODEL=$MODEL"
echo "======================================"

# Phase 1: Bias vector (full thinking traces, no perplexity selection)
echo ""
echo "=== Phase 1: Global bias vector ==="
python optimize_steering_vectors.py \
    --model "$MODEL" \
    --max_iters "$MAX_ITERS" \
    --n_training_examples "$N_TRAIN" \
    --n_eval_examples "$N_EVAL" \
    --optim_minibatch_size "$MINI_BS" \
    --layer "$LAYER" \
    --steering_vector_idx -1 \
    --lr "$LR" \
    --save_path "$SAVE_DIR"

echo ""
echo "Bias training complete. File: ${SAVE_DIR}/qwen2.5-0.5b_bias_linear.pt"

# Phase 2: Category vectors (activation + perplexity selection, bias pre-applied)
for CLUSTER in $(seq 0 $((N_CATS - 1))); do
    echo ""
    echo "=== Phase 2: Cat idx${CLUSTER} ==="
    python optimize_steering_vectors.py \
        --model "$MODEL" \
        --max_iters "$MAX_ITERS" \
        --n_training_examples "$N_TRAIN" \
        --n_eval_examples "$N_EVAL" \
        --optim_minibatch_size "$MINI_BS" \
        --layer "$LAYER" \
        --steering_vector_idx "$CLUSTER" \
        --lr "$LR" \
        --use_activation_perplexity_selection \
        --save_path "$SAVE_DIR"
    echo "Cat idx${CLUSTER} done: ${SAVE_DIR}/qwen2.5-0.5b_idx${CLUSTER}_linear.pt"
done

# Write layer_map.json so hybrid_eval.py picks up the right layer
echo ""
echo "=== Writing layer_map.json ==="
python write_layer_map.py --save_dir "$SAVE_DIR" --layer "$LAYER" --n_cats "$N_CATS"

echo ""
echo "======================================"
echo " ORZ-0.5B training COMPLETE"
echo " Vectors: $SAVE_DIR"
echo "======================================"
