#!/bin/bash
# Full clean pipeline: pairs → vectors → hybrid
set -euo pipefail

cd /workspace/thinking-llms-interp/train-vectors
source ../.venv/bin/activate
export $(cat ../.env | xargs)
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# ---- Model config ----
MODEL="Qwen/Qwen2.5-7B"
MODEL_SHORT="qwen2.5-7b"
THINKING_MODEL="Open-Reasoner-Zero/Open-Reasoner-Zero-7B"
THINKING_SHORT="open-reasoner-zero-7b"

SAE_LAYER=20
N_CLUSTERS=10

# ---- Data sizes ----
N_TRAIN=2500
N_EVAL=100
MAX_NEW_TOKENS=1024
BATCH_SIZE=256
MAX_CONCURRENT=100
SEED=42
API_MODEL="gpt-4.1"

# ---- Steering vector config ----
DOM_BATCH_SIZE=48
DOM_ATTRIB_BATCH_SIZE=8
N_ATTRIB=50
STEER_COEFFS="0.5,1.0,2.0,3.0,5.0,8.0"
EVAL_MAX_TOKENS=64
GEN_BATCH_SIZE=256

# ---- Hybrid config ----
HYBRID_DATASET="math500"
HYBRID_N_TASKS=0
HYBRID_BATCH_GEN=128
HYBRID_BATCH_HYBRID=128
HYBRID_MAX_TOKENS=2048
JUDGE_MODEL="openai/gpt-5.2"

# ---- Derived paths ----
PAIRS_DIR="results/synthetic_pairs"
VECTORS_DIR="results/diff_of_means"

TRAIN_PAIRS="${PAIRS_DIR}/synthetic_pairs_${MODEL_SHORT}_${N_CLUSTERS}clusters_train.json"
EVAL_PAIRS="${PAIRS_DIR}/synthetic_pairs_${MODEL_SHORT}_${N_CLUSTERS}clusters_eval.json"
VECTORS_PT="${VECTORS_DIR}/dom_vectors_multilayer_${MODEL_SHORT}.pt"
BEST_COEFFS="${VECTORS_DIR}/dom_best_coeffs_${MODEL_SHORT}.json"

mkdir -p "$PAIRS_DIR" "$VECTORS_DIR/figures"

# ==============================================================================
# STEP 1: Generate TRAINING pairs
# ==============================================================================
echo "============================================================"
echo "[1/4] Generate train pairs (n=${N_TRAIN})"
echo "============================================================"
if [ -f "${TRAIN_PAIRS}" ]; then
    echo "  -> Already exists, skipping."
else
    python -u generate_pairs.py \
        --model "${MODEL}" \
        --api_model "${API_MODEL}" \
        --thinking_model "${THINKING_MODEL}" \
        --sae_layer "${SAE_LAYER}" \
        --n_clusters "${N_CLUSTERS}" \
        --n_questions "${N_TRAIN}" \
        --max_new_tokens "${MAX_NEW_TOKENS}" \
        --batch_size "${BATCH_SIZE}" \
        --max_concurrent "${MAX_CONCURRENT}" \
        --seed "${SEED}" \
        --question_offset 0 \
        --output_suffix "_train" \
        --save_dir "${PAIRS_DIR}" \
        2>&1
fi

# ==============================================================================
# STEP 2: Generate EVAL pairs (disjoint questions)
# ==============================================================================
echo ""
echo "============================================================"
echo "[2/4] Generate eval pairs (n=${N_EVAL}, offset=${N_TRAIN})"
echo "============================================================"
if [ -f "${EVAL_PAIRS}" ]; then
    echo "  -> Already exists, skipping."
else
    python -u generate_pairs.py \
        --model "${MODEL}" \
        --api_model "${API_MODEL}" \
        --thinking_model "${THINKING_MODEL}" \
        --sae_layer "${SAE_LAYER}" \
        --n_clusters "${N_CLUSTERS}" \
        --n_questions "${N_EVAL}" \
        --max_new_tokens "${MAX_NEW_TOKENS}" \
        --batch_size "${BATCH_SIZE}" \
        --max_concurrent "${MAX_CONCURRENT}" \
        --seed "${SEED}" \
        --question_offset "${N_TRAIN}" \
        --output_suffix "_eval" \
        --save_dir "${PAIRS_DIR}" \
        2>&1
fi

# ==============================================================================
# STEP 3: Compute vectors + attribution + coefficient sweep
# ==============================================================================
echo ""
echo "============================================================"
echo "[3/4] Compute steering vectors + attribution + coeff sweep"
echo "============================================================"
if [ -f "${BEST_COEFFS}" ]; then
    echo "  -> Already exists, skipping."
else
    SKIP_FLAGS=""
    if [ -f "${VECTORS_PT}" ]; then
        SKIP_FLAGS="${SKIP_FLAGS} --skip_vectors"
    fi
    if [ -f "${VECTORS_DIR}/dom_attribution_${MODEL_SHORT}.json" ]; then
        SKIP_FLAGS="${SKIP_FLAGS} --skip_attribution"
    fi

    python -u compute_vectors.py \
        --model "${MODEL}" \
        --pairs_file "${TRAIN_PAIRS}" \
        --eval_pairs_file "${EVAL_PAIRS}" \
        --save_dir "${VECTORS_DIR}" \
        --n_eval_questions "${N_EVAL}" \
        --batch_size "${DOM_BATCH_SIZE}" \
        --attrib_batch_size "${DOM_ATTRIB_BATCH_SIZE}" \
        --gen_batch_size "${GEN_BATCH_SIZE}" \
        --n_attribution_examples "${N_ATTRIB}" \
        --steer_coeffs "${STEER_COEFFS}" \
        --eval_max_tokens "${EVAL_MAX_TOKENS}" \
        --api_model "${API_MODEL}" \
        --max_concurrent "${MAX_CONCURRENT}" \
        --min_layer_frac 0.2 \
        --seed "${SEED}" \
        ${SKIP_FLAGS} \
        2>&1
fi

# ==============================================================================
# STEP 4: Hybrid evaluation
# ==============================================================================
echo ""
echo "============================================================"
echo "[4/4] Hybrid evaluation on ${HYBRID_DATASET} (n_tasks=${HYBRID_N_TASKS})"
echo "============================================================"
cd /workspace/thinking-llms-interp/hybrid
python -u hybrid_eval.py \
    --dataset "${HYBRID_DATASET}" \
    --thinking_model "${THINKING_MODEL}" \
    --base_model "${MODEL}" \
    --sae_layer "${SAE_LAYER}" \
    --n_clusters "${N_CLUSTERS}" \
    --n_tasks "${HYBRID_N_TASKS}" \
    --batch_gen_size "${HYBRID_BATCH_GEN}" \
    --hybrid_gen_batch_size "${HYBRID_BATCH_HYBRID}" \
    --max_new_tokens "${HYBRID_MAX_TOKENS}" \
    --max_thinking_tokens "${HYBRID_MAX_TOKENS}" \
    --dom_vectors_dir "/workspace/thinking-llms-interp/train-vectors/${VECTORS_DIR}" \
    --dom_vectors_model_short "${MODEL_SHORT}" \
    --coeff_sweep \
    --normalize_to_mean_norm \
    --judge_model "${JUDGE_MODEL}" \
    --max_concurrent "${MAX_CONCURRENT}" \
    --results_suffix "dom-vectors-coeff-sweep" \
    2>&1

echo ""
echo "============================================================"
echo "PIPELINE COMPLETE"
echo "============================================================"

# ==============================================================================
# STEP 5 (optional): Random-vector baseline
# ==============================================================================
# Uncomment below to run a random-noise control with the same setup.
# python -u hybrid_eval.py \
#     --dataset "${HYBRID_DATASET}" \
#     --thinking_model "${THINKING_MODEL}" \
#     --base_model "${MODEL}" \
#     --sae_layer "${SAE_LAYER}" \
#     --n_clusters "${N_CLUSTERS}" \
#     --n_tasks "${HYBRID_N_TASKS}" \
#     --batch_gen_size "${HYBRID_BATCH_GEN}" \
#     --hybrid_gen_batch_size "${HYBRID_BATCH_HYBRID}" \
#     --max_new_tokens "${HYBRID_MAX_TOKENS}" \
#     --max_thinking_tokens "${HYBRID_MAX_TOKENS}" \
#     --dom_vectors_dir "/workspace/thinking-llms-interp/train-vectors/${VECTORS_DIR}" \
#     --dom_vectors_model_short "${MODEL_SHORT}" \
#     --coeff_sweep \
#     --normalize_to_mean_norm \
#     --random_vectors \
#     --judge_model "${JUDGE_MODEL}" \
#     --max_concurrent "${MAX_CONCURRENT}" \
#     --results_suffix "dom-vectors-random-baseline" \
#     2>&1
