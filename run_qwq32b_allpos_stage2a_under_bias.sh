#!/bin/bash
# Stage 2a: re-collect disagreements UNDER the new (all-pos, CE, cap=25) bias.
# Filter mode: disagreement (default) -- keep only positions where
# base+bias top-K still misses thinking's top-1. These are the residual
# signal cats should fit.
set -uo pipefail
cd /workspace/thinking-llms-interp/train-vectors
source ../.venv/bin/activate
source ../.env_exports.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY=error

BIAS_DIR="results/vars/correction_vectors_qwq32b_biasfirst_allpos_stage1_ce_cap25"
BIAS_PATH="$BIAS_DIR/qwen2.5-32b_bias_global.pt"
[ -f "$BIAS_PATH" ] || { echo "ERROR: bias missing at $BIAS_PATH"; exit 1; }

SAVE_DIR="results/vars/correction_vectors_qwq32b_biasfirst_allpos_stage2a_under_bias"
mkdir -p "$SAVE_DIR"
LOG="/tmp/qwq32b_allpos_stage2a_under_bias.log"

if [ -f "$SAVE_DIR/disagreements.pt" ]; then
    echo "Already collected. Skipping."
    exit 0
fi

CUDA_VISIBLE_DEVICES=${GPUS:-1,2} python -u optimize_correction_vectors.py \
    --base_model "Qwen/Qwen2.5-32B" \
    --thinking_model "Qwen/QwQ-32B" \
    --thinking_model_short "qwq-32b" \
    --steer_layer 38 \
    --save_dir "$SAVE_DIR" \
    --topk 50 --train_topk 3 --kl_mode ce \
    --n_responses 20000 \
    --max_seq_len 1536 --max_positions_per_example 64 \
    --seed 42 --holdout_frac 0.1 \
    --collection_mode disagreement \
    --frozen_bias_path "$BIAS_PATH" --frozen_bias_layer 38 \
    --collect_only \
    2>&1 | tee "$LOG"

echo "DONE Stage 2a (under-bias)."
ls -la "$SAVE_DIR"
