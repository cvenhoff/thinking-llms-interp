#!/bin/bash
# Stage 2 bias-first for Qwen2.5-32B + DeepSeek-R1-Distill-Qwen-32B.
set -euo pipefail
cd /workspace/thinking-llms-interp/train-vectors
source ../.venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY=error

STAGE1_DIR="results/vars/correction_vectors_dsqwen32b_biasfirst_stage1"
BIAS_PATH="$STAGE1_DIR/qwen2.5-32b_bias_global.pt"
if [ ! -f "$BIAS_PATH" ]; then
    echo "ERROR: bias not found at $BIAS_PATH"; exit 1
fi

SAVE_DIR="results/vars/correction_vectors_dsqwen32b_biasfirst_stage2"
mkdir -p "$SAVE_DIR"

LOG2A="/tmp/dsqwen32b_biasfirst_stage2a_collect.log"
if [ ! -f "$SAVE_DIR/disagreements.pt" ]; then
    echo "===== Stage 2a: collecting disagreements under bias steering ====="
    python -u optimize_correction_vectors.py \
        --base_model "Qwen/Qwen2.5-32B" \
        --thinking_model "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" \
        --thinking_model_short "deepseek-r1-distill-qwen-32b" \
        --steer_layer 38 \
        --save_dir "$SAVE_DIR" \
        --topk 50 --train_topk 3 --kl_mode topk \
        --n_responses 20000 \
        --max_seq_len 2048 --max_positions_per_example 64 \
        --seed 42 --holdout_frac 0.1 \
        --frozen_bias_path "$BIAS_PATH" \
        --frozen_bias_layer 38 \
        --collect_only \
        2>&1 | tee "$LOG2A"
else
    echo "Stage 2a already done, skipping."
fi

LOG2B="/tmp/dsqwen32b_biasfirst_stage2b_train.log"
echo "===== Stage 2b: training cat vectors with frozen bias ====="
python -u optimize_correction_vectors.py \
    --base_model "Qwen/Qwen2.5-32B" \
    --thinking_model "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" \
    --thinking_model_short "deepseek-r1-distill-qwen-32b" \
    --steer_layer 38 \
    --save_dir "$SAVE_DIR" \
    --topk 50 --train_topk 3 --kl_mode topk \
    --max_seq_len 2048 --max_positions_per_example 64 \
    --n_epochs 5 --example_batch_size 16 \
    --lr 0.01 --weight_decay 0.0 --max_norm 0.0 \
    --seed 42 --holdout_frac 0.1 \
    --min_disagreements 1 --min_disagreements_ratio 0.0 --min_category_share 0.0 \
    --frozen_bias_path "$BIAS_PATH" \
    --frozen_bias_layer 38 \
    --load_collected \
    2>&1 | tee "$LOG2B"

echo
echo "DONE Stage 2. Cats at: $SAVE_DIR  (bias at: $STAGE1_DIR)"
