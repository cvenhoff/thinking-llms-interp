#!/bin/bash
# Stage 0 (data) for Qwen2.5-32B + DeepSeek-R1-Distill-Qwen-32B:
#  - collect initial disagreements WITHOUT any bias (writes disagreements.pt
#    that Stage 1 then trains a bias on).
set -euo pipefail
cd /workspace/thinking-llms-interp/train-vectors
source ../.venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY=error

SAVE_DIR="results/vars/correction_vectors_dsqwen32b_biasfirst_stage1"
mkdir -p "$SAVE_DIR"
LOG="/tmp/dsqwen32b_biasfirst_stage0_collect.log"

if [ -f "$SAVE_DIR/disagreements.pt" ]; then
    echo "Stage 0 already done (disagreements.pt exists), skipping."
    exit 0
fi

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
    --collect_only \
    2>&1 | tee "$LOG"

echo "DONE Stage 0. disagreements.pt at: $SAVE_DIR"
