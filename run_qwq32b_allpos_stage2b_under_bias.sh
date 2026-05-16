#!/bin/bash
# Stage 2b: train cats on disagreements collected UNDER the new bias.
# kl_mode=ce, max_norm=6 (matches rescale-6 best result), frozen new bias.
set -uo pipefail
cd /workspace/thinking-llms-interp/train-vectors
source ../.venv/bin/activate
source ../.env_exports.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY=error

SRC_DIR="results/vars/correction_vectors_qwq32b_biasfirst_allpos_stage2a_under_bias"
BIAS_DIR="results/vars/correction_vectors_qwq32b_biasfirst_allpos_stage1_ce_cap25"
TAG=${TAG:-allpos_under_bias_stage2b_ce_cap6}
DST_DIR="results/vars/correction_vectors_qwq32b_biasfirst_${TAG}"
MAX_NORM=${MAX_NORM:-6.0}
N_EPOCHS=${N_EPOCHS:-2}
LR=${LR:-0.005}
BS_PER_GPU=${BS_PER_GPU:-8}

mkdir -p "$DST_DIR"
if [ ! -f "$DST_DIR/disagreements.pt" ]; then
    cp "$SRC_DIR/disagreements.pt" "$DST_DIR/"
fi

LOG="/tmp/qwq32b_${TAG}.log"
echo "===== Stage 2b under-bias data (kl=ce, cap=${MAX_NORM}, lr=${LR}, ep=${N_EPOCHS}) ====="

CUDA_VISIBLE_DEVICES=${GPUS:-1,2} python -u optimize_correction_vectors.py \
    --base_model "Qwen/Qwen2.5-32B" \
    --thinking_model "Qwen/QwQ-32B" \
    --thinking_model_short "qwq-32b" \
    --steer_layer 38 \
    --save_dir "$DST_DIR" \
    --topk 50 --train_topk 3 --kl_mode ce \
    --max_seq_len 1536 --max_positions_per_example 64 \
    --n_epochs "$N_EPOCHS" --example_batch_size "$BS_PER_GPU" \
    --lr "$LR" --weight_decay 0.0 --max_norm "$MAX_NORM" \
    --seed 42 --holdout_frac 0.1 \
    --min_disagreements 1 --min_disagreements_ratio 0.0 --min_category_share 0.0 \
    --frozen_bias_path "$BIAS_DIR/qwen2.5-32b_bias_global.pt" --frozen_bias_layer 38 \
    --load_collected \
    2>&1 | tee "$LOG"

echo "DONE Stage 2b ($TAG)."
ls -la "$DST_DIR" | grep -E "best|metrics|idx" | tail
