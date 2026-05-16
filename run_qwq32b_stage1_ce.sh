#!/bin/bash
# QwQ-32B Stage 1 (global bias) retrain with CE loss + max_norm cap.
# Reuses cached Stage 0 disagreements.pt.
set -uo pipefail
cd /workspace/thinking-llms-interp/train-vectors
source ../.venv/bin/activate
source ../.env_exports.sh

SRC_DIR="results/vars/correction_vectors_qwq32b_biasfirst_stage1"
TAG=${TAG:-stage1_ce_cap25}
DST_DIR="results/vars/correction_vectors_qwq32b_biasfirst_${TAG}"
MAX_NORM=${MAX_NORM:-25.0}
N_EPOCHS=${N_EPOCHS:-2}
LR=${LR:-0.005}

mkdir -p "$DST_DIR"
if [ ! -f "$DST_DIR/disagreements.pt" ]; then
    cp "$SRC_DIR/disagreements.pt" "$DST_DIR/"
fi

LOG="/tmp/qwq32b_${TAG}.log"
echo "===== Stage 1 retrain (kl_mode=ce, max_norm=${MAX_NORM}, lr=${LR}, n_epochs=${N_EPOCHS}) ====="
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

# DDP across 2 GPUs (rank-0 trainable params, V/b grads all-reduced)
NPROC=${NPROC:-2}
torchrun --standalone --nproc_per_node="$NPROC" \
    optimize_correction_vectors.py \
    --base_model "Qwen/Qwen2.5-32B" \
    --thinking_model "Qwen/QwQ-32B" \
    --thinking_model_short "qwq-32b" \
    --steer_layer 38 \
    --save_dir "$DST_DIR" \
    --topk 50 --train_topk 3 --kl_mode ce \
    --max_seq_len 1536 --max_positions_per_example 64 \
    --n_epochs "$N_EPOCHS" --example_batch_size 8 \
    --lr "$LR" --weight_decay 0.0 --max_norm "$MAX_NORM" \
    --seed 42 --holdout_frac 0.1 \
    --min_disagreements 1 --min_disagreements_ratio 0.0 --min_category_share 0.0 \
    --load_collected --skip_cats_phase --train_global_bias \
    2>&1 | tee "$LOG"

echo "DONE Stage 1 ($TAG)."
ls -la "$DST_DIR"
