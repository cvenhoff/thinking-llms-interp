#!/bin/bash
# Stage 1 bias-first for Qwen2.5-32B + QwQ-32B - DDP across 3 H200s.
# Each rank holds a full copy of the (frozen) base model (~64 GB);
# V/b grads are all-reduced across ranks after each backward.
# Effective batch = BS_PER_GPU * 3.
set -euo pipefail
cd /workspace/thinking-llms-interp/train-vectors
source ../.venv/bin/activate
source ../.env_exports.sh

SAVE_DIR="results/vars/correction_vectors_qwq32b_biasfirst_stage1"
mkdir -p "$SAVE_DIR"
LOG="/tmp/qwq32b_biasfirst_stage1.log"
BS_PER_GPU=${BS_PER_GPU:-${BS:-8}}
N_EPOCHS=${N_EPOCHS:-2}
NPROC=${NPROC:-3}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-1536}

SRC_DUMP=results/vars/correction_vectors_qwq32b/disagreements.pt
if [ -f "$SRC_DUMP" ] && [ ! -f "$SAVE_DIR/disagreements.pt" ]; then
    cp "$SRC_DUMP" "$SAVE_DIR/disagreements.pt"
fi

# torchrun spawns NPROC processes with RANK / WORLD_SIZE / LOCAL_RANK
# env vars; the trainer uses these to (a) load the base model on its
# own GPU via device_map={"": local_rank}, (b) shard training buckets
# across ranks, (c) all-reduce V.grad / b.grad after each backward.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
torchrun --standalone --nproc_per_node="$NPROC" \
    optimize_correction_vectors.py \
    --base_model "Qwen/Qwen2.5-32B" \
    --thinking_model "Qwen/QwQ-32B" \
    --thinking_model_short "qwq-32b" \
    --steer_layer 38 \
    --save_dir "$SAVE_DIR" \
    --topk 50 --train_topk 3 --kl_mode topk \
    --max_seq_len "$MAX_SEQ_LEN" --max_positions_per_example 64 \
    --n_epochs "$N_EPOCHS" --example_batch_size "$BS_PER_GPU" \
    --lr 0.01 --weight_decay 0.0 --max_norm 0.0 \
    --seed 42 --holdout_frac 0.1 \
    --min_disagreements 1 --min_disagreements_ratio 0.0 --min_category_share 0.0 \
    --load_collected --skip_cats_phase --train_global_bias \
    2>&1 | tee "$LOG"

echo "DONE Stage 1. Bias at: $SAVE_DIR"
ls -la "$SAVE_DIR"
