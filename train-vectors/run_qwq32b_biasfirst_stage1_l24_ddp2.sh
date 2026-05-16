#!/bin/bash
# Stage 1 bias-first, QwQ-32B, LAYER 24, DDP NPROC=2 on GPU 0+1.
# Restart strategy after the NPROC=3 DDP hung at step 900: tighter
# MAX_SEQ_LEN to bound batch variance, fewer ranks to bound NCCL
# allreduce volume, but slightly larger BS to keep throughput.  GPU 2
# is reserved for the biasonly eval running in parallel.
set -euo pipefail
cd /workspace/thinking-llms-interp/train-vectors
source ../.venv/bin/activate
source ../.env_exports.sh

LAYER=24
SAVE_DIR="results/vars/correction_vectors_qwq32b_biasfirst_stage1_l${LAYER}_v2"
mkdir -p "$SAVE_DIR"
LOG="/tmp/qwq32b_biasfirst_stage1_l${LAYER}_v2.log"
BS_PER_GPU=${BS_PER_GPU:-${BS:-5}}
N_EPOCHS=${N_EPOCHS:-1}
NPROC=${NPROC:-2}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-1280}

# Re-use the layer-agnostic disagreements dump.
SRC_DUMP=results/vars/correction_vectors_qwq32b_biasfirst_stage1/disagreements.pt
if [ -f "$SRC_DUMP" ] && [ ! -f "$SAVE_DIR/disagreements.pt" ]; then
    echo "Reusing layer-agnostic Stage-1 disagreements from $SRC_DUMP"
    cp "$SRC_DUMP" "$SAVE_DIR/disagreements.pt"
fi

# Reserve GPUs 0 and 1 for this DDP run; GPU 2 used by biasonly eval.
export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
torchrun --standalone --nproc_per_node="$NPROC" \
    optimize_correction_vectors.py \
    --base_model "Qwen/Qwen2.5-32B" \
    --thinking_model "Qwen/QwQ-32B" \
    --thinking_model_short "qwq-32b" \
    --steer_layer "$LAYER" \
    --save_dir "$SAVE_DIR" \
    --topk 50 --train_topk 3 --kl_mode topk \
    --max_seq_len "$MAX_SEQ_LEN" --max_positions_per_example 64 \
    --n_epochs "$N_EPOCHS" --example_batch_size "$BS_PER_GPU" \
    --lr 0.01 --weight_decay 0.0 --max_norm 0.0 \
    --seed 42 --holdout_frac 0.1 \
    --min_disagreements 1 --min_disagreements_ratio 0.0 --min_category_share 0.0 \
    --load_collected --skip_cats_phase --train_global_bias \
    2>&1 | tee "$LOG"

echo "DONE Stage 1 (L${LAYER} v2). Bias at: $SAVE_DIR"
ls -la "$SAVE_DIR"
