#!/bin/bash
# Stage 1 bias-first, QwQ-32B + Qwen2.5-32B base, at LAYER 24.
# Same recipe as the layer-38 script, but injecting at residual stream
# layer 24 (matching the historical empirically-validated steer depth
# for QwQ-32B).  Re-uses the layer-agnostic Stage-1 disagreements dump.
set -euo pipefail
cd /workspace/thinking-llms-interp/train-vectors
source ../.venv/bin/activate
source ../.env_exports.sh

LAYER=24
SAVE_DIR="results/vars/correction_vectors_qwq32b_biasfirst_stage1_l${LAYER}"
mkdir -p "$SAVE_DIR"
LOG="/tmp/qwq32b_biasfirst_stage1_l${LAYER}.log"
BS_PER_GPU=${BS_PER_GPU:-${BS:-4}}
N_EPOCHS=${N_EPOCHS:-1}
NPROC=${NPROC:-3}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-1536}

# Re-use the layer-38 Stage-1 disagreements dump: it was collected with
# frozen_bias=None so it is independent of steer_layer (only the base
# model's argmax matters).  Copy if not already present.
SRC_DUMP=results/vars/correction_vectors_qwq32b_biasfirst_stage1/disagreements.pt
if [ -f "$SRC_DUMP" ] && [ ! -f "$SAVE_DIR/disagreements.pt" ]; then
    echo "Reusing layer-agnostic Stage-1 disagreements from $SRC_DUMP"
    cp "$SRC_DUMP" "$SAVE_DIR/disagreements.pt"
fi

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

echo "DONE Stage 1 (L${LAYER}). Bias at: $SAVE_DIR"
ls -la "$SAVE_DIR"
