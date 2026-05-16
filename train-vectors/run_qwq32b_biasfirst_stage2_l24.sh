#!/bin/bash
# Stage 2 bias-first, QwQ-32B + Qwen2.5-32B base, at LAYER 24.
#   2a. Re-collect disagreements with the LAYER-24 bias hooked into base.
#       (single-process, pipeline-parallel inference)
#   2b. Train cats as residuals on top of frozen LAYER-24 bias (DDP).
set -euo pipefail
cd /workspace/thinking-llms-interp/train-vectors
source ../.venv/bin/activate
source ../.env_exports.sh

LAYER=24
BS_PER_GPU=${BS_PER_GPU:-${BS:-3}}
N_EPOCHS=${N_EPOCHS:-1}
NPROC=${NPROC:-3}
N_RECOLLECT=${N_RECOLLECT:-3000}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-1536}

STAGE1_DIR="results/vars/correction_vectors_qwq32b_biasfirst_stage1_l${LAYER}"
BIAS_PATH="$STAGE1_DIR/qwen2.5-32b_bias_global.pt"
if [ ! -f "$BIAS_PATH" ]; then
    echo "ERROR: bias not found at $BIAS_PATH"; exit 1
fi

SAVE_DIR="results/vars/correction_vectors_qwq32b_biasfirst_stage2_l${LAYER}"
mkdir -p "$SAVE_DIR"

LOG2A="/tmp/qwq32b_biasfirst_stage2a_l${LAYER}_collect.log"
if [ ! -f "$SAVE_DIR/disagreements.pt" ]; then
    echo "===== Stage 2a (L${LAYER}): re-collecting disagreements under bias ====="
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    python -u optimize_correction_vectors.py \
        --base_model "Qwen/Qwen2.5-32B" \
        --thinking_model "Qwen/QwQ-32B" \
        --thinking_model_short "qwq-32b" \
        --steer_layer "$LAYER" \
        --save_dir "$SAVE_DIR" \
        --topk 50 --train_topk 3 --kl_mode topk \
        --n_responses "$N_RECOLLECT" \
        --max_seq_len "$MAX_SEQ_LEN" --max_positions_per_example 64 \
        --seed 42 --holdout_frac 0.1 \
        --frozen_bias_path "$BIAS_PATH" \
        --frozen_bias_layer "$LAYER" \
        --collect_only \
        2>&1 | tee "$LOG2A"
else
    echo "Stage 2a (L${LAYER}) already done (disagreements.pt exists), skipping."
fi

LOG2B="/tmp/qwq32b_biasfirst_stage2b_l${LAYER}_train.log"
echo "===== Stage 2b (L${LAYER}): training cat vectors with frozen bias (DDP x$NPROC) ====="
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
    --frozen_bias_path "$BIAS_PATH" \
    --frozen_bias_layer "$LAYER" \
    --load_collected \
    2>&1 | tee "$LOG2B"

echo
echo "DONE Stage 2 (L${LAYER}). Cats at: $SAVE_DIR  (bias at: $STAGE1_DIR)"
