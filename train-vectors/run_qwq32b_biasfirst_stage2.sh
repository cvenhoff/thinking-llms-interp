#!/bin/bash
# Stage 2 bias-first for Qwen2.5-32B + QwQ-32B - DDP across 3 H200s.
#   2a. Re-collect disagreements with Stage-1 bias hooked into base.
#       (single-process, pipeline-parallel inference - already fast)
#   2b. Train cats as residuals on top of frozen bias (DDP).
set -euo pipefail
cd /workspace/thinking-llms-interp/train-vectors
source ../.venv/bin/activate
source ../.env_exports.sh
BS_PER_GPU=${BS_PER_GPU:-${BS:-8}}
N_EPOCHS=${N_EPOCHS:-2}
NPROC=${NPROC:-3}
N_RECOLLECT=${N_RECOLLECT:-5000}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-1536}

STAGE1_DIR="results/vars/correction_vectors_qwq32b_biasfirst_stage1"
BIAS_PATH="$STAGE1_DIR/qwen2.5-32b_bias_global.pt"
if [ ! -f "$BIAS_PATH" ]; then
    echo "ERROR: bias not found at $BIAS_PATH"; exit 1
fi

SAVE_DIR="results/vars/correction_vectors_qwq32b_biasfirst_stage2"
mkdir -p "$SAVE_DIR"

LOG2A="/tmp/qwq32b_biasfirst_stage2a_collect.log"
if [ ! -f "$SAVE_DIR/disagreements.pt" ]; then
    echo "===== Stage 2a: collecting disagreements under bias steering ====="
    # Collection is inference-only (no gradients).  Pipeline-parallel
    # via device_map=auto fits both base+thinking on 3 GPUs and runs at
    # ~3.5 it/s; cutting n_responses from 20000 -> N_RECOLLECT keeps
    # this stage well under 30 min.
    export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
    python -u optimize_correction_vectors.py \
        --base_model "Qwen/Qwen2.5-32B" \
        --thinking_model "Qwen/QwQ-32B" \
        --thinking_model_short "qwq-32b" \
        --steer_layer 38 \
        --save_dir "$SAVE_DIR" \
        --topk 50 --train_topk 3 --kl_mode topk \
        --n_responses "$N_RECOLLECT" \
        --max_seq_len "$MAX_SEQ_LEN" --max_positions_per_example 64 \
        --seed 42 --holdout_frac 0.1 \
        --frozen_bias_path "$BIAS_PATH" \
        --frozen_bias_layer 38 \
        --collect_only \
        2>&1 | tee "$LOG2A"
else
    echo "Stage 2a already done (disagreements.pt exists), skipping."
fi

LOG2B="/tmp/qwq32b_biasfirst_stage2b_train.log"
echo "===== Stage 2b: training cat vectors with frozen bias (DDP x$NPROC) ====="
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
    --frozen_bias_path "$BIAS_PATH" \
    --frozen_bias_layer 38 \
    --load_collected \
    2>&1 | tee "$LOG2B"

echo
echo "DONE Stage 2. Cats at: $SAVE_DIR  (bias at: $STAGE1_DIR)"
