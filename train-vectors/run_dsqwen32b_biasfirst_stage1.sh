#!/bin/bash
# Stage 1 bias-first for Qwen2.5-32B + DeepSeek-R1-Distill-Qwen-32B.
#
# DEVIATION FROM QwQ STAGE 1: this stage runs SINGLE-PROCESS with
# device_map="auto" (pipeline-parallel base + thinking across all 3
# H200s), not DDP.  In two end-to-end DDP attempts the run repeatedly
# hung in the very first scalar allreduce that follows the training
# loop -- a NCCL stall we could not narrow down with the 2 h
# collective timeout in place.  Pipeline-parallel skips NCCL entirely
# and is what stage 2a (collection) already uses successfully on this
# hardware.  It is ~1.5x slower per step than DDP-3 but reliably
# completes the full holdout-eval / best-snapshot selection.
set -euo pipefail
cd /workspace/thinking-llms-interp/train-vectors
source ../.venv/bin/activate
source ../.env_exports.sh
# Single-process pipeline-parallel: BS_PER_GPU is the *global* batch
# now (not per-GPU), so we bump the default a bit.
BS=${DSQ_S1_BS:-${BS:-8}}
N_EPOCHS=${DSQ_N_EPOCHS:-2}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-1536}

SAVE_DIR="results/vars/correction_vectors_dsqwen32b_biasfirst_stage1"
LOG="/tmp/dsqwen32b_biasfirst_stage1.log"

if [ ! -f "$SAVE_DIR/disagreements.pt" ]; then
    echo "ERROR: Stage 0 disagreements.pt missing in $SAVE_DIR"; exit 1
fi

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
python -u optimize_correction_vectors.py \
    --base_model "Qwen/Qwen2.5-32B" \
    --thinking_model "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" \
    --thinking_model_short "deepseek-r1-distill-qwen-32b" \
    --steer_layer 38 \
    --save_dir "$SAVE_DIR" \
    --topk 50 --train_topk 3 --kl_mode topk \
    --max_seq_len "$MAX_SEQ_LEN" --max_positions_per_example 64 \
    --n_epochs "$N_EPOCHS" --example_batch_size "$BS" \
    --lr 0.01 --weight_decay 0.0 --max_norm 0.0 \
    --seed 42 --holdout_frac 0.1 \
    --min_disagreements 1 --min_disagreements_ratio 0.0 --min_category_share 0.0 \
    --load_collected --skip_cats_phase --train_global_bias \
    2>&1 | tee "$LOG"

echo "DONE Stage 1. Bias at: $SAVE_DIR"
ls -la "$SAVE_DIR"
