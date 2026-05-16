#!/bin/bash
# Re-collect Stage 0 with entropy_threshold=0.0 -> ALL positions (no disagreement filter).
# This matches the OLD-style training that trained on every token in annotated spans.
set -uo pipefail
cd /workspace/thinking-llms-interp/train-vectors
source ../.venv/bin/activate
source ../.env_exports.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY=error

SAVE_DIR="results/vars/correction_vectors_qwq32b_biasfirst_allpos_stage0"
LOG="/tmp/qwq32b_allpos_stage0.log"
mkdir -p "$SAVE_DIR"

if [ -f "$SAVE_DIR/disagreements.pt" ]; then
    echo "Already collected. Skipping."
    exit 0
fi

CUDA_VISIBLE_DEVICES=${GPUS:-1,2} python -u optimize_correction_vectors.py \
    --base_model "Qwen/Qwen2.5-32B" \
    --thinking_model "Qwen/QwQ-32B" \
    --thinking_model_short "qwq-32b" \
    --steer_layer 38 \
    --save_dir "$SAVE_DIR" \
    --topk 50 --train_topk 3 --kl_mode ce \
    --n_responses 20000 \
    --max_seq_len 1536 --max_positions_per_example 64 \
    --seed 42 --holdout_frac 0.1 \
    --collection_mode entropy --entropy_threshold 0.0 \
    --collect_only \
    2>&1 | tee "$LOG"

echo "DONE all-positions collection."
ls -la "$SAVE_DIR"
