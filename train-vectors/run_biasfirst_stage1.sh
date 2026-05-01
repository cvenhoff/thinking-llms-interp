#!/bin/bash
# Stage 1 of bias-first pipeline: train ONLY a global bias vector on
# the original (no-bias) collected disagreements, using the SAME
# hyperparameters as v3.  Output: <save_dir>/<model>_bias_global.pt
# + <save_dir>/bias_layer.json which Stage 2 will load with
# --frozen_bias_path.
set -euo pipefail
cd /workspace/thinking-llms-interp/train-vectors
source ../.venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY=error

SAVE_DIR="results/vars/correction_vectors_orz7b_full_e5_biasfirst_stage1"
mkdir -p "$SAVE_DIR"
LOG="/tmp/biasfirst_stage1.log"

# Re-use the v3 disagreements dump (same collection, no bias on collect).
SRC_DUMP=results/vars/correction_vectors_orz7b_full_e5_v3/disagreements.pt
if [ -f "$SRC_DUMP" ] && [ ! -f "$SAVE_DIR/disagreements.pt" ]; then
    cp "$SRC_DUMP" "$SAVE_DIR/disagreements.pt"
fi

python -u optimize_correction_vectors.py \
    --base_model "Qwen/Qwen2.5-7B" \
    --thinking_model "Open-Reasoner-Zero/Open-Reasoner-Zero-7B" \
    --thinking_model_short "open-reasoner-zero-7b" \
    --steer_layer 16 \
    --save_dir "$SAVE_DIR" \
    --topk 50 \
    --train_topk 3 \
    --kl_mode topk \
    --max_seq_len 2048 \
    --max_positions_per_example 64 \
    --n_epochs 5 \
    --example_batch_size 16 \
    --lr 0.01 \
    --weight_decay 0.0 \
    --max_norm 0.0 \
    --seed 42 \
    --holdout_frac 0.1 \
    --min_disagreements 1 \
    --min_disagreements_ratio 0.0 \
    --min_category_share 0.0 \
    --load_collected \
    --skip_cats_phase \
    --train_global_bias \
    2>&1 | tee "$LOG"

echo
echo "DONE Stage 1. Bias at: $SAVE_DIR"
ls -la "$SAVE_DIR"
