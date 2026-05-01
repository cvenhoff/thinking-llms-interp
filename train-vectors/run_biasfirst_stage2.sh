#!/bin/bash
# Stage 2 of bias-first pipeline (split into 2a/2b for VRAM hygiene):
#   2a. Re-collect disagreements with the Stage-1 bias hooked into the
#       BASE model at the steering layer (all positions).  Resulting
#       positions are where BASE+BIAS still disagrees with thinking.
#       Writes <save_dir>/disagreements.pt then exits so the OS frees
#       the thinking model's VRAM.
#   2b. Train per-category vectors V[c] with the hook applying
#       ``bias_frozen + V[cat[p]]`` at each disagreement position.
#       The optimizer learns the category-specific RESIDUAL on top of
#       the static bias.  Saves cats to <save_dir>; bias remains in
#       the Stage-1 dir.
set -euo pipefail
cd /workspace/thinking-llms-interp/train-vectors
source ../.venv/bin/activate
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY=error

STAGE1_DIR="results/vars/correction_vectors_orz7b_full_e5_biasfirst_stage1"
BIAS_PATH="$STAGE1_DIR/qwen2.5-7b_bias_global.pt"
if [ ! -f "$BIAS_PATH" ]; then
    echo "ERROR: bias not found at $BIAS_PATH"; exit 1
fi

SAVE_DIR="results/vars/correction_vectors_orz7b_full_e5_biasfirst_stage2"
mkdir -p "$SAVE_DIR"

# ---- 2a: collect disagreements under bias steering -------------------
LOG2A="/tmp/biasfirst_stage2a_collect.log"
if [ ! -f "$SAVE_DIR/disagreements.pt" ]; then
    echo "===== Stage 2a: collecting disagreements under bias steering ====="
    python -u optimize_correction_vectors.py \
        --base_model "Qwen/Qwen2.5-7B" \
        --thinking_model "Open-Reasoner-Zero/Open-Reasoner-Zero-7B" \
        --thinking_model_short "open-reasoner-zero-7b" \
        --steer_layer 16 \
        --save_dir "$SAVE_DIR" \
        --topk 50 \
        --train_topk 3 \
        --kl_mode topk \
        --n_responses 20000 \
        --max_seq_len 2048 \
        --max_positions_per_example 64 \
        --seed 42 \
        --holdout_frac 0.1 \
        --frozen_bias_path "$BIAS_PATH" \
        --frozen_bias_layer 16 \
        --collect_only \
        2>&1 | tee "$LOG2A"
else
    echo "Stage 2a already done (disagreements.pt exists), skipping."
fi

# ---- 2b: train cat vectors with frozen bias --------------------------
LOG2B="/tmp/biasfirst_stage2b_train.log"
echo "===== Stage 2b: training cat vectors with frozen bias ====="
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
    --frozen_bias_path "$BIAS_PATH" \
    --frozen_bias_layer 16 \
    --load_collected \
    2>&1 | tee "$LOG2B"

echo
echo "DONE Stage 2. Cats at: $SAVE_DIR  (bias at: $STAGE1_DIR)"
