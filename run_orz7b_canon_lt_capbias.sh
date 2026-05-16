#!/bin/bash
# ORZ-7B / Qwen2.5-7B with last-token annotations AND a capped Stage-1
# bias (max_norm=10 instead of unbounded ~18) so cat residuals have room
# to add lift over biasonly.  Uses GPUs 1+2 only (GPU 0 is busy with
# QwQ-32B annotation).
set -uo pipefail
cd /workspace/thinking-llms-interp
source .venv/bin/activate
source .env_exports.sh

export CUDA_VISIBLE_DEVICES=1,2
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

LAYER=16
SAVE1_FROM_TV="results/vars/correction_vectors_orz7b_biasfirst_stage1_canon_lt_cap10"
SAVE2_FROM_TV="results/vars/correction_vectors_orz7b_biasfirst_stage2_canon_lt_cap10"
SAVE1="train-vectors/$SAVE1_FROM_TV"
SAVE2="train-vectors/$SAVE2_FROM_TV"
mkdir -p "$SAVE1" "$SAVE2"

LOG1="/tmp/orz7b_lt_cap10_stage1.log"
LOG2A="/tmp/orz7b_lt_cap10_stage2a.log"
LOG2B="/tmp/orz7b_lt_cap10_stage2b.log"

# ---- Stage 1: smaller bias cap, fewer epochs ----
if [ ! -f "$SAVE1/qwen2.5-7b_bias_global.pt" ]; then
    echo "===== Stage 1 (LT cap10): bias with max_norm=10 ====="
    cd /workspace/thinking-llms-interp/train-vectors
    python -u optimize_correction_vectors.py \
        --base_model "Qwen/Qwen2.5-7B" \
        --thinking_model "Open-Reasoner-Zero/Open-Reasoner-Zero-7B" \
        --thinking_model_short "open-reasoner-zero-7b" \
        --steer_layer "$LAYER" \
        --save_dir "$SAVE1_FROM_TV" \
        --topk 50 --train_topk 3 --kl_mode topk \
        --n_responses 20000 \
        --max_seq_len 2048 --max_positions_per_example 64 \
        --n_epochs 3 --example_batch_size 16 \
        --lr 0.01 --weight_decay 0.0 --max_norm 10.0 \
        --seed 42 --holdout_frac 0.1 \
        --min_disagreements 1 --min_disagreements_ratio 0.0 --min_category_share 0.0 \
        --skip_cats_phase --train_global_bias \
        2>&1 | tee "$LOG1"
    cd /workspace/thinking-llms-interp
fi

BIAS_PATH_FROM_TV="$SAVE1_FROM_TV/qwen2.5-7b_bias_global.pt"
[ -f "$SAVE1/qwen2.5-7b_bias_global.pt" ] || { echo "Stage 1 failed"; exit 1; }

# ---- Stage 2a ----
if [ ! -f "$SAVE2/disagreements.pt" ]; then
    echo "===== Stage 2a (LT cap10): re-collect under capped bias ====="
    cd /workspace/thinking-llms-interp/train-vectors
    python -u optimize_correction_vectors.py \
        --base_model "Qwen/Qwen2.5-7B" \
        --thinking_model "Open-Reasoner-Zero/Open-Reasoner-Zero-7B" \
        --thinking_model_short "open-reasoner-zero-7b" \
        --steer_layer "$LAYER" \
        --save_dir "$SAVE2_FROM_TV" \
        --topk 50 --train_topk 3 --kl_mode topk \
        --n_responses 20000 \
        --max_seq_len 2048 --max_positions_per_example 64 \
        --seed 42 --holdout_frac 0.1 \
        --frozen_bias_path "$BIAS_PATH_FROM_TV" --frozen_bias_layer "$LAYER" \
        --collect_only \
        2>&1 | tee "$LOG2A"
    cd /workspace/thinking-llms-interp
fi

# ---- Stage 2b ----
echo "===== Stage 2b (LT cap10): train cats ====="
cd /workspace/thinking-llms-interp/train-vectors
python -u optimize_correction_vectors.py \
    --base_model "Qwen/Qwen2.5-7B" \
    --thinking_model "Open-Reasoner-Zero/Open-Reasoner-Zero-7B" \
    --thinking_model_short "open-reasoner-zero-7b" \
    --steer_layer "$LAYER" \
    --save_dir "$SAVE2_FROM_TV" \
    --topk 50 --train_topk 3 --kl_mode topk \
    --max_seq_len 2048 --max_positions_per_example 64 \
    --n_epochs 5 --example_batch_size 16 \
    --lr 0.01 --weight_decay 0.0 --max_norm 0.0 \
    --seed 42 --holdout_frac 0.1 \
    --min_disagreements 1 --min_disagreements_ratio 0.0 --min_category_share 0.0 \
    --frozen_bias_path "$BIAS_PATH_FROM_TV" --frozen_bias_layer "$LAYER" \
    --load_collected \
    2>&1 | tee "$LOG2B"
cd /workspace/thinking-llms-interp

echo "DONE training (LT cap10)"
ls -la "$SAVE1" "$SAVE2"
