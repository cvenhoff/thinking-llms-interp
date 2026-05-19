#!/bin/bash
# Experiment A: Train cat vectors on ALL disagreements with frozen bias
# (removes the stage1.5 filter - the core bug)
set -e
cd /workspace/thinking-llms-interp/train-vectors
source ../.venv/bin/activate

BASE="Qwen/Qwen2.5-1.5B"
THINK="open-reasoner-zero/open-reasoner-zero-1.5b"
DATA="/workspace/thinking-llms-interp/data/math500_train_responses_orz_1.5b.jsonl"
SAE_PATH="/workspace/thinking-llms-interp/saes/orz-1.5b-layer18"
SAE_LAYER=18
SEED=42
OUTDIR="results/vars/correction_vectors_orz-1.5b_expA_allpos"
BIAS_PATH="results/vars/correction_vectors_orz-1.5b_s15_stage1/qwen2.5-1.5b_bias_global.pt"
LOG_PREFIX="/workspace/tmp/expA_cats_allpos"

mkdir -p "$OUTDIR" /workspace/tmp

echo "[ExpA] Training cat vectors on ALL disagreements with frozen bias (no stage1.5 filter)"
echo "[ExpA] Bias frozen from: $BIAS_PATH"
echo "[ExpA] Output: $OUTDIR"

CUDA_VISIBLE_DEVICES=1,2 torchrun \
    --standalone --nproc_per_node=2 \
    optimize_correction_vectors.py \
    --base_model        "$BASE" \
    --thinking_model    "$THINK" \
    --data_path         "$DATA" \
    --sae_path          "$SAE_PATH" \
    --sae_layer         $SAE_LAYER \
    --save_dir          "$OUTDIR" \
    --seed              $SEED \
    --kl_mode           topk \
    --train_topk        3 \
    --topk              50 \
    --example_batch_size 32 \
    --max_positions_per_cat 1000 \
    --per_cat_loss \
    --collect_batch_size 16 \
    --frozen_bias_path  "$BIAS_PATH" \
    --num_epochs        10 \
    --lr                1e-3 \
    --train_global_bias false \
    2>&1 | tee "${LOG_PREFIX}.log"

echo "[ExpA] DONE. Cats saved to $OUTDIR"
