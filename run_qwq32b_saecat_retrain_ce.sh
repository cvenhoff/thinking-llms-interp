#!/bin/bash
# QwQ-32B Stage 2b retrain with full-vocab CE loss + max_norm cap.
# Tests fix for Bug 1 (top-K renorm leak): kl_mode=ce uses full-vocab
# softmax then -log p[target], same as the old pipeline's CE loss.
# Reuses cached disagreements from the original saecat Stage 2a run.

set -uo pipefail
cd /workspace/thinking-llms-interp
source .venv/bin/activate
source .env_exports.sh

LAYER=38
SAE_LAYER=27
N_CLUSTERS=10
GPUS=${GPUS:-1,2}
MAX_NORM=${MAX_NORM:-15.0}
N_EPOCHS=${N_EPOCHS:-3}
LR=${LR:-0.005}
TAG=${TAG:-saecat_ce_cap${MAX_NORM}}

SAVE1_FROM_TV="results/vars/correction_vectors_qwq32b_biasfirst_stage1"
SAVE2_SRC_FROM_TV="results/vars/correction_vectors_qwq32b_biasfirst_stage2_saecat"
SAVE2_FROM_TV="results/vars/correction_vectors_qwq32b_biasfirst_stage2_${TAG}"
SAVE1="train-vectors/$SAVE1_FROM_TV"
SAVE2_SRC="train-vectors/$SAVE2_SRC_FROM_TV"
SAVE2="train-vectors/$SAVE2_FROM_TV"

mkdir -p "$SAVE2"
if [ ! -f "$SAVE2/disagreements.pt" ]; then
    if [ -f "$SAVE2_SRC/disagreements.pt" ]; then
        echo "Reusing cached disagreements from $SAVE2_SRC"
        cp "$SAVE2_SRC/disagreements.pt" "$SAVE2/"
    else
        echo "ERROR: no cached disagreements at $SAVE2_SRC/disagreements.pt"
        exit 1
    fi
fi

BIAS_PATH_FROM_TV="$SAVE1_FROM_TV/qwen2.5-32b_bias_global.pt"
[ -f "$SAVE1/qwen2.5-32b_bias_global.pt" ] || { echo "Stage 1 bias missing"; exit 1; }

LOG="/tmp/qwq32b_${TAG}.log"
echo "===== Stage 2b retrain (kl_mode=ce, max_norm=${MAX_NORM}, lr=${LR}, n_epochs=${N_EPOCHS}, tag=${TAG}) ====="
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

cd /workspace/thinking-llms-interp/train-vectors
CUDA_VISIBLE_DEVICES="$GPUS" python -u optimize_correction_vectors.py \
    --base_model "Qwen/Qwen2.5-32B" \
    --thinking_model "Qwen/QwQ-32B" \
    --thinking_model_short "qwq-32b" \
    --steer_layer "$LAYER" \
    --save_dir "$SAVE2_FROM_TV" \
    --topk 50 --train_topk 3 --kl_mode ce \
    --max_seq_len 1536 --max_positions_per_example 64 \
    --n_epochs "$N_EPOCHS" --example_batch_size 8 \
    --lr "$LR" --weight_decay 0.0 --max_norm "$MAX_NORM" \
    --seed 42 --holdout_frac 0.1 \
    --min_disagreements 1 --min_disagreements_ratio 0.0 --min_category_share 0.0 \
    --frozen_bias_path "$BIAS_PATH_FROM_TV" --frozen_bias_layer "$LAYER" \
    --load_collected \
    2>&1 | tee "$LOG"
echo "DONE retrain ($TAG). Output: $SAVE2"
ls -la "$SAVE2"
