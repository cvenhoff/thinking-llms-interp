#!/bin/bash
# QwQ-32B applying the EXACT ORZ-7B canonical recipe (disagreement-only at
# every stage, kl_mode=topk, 5 epochs, lr=0.01, bs=16) but with max_norm
# caps to prevent QwQ's runaway bias/cat growth observed without caps
# (ORZ ended at norm 18 / 12; QwQ uncapped reached 135 / 130).
#
#   Stage 1 : 20K rollouts -> bias on disagreement-only positions
#             (kl=topk, lr=0.01, 5 ep, bs=16, max_norm=25)
#   Stage 2a: re-collect 20K rollouts UNDER stage-1 bias, disagreement-only
#   Stage 2b: cats on Stage 2a data with frozen bias
#             (kl=topk, lr=0.01, 5 ep, bs=16, max_norm=12)
set -uo pipefail
cd /workspace/thinking-llms-interp/train-vectors
source ../.venv/bin/activate
source ../.env_exports.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY=error
# NCCL stability for long DDP runs on H200s
export NCCL_TIMEOUT=3600
export TORCH_NCCL_BLOCKING_WAIT=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_IB_DISABLE=1

LAYER=38
SAE_LAYER=27
N_CLUSTERS=10
GPUS=${GPUS:-1,2}
NPROC=${NPROC:-2}
BS_PER_GPU=${BS_PER_GPU:-8}
N_EPOCHS=${N_EPOCHS:-5}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-1536}

SAVE1="results/vars/correction_vectors_qwq32b_biasfirst_orzcanon_stage1"
SAVE2="results/vars/correction_vectors_qwq32b_biasfirst_orzcanon_stage2"
mkdir -p "$SAVE1" "$SAVE2"

# ---- Stage 0: collect disagreement-only positions (no bias) ----
LOG0="/tmp/qwq32b_orzcanon_stage0.log"
if [ ! -f "$SAVE1/disagreements.pt" ]; then
    echo "===== Stage 0: collect 20K rollouts, disagreement-only ====="
    CUDA_VISIBLE_DEVICES="$GPUS" python -u optimize_correction_vectors.py \
        --base_model "Qwen/Qwen2.5-32B" \
        --thinking_model "Qwen/QwQ-32B" \
        --thinking_model_short "qwq-32b" \
        --steer_layer "$LAYER" \
        --save_dir "$SAVE1" \
        --topk 50 --train_topk 3 --kl_mode topk \
        --n_responses 20000 \
        --max_seq_len "$MAX_SEQ_LEN" --max_positions_per_example 64 \
        --seed 42 --holdout_frac 0.1 \
        --collection_mode disagreement \
        --collect_only \
        2>&1 | tee "$LOG0"
else
    echo "Stage 0 already done."
fi

# ---- Stage 1: train bias on Stage 0 (disagreement-only) data ----
LOG1="/tmp/qwq32b_orzcanon_stage1.log"
if [ ! -f "$SAVE1/qwen2.5-32b_bias_global.pt" ]; then
    echo "===== Stage 1: bias (kl=topk, max_norm=25, ${N_EPOCHS} ep) ====="
    torchrun --standalone --nproc_per_node="$NPROC" \
        optimize_correction_vectors.py \
        --base_model "Qwen/Qwen2.5-32B" \
        --thinking_model "Qwen/QwQ-32B" \
        --thinking_model_short "qwq-32b" \
        --steer_layer "$LAYER" \
        --save_dir "$SAVE1" \
        --topk 50 --train_topk 3 --kl_mode topk \
        --max_seq_len "$MAX_SEQ_LEN" --max_positions_per_example 64 \
        --n_epochs "$N_EPOCHS" --example_batch_size "$BS_PER_GPU" \
        --lr 0.01 --weight_decay 0.0 --max_norm 25.0 \
        --seed 42 --holdout_frac 0.1 \
        --min_disagreements 1 --min_disagreements_ratio 0.0 --min_category_share 0.0 \
        --collection_mode disagreement \
        --load_collected --skip_cats_phase --train_global_bias \
        2>&1 | tee "$LOG1"
else
    echo "Stage 1 already done."
fi

BIAS_PATH="$SAVE1/qwen2.5-32b_bias_global.pt"
[ -f "$BIAS_PATH" ] || { echo "Stage 1 failed"; exit 1; }

# ---- Stage 2a: re-collect under Stage 1 bias, disagreement-only ----
LOG2A="/tmp/qwq32b_orzcanon_stage2a.log"
if [ ! -f "$SAVE2/disagreements.pt" ]; then
    echo "===== Stage 2a: recollect under bias (disagreement-only) ====="
    CUDA_VISIBLE_DEVICES="$GPUS" python -u optimize_correction_vectors.py \
        --base_model "Qwen/Qwen2.5-32B" \
        --thinking_model "Qwen/QwQ-32B" \
        --thinking_model_short "qwq-32b" \
        --steer_layer "$LAYER" \
        --save_dir "$SAVE2" \
        --topk 50 --train_topk 3 --kl_mode topk \
        --n_responses 20000 \
        --max_seq_len "$MAX_SEQ_LEN" --max_positions_per_example 64 \
        --seed 42 --holdout_frac 0.1 \
        --collection_mode disagreement \
        --frozen_bias_path "$BIAS_PATH" --frozen_bias_layer "$LAYER" \
        --collect_only \
        2>&1 | tee "$LOG2A"
else
    echo "Stage 2a already done."
fi

# ---- Stage 2b: cats with frozen bias ----
LOG2B="/tmp/qwq32b_orzcanon_stage2b.log"
echo "===== Stage 2b: cats (kl=topk, max_norm=12, ${N_EPOCHS} ep) ====="
torchrun --standalone --nproc_per_node="$NPROC" \
    optimize_correction_vectors.py \
    --base_model "Qwen/Qwen2.5-32B" \
    --thinking_model "Qwen/QwQ-32B" \
    --thinking_model_short "qwq-32b" \
    --steer_layer "$LAYER" \
    --save_dir "$SAVE2" \
    --topk 50 --train_topk 3 --kl_mode topk \
    --max_seq_len "$MAX_SEQ_LEN" --max_positions_per_example 64 \
    --n_epochs "$N_EPOCHS" --example_batch_size "$BS_PER_GPU" \
    --lr 0.01 --weight_decay 0.0 --max_norm 12.0 \
    --seed 42 --holdout_frac 0.1 \
    --min_disagreements 1 --min_disagreements_ratio 0.0 --min_category_share 0.0 \
    --frozen_bias_path "$BIAS_PATH" --frozen_bias_layer "$LAYER" \
    --load_collected \
    2>&1 | tee "$LOG2B"

echo
echo "DONE QwQ-32B ORZ-canonical. Bias=$SAVE1/qwen2.5-32b_bias_global.pt  Cats=$SAVE2/"
ls -la "$SAVE1" | grep -E "bias|metrics" | tail
ls -la "$SAVE2" | grep -E "best|metrics|idx" | tail
