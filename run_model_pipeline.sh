#!/bin/bash
# Parameterized ORZ-recipe pipeline for one thinking model.
# Runs: stage0 (collect) → stage1 (bias) → stage2a (recollect) → stage2b (cats) → eval
#
# Usage:
#   THINKING_MODEL=... THINKING_SHORT=... bash run_model_pipeline.sh
#
# Required env vars:
#   THINKING_MODEL       e.g. "Open-Reasoner-Zero/Open-Reasoner-Zero-32B"
#   THINKING_SHORT       e.g. "orz-32b"   (used in save-dir names and result files)
#
# Optional env vars (all have defaults):
#   STAGE0_DISAGREE_PT   path to a pre-collected disagreements.pt to skip stage0
#   SKIP_STAGE0          set to 1 to skip stage0 (requires STAGE0_DISAGREE_PT)
#   EVAL_ONLY            set to 1 to skip training and go straight to eval
#   NPROC                DDP processes (default: 3)
#   BS_PER_GPU           example batch size per GPU (default: 8)

set -uo pipefail

: "${THINKING_MODEL:?Need THINKING_MODEL env var}"
: "${THINKING_SHORT:?Need THINKING_SHORT env var}"

cd /workspace/thinking-llms-interp
source .venv/bin/activate
source .env_exports.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY=error
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=86400
export TORCH_NCCL_ASYNC_ERROR_HANDLING=0
export NCCL_IB_DISABLE=1

BASE_MODEL="Qwen/Qwen2.5-32B"
STEER_LAYER=38
SAE_LAYER=27
N_CLUSTERS=10
N_RESPONSES=20000
MAX_SEQ_LEN=${MAX_SEQ_LEN:-1024}
N_EPOCHS=5
LR=0.01
BIAS_MAX_NORM=25.0
CAT_MAX_NORM=12.0
NPROC=${NPROC:-3}
BS_PER_GPU=${BS_PER_GPU:-5}
SKIP_STAGE0=${SKIP_STAGE0:-0}
EVAL_ONLY=${EVAL_ONLY:-0}

TAG="${THINKING_SHORT}"
SAVE1="train-vectors/results/vars/correction_vectors_${TAG}_canon_stage1"
SAVE2="train-vectors/results/vars/correction_vectors_${TAG}_canon_stage2"
mkdir -p "$SAVE1" "$SAVE2"

echo "========================================================"
echo "  MODEL PIPELINE: ${THINKING_SHORT}"
echo "  thinking: $THINKING_MODEL"
echo "  base:     $BASE_MODEL"
echo "  stage1:   $SAVE1"
echo "  stage2:   $SAVE2"
echo "========================================================"

if [ "${EVAL_ONLY}" = "0" ]; then

  # ---- Stage 0: collect disagreements ----
  if [ "${SKIP_STAGE0}" = "1" ] && [ -n "${STAGE0_DISAGREE_PT:-}" ]; then
    echo "Stage 0: symlinking pre-collected data from $STAGE0_DISAGREE_PT"
    ln -sf "$(realpath "$STAGE0_DISAGREE_PT")" "$SAVE1/disagreements.pt"
  elif [ -f "$SAVE1/disagreements.pt" ]; then
    echo "Stage 0: already done."
  else
    echo "===== Stage 0: collect ${N_RESPONSES} rollouts, disagreement-only ====="
    cd /workspace/thinking-llms-interp/train-vectors
    CUDA_VISIBLE_DEVICES=0,1,2 python -u optimize_correction_vectors.py \
        --base_model "$BASE_MODEL" \
        --thinking_model "$THINKING_MODEL" \
        --thinking_model_short "$THINKING_SHORT" \
        --steer_layer "$STEER_LAYER" \
        --save_dir "results/vars/correction_vectors_${TAG}_canon_stage1" \
        --topk 50 --train_topk 3 --kl_mode ce \
        --n_responses "$N_RESPONSES" \
        --max_seq_len "$MAX_SEQ_LEN" --max_positions_per_example 64 \
        --seed 42 --holdout_frac 0.1 \
        --collection_mode disagreement \
        --collect_only \
        2>&1 | tee "/workspace/tmp/${TAG}_stage0.log"
    cd /workspace/thinking-llms-interp
  fi
  [ -f "$SAVE1/disagreements.pt" ] || { echo "Stage 0 FAILED"; exit 1; }

  # ---- Stage 1: train global bias ----
  if [ -f "$SAVE1/qwen2.5-32b_bias_global.pt" ]; then
    echo "Stage 1: already done."
  else
    echo "===== Stage 1: bias (CE, max_norm=${BIAS_MAX_NORM}, ${N_EPOCHS} ep) ====="
    cd /workspace/thinking-llms-interp/train-vectors
    torchrun --standalone --nproc_per_node="$NPROC" \
        optimize_correction_vectors.py \
        --base_model "$BASE_MODEL" \
        --thinking_model "$THINKING_MODEL" \
        --thinking_model_short "$THINKING_SHORT" \
        --steer_layer "$STEER_LAYER" \
        --save_dir "results/vars/correction_vectors_${TAG}_canon_stage1" \
        --topk 50 --train_topk 3 --kl_mode ce \
        --max_seq_len "$MAX_SEQ_LEN" --max_positions_per_example 64 \
        --n_epochs "$N_EPOCHS" --example_batch_size "$BS_PER_GPU" \
        --lr "$LR" --weight_decay 0.0 --max_norm "$BIAS_MAX_NORM" \
        --seed 42 --holdout_frac 0.1 \
        --min_disagreements 1 --min_disagreements_ratio 0.0 --min_category_share 0.0 \
        --collection_mode disagreement \
        --load_collected --skip_cats_phase --train_global_bias \
        2>&1 | tee "/workspace/tmp/${TAG}_stage1.log"
    cd /workspace/thinking-llms-interp
  fi
  BIAS_PATH="$SAVE1/qwen2.5-32b_bias_global.pt"
  [ -f "$BIAS_PATH" ] || { echo "Stage 1 FAILED"; exit 1; }
  echo "  bias norm: $(python3 -c "import torch; v=torch.load('$BIAS_PATH',map_location='cpu',weights_only=True); print(f'{v.norm().item():.2f}')" 2>/dev/null)"

  # ---- Stage 2a: re-collect under bias ----
  if [ -f "$SAVE2/disagreements.pt" ]; then
    echo "Stage 2a: already done."
  else
    echo "===== Stage 2a: re-collect ${N_RESPONSES} under bias ====="
    cd /workspace/thinking-llms-interp/train-vectors
    CUDA_VISIBLE_DEVICES=0,1,2 python -u optimize_correction_vectors.py \
        --base_model "$BASE_MODEL" \
        --thinking_model "$THINKING_MODEL" \
        --thinking_model_short "$THINKING_SHORT" \
        --steer_layer "$STEER_LAYER" \
        --save_dir "results/vars/correction_vectors_${TAG}_canon_stage2" \
        --topk 50 --train_topk 3 --kl_mode ce \
        --n_responses "$N_RESPONSES" \
        --max_seq_len "$MAX_SEQ_LEN" --max_positions_per_example 64 \
        --seed 43 --holdout_frac 0.1 \
        --collection_mode disagreement \
        --frozen_bias_path "results/vars/correction_vectors_${TAG}_canon_stage1/qwen2.5-32b_bias_global.pt" \
        --frozen_bias_layer "$STEER_LAYER" \
        --collect_only \
        2>&1 | tee "/workspace/tmp/${TAG}_stage2a.log"
    cd /workspace/thinking-llms-interp
  fi
  [ -f "$SAVE2/disagreements.pt" ] || { echo "Stage 2a FAILED"; exit 1; }

  # ---- Stage 2b: train category vectors ----
  if [ -f "$SAVE2/qwen2.5-32b_idx0_linear.pt" ]; then
    echo "Stage 2b: already done."
  else
    echo "===== Stage 2b: cats (CE, max_norm=${CAT_MAX_NORM}, ${N_EPOCHS} ep) ====="
    cd /workspace/thinking-llms-interp/train-vectors
    torchrun --standalone --nproc_per_node="$NPROC" \
        optimize_correction_vectors.py \
        --base_model "$BASE_MODEL" \
        --thinking_model "$THINKING_MODEL" \
        --thinking_model_short "$THINKING_SHORT" \
        --steer_layer "$STEER_LAYER" \
        --save_dir "results/vars/correction_vectors_${TAG}_canon_stage2" \
        --topk 50 --train_topk 3 --kl_mode ce \
        --max_seq_len "$MAX_SEQ_LEN" --max_positions_per_example 64 \
        --n_epochs "$N_EPOCHS" --example_batch_size "$BS_PER_GPU" \
        --lr "$LR" --weight_decay 0.0 --max_norm "$CAT_MAX_NORM" \
        --seed 42 --holdout_frac 0.1 \
        --min_disagreements 1 --min_disagreements_ratio 0.0 --min_category_share 0.0 \
        --collection_mode disagreement \
        --frozen_bias_path "results/vars/correction_vectors_${TAG}_canon_stage1/qwen2.5-32b_bias_global.pt" \
        --frozen_bias_layer "$STEER_LAYER" \
        --load_collected \
        2>&1 | tee "/workspace/tmp/${TAG}_stage2b.log"
    cd /workspace/thinking-llms-interp
  fi
  [ -f "$SAVE2/qwen2.5-32b_idx0_linear.pt" ] || { echo "Stage 2b FAILED"; exit 1; }

fi  # EVAL_ONLY

# ---- Eval ----
echo
echo "===== Eval (pg, n=128, math500, 3 judge reps) ====="
cd /workspace/thinking-llms-interp/hybrid

BIAS_PATH_ABS="/workspace/thinking-llms-interp/$SAVE1/qwen2.5-32b_bias_global.pt"
CATS_DIR_ABS="/workspace/thinking-llms-interp/$SAVE2"

run_eval () {
    local EVAL_TAG="$1"
    local EXTRA="$2"
    rm -f "results/rolling/rolling_qwen2.5-32b_math500_${EVAL_TAG}.jsonl"
    rm -f "results/summary_qwen2.5-32b_math500_${EVAL_TAG}.json"
    rm -f "results/judge_reps_qwen2.5-32b_math500_${EVAL_TAG}.json"
    echo "  --- ${EVAL_TAG} ---"
    uv run python hybrid_eval.py \
        --dataset math500 --n_tasks 128 \
        --max_new_tokens 2000 --max_thinking_tokens 2000 \
        --batch_gen_size 4 --hybrid_gen_batch_size 4 \
        --base_model "$BASE_MODEL" \
        --thinking_model "$THINKING_MODEL" \
        --sae_layer "$SAE_LAYER" --n_clusters "$N_CLUSTERS" \
        --disable_sae_mean \
        --dom_vectors_dir ../train-vectors/results/diff_of_means \
        --dom_vectors_model_short qwen2.5-32b \
        --old_vectors_dir "$CATS_DIR_ABS" \
        --old_vectors_layer "$STEER_LAYER" \
        --bias_vector_path "$BIAS_PATH_ABS" \
        --bias_layer "$STEER_LAYER" \
        --coef_sweep "0.5,1.0,1.5,2.0,2.5,3.0" \
        --coef_select pg \
        --judge_repetitions 3 \
        --results_suffix "${EVAL_TAG}" \
        $EXTRA 2>&1 | tee "/workspace/tmp/${EVAL_TAG}.log"
}

run_eval "${TAG}-pg-learn-128"     ""
run_eval "${TAG}-pg-rand-128"      "--randomize_vectors --random_seed 42"
run_eval "${TAG}-pg-biasonly-128"  "--bias_only"

echo
echo "========================================================"
echo "  DONE: ${THINKING_SHORT}"
echo "========================================================"
echo
cd /workspace/thinking-llms-interp/hybrid
for COND in learn rand biasonly; do
    F="results/rolling/rolling_qwen2.5-32b_math500_${TAG}-pg-${COND}-128.jsonl"
    [ -f "$F" ] || continue
    python3 -c "
import json, sys
rows = [json.loads(l) for l in open('$F')]
n = len(rows)
aT = sum(1 for r in rows if r['judges']['thinking']['correct'])/n
aB = sum(1 for r in rows if r['judges']['base']['correct'])/n
aH = sum(1 for r in rows if r['judges']['hybrid']['correct'])/n
gap = (aH-aB)/max(1e-9,aT-aB) if aT > aB else float('nan')
print(f'  ${COND:<10} T={aT:.3f}  B={aB:.3f}  H={aH:.3f}  gap={gap:+.3f}  (n={n})')
"
done
