#!/bin/bash
# Quick sanity check: bias-first pipeline end-to-end on ORZ-7B / Qwen2.5-7B
# (the original recipe-validated pair).  Reduced data so we can finish in
# ~1.5 h.  If this gives >50% gap recovered we know the pipeline still
# works and the QwQ-32B regression is model-specific, not a bug.
#
# Reductions vs the canonical recipe:
#   --n_responses 3000  (was 20000)  for stage 2a
#   --n_epochs 3        (was 5)
#   eval n_tasks 32     (was 128)
# Other hyperparameters kept identical to canonical.
set -uo pipefail
cd /workspace/thinking-llms-interp
source .venv/bin/activate
source .env_exports.sh

LAYER=16
SAVE1="train-vectors/results/vars/correction_vectors_orz7b_biasfirst_stage1_sanity"
SAVE2="train-vectors/results/vars/correction_vectors_orz7b_biasfirst_stage2_sanity"
mkdir -p "$SAVE1" "$SAVE2"

LOG1="/tmp/orz7b_sanity_stage1.log"
LOG2A="/tmp/orz7b_sanity_stage2a.log"
LOG2B="/tmp/orz7b_sanity_stage2b.log"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

# ---- Stage 1: collect disagreements (no bias) + train global bias ----
# Single GPU is plenty for 7B; we keep collection on GPU 0.
LOG_LATEST="$LOG1"
SAVE1_FROM_TV="results/vars/correction_vectors_orz7b_biasfirst_stage1_sanity"
SAVE2_FROM_TV="results/vars/correction_vectors_orz7b_biasfirst_stage2_sanity"

if [ ! -f "$SAVE1/qwen2.5-7b_bias_global.pt" ]; then
    echo "===== Stage 1: collect + train global bias (L${LAYER}) ====="
    cd /workspace/thinking-llms-interp/train-vectors
    CUDA_VISIBLE_DEVICES=0,1,2 python -u optimize_correction_vectors.py \
        --base_model "Qwen/Qwen2.5-7B" \
        --thinking_model "Open-Reasoner-Zero/Open-Reasoner-Zero-7B" \
        --thinking_model_short "open-reasoner-zero-7b" \
        --steer_layer "$LAYER" \
        --save_dir "$SAVE1_FROM_TV" \
        --topk 50 --train_topk 3 --kl_mode topk \
        --n_responses 3000 \
        --max_seq_len 2048 --max_positions_per_example 64 \
        --n_epochs 3 --example_batch_size 16 \
        --lr 0.01 --weight_decay 0.0 --max_norm 0.0 \
        --seed 42 --holdout_frac 0.1 \
        --min_disagreements 1 --min_disagreements_ratio 0.0 --min_category_share 0.0 \
        --skip_cats_phase --train_global_bias \
        2>&1 | tee "$LOG1"
    cd /workspace/thinking-llms-interp
else
    echo "Stage 1 already done."
fi

BIAS_PATH="$SAVE1/qwen2.5-7b_bias_global.pt"
BIAS_PATH_FROM_TV="$SAVE1_FROM_TV/qwen2.5-7b_bias_global.pt"
[ -f "$BIAS_PATH" ] || { echo "Stage 1 failed - no bias produced"; exit 1; }

# ---- Stage 2a: re-collect under bias ----
if [ ! -f "$SAVE2/disagreements.pt" ]; then
    echo "===== Stage 2a: re-collect under bias (L${LAYER}) ====="
    cd /workspace/thinking-llms-interp/train-vectors
    CUDA_VISIBLE_DEVICES=0,1,2 python -u optimize_correction_vectors.py \
        --base_model "Qwen/Qwen2.5-7B" \
        --thinking_model "Open-Reasoner-Zero/Open-Reasoner-Zero-7B" \
        --thinking_model_short "open-reasoner-zero-7b" \
        --steer_layer "$LAYER" \
        --save_dir "$SAVE2_FROM_TV" \
        --topk 50 --train_topk 3 --kl_mode topk \
        --n_responses 3000 \
        --max_seq_len 2048 --max_positions_per_example 64 \
        --seed 42 --holdout_frac 0.1 \
        --frozen_bias_path "$BIAS_PATH_FROM_TV" --frozen_bias_layer "$LAYER" \
        --collect_only \
        2>&1 | tee "$LOG2A"
    cd /workspace/thinking-llms-interp
else
    echo "Stage 2a already done."
fi

# ---- Stage 2b: train cats with frozen bias ----
echo "===== Stage 2b: train cats with frozen bias (L${LAYER}) ====="
cd /workspace/thinking-llms-interp/train-vectors
CUDA_VISIBLE_DEVICES=0,1,2 python -u optimize_correction_vectors.py \
    --base_model "Qwen/Qwen2.5-7B" \
    --thinking_model "Open-Reasoner-Zero/Open-Reasoner-Zero-7B" \
    --thinking_model_short "open-reasoner-zero-7b" \
    --steer_layer "$LAYER" \
    --save_dir "$SAVE2_FROM_TV" \
    --topk 50 --train_topk 3 --kl_mode topk \
    --max_seq_len 2048 --max_positions_per_example 64 \
    --n_epochs 3 --example_batch_size 16 \
    --lr 0.01 --weight_decay 0.0 --max_norm 0.0 \
    --seed 42 --holdout_frac 0.1 \
    --min_disagreements 1 --min_disagreements_ratio 0.0 --min_category_share 0.0 \
    --frozen_bias_path "$BIAS_PATH_FROM_TV" --frozen_bias_layer "$LAYER" \
    --load_collected \
    2>&1 | tee "$LOG2B"
cd /workspace/thinking-llms-interp

echo "DONE training"
ls -la "$SAVE1" "$SAVE2"

# ---- Eval: 3 conditions in parallel on 3 GPUs ----
echo
echo "===== Eval (math500, 32 tasks, judge x3) ====="
N_TASKS=${N_TASKS:-32}

run_one_gpu () {
    local GPU="$1"
    local TAG="$2"
    local EXTRA="$3"
    cd /workspace/thinking-llms-interp/hybrid
    rm -f "results/rolling/rolling_qwen2.5-7b_math500_${TAG}.jsonl"
    rm -f "results/summary_qwen2.5-7b_math500_${TAG}.json"
    rm -f "results/judge_reps_qwen2.5-7b_math500_${TAG}.json"
    echo "===== ${TAG} on GPU ${GPU} ====="
    CUDA_VISIBLE_DEVICES="$GPU" python hybrid_eval.py \
        --dataset math500 --n_tasks "$N_TASKS" \
        --max_new_tokens 2000 --max_thinking_tokens 2000 \
        --batch_gen_size 16 --hybrid_gen_batch_size 16 \
        --base_model Qwen/Qwen2.5-7B \
        --thinking_model Open-Reasoner-Zero/Open-Reasoner-Zero-7B \
        --sae_layer 20 --n_clusters 10 \
        --dom_vectors_dir ../train-vectors/results/diff_of_means \
        --dom_vectors_model_short qwen2.5-7b \
        --old_vectors_dir "../$SAVE2" \
        --old_vectors_layer "$LAYER" \
        --bias_vector_path "../$BIAS_PATH" \
        --bias_layer "$LAYER" \
        --coef_sweep "0.5,1.0,1.5" \
        --judge_repetitions 3 \
        --results_suffix "${TAG}" \
        $EXTRA 2>&1 | tee "/tmp/${TAG}.log"
}

cd /workspace/thinking-llms-interp
run_one_gpu 0 "orz-bf-learn-sanity-${N_TASKS}"     ""                                    &
PID0=$!
run_one_gpu 1 "orz-bf-rand-sanity-${N_TASKS}"      "--randomize_vectors --random_seed 42" &
PID1=$!
run_one_gpu 2 "orz-bf-biasonly-sanity-${N_TASKS}"  "--bias_only"                          &
PID2=$!

wait $PID0; CODE0=$?
wait $PID1; CODE1=$?
wait $PID2; CODE2=$?
echo "Exit codes: learn=$CODE0  rand=$CODE1  biasonly=$CODE2"
