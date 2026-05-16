#!/bin/bash
# ORZ-7B / Qwen2.5-7B canonical bias-first recipe with PER-POSITION
# SAE-based category assignment during disagreement collection.
#
# This eliminates the train/eval category mismatch that caused cats to
# train on a wildly different distribution than they fire on at eval:
# at training time we now classify each disagreement position by running
# the THINKING-MODEL activation at sae_classify_layer through the SAE,
# exactly mirroring hybrid_eval.py's last-token classification.
#
# Vectors are saved to a NEW directory so we can compare 1:1 against the
# previous (sentence-mean / sentence-last-token) annotation runs.

set -uo pipefail
cd /workspace/thinking-llms-interp
source .venv/bin/activate
source .env_exports.sh

LAYER=16          # steering layer (ORZ canonical)
SAE_LAYER=20      # SAE classification layer (matches hybrid_eval.py)
N_CLUSTERS=10

SAVE1_FROM_TV="results/vars/correction_vectors_orz7b_biasfirst_stage1_saecat"
SAVE2_FROM_TV="results/vars/correction_vectors_orz7b_biasfirst_stage2_saecat"
SAVE1="train-vectors/$SAVE1_FROM_TV"
SAVE2="train-vectors/$SAVE2_FROM_TV"
mkdir -p "$SAVE1" "$SAVE2"

LOG1="/tmp/orz7b_saecat_stage1.log"
LOG2A="/tmp/orz7b_saecat_stage2a.log"
LOG2B="/tmp/orz7b_saecat_stage2b.log"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

# ---- Stage 1 (bias only -- categories irrelevant; SAE not used) ----
if [ ! -f "$SAVE1/qwen2.5-7b_bias_global.pt" ]; then
    echo "===== Stage 1: 20K + global bias (L${LAYER}) ====="
    cd /workspace/thinking-llms-interp/train-vectors
    CUDA_VISIBLE_DEVICES=0,1,2 python -u optimize_correction_vectors.py \
        --base_model "Qwen/Qwen2.5-7B" \
        --thinking_model "Open-Reasoner-Zero/Open-Reasoner-Zero-7B" \
        --thinking_model_short "open-reasoner-zero-7b" \
        --steer_layer "$LAYER" \
        --save_dir "$SAVE1_FROM_TV" \
        --topk 50 --train_topk 3 --kl_mode topk \
        --n_responses 20000 \
        --max_seq_len 2048 --max_positions_per_example 64 \
        --n_epochs 5 --example_batch_size 16 \
        --lr 0.01 --weight_decay 0.0 --max_norm 0.0 \
        --seed 42 --holdout_frac 0.1 \
        --min_disagreements 1 --min_disagreements_ratio 0.0 --min_category_share 0.0 \
        --skip_cats_phase --train_global_bias \
        2>&1 | tee "$LOG1"
    cd /workspace/thinking-llms-interp
else
    echo "Stage 1 already done."
fi

BIAS_PATH_FROM_TV="$SAVE1_FROM_TV/qwen2.5-7b_bias_global.pt"
[ -f "$SAVE1/qwen2.5-7b_bias_global.pt" ] || { echo "Stage 1 failed"; exit 1; }

# ---- Stage 2a (re-collect with per-position SAE cats) ----
if [ ! -f "$SAVE2/disagreements.pt" ]; then
    echo "===== Stage 2a: re-collect 20K under bias with SAE-cats (sae L${SAE_LAYER}) ====="
    cd /workspace/thinking-llms-interp/train-vectors
    CUDA_VISIBLE_DEVICES=0,1,2 python -u optimize_correction_vectors.py \
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
        --sae_classify_layer "$SAE_LAYER" --sae_n_clusters "$N_CLUSTERS" \
        --collect_only \
        2>&1 | tee "$LOG2A"
    cd /workspace/thinking-llms-interp
else
    echo "Stage 2a already done."
fi

# ---- Stage 2b (train cats with frozen bias, SAE-cats) ----
echo "===== Stage 2b: cats with frozen bias (SAE-cats) ====="
cd /workspace/thinking-llms-interp/train-vectors
CUDA_VISIBLE_DEVICES=0,1,2 python -u optimize_correction_vectors.py \
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
    --sae_classify_layer "$SAE_LAYER" --sae_n_clusters "$N_CLUSTERS" \
    --load_collected \
    2>&1 | tee "$LOG2B"
cd /workspace/thinking-llms-interp

echo "DONE training (saecat)"
ls -la "$SAVE1" "$SAVE2"

# ---- Eval (math500, judge x3) ----
echo
N_TASKS=${N_TASKS:-48}
echo "===== Eval (math500, ${N_TASKS} tasks, judge x3) ====="

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
        --sae_layer "$SAE_LAYER" --n_clusters "$N_CLUSTERS" \
        --dom_vectors_dir ../train-vectors/results/diff_of_means \
        --dom_vectors_model_short qwen2.5-7b \
        --old_vectors_dir "../$SAVE2" \
        --old_vectors_layer "$LAYER" \
        --bias_vector_path "../$SAVE1/qwen2.5-7b_bias_global.pt" \
        --bias_layer "$LAYER" \
        --coef_sweep "0.5,1.0,1.5" \
        --judge_repetitions 3 \
        --results_suffix "${TAG}" \
        $EXTRA 2>&1 | tee "/tmp/${TAG}.log"
}

cd /workspace/thinking-llms-interp
run_one_gpu 0 "orz-bf-learn-saecat-${N_TASKS}"     ""                                    &
PID0=$!
run_one_gpu 1 "orz-bf-rand-saecat-${N_TASKS}"      "--randomize_vectors --random_seed 42" &
PID1=$!
run_one_gpu 2 "orz-bf-biasonly-saecat-${N_TASKS}"  "--bias_only"                          &
PID2=$!

wait $PID0; CODE0=$?
wait $PID1; CODE1=$?
wait $PID2; CODE2=$?
echo "Exit codes: learn=$CODE0  rand=$CODE1  biasonly=$CODE2"
