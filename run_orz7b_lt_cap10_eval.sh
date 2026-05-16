#!/bin/bash
# Eval ORZ-7B LT cap10 (smaller bias) - 3 conditions in parallel.
set -uo pipefail
cd /workspace/thinking-llms-interp
source .venv/bin/activate
source .env_exports.sh

LAYER=16
SAVE1="train-vectors/results/vars/correction_vectors_orz7b_biasfirst_stage1_canon_lt_cap10"
SAVE2="train-vectors/results/vars/correction_vectors_orz7b_biasfirst_stage2_canon_lt_cap10"
N_TASKS=${N_TASKS:-48}

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY=error

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
        --bias_vector_path "../$SAVE1/qwen2.5-7b_bias_global.pt" \
        --bias_layer "$LAYER" \
        --coef_sweep "0.5,1.0,1.5" \
        --judge_repetitions 3 \
        --results_suffix "${TAG}" \
        $EXTRA 2>&1 | tee "/tmp/${TAG}.log"
}

cd /workspace/thinking-llms-interp
run_one_gpu 0 "orz-bf-learn-cap10-${N_TASKS}"     ""                                    &
PID0=$!
run_one_gpu 1 "orz-bf-rand-cap10-${N_TASKS}"      "--randomize_vectors --random_seed 42" &
PID1=$!
run_one_gpu 2 "orz-bf-biasonly-cap10-${N_TASKS}"  "--bias_only"                          &
PID2=$!

wait $PID0; CODE0=$?
wait $PID1; CODE1=$?
wait $PID2; CODE2=$?
echo "Exit codes: learn=$CODE0  rand=$CODE1  biasonly=$CODE2"
