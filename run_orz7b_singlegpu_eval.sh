#!/bin/bash
# Eval the cats trained by run_orz7b_singlegpu_diag.sh against the
# canon Stage-1 bias.  Runs all 3 conditions in parallel on GPUs 0/1/2,
# matches the same coef sweep + judge_repetitions=3 as run_orz7b_canon.sh.
set -uo pipefail
cd /workspace/thinking-llms-interp
source .venv/bin/activate
source .env_exports.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

LAYER=16
SAVE1="train-vectors/results/vars/correction_vectors_orz7b_biasfirst_stage1_canon"
SAVE2="train-vectors/results/vars/correction_vectors_orz7b_biasfirst_stage2_singlegpu_diag"

[ -f "$SAVE1/qwen2.5-7b_bias_global.pt" ] || { echo "Missing bias"; exit 1; }
[ -f "$SAVE2/qwen2.5-7b_idx9_linear.pt" ] || { echo "Missing cats"; exit 1; }

N_TASKS=${N_TASKS:-48}
echo "===== Eval (math500, ${N_TASKS} tasks, judge x3, single-GPU-trained cats) ====="

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
run_one_gpu 0 "orz-bf-learn-singlegpu-${N_TASKS}"     ""                                    &
PID0=$!
run_one_gpu 1 "orz-bf-rand-singlegpu-${N_TASKS}"      "--randomize_vectors --random_seed 42" &
PID1=$!
run_one_gpu 2 "orz-bf-biasonly-singlegpu-${N_TASKS}"  "--bias_only"                          &
PID2=$!

wait $PID0; CODE0=$?
wait $PID1; CODE1=$?
wait $PID2; CODE2=$?
echo "Exit codes: learn=$CODE0  rand=$CODE1  biasonly=$CODE2"
