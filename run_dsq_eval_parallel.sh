#!/bin/bash
# DSQ-32B hybrid eval: 3 conditions in parallel, one per GPU.
set -uo pipefail
cd /workspace/thinking-llms-interp
source .venv/bin/activate
source .env_exports.sh

CATS_DIR="train-vectors/results/vars/correction_vectors_dsqwen32b_biasfirst_stage2"
BIAS_PATH="train-vectors/results/vars/correction_vectors_dsqwen32b_biasfirst_stage1/qwen2.5-32b_bias_global.pt"

if [ ! -f "$BIAS_PATH" ]; then
    echo "ERROR: bias not found at $BIAS_PATH"; exit 1
fi
if [ ! -f "$CATS_DIR/qwen2.5-32b_idx0_linear.pt" ]; then
    echo "ERROR: cats not found in $CATS_DIR"; exit 1
fi

run_one_gpu () {
    local GPU="$1"
    local TAG="$2"
    local EXTRA="$3"
    cd /workspace/thinking-llms-interp/hybrid
    rm -f "results/rolling/rolling_qwen2.5-32b_math500_${TAG}.jsonl"
    rm -f "results/summary_qwen2.5-32b_math500_${TAG}.json"
    rm -f "results/judge_reps_qwen2.5-32b_math500_${TAG}.json"
    echo "===== ${TAG} on GPU ${GPU} ====="
    CUDA_VISIBLE_DEVICES="$GPU" python hybrid_eval.py \
        --dataset math500 --n_tasks 128 \
        --max_new_tokens 2000 --max_thinking_tokens 2000 \
        --batch_gen_size 4 --hybrid_gen_batch_size 4 \
        --base_model Qwen/Qwen2.5-32B \
        --thinking_model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
        --sae_layer 27 --n_clusters 15 \
        --dom_vectors_dir ../train-vectors/results/diff_of_means \
        --dom_vectors_model_short qwen2.5-32b \
        --old_vectors_dir "../$CATS_DIR" \
        --old_vectors_layer 38 \
        --bias_vector_path "../$BIAS_PATH" \
        --bias_layer 38 \
        --coef_sweep "0.5,1.0,1.5" \
        --judge_repetitions 3 \
        --results_suffix "${TAG}" \
        $EXTRA 2>&1 | tee "/tmp/${TAG}.log"
}

run_one_gpu 0 "dsq-bf-learn-128"     ""                                    &
PID0=$!
run_one_gpu 1 "dsq-bf-rand-128"      "--randomize_vectors --random_seed 42" &
PID1=$!
run_one_gpu 2 "dsq-bf-biasonly-128"  "--bias_only"                          &
PID2=$!

echo "Launched: learn=$PID0 rand=$PID1 biasonly=$PID2"
wait $PID0; CODE0=$?
wait $PID1; CODE1=$?
wait $PID2; CODE2=$?
echo "Done.  exit codes: learn=$CODE0  rand=$CODE1  biasonly=$CODE2"
