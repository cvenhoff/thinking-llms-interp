#!/bin/bash
# ORZ-7B SAE-cat eval at n=128 tasks (matches historical eval size).
# Reuses already-trained vectors in correction_vectors_orz7b_biasfirst_stage2_saecat.
# Runs on GPU 0 only (GPUs 1+2 reserved for QwQ-32B SAE-cat pipeline).

set -uo pipefail
cd /workspace/thinking-llms-interp
source .venv/bin/activate
source .env_exports.sh

LAYER=16
SAE_LAYER=20
N_CLUSTERS=10
GPU=${GPU:-0}
N_TASKS=${N_TASKS:-128}

SAVE1="train-vectors/results/vars/correction_vectors_orz7b_biasfirst_stage1_saecat"
SAVE2="train-vectors/results/vars/correction_vectors_orz7b_biasfirst_stage2_saecat"

[ -f "$SAVE1/qwen2.5-7b_bias_global.pt" ] || { echo "Bias missing"; exit 1; }
[ -f "$SAVE2/qwen2.5-7b_idx0_linear.pt" ] || { echo "Cats missing"; exit 1; }

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

run_one () {
    local TAG="$1"
    local EXTRA="$2"
    cd /workspace/thinking-llms-interp/hybrid
    rm -f "results/rolling/rolling_qwen2.5-7b_math500_${TAG}.jsonl"
    rm -f "results/summary_qwen2.5-7b_math500_${TAG}.json"
    rm -f "results/judge_reps_qwen2.5-7b_math500_${TAG}.json"
    echo "===== ${TAG} (GPU ${GPU}, n=${N_TASKS}) ====="
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
    cd /workspace/thinking-llms-interp
}

run_one "orz-bf-learn-saecat-${N_TASKS}"     ""
run_one "orz-bf-rand-saecat-${N_TASKS}"      "--randomize_vectors --random_seed 42"
run_one "orz-bf-biasonly-saecat-${N_TASKS}"  "--bias_only"

echo "ALL ORZ-7B n=${N_TASKS} SAE-CAT EVAL DONE."
