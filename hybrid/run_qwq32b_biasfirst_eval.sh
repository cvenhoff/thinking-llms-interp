#!/bin/bash
# Hybrid eval for the QwQ-32B / Qwen2.5-32B bias-first pipeline.
#  - learned :   per-category vectors trained as residuals on top of frozen bias
#  - rand    :   per-category vectors replaced with norm-preserving random dirs
#  - biasonly:   per-category vectors zeroed, only the global bias remains
set -euo pipefail
cd /workspace/thinking-llms-interp/hybrid

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY=error

CATS_DIR="../train-vectors/results/vars/correction_vectors_qwq32b_biasfirst_stage2"
BIAS_PATH="../train-vectors/results/vars/correction_vectors_qwq32b_biasfirst_stage1/qwen2.5-32b_bias_global.pt"

if [ ! -f "$BIAS_PATH" ]; then
    echo "ERROR: bias not found at $BIAS_PATH"; exit 1
fi
if [ ! -f "$CATS_DIR/qwen2.5-32b_idx0_linear.pt" ]; then
    echo "ERROR: cats not found in $CATS_DIR"; exit 1
fi

run_one () {
    local TAG="$1"
    local EXTRA="$2"
    rm -f "results/rolling/rolling_qwen2.5-32b_math500_${TAG}.jsonl"
    rm -f "results/summary_qwen2.5-32b_math500_${TAG}.json"
    echo "===== ${TAG} ====="
    uv run python hybrid_eval.py \
        --dataset math500 --n_tasks 128 \
        --max_new_tokens 2000 --max_thinking_tokens 2000 \
        --batch_gen_size 4 --hybrid_gen_batch_size 4 \
        --base_model Qwen/Qwen2.5-32B \
        --thinking_model Qwen/QwQ-32B \
        --sae_layer 27 --n_clusters 10 \
        --disable_sae_mean \
        --dom_vectors_dir ../train-vectors/results/diff_of_means \
        --dom_vectors_model_short qwen2.5-32b \
        --old_vectors_dir "$CATS_DIR" \
        --old_vectors_layer 38 \
        --bias_vector_path "$BIAS_PATH" \
        --bias_layer 38 \
        --coef_sweep "0.5,1.0,1.5" \
        --results_suffix "${TAG}" \
        $EXTRA 2>&1 | tee "/tmp/${TAG}.log"
}

run_one "qwq-bf-learn-128"     ""
run_one "qwq-bf-rand-128"      "--randomize_vectors --random_seed 42"
run_one "qwq-bf-biasonly-128"  "--bias_only"

echo
echo "ALL QWQ EVAL DONE."
