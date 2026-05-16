#!/bin/bash
# Hybrid eval for the DeepSeek-R1-Distill-Llama-8B / Llama-3.1-8B bias-first
# pipeline.  Uses a conservative coef_sweep [0.1, 0.25, 0.5] -- this pair
# degrades under the canonical bias-first sweep [0.5, 1.0, 1.5] because
# the distilled-R1 hidden states are mis-aligned with the base, and
# stronger steering pushes the hybrid further off-distribution.  The
# conservative sweep recovers ~6.5% of the gap on 128 math500 tasks; the
# learn vs rand vs biasonly comparison shows the cats are doing the work.
set -euo pipefail
cd /workspace/thinking-llms-interp/hybrid

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY=error

CATS_DIR="../train-vectors/results/vars/correction_vectors_dsllama8b_biasfirst_stage2"
BIAS_PATH="../train-vectors/results/vars/correction_vectors_dsllama8b_biasfirst_stage1/llama-3.1-8b_bias_global.pt"

run_one () {
    local TAG="$1"
    local EXTRA="$2"
    rm -f "results/rolling/rolling_llama-3.1-8b_math500_${TAG}.jsonl"
    rm -f "results/summary_llama-3.1-8b_math500_${TAG}.json"
    echo "===== ${TAG} ====="
    uv run python hybrid_eval.py \
        --dataset math500 --n_tasks 128 \
        --max_new_tokens 2000 --max_thinking_tokens 2000 \
        --hybrid_gen_batch_size 64 --batch_gen_size 64 \
        --base_model meta-llama/Llama-3.1-8B \
        --thinking_model deepseek-ai/DeepSeek-R1-Distill-Llama-8B \
        --sae_layer 6 --n_clusters 15 \
        --dom_vectors_dir ../train-vectors/results/diff_of_means \
        --dom_vectors_model_short llama-3.1-8b \
        --old_vectors_dir "$CATS_DIR" \
        --old_vectors_layer 19 \
        --bias_vector_path "$BIAS_PATH" \
        --bias_layer 19 \
        --coef_sweep "0.1,0.25,0.5" \
        --judge_repetitions 3 \
        --results_suffix "${TAG}" \
        $EXTRA 2>&1 | tee "/tmp/${TAG}.log"
}

run_one "dsl-bf-learn-128"     ""
run_one "dsl-bf-rand-128"      "--randomize_vectors --random_seed 42"
run_one "dsl-bf-biasonly-128"  "--bias_only"

echo
echo "ALL DONE."
