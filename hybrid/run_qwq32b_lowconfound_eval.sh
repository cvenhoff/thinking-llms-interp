#!/bin/bash
# Low-confound coefficient-selection eval for the QwQ-32B / Qwen2.5-32B
# bias-first pipeline, using the new --coef_select=think_top1_match_maxconf.
#
# Why this exists:
#   With pg / kl_top3 / think_top1 the random ablation can sometimes match
#   or beat the learned vectors, because those rules let the sweep pick a
#   coef that just happens to push base towards a token with high think-
#   model logp -- even when the steered argmax never equals the thinking
#   token.  think_top1_match_maxconf fixes this by:
#     1. only crediting coefs whose steered argmax EXACTLY equals the
#        thinking model's top-1 token T,
#     2. among those matching coefs picking the one with highest
#        log p_steered(T) (highest-confidence match),
#     3. otherwise leaving the row UNSTEERED.
#   Random vectors fail (1) ~(V-1)/V of the time per coef and so usually
#   fall through to the base argmax -> random gap recovery should drop
#   to ~0%.  Trained vectors that actually point in T's direction get
#   credit at the most confident coef.
#
# Vectors used (current best for QwQ-32B):
#   - bias:  ../train-vectors/results/vars/correction_vectors_qwq32b_biasfirst_allpos_stage1_ce_cap25
#   - cats:  ../train-vectors/results/vars/correction_vectors_qwq32b_biasfirst_allpos_stage2_rescale6
set -euo pipefail
cd /workspace/thinking-llms-interp/hybrid

# Pull HF cache + API keys onto /workspace (mfs) so we don't blow the
# 20G overlay rootfs that /workspace-vast lives on.
. /workspace/thinking-llms-interp/.env_exports.sh

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY=error

CATS_DIR="../train-vectors/results/vars/correction_vectors_qwq32b_biasfirst_allpos_stage2_rescale6"
BIAS_PATH="../train-vectors/results/vars/correction_vectors_qwq32b_biasfirst_allpos_stage1_ce_cap25/qwen2.5-32b_bias_global.pt"

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
        --coef_sweep "0.5,1.0,1.5,2.0,2.5,3.0" \
        --coef_select think_top1_match_maxconf \
        --judge_repetitions 3 \
        --results_suffix "${TAG}" \
        $EXTRA 2>&1 | tee "/tmp/${TAG}.log"
}

run_one "qwq-lc-learn-128"     ""
run_one "qwq-lc-rand-128"      "--randomize_vectors --random_seed 42"
run_one "qwq-lc-biasonly-128"  "--bias_only"

echo
echo "ALL QWQ LOW-CONFOUND EVAL DONE."
echo
echo "Summaries:"
for TAG in qwq-lc-learn-128 qwq-lc-rand-128 qwq-lc-biasonly-128; do
    F="results/summary_qwen2.5-32b_math500_${TAG}.json"
    if [ -f "$F" ]; then
        echo "----- $TAG -----"
        cat "$F"
        echo
    fi
done
