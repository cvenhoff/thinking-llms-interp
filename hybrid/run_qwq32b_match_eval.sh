#!/bin/bash
# Low-confound coefficient-selection eval: think_top1_match
# (strict argmax==T gate, SMALLEST matching coef tiebreak).
#
# Why this rule (vs the maxconf variant we tried first):
#   The maxconf tiebreak (highest log p_steered(T) among matching coefs)
#   concentrates ~75% of committed steerings at coef=3.0 for BOTH trained
#   and random vectors, because at norm ~93 the perturbation effectively
#   randomises the next-token distribution and the strict argmax==T gate
#   stops being exclusive (random match rate 13.6%, trained 13.9% --
#   indistinguishable; trained gap -15%, random gap +20% on n=128/100
#   partial, see results/maxconf_negctrl/).
#
#   `think_top1_match` (smallest matching coef) keeps the same strict
#   gate but biases AWAY from large coefs.  Trained vectors that genuinely
#   point at T are expected to satisfy argmax==T at small coefs (where
#   random rarely does), so the rule should now separate the two:
#     - trained:  many small-coef matches    -> commits at small coef ->
#                 mild residual perturbation, trajectory preserved
#     - random:   matches only happen by chance at large coef ->
#                 commits at large coef on the rare match -> ~no signal
#
# Vectors used (current best for QwQ-32B; same as maxconf run):
#   - bias:  ../train-vectors/.../correction_vectors_qwq32b_biasfirst_allpos_stage1_ce_cap25
#   - cats:  ../train-vectors/.../correction_vectors_qwq32b_biasfirst_allpos_stage2_rescale6
set -euo pipefail
cd /workspace/thinking-llms-interp/hybrid

. /workspace/thinking-llms-interp/.env_exports.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY=error

CATS_DIR="../train-vectors/results/vars/correction_vectors_qwq32b_biasfirst_allpos_stage2_rescale6"
BIAS_PATH="../train-vectors/results/vars/correction_vectors_qwq32b_biasfirst_allpos_stage1_ce_cap25/qwen2.5-32b_bias_global.pt"

if [ ! -f "$BIAS_PATH" ]; then echo "ERROR: bias not found at $BIAS_PATH"; exit 1; fi
if [ ! -f "$CATS_DIR/qwen2.5-32b_idx0_linear.pt" ]; then echo "ERROR: cats not found"; exit 1; fi

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
        --coef_select think_top1_match \
        --judge_repetitions 3 \
        --results_suffix "${TAG}" \
        $EXTRA 2>&1 | tee "/tmp/${TAG}.log"
}

run_one "qwq-mt-learn-128"     ""
run_one "qwq-mt-rand-128"      "--randomize_vectors --random_seed 42"
run_one "qwq-mt-biasonly-128"  "--bias_only"

echo
echo "ALL QWQ THINK-TOP1-MATCH EVAL DONE."
echo
for TAG in qwq-mt-learn-128 qwq-mt-rand-128 qwq-mt-biasonly-128; do
    F="results/summary_qwen2.5-32b_math500_${TAG}.json"
    if [ -f "$F" ]; then echo "----- $TAG -----"; cat "$F"; echo; fi
done
