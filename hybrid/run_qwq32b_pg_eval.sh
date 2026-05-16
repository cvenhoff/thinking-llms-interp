#!/bin/bash
# Hybrid eval under the paper's pg (perplexity-guardrail) selection rule:
#   for each disagreement, pick the coef whose steered-base argmax token
#   has the highest log-prob under the thinking model's distribution.
#
# Why this run, after the two argmax-gate negative-controls:
#   - think_top1_match_maxconf  : random gap +20%, trained -15%, both pile
#                                 ~75% at coef=3.0  (results/maxconf_negctrl/)
#   - think_top1_match (smallest): random +23.8%, trained -9.5%, both have
#                                 ~13% match rate, ~52% at coef>=2.5
#                                 (results/match_negctrl/)
#   Both gate-based rules fail to separate trained from random at this
#   vector norm, because the strict argmax==T gate only fires in a
#   perturbation regime where directional information is washed out.
#
#   pg uses a CONTINUOUS scoring signal (log p_think(argmax(steered))) so
#   it does not require an argmax flip and rewards perturbations that
#   shift mass toward thinking-plausible tokens.  This is the regime
#   where directional structure of trained vectors *should* matter, and
#   the historical claim ">90% gap" was measured under pg.
#
# Vectors used (current best for QwQ-32B):
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
        --coef_select pg \
        --judge_repetitions 3 \
        --results_suffix "${TAG}" \
        $EXTRA 2>&1 | tee "/tmp/${TAG}.log"
}

run_one "qwq-pg-learn-128"     ""
run_one "qwq-pg-rand-128"      "--randomize_vectors --random_seed 42"
run_one "qwq-pg-biasonly-128"  "--bias_only"

echo
echo "ALL QWQ PG EVAL DONE."
echo
for TAG in qwq-pg-learn-128 qwq-pg-rand-128 qwq-pg-biasonly-128; do
    F="results/summary_qwen2.5-32b_math500_${TAG}.json"
    if [ -f "$F" ]; then echo "----- $TAG -----"; cat "$F"; echo; fi
done
