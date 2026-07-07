#!/usr/bin/env bash
# Hybrid eval on the Hendrycks-MATH holdout (disjoint from the training mix and
# MATH500) for one pair, using the selected best-of-3 vectors. Same recipe as
# the OOD evals. Writes mlp_eval_hendrycks_holdout_qa_instr<VARIANT>_h512/<CONFIG>/.
#
# Env: CONFIG ; optional VARIANT (default _holdoutsel)
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
ROOT="${THINKING_LLMS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${ROOT}"
source .env_exports.sh 2>/dev/null || true
source .venv/bin/activate 2>/dev/null || true
source "${ROOT}/configs.sh"

: "${CONFIG:?}"
cfg_load "${CONFIG}"
VARIANT="${VARIANT:-_holdoutsel}"
JUDGE_MODEL="${JUDGE_MODEL:-anthropic/claude-sonnet-4-6}"
DATASET=hendrycks_holdout
GOLD="${ROOT}/data/hendrycks_holdout_eval/eval.jsonl"
CACHE="${ROOT}/hybrid/results/response_cache_hendrycks_holdout"
VEC_DIR="${ROOT}/mlp_vectors_qa_instr${VARIANT}_h${MLP_HIDDEN}/${CONFIG}"
EVAL_DIR="${ROOT}/mlp_eval_hendrycks_holdout_qa_instr${VARIANT}_h${MLP_HIDDEN}/${CONFIG}"
mkdir -p "${EVAL_DIR}"

echo "== eval-hendrycks | ${CONFIG} variant=${VARIANT} | $(date -u) =="
[[ -f "${VEC_DIR}/cat_coef_mlp.pt" ]] || { echo "FATAL: missing vectors ${VEC_DIR}"; exit 2; }
[[ -f "${GOLD}" ]] || { echo "FATAL: missing gold ${GOLD}"; exit 2; }
N_EX=$(wc -l < "${GOLD}")
for S in 0 1 2; do
    F="${CACHE}/thinking_${TS}_${DATASET}_temp0.6_max2048_s${S}.jsonl"
    { [[ -f "${F}" && $(wc -l < "${F}") -ge ${N_EX} ]]; } || { echo "FATAL: missing think s${S}"; exit 2; }
done
[[ -f "${CACHE}/base_${BS}_${DATASET}_temp0_max2048.jsonl" ]] || { echo "FATAL: missing base cache"; exit 2; }

N_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)
TWO=""; [[ ${N_GPUS} -ge 2 ]] && TWO="--two_gpu_split"

cd hybrid
python hybrid_eval.py \
    --dataset "${DATASET}" --hendrycks_holdout_file "${GOLD}" \
    --thinking_model "${TM}" --base_model "${BM}" \
    --sae_layer ${SAEL} --n_clusters ${NK} \
    --dom_vectors_dir "${VEC_DIR}" --old_vectors_dir "${VEC_DIR}" --old_vectors_layer ${SL} \
    --coef_select mlp --mlp_coef_path "${VEC_DIR}/cat_coef_mlp.pt" --mlp_config_path "${VEC_DIR}/mlp_config.json" \
    --max_new_tokens 2048 --max_thinking_tokens 2048 \
    --temperature 0.0 --decode_temperature 0 \
    --hybrid_gen_batch_size ${BS_HYBRID} \
    --judge_repetitions 3 --judge_model "${JUDGE_MODEL}" \
    --results_dir "${EVAL_DIR}" --response_cache_dir "${CACHE}" --results_suffix final \
    --think_cache_temp_label 0.6 --think_cache_max_tokens 2048 --think_cache_sample_idx 0 \
    --base_cache_temp_label 0 --base_cache_max_tokens 2048 --base_cache_sample_idx -1 \
    --hybrid_cache_sample_idx -1 \
    --think_prompt_family auto --math_directive --base_prompt_style qa_instr \
    --pure_steer_base_eos ${TWO}
cd "${ROOT}"

echo "--- judging extra think samples s1,s2 ---"
python hybrid/judge_extra_think_samples.py \
    --cache_dir "${CACHE}" --think_short "${TS}" --base_id "${BS}" \
    --dataset "${DATASET}" --gold_file "${GOLD}" \
    --temp_label 0.6 --max_tokens 2048 --sample_ids 1,2 \
    --judge_repetitions 3 --judge_model "${JUDGE_MODEL}" \
    --out_dir "${EVAL_DIR}" --results_suffix final

echo "--- aggregating ---"
python hybrid/aggregate_samples_final.py \
    --eval_dir "${EVAL_DIR}" --base_id "${BS}" --dataset "${DATASET}" --suffix final
echo "== DONE eval-hendrycks ${CONFIG} $(date -u) =="
