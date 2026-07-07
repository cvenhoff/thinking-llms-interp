#!/usr/bin/env bash
# Out-of-distribution hybrid eval on MATH500 or GSM8K for one pair, using the
# selected best-of-3 vectors. Builds the hybrid rollouts, judges base/think/
# hybrid answers with an LLM judge (x3), and aggregates the 3 think samples.
# Writes artifacts/mlp_eval_qa_instr<VARIANT>_h512/<CONFIG>/.
#
# Env: CONFIG, DATASET (math500|gsm8k) ; optional VARIANT (default _holdoutsel)
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
ROOT="${THINKING_LLMS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${ROOT}"
source .env_exports.sh 2>/dev/null || true
source .venv/bin/activate 2>/dev/null || true
source "${ROOT}/configs.sh"

: "${CONFIG:?}"; : "${DATASET:?}"
cfg_load "${CONFIG}"
VARIANT="${VARIANT:-_holdoutsel}"
DECODE_T="${DECODE_T:-0}"
JUDGE_MODEL="${JUDGE_MODEL:-anthropic/claude-sonnet-4-6}"
case "${DATASET}" in math500|gsm8k) ;; *) echo "FATAL: DATASET must be math500|gsm8k"; exit 2 ;; esac

VEC_DIR="${ROOT}/artifacts/mlp_vectors_qa_instr${VARIANT}_h${MLP_HIDDEN}/${CONFIG}"
EVAL_DIR="${ROOT}/artifacts/mlp_eval_qa_instr${VARIANT}_h${MLP_HIDDEN}/${CONFIG}"
CACHE_THINK="${ROOT}/hybrid/results/response_cache_final"
CACHE_BASE="${ROOT}/hybrid/results/response_cache_base_qa_instr"
MERGED="${ROOT}/hybrid/results/response_cache_qa_instr${VARIANT}_h${MLP_HIDDEN}_merged/${CONFIG}"
mkdir -p "${EVAL_DIR}" "${MERGED}"

echo "== eval-ood | ${CONFIG} ${DATASET} variant=${VARIANT} | $(date -u) =="
[[ -f "${VEC_DIR}/cat_coef_mlp.pt" && -f "${VEC_DIR}/mlp_config.json" ]] || { echo "FATAL: missing vectors in ${VEC_DIR}"; exit 2; }

for S in 0 1 2; do
    ln -sfn "${CACHE_THINK}/thinking_${TS}_${DATASET}_temp0.6_max2048_s${S}.jsonl" \
            "${MERGED}/thinking_${TS}_${DATASET}_temp0.6_max2048_s${S}.jsonl"
done
SRC_BASE="${CACHE_BASE}/base_qa_instr_${BS}_${DATASET}_temp0_max2048.jsonl"
[[ -f "${SRC_BASE}" ]] || { echo "FATAL: missing base rollouts ${SRC_BASE}"; exit 2; }
ln -sfn "${SRC_BASE}" "${MERGED}/base_${BS}_${DATASET}_temp0_max2048.jsonl"
for S in 0 1 2; do [[ -f "${CACHE_THINK}/thinking_${TS}_${DATASET}_temp0.6_max2048_s${S}.jsonl" ]] || { echo "FATAL: missing think s${S}"; exit 2; }; done

N_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)
TWO=""; [[ ${N_GPUS} -ge 2 ]] && TWO="--two_gpu_split"

cd hybrid
python hybrid_eval.py \
    --dataset "${DATASET}" \
    --thinking_model "${TM}" --base_model "${BM}" \
    --sae_layer ${SAEL} --n_clusters ${NK} \
    --dom_vectors_dir "${VEC_DIR}" --old_vectors_dir "${VEC_DIR}" --old_vectors_layer ${SL} \
    --coef_select mlp --mlp_coef_path "${VEC_DIR}/cat_coef_mlp.pt" --mlp_config_path "${VEC_DIR}/mlp_config.json" \
    --max_new_tokens 2048 --max_thinking_tokens 2048 \
    --temperature 0.0 --decode_temperature ${DECODE_T} --decode_seed 0 \
    --hybrid_gen_batch_size ${BS_HYBRID} \
    --judge_repetitions 3 --judge_model "${JUDGE_MODEL}" \
    --results_dir "${EVAL_DIR}" --response_cache_dir "${MERGED}" --results_suffix final \
    --think_cache_temp_label 0.6 --think_cache_max_tokens 2048 --think_cache_sample_idx 0 \
    --base_cache_temp_label 0 --base_cache_max_tokens 2048 --base_cache_sample_idx -1 \
    --hybrid_cache_sample_idx -1 \
    --think_prompt_family auto --math_directive --base_prompt_style qa_instr \
    --pure_steer_base_eos ${TWO}
cd "${ROOT}"

echo "--- judging extra think samples s1,s2 ---"
python hybrid/judge_extra_think_samples.py \
    --cache_dir "${CACHE_THINK}" --think_short "${TS}" --base_id "${BS}" \
    --dataset "${DATASET}" --temp_label 0.6 --max_tokens 2048 --sample_ids 1,2 \
    --judge_repetitions 3 --judge_model "${JUDGE_MODEL}" \
    --out_dir "${EVAL_DIR}" --results_suffix final

echo "--- aggregating ---"
python hybrid/aggregate_samples_final.py \
    --eval_dir "${EVAL_DIR}" --base_id "${BS}" --dataset "${DATASET}" --suffix final
echo "== DONE eval-ood ${CONFIG} ${DATASET} $(date -u) =="
