#!/usr/bin/env bash
# Negative-control ablation on the canonical selected vectors: base/think arms
# are unchanged; only the hybrid arm's steering is perturbed at inference.
#   randcat  : --random_firing            (steer with a random SAE category)
#   randV    : --randomize_vectors        (random steering directions)
#   mlponly  : --randomize_vectors --random_firing  (only the MLP magnitude is real)
#   randpos  : --random_steer_prob <p>    (steer random positions at empirical rate)
#
# Env: CONFIG (orz-1.5b|orz-32b), DATASET (math500|gsm8k), ABLATION
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
ROOT="${THINKING_LLMS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${ROOT}"
source .env_exports.sh 2>/dev/null || true
source .venv/bin/activate 2>/dev/null || true
source "${ROOT}/configs.sh"

: "${CONFIG:?}"; : "${DATASET:?}"; : "${ABLATION:?}"
cfg_load "${CONFIG}"
JUDGE_MODEL="${JUDGE_MODEL:-anthropic/claude-sonnet-4-6}"
case "${DATASET}" in math500|gsm8k) ;; *) echo "FATAL: bad DATASET"; exit 2 ;; esac

# Empirical steered-position fraction from the canonical run (for randpos).
declare -A PSTEER
PSTEER[orz-1.5b/math500]=0.0493; PSTEER[orz-1.5b/gsm8k]=0.0885
PSTEER[orz-32b/math500]=0.1225;  PSTEER[orz-32b/gsm8k]=0.1151

EXTRA=()
case "${ABLATION}" in
    randcat) EXTRA=(--random_firing --random_seed 0) ;;
    randV)   EXTRA=(--randomize_vectors --random_seed 0) ;;
    mlponly) EXTRA=(--randomize_vectors --random_firing --random_seed 0) ;;
    randpos) P="${PSTEER[${CONFIG}/${DATASET}]:?no empirical p for ${CONFIG}/${DATASET}}"; EXTRA=(--random_steer_prob "${P}" --random_seed 0) ;;
    *) echo "FATAL: unknown ABLATION=${ABLATION}"; exit 2 ;;
esac

VEC_DIR="${ROOT}/artifacts/mlp_vectors_qa_instr_holdoutsel_h${MLP_HIDDEN}/${CONFIG}"
EVAL_DIR="${ROOT}/artifacts/mlp_eval_qa_instr_holdoutsel_ablations/${CONFIG}-${ABLATION}/${DATASET}"
CACHE_THINK="${ROOT}/hybrid/results/response_cache_final"
CACHE_BASE="${ROOT}/hybrid/results/response_cache_base_qa_instr"
MERGED="${ROOT}/hybrid/results/response_cache_holdoutsel_abl_merged/${CONFIG}"
mkdir -p "${EVAL_DIR}" "${MERGED}"

echo "== eval-ablation | ${ABLATION} ${CONFIG} ${DATASET} | $(date -u) =="
[[ -f "${VEC_DIR}/cat_coef_mlp.pt" ]] || { echo "FATAL: missing vectors ${VEC_DIR}"; exit 2; }
for S in 0 1 2; do
    ln -sfn "${CACHE_THINK}/thinking_${TS}_${DATASET}_temp0.6_max2048_s${S}.jsonl" \
            "${MERGED}/thinking_${TS}_${DATASET}_temp0.6_max2048_s${S}.jsonl"
done
SRC_BASE="${CACHE_BASE}/base_qa_instr_${BS}_${DATASET}_temp0_max2048.jsonl"
[[ -f "${SRC_BASE}" ]] || { echo "FATAL: missing base ${SRC_BASE}"; exit 2; }
ln -sfn "${SRC_BASE}" "${MERGED}/base_${BS}_${DATASET}_temp0_max2048.jsonl"

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
    --temperature 0.0 --decode_temperature 0 \
    --hybrid_gen_batch_size ${BS_HYBRID} \
    --judge_repetitions 3 --judge_model "${JUDGE_MODEL}" \
    --results_dir "${EVAL_DIR}" --response_cache_dir "${MERGED}" --results_suffix "abl_${ABLATION}" \
    --think_cache_temp_label 0.6 --think_cache_max_tokens 2048 --think_cache_sample_idx 0 \
    --base_cache_temp_label 0 --base_cache_max_tokens 2048 --base_cache_sample_idx -1 \
    --hybrid_cache_sample_idx -1 \
    --think_prompt_family auto --math_directive --base_prompt_style qa_instr \
    --pure_steer_base_eos "${EXTRA[@]}" ${TWO}
cd "${ROOT}"
echo "== DONE eval-ablation ${ABLATION} ${CONFIG} ${DATASET} $(date -u) =="
