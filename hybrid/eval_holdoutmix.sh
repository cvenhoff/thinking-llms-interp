#!/usr/bin/env bash
# Hybrid gap-recovery on the 788-question training-mix holdout, used as the
# best-of-3 vector selection signal. Same recipe as the OOD evals (qa_instr base
# greedy, think temp0.6 s0, LLM judge x3). Writes
# artifacts/mlp_eval_holdoutmix/<CONFIG>/<RUNTAG>/.
#
# Env: CONFIG, VEC_DIR (the run's vectors), RUNTAG (run1|run2|run3)
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
ROOT="${THINKING_LLMS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${ROOT}"
source .env_exports.sh 2>/dev/null || true
source .venv/bin/activate 2>/dev/null || true
source "${ROOT}/configs.sh"

: "${CONFIG:?}"; : "${VEC_DIR:?}"; : "${RUNTAG:?}"
cfg_load "${CONFIG}"
JUDGE_MODEL="${JUDGE_MODEL:-anthropic/claude-sonnet-4-6}"
GOLD="${ROOT}/data/trainmix_holdout_eval/eval.jsonl"
EVAL_DIR="${ROOT}/artifacts/mlp_eval_holdoutmix/${CONFIG}/${RUNTAG}"
CACHE="${ROOT}/hybrid/results/response_cache_final"
mkdir -p "${EVAL_DIR}"

echo "== eval-holdoutmix | ${CONFIG}/${RUNTAG} | $(date -u) =="
[[ -f "${VEC_DIR}/cat_coef_mlp.pt" && -f "${VEC_DIR}/mlp_config.json" ]] || { echo "FATAL: missing vectors ${VEC_DIR}"; exit 2; }
[[ -f "${GOLD}" ]] || { echo "FATAL: missing gold ${GOLD}"; exit 2; }
[[ -f "${CACHE}/thinking_${TS}_holdoutmix_temp0.6_max2048_s0.jsonl" ]] || { echo "FATAL: missing holdoutmix think cache"; exit 2; }
[[ -f "${CACHE}/base_${BS}_holdoutmix_temp0_max2048.jsonl" ]] || { echo "FATAL: missing holdoutmix base cache"; exit 2; }

N_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)
TWO=""; [[ ${N_GPUS} -ge 2 ]] && TWO="--two_gpu_split"

cd hybrid
python hybrid_eval.py \
    --dataset holdoutmix --holdoutmix_file "${GOLD}" \
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
echo "== DONE eval-holdoutmix ${CONFIG}/${RUNTAG} $(date -u) =="
