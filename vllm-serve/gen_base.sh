#!/usr/bin/env bash
# Greedy base-model rollouts with the qa_instr prompt
#   "Answer the following question:\nQ: {q}\nA:"
# for one base model via vLLM (temp 0, max 2048). Written as:
#   response_cache_base_qa_instr/base_qa_instr_<bs>_math500_temp0_max2048.jsonl
#   response_cache_base_qa_instr/base_qa_instr_<bs>_gsm8k_temp0_max2048.jsonl
#   response_cache_final/base_<bs>_holdoutmix_temp0_max2048.jsonl   (selection)
#
# Env (set by run_rollouts.sh, or by hand):
#   BASE_MODEL, BASE_SHORT, TP, VLLM_PORT
set -euo pipefail
export PYTHONUNBUFFERED=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
ROOT="${THINKING_LLMS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${ROOT}"
source .env_exports.sh 2>/dev/null || true
source .venv_vllm/bin/activate 2>/dev/null || source .venv/bin/activate 2>/dev/null || true

: "${BASE_MODEL:?}"; : "${BASE_SHORT:?}"; TP="${TP:-1}"; VLLM_PORT="${VLLM_PORT:-9100}"
CACHE="${ROOT}/hybrid/results/response_cache_base_qa_instr"
CACHE_FINAL="${ROOT}/hybrid/results/response_cache_final"
HOLDOUT_FILE="${ROOT}/data/trainmix_holdout_eval/eval.jsonl"
mkdir -p "${CACHE}" "${CACHE_FINAL}"
MAX_TOK=2048; N_MATH=500; N_GSM=1319; N_HOLD=$(wc -l < "${HOLDOUT_FILE}")

echo "== gen-base | ${BASE_MODEL} (${BASE_SHORT}) tp=${TP} | $(date -u) =="

python -m vllm.entrypoints.openai.api_server \
    --model "${BASE_MODEL}" --port ${VLLM_PORT} \
    --tensor-parallel-size ${TP} --max-model-len 6144 \
    --gpu-memory-utilization ${GPU_MEM_UTIL:-0.85} \
    --trust-remote-code --enable-prefix-caching --disable-custom-all-reduce 2>&1 &
VLLM_PID=$!
trap 'kill ${VLLM_PID} 2>/dev/null || true' EXIT
READY=0; MAX_WAIT=$((TP > 1 ? 360 : 180))
for i in $(seq 1 ${MAX_WAIT}); do
    sleep 10
    curl -s "http://localhost:${VLLM_PORT}/health" >/dev/null 2>&1 && { echo "[vllm] ready in $((i*10))s"; READY=1; break; }
    kill -0 ${VLLM_PID} 2>/dev/null || { echo "ERROR: vLLM died"; break; }
done
[[ ${READY} -eq 1 ]] || { echo "FATAL: vLLM did not start"; exit 1; }
sleep 10

gen() { # dataset n_examples [questions_file]
    local ds="$1" need="$2" file="${3:-}"
    local out="${CACHE}/base_qa_instr_${BASE_SHORT}_${ds}_temp0_max${MAX_TOK}.jsonl"
    local have=0; [[ -f "${out}" ]] && have=$(wc -l < "${out}")
    if [[ ${have} -lt ${need} ]]; then
        echo "--- base ${ds} (have=${have} need=${need}) ---"
        python "${ROOT}/vllm-serve/gen_base_qa_instr.py" \
            --base_url "http://localhost:${VLLM_PORT}/v1" \
            --base_model "${BASE_MODEL}" --base_short "${BASE_SHORT}" \
            --dataset "${ds}" ${file:+--questions_file "${file}"} --n_examples ${need} \
            --max_tokens ${MAX_TOK} --temperature 0.0 --top_p 1.0 \
            --output_dir "${CACHE}" --max_concurrent 64 --resume
    else
        echo "SKIP base ${ds}: ${have}>=${need}"
    fi
}

gen math500    "${N_MATH}"
gen gsm8k      "${N_GSM}"
gen holdoutmix "${N_HOLD}" "${HOLDOUT_FILE}"
# Expose the holdout base rollouts under the name the selection eval expects.
ln -sfn "${CACHE}/base_qa_instr_${BASE_SHORT}_holdoutmix_temp0_max${MAX_TOK}.jsonl" \
        "${CACHE_FINAL}/base_${BASE_SHORT}_holdoutmix_temp0_max${MAX_TOK}.jsonl"
echo "== DONE gen-base ${BASE_SHORT} $(date -u) =="
