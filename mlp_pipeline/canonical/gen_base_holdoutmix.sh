#!/usr/bin/env bash
# Boot vLLM for BASE_MODEL once and generate greedy BASE rollouts (qa_instr
# prompt) for the 788-Q trainmix VAL holdout, then expose the file to the hybrid
# eval cache as base_<bs>_holdoutmix_temp0_max2048.jsonl. Matches the vLLM base
# generation used for math500/gsm8k (fast; not the slow in-hybrid_eval path).
#
# Required env: BASE_MODEL, BASE_SHORT, TP, VLLM_PORT
set -euo pipefail
export PYTHONUNBUFFERED=1
ROOT=/workspace-vast/constantinv/thinking-llms-interp
cd "${ROOT}"; mkdir -p "${ROOT}/slurm_logs/final_final"
source .env_exports.sh 2>/dev/null || true
source .venv_vllm/bin/activate
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

: "${BASE_MODEL:?}"; : "${BASE_SHORT:?}"; TP="${TP:-1}"; VLLM_PORT="${VLLM_PORT:-9300}"
CACHE_QAI="${ROOT}/hybrid/results/response_cache_base_qa_instr"
CACHE_FINAL="${ROOT}/hybrid/results/response_cache_final"
GOLD="${ROOT}/data/trainmix_holdout_eval/eval.jsonl"
mkdir -p "${CACHE_QAI}"
NEED=$(wc -l < "${GOLD}")
MAX_TOK=2048
OUT="${CACHE_QAI}/base_qa_instr_${BASE_SHORT}_holdoutmix_temp0_max${MAX_TOK}.jsonl"
DST="${CACHE_FINAL}/base_${BASE_SHORT}_holdoutmix_temp0_max${MAX_TOK}.jsonl"

echo "== gen-base-holdoutmix | ${BASE_MODEL} (${BASE_SHORT}) TP=${TP} port=${VLLM_PORT} need=${NEED} | $(date -u) =="
if [[ -f "${DST}" ]] && [[ $(wc -l < "${DST}") -ge ${NEED} ]]; then
    echo "base holdout cache already complete at ${DST}; nothing to do."; exit 0
fi
if [[ -f "${OUT}" ]] && [[ $(wc -l < "${OUT}") -ge ${NEED} ]]; then
    echo "vLLM output already complete; linking to cache."
    ln -sfn "${OUT}" "${DST}"; exit 0
fi

python -m vllm.entrypoints.openai.api_server \
    --model "${BASE_MODEL}" --port ${VLLM_PORT} \
    --tensor-parallel-size ${TP} --max-model-len 6144 \
    --gpu-memory-utilization ${GPU_MEM_UTIL:-0.85} \
    --trust-remote-code --enable-prefix-caching \
    --disable-custom-all-reduce 2>&1 &
VLLM_PID=$!
trap "kill ${VLLM_PID} 2>/dev/null || true" EXIT

READY=0; MAX_WAIT=$((TP > 1 ? 420 : 220))
for i in $(seq 1 ${MAX_WAIT}); do
    sleep 10
    curl -s "http://localhost:${VLLM_PORT}/health" >/dev/null 2>&1 && { echo "[health] ready after $((i*10))s"; READY=1; break; }
    kill -0 ${VLLM_PID} 2>/dev/null || { echo "ERROR: vLLM died"; break; }
done
# NB: not a hard FATAL -- vLLM failing to boot is usually transient node
# contention, so exit non-zero WITHOUT the FATAL marker so the chain retries.
[[ ${READY} -eq 1 ]] || { echo "ERR-RETRY: vLLM did not start (transient)"; exit 3; }
sleep 10

python mlp_pipeline/canonical/gen_base_qa_instr.py \
    --base_url "http://localhost:${VLLM_PORT}/v1" \
    --base_model "${BASE_MODEL}" --base_short "${BASE_SHORT}" \
    --dataset holdoutmix --questions_file "${GOLD}" --n_examples ${NEED} \
    --max_tokens ${MAX_TOK} --temperature 0.0 --top_p 1.0 \
    --output_dir "${CACHE_QAI}" --max_concurrent 64 --resume

GOT=$([[ -f "${OUT}" ]] && wc -l < "${OUT}" || echo 0)
[[ ${GOT} -ge ${NEED} ]] || { echo "FATAL: only ${GOT}/${NEED} base rollouts generated"; exit 2; }
ln -sfn "${OUT}" "${DST}"
echo "== DONE gen-base-holdoutmix ${BASE_SHORT}: ${GOT} rollouts -> ${DST} $(date -u) =="
