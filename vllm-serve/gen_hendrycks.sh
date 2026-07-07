#!/usr/bin/env bash
# Rollouts on the Hendrycks-MATH holdout (disjoint from the training mix and from
# MATH500), same recipe as the OOD math500/gsm8k evals. Written to
# hybrid/results/response_cache_hendrycks_holdout/ :
#   base : base_<short>_hendrycks_holdout_temp0_max2048.jsonl   (greedy, qa_instr)
#   think: thinking_<short>_hendrycks_holdout_temp0.6_max2048_s{0,1,2}.jsonl
#
# Env: ROLE (base|think), MODEL, SHORT, TP, VLLM_PORT
set -euo pipefail
export PYTHONUNBUFFERED=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
ROOT="${THINKING_LLMS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${ROOT}"
source .env_exports.sh 2>/dev/null || true
source .venv_vllm/bin/activate 2>/dev/null || source .venv/bin/activate 2>/dev/null || true

: "${ROLE:?}"; : "${MODEL:?}"; : "${SHORT:?}"; : "${TP:?}"; VLLM_PORT="${VLLM_PORT:-9700}"
MAX_TOK=2048; BASE_SEED=42
DATASET="hendrycks_holdout"
GOLD="${ROOT}/data/hendrycks_holdout_eval/eval.jsonl"
CACHE="${ROOT}/hybrid/results/response_cache_hendrycks_holdout"
mkdir -p "${CACHE}"
N_EX=$(wc -l < "${GOLD}")

echo "== gen-hendrycks | ROLE=${ROLE} ${MODEL} (${SHORT}) tp=${TP} N=${N_EX} | $(date -u) =="

if [[ "${ROLE}" == "base" ]]; then
    DST="${CACHE}/base_${SHORT}_${DATASET}_temp0_max${MAX_TOK}.jsonl"
    [[ -f "${DST}" && $(wc -l < "${DST}") -ge ${N_EX} ]] && { echo "base complete; nothing to do."; exit 0; }
else
    OK=1; for S in 0 1 2; do F="${CACHE}/thinking_${SHORT}_${DATASET}_temp0.6_max${MAX_TOK}_s${S}.jsonl"; { [[ -f "${F}" && $(wc -l < "${F}") -ge ${N_EX} ]]; } || { OK=0; break; }; done
    [[ ${OK} -eq 1 ]] && { echo "think s0/s1/s2 complete; nothing to do."; exit 0; }
fi

python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --port ${VLLM_PORT} \
    --tensor-parallel-size ${TP} --max-model-len 8192 --generation-config vllm \
    --gpu-memory-utilization ${GPU_MEM_UTIL:-0.90} \
    --trust-remote-code --enable-prefix-caching --disable-custom-all-reduce 2>&1 &
VLLM_PID=$!
trap 'kill ${VLLM_PID} 2>/dev/null || true' EXIT
READY=0; MAX_WAIT=$((TP > 1 ? 480 : 240))
for i in $(seq 1 ${MAX_WAIT}); do
    sleep 10
    curl -s "http://localhost:${VLLM_PORT}/health" >/dev/null 2>&1 && { echo "[vllm] ready in $((i*10))s"; READY=1; break; }
    kill -0 ${VLLM_PID} 2>/dev/null || { echo "ERROR: vLLM died"; break; }
done
[[ ${READY} -eq 1 ]] || { echo "FATAL: vLLM did not start"; exit 1; }
sleep 10

if [[ "${ROLE}" == "base" ]]; then
    OUT="${CACHE}/base_qa_instr_${SHORT}_${DATASET}_temp0_max${MAX_TOK}.jsonl"
    python "${ROOT}/vllm-serve/gen_base_qa_instr.py" \
        --base_url "http://localhost:${VLLM_PORT}/v1" \
        --base_model "${MODEL}" --base_short "${SHORT}" \
        --dataset "${DATASET}" --questions_file "${GOLD}" --n_examples ${N_EX} \
        --max_tokens ${MAX_TOK} --temperature 0.0 --top_p 1.0 \
        --output_dir "${CACHE}" --max_concurrent 64 --resume
    [[ -f "${OUT}" && $(wc -l < "${OUT}") -ge ${N_EX} ]] || { echo "FATAL: base rollouts incomplete"; exit 2; }
    ln -sfn "${OUT}" "${CACHE}/base_${SHORT}_${DATASET}_temp0_max${MAX_TOK}.jsonl"
else
    cd "${ROOT}/vllm-serve"
    for S in 0 1 2; do
        F="${CACHE}/thinking_${SHORT}_${DATASET}_temp0.6_max${MAX_TOK}_s${S}.jsonl"
        [[ -f "${F}" && $(wc -l < "${F}") -ge ${N_EX} ]] && { echo "SKIP think s${S}"; continue; }
        python generate_rollouts.py \
            --base_url "http://localhost:${VLLM_PORT}/v1" \
            --model "${MODEL}" --model_short "${SHORT}" --role thinking \
            --dataset "${DATASET}" --dataset_file "${GOLD}" --n_examples ${N_EX} \
            --max_tokens ${MAX_TOK} --temperature 0.6 --top_p 0.95 \
            --seed $((BASE_SEED + S)) --sample_idx ${S} --math_directive_mode always \
            --output_dir "${CACHE}" --max_concurrent 64 --resume
    done
fi
echo "== DONE gen-hendrycks ${ROLE} ${SHORT} $(date -u) =="
