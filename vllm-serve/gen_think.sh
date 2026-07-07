#!/usr/bin/env bash
# Thinking-model rollouts for one model via vLLM, written to
# hybrid/results/response_cache_final/ :
#   thinking_<short>_trainmix_temp0.6_max2048_s0.jsonl        (vector training)
#   thinking_<short>_math500_temp0.6_max2048_s{0,1,2}.jsonl   (OOD eval)
#   thinking_<short>_gsm8k_temp0.6_max2048_s{0,1,2}.jsonl     (OOD eval)
#   thinking_<short>_holdoutmix_temp0.6_max2048_s0.jsonl      (best-of-3 selection)
#
# Env (set by run_rollouts.sh, or by hand):
#   MODEL      HF model id        SHORT   short name for cache filenames
#   FMT        rollout format (orz|r1|qwq)   TP  vLLM tensor-parallel size
#   VLLM_PORT  free TCP port (default 8000)
set -euo pipefail
export PYTHONUNBUFFERED=1
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1
ROOT="${THINKING_LLMS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${ROOT}"
source .env_exports.sh 2>/dev/null || true
source .venv_vllm/bin/activate 2>/dev/null || source .venv/bin/activate 2>/dev/null || true

: "${MODEL:?}"; : "${SHORT:?}"; : "${FMT:?}"; TP="${TP:-1}"; VLLM_PORT="${VLLM_PORT:-8000}"
CACHE="${ROOT}/hybrid/results/response_cache_final"
TRAIN_FILE="${ROOT}/data/training_mix_v1/train.jsonl"
VAL_FILE="${ROOT}/data/training_mix_v1/val.jsonl"
HOLDOUT_FILE="${ROOT}/data/trainmix_holdout_eval/eval.jsonl"
mkdir -p "${CACHE}"

TEMP=0.6; TOP_P=0.95; MAX_TOK=2048; BASE_SEED=42
N_MATH=500; N_GSM=1319
N_TRAIN=$(( $(wc -l < "${TRAIN_FILE}") + $(wc -l < "${VAL_FILE}") ))
N_HOLD=$(wc -l < "${HOLDOUT_FILE}")

echo "== gen-think | ${MODEL} (${SHORT}) fmt=${FMT} tp=${TP} | $(date -u) =="

COMBINED=$(mktemp /tmp/trainmix_combined_XXXXXX.jsonl)
cat "${TRAIN_FILE}" "${VAL_FILE}" > "${COMBINED}"
trap 'rm -f "${COMBINED}"; [[ -n "${VLLM_PID:-}" ]] && kill "${VLLM_PID}" 2>/dev/null || true' EXIT

python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --port ${VLLM_PORT} \
    --tensor-parallel-size ${TP} --max-model-len 4096 \
    --gpu-memory-utilization ${GPU_MEM_UTIL:-0.85} \
    --trust-remote-code --enable-prefix-caching --disable-custom-all-reduce 2>&1 &
VLLM_PID=$!
READY=0; MAX_WAIT=$((TP > 1 ? 360 : 180))
for i in $(seq 1 ${MAX_WAIT}); do
    sleep 10
    curl -s "http://localhost:${VLLM_PORT}/health" >/dev/null 2>&1 && { echo "[vllm] ready in $((i*10))s"; READY=1; break; }
    kill -0 ${VLLM_PID} 2>/dev/null || { echo "ERROR: vLLM died"; break; }
done
[[ ${READY} -eq 1 ]] || { echo "FATAL: vLLM did not start"; exit 1; }
sleep 10

cd "${ROOT}/vllm-serve"
gen() { # dataset n_examples sample_idx math_mode [dataset_file]
    local ds="$1" need="$2" s="$3" mdir="$4" file="${5:-}"
    local out="${CACHE}/thinking_${SHORT}_${ds}_temp${TEMP}_max${MAX_TOK}_s${s}.jsonl"
    local have=0; [[ -f "${out}" ]] && have=$(wc -l < "${out}")
    [[ ${have} -ge ${need} ]] && { echo "SKIP ${ds} s${s}: ${have}>=${need}"; return; }
    echo "--- ${ds} s=${s} (have=${have} need=${need}) ---"
    python generate_rollouts.py \
        --base_url "http://localhost:${VLLM_PORT}/v1" \
        --model "${MODEL}" --model_short "${SHORT}" --format "${FMT}" --role thinking \
        --dataset "${ds}" ${file:+--dataset_file "${file}"} --n_examples ${need} \
        --max_tokens ${MAX_TOK} --temperature ${TEMP} --top_p ${TOP_P} \
        --seed $((BASE_SEED + s)) --sample_idx ${s} \
        --math_directive_mode "${mdir}" \
        --output_dir "${CACHE}" --max_concurrent 64 --resume
}

gen trainmix   "${N_TRAIN}" 0 auto   "${COMBINED}"
gen holdoutmix "${N_HOLD}"  0 auto   "${HOLDOUT_FILE}"
for S in 0 1 2; do
    gen math500 "${N_MATH}" "${S}" always
    gen gsm8k   "${N_GSM}"  "${S}" always
done
echo "== DONE gen-think ${SHORT} $(date -u) =="
