#!/usr/bin/env bash
# Boot vLLM for BASE_MODEL once, then run gen_base_qa_instr.py for
# math500 + gsm8k at max=2048, temp=0.0 with the qa_instr prompt:
#   Answer the following question:\nQ: {q}\nA:
#
# Required env: BASE_MODEL, BASE_SHORT, TP, VLLM_PORT
#
#SBATCH --job-name=gen-base-qa-instr
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --cpus-per-task=8
#SBATCH --time=4:00:00
#SBATCH --output=/workspace-vast/constantinv/thinking-llms-interp/slurm_logs/final_final/%x-%j.out
#SBATCH --error=/workspace-vast/constantinv/thinking-llms-interp/slurm_logs/final_final/%x-%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
ROOT=/workspace-vast/constantinv/thinking-llms-interp
cd "${ROOT}"
mkdir -p "${ROOT}/slurm_logs/final_final"

source .env_exports.sh 2>/dev/null || true
source .venv_vllm/bin/activate

export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

CACHE="${CACHE_DIR:-${ROOT}/hybrid/results/response_cache_base_qa_instr}"
mkdir -p "${CACHE}"
VLLM_PORT=${VLLM_PORT:-9100}

TEMP=0.0
TOP_P=1.0
MAX_TOK=2048
N_MATH=500
N_GSM=1319

echo "=========================================="
echo "gen-base-qa-instr | BASE_MODEL=${BASE_MODEL} (${BASE_SHORT})"
echo "Job ${SLURM_JOB_ID:-local} | Node=$(hostname) | $(date -u)"
echo "TP=${TP} VLLM_PORT=${VLLM_PORT}  temp=${TEMP} max_tokens=${MAX_TOK}"
echo "CACHE=${CACHE}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | head -1 || true
echo "=========================================="

ALL_OK=1
for DS_PAIR in "math500 ${N_MATH}" "gsm8k ${N_GSM}"; do
    read -r DS NEED <<< "${DS_PAIR}"
    F="${CACHE}/base_qa_instr_${BASE_SHORT}_${DS}_temp0_max${MAX_TOK}.jsonl"
    if [[ ! -f "${F}" ]] || [[ $(wc -l < "${F}") -lt ${NEED} ]]; then
        ALL_OK=0; break
    fi
done
if [[ ${ALL_OK} -eq 1 ]]; then
    echo "All target files complete for ${BASE_SHORT}. Nothing to do."
    exit 0
fi

python -m vllm.entrypoints.openai.api_server \
    --model "${BASE_MODEL}" \
    --port ${VLLM_PORT} \
    --tensor-parallel-size ${TP} \
    --max-model-len 4096 \
    --gpu-memory-utilization ${GPU_MEM_UTIL:-0.85} \
    --trust-remote-code \
    --enable-prefix-caching \
    --disable-custom-all-reduce 2>&1 &
VLLM_PID=$!
trap "kill ${VLLM_PID} 2>/dev/null || true" EXIT

READY=0
MAX_WAIT=$((TP > 1 ? 360 : 180))
for i in $(seq 1 ${MAX_WAIT}); do
    sleep 10
    if curl -s "http://localhost:${VLLM_PORT}/health" >/dev/null 2>&1; then
        echo "[health] ready after $((i*10))s"; READY=1; break
    fi
    if ! kill -0 ${VLLM_PID} 2>/dev/null; then
        echo "ERROR: vLLM process died"; break
    fi
done
if [[ ${READY} -eq 0 ]]; then
    echo "FATAL: vLLM did not start"
    exit 1
fi
sleep 10

for DS_PAIR in "math500 ${N_MATH}" "gsm8k ${N_GSM}"; do
    read -r DS NEED <<< "${DS_PAIR}"
    OUTFILE="${CACHE}/base_qa_instr_${BASE_SHORT}_${DS}_temp0_max${MAX_TOK}.jsonl"
    EXIST=0; [[ -f "${OUTFILE}" ]] && EXIST=$(wc -l < "${OUTFILE}")
    if [[ ${EXIST} -ge ${NEED} ]]; then
        echo "SKIP ${BASE_SHORT}/${DS}: ${EXIST} >= ${NEED}"; continue
    fi
    echo ""
    echo "--- ${BASE_SHORT}/${DS} (have=${EXIST}, need=${NEED}) ---"
    python mlp_pipeline/canonical/gen_base_qa_instr.py \
        --base_url "http://localhost:${VLLM_PORT}/v1" \
        --base_model "${BASE_MODEL}" --base_short "${BASE_SHORT}" \
        --dataset "${DS}" --n_examples ${NEED} \
        --max_tokens ${MAX_TOK} --temperature ${TEMP} --top_p ${TOP_P} \
        --output_dir "${CACHE}" --max_concurrent 64 --resume
done

echo ""
echo "=========================================="
echo "DONE gen-base-qa-instr ${BASE_SHORT}  $(date -u)"
ls -lh "${CACHE}" | grep "_${BASE_SHORT}_.*_max${MAX_TOK}" || true
echo "=========================================="
