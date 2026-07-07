#!/usr/bin/env bash
# final_final gen-think (MAX_TOK=4096): think-model rollouts via vLLM.
# Produces, into hybrid/results/response_cache_final/ :
#   thinking_<short>_trainmix_temp0.6_max4096_s0.jsonl
#   thinking_<short>_math500_temp0.6_max4096_s{0,1,2}.jsonl
#   thinking_<short>_gsm8k_temp0.6_max4096_s{0,1,2}.jsonl
#
# Required env vars:
#   MODEL          HF model id
#   SHORT          short model name (cache filenames)
#   FMT            generate_rollouts.py format key (orz/r1/qwq/passthrough)
#   TP             vLLM tensor-parallel-size
#   VLLM_PORT      free TCP port for vLLM
#
#SBATCH --job-name=gen-think-finalfinal
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --cpus-per-task=8
#SBATCH --time=12:00:00
#SBATCH --output=/workspace-vast/constantinv/thinking-llms-interp/slurm_logs/final_final/%x-%j.out
#SBATCH --error=/workspace-vast/constantinv/thinking-llms-interp/slurm_logs/final_final/%x-%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
ROOT=/workspace-vast/constantinv/thinking-llms-interp
cd "${ROOT}"
mkdir -p "${ROOT}/slurm_logs/final_final"

source .env_exports.sh 2>/dev/null || true
source .venv_vllm/bin/activate

# Some models declare smaller max_position_embeddings than 8192.  Allow
# vLLM to override; for our prompts (<1000 tokens + max 4096 output) we
# stay well within trained ranges in practice.
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

CACHE="${ROOT}/hybrid/results/response_cache_final"
mkdir -p "${CACHE}"
TRAIN_FILE="${ROOT}/data/training_mix_v1/train.jsonl"
VAL_FILE="${ROOT}/data/training_mix_v1/val.jsonl"
VLLM_PORT=${VLLM_PORT:-8000}

TOTAL_TRAIN=$(wc -l < "${TRAIN_FILE}")
TOTAL_VAL=$(wc -l < "${VAL_FILE}")
TOTAL_NEEDED=$((TOTAL_TRAIN + TOTAL_VAL))

# Sampling parameters
TEMP=0.6
TOP_P=0.95
MAX_TOK=4096
BASE_SEED=42
SAMPLES_BENCH=(0 1 2)   # math500/gsm8k
SAMPLES_TRAIN=(0)       # trainmix (1 sample only)
N_MATH=500
N_GSM=1319

echo "=========================================="
echo "final_final gen-think | MODEL=${MODEL} (${SHORT})"
echo "Job ${SLURM_JOB_ID:-local} | Node=$(hostname) | $(date -u)"
echo "TP=${TP} | FMT=${FMT} | trainmix=${TOTAL_NEEDED}"
echo "temp=${TEMP} top_p=${TOP_P} max_tokens=${MAX_TOK} base_seed=${BASE_SEED}"
echo "CACHE=${CACHE}"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | head -1 || true
echo "=========================================="

# Skip vLLM startup entirely if all target files are already present.
ALL_OK=1
for S in "${SAMPLES_TRAIN[@]}"; do
    F="${CACHE}/thinking_${SHORT}_trainmix_temp${TEMP}_max${MAX_TOK}_s${S}.jsonl"
    if [[ ! -f "${F}" ]] || [[ $(wc -l < "${F}") -lt ${TOTAL_NEEDED} ]]; then
        ALL_OK=0; break
    fi
done
if [[ ${ALL_OK} -eq 1 ]]; then
    for S in "${SAMPLES_BENCH[@]}"; do
        for DS_PAIR in "math500 ${N_MATH}" "gsm8k ${N_GSM}"; do
            read -r DS NEED <<< "${DS_PAIR}"
            F="${CACHE}/thinking_${SHORT}_${DS}_temp${TEMP}_max${MAX_TOK}_s${S}.jsonl"
            if [[ ! -f "${F}" ]] || [[ $(wc -l < "${F}") -lt ${NEED} ]]; then
                ALL_OK=0; break 2
            fi
        done
    done
fi
if [[ ${ALL_OK} -eq 1 ]]; then
    echo "All required rollouts already present for ${SHORT}. Nothing to do."
    exit 0
fi

# ---- Start vLLM ----
COMBINED_FILE=$(mktemp /tmp/trainmix_combined_XXXXXX.jsonl)
cat "${TRAIN_FILE}" "${VAL_FILE}" > "${COMBINED_FILE}"
trap "rm -f '${COMBINED_FILE}'" EXIT

# max-model-len 8192 to accommodate prompt + 4096 output tokens
python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --port ${VLLM_PORT} \
    --tensor-parallel-size ${TP} \
    --max-model-len 8192 \
    --gpu-memory-utilization ${GPU_MEM_UTIL:-0.85} \
    --trust-remote-code \
    --enable-prefix-caching \
    --disable-custom-all-reduce 2>&1 &
VLLM_PID=$!

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
    echo "FATAL: vLLM did not start in $((MAX_WAIT*10/60)) min"
    kill ${VLLM_PID} 2>/dev/null || true; wait ${VLLM_PID} 2>/dev/null || true
    exit 1
fi
for i in $(seq 1 60); do
    sleep 5
    OUT=$(curl -s "http://localhost:${VLLM_PORT}/v1/models" 2>/dev/null || echo "")
    if python3 -c "import sys,json; d=json.loads(sys.argv[1]); assert d.get('data')" "${OUT}" 2>/dev/null; then
        echo "[models] /v1/models JSON ok"; break
    fi
done
sleep 10

# ---- Trainmix (1 sample) ----
for S in "${SAMPLES_TRAIN[@]}"; do
    OUTFILE="${CACHE}/thinking_${SHORT}_trainmix_temp${TEMP}_max${MAX_TOK}_s${S}.jsonl"
    EXIST=0; [[ -f "${OUTFILE}" ]] && EXIST=$(wc -l < "${OUTFILE}")
    if [[ ${EXIST} -ge ${TOTAL_NEEDED} ]]; then
        echo "SKIP trainmix_s${S}: ${EXIST} >= ${TOTAL_NEEDED}"; continue
    fi
    echo "--- trainmix s=${S} (have=${EXIST}, need=${TOTAL_NEEDED}) ---"
    cd vllm-serve
    python generate_rollouts.py \
        --base_url "http://localhost:${VLLM_PORT}/v1" \
        --model "${MODEL}" --model_short "${SHORT}" --format "${FMT}" \
        --dataset trainmix --dataset_file "${COMBINED_FILE}" \
        --n_examples ${TOTAL_NEEDED} \
        --max_tokens ${MAX_TOK} --temperature ${TEMP} --top_p ${TOP_P} \
        --seed $((BASE_SEED + S)) --sample_idx ${S} \
        --math_directive_mode auto \
        --output_dir "${CACHE}" --max_concurrent 64 --resume
    cd "${ROOT}"
done

# ---- math500 / gsm8k (3 samples each) ----
for S in "${SAMPLES_BENCH[@]}"; do
    for DS_PAIR in "math500 ${N_MATH}" "gsm8k ${N_GSM}"; do
        read -r DS NEED <<< "${DS_PAIR}"
        OUTFILE="${CACHE}/thinking_${SHORT}_${DS}_temp${TEMP}_max${MAX_TOK}_s${S}.jsonl"
        EXIST=0; [[ -f "${OUTFILE}" ]] && EXIST=$(wc -l < "${OUTFILE}")
        if [[ ${EXIST} -ge ${NEED} ]]; then
            echo "SKIP ${DS}_s${S}: ${EXIST} >= ${NEED}"; continue
        fi
        echo "--- ${DS} s=${S} (have=${EXIST}, need=${NEED}) ---"
        cd vllm-serve
        python generate_rollouts.py \
            --base_url "http://localhost:${VLLM_PORT}/v1" \
            --model "${MODEL}" --model_short "${SHORT}" --format "${FMT}" \
            --dataset "${DS}" --n_examples ${NEED} \
            --max_tokens ${MAX_TOK} --temperature ${TEMP} --top_p ${TOP_P} \
            --seed $((BASE_SEED + S)) --sample_idx ${S} \
            --math_directive_mode always \
            --output_dir "${CACHE}" --max_concurrent 64 --resume
        cd "${ROOT}"
    done
done

kill ${VLLM_PID} 2>/dev/null || true
wait ${VLLM_PID} 2>/dev/null || true

# ── Symlink the produced max=4096 think rollouts into the merged-cache
# dir so eval_final_final.sh's ${EVAL_DIR}/response_cache -> CACHE_MERGED
# symlink picks them up automatically (eval looks for files matching
# thinking_${TS}_${DS}_temp${ROLL_TEMP}_max${ROLL_MAX}_s${S}.jsonl
# inside response_cache).
MERGED="${ROOT}/hybrid/results/response_cache_final_final_merged"
mkdir -p "${MERGED}"
for S in 0; do
    F="${CACHE}/thinking_${SHORT}_trainmix_temp${TEMP}_max${MAX_TOK}_s${S}.jsonl"
    [[ -f "${F}" ]] && ln -sfn "${F}" "${MERGED}/$(basename "${F}")"
done
for S in 0 1 2; do
    for DS in math500 gsm8k; do
        F="${CACHE}/thinking_${SHORT}_${DS}_temp${TEMP}_max${MAX_TOK}_s${S}.jsonl"
        [[ -f "${F}" ]] && ln -sfn "${F}" "${MERGED}/$(basename "${F}")"
    done
done
echo "[merged] linked thinking_${SHORT}_*_max${MAX_TOK}_*.jsonl -> ${MERGED}/"

echo ""
echo "=========================================="
echo "DONE final_final gen-think ${SHORT}  $(date -u)"
echo "Files now in ${CACHE}:"
ls -lh "${CACHE}" | grep "thinking_${SHORT}_.*max${MAX_TOK}" || true
echo "=========================================="
