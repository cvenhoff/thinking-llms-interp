#!/usr/bin/env bash
# Boot vLLM for ONE model and generate hendrycks-MATH holdout rollouts for a
# single role (base|think), using the SAME recipe as math500/gsm8k:
#   base : qa_instr prompt ("Answer the following question:\nQ:...\nA:"),
#          greedy temp0, 1 rollout   (via gen_base_qa_instr.py)
#   think: chat template + math directive, temp0.6 top_p0.95, samples s0/s1/s2
#          (via vllm-serve/generate_rollouts.py, --generation-config vllm)
#
# Required env: ROLE (base|think), MODEL, SHORT, TP, VLLM_PORT
# Optional env: MAX_TOK(2048), MAX_MODEL_LEN, GPU_MEM_UTIL, N_EX, BASE_SEED(42)
set -euo pipefail
export PYTHONUNBUFFERED=1
ROOT=/workspace-vast/constantinv/thinking-llms-interp
cd "${ROOT}"; mkdir -p "${ROOT}/slurm_logs/final_final"
set -a; source .env 2>/dev/null || true; set +a
source .env_exports.sh 2>/dev/null || true
source .venv_vllm/bin/activate
export VLLM_ALLOW_LONG_MAX_MODEL_LEN=1

: "${ROLE:?}"; : "${MODEL:?}"; : "${SHORT:?}"; : "${TP:?}"; VLLM_PORT="${VLLM_PORT:-9700}"
MAX_TOK="${MAX_TOK:-2048}"; MAX_MODEL_LEN="${MAX_MODEL_LEN:-8192}"
BASE_SEED="${BASE_SEED:-42}"
DATASET="hendrycks_holdout"
GOLD="${ROOT}/data/hendrycks_holdout_eval/eval.jsonl"
CACHE="${ROOT}/hybrid/results/response_cache_hendrycks_holdout"
mkdir -p "${CACHE}"
N_EX="${N_EX:-$(wc -l < "${GOLD}")}"

echo "== hendrycks-holdout gen | ROLE=${ROLE} MODEL=${MODEL} (${SHORT}) TP=${TP} port=${VLLM_PORT} N=${N_EX} | $(date -u) =="
[[ -f "${GOLD}" ]] || { echo "FATAL: missing gold ${GOLD}"; exit 2; }

if [[ "${ROLE}" == "base" ]]; then
    SAMPLES="-1"
    OUT="${CACHE}/base_qa_instr_${SHORT}_${DATASET}_temp0_max${MAX_TOK}.jsonl"
    DST="${CACHE}/base_${SHORT}_${DATASET}_temp0_max${MAX_TOK}.jsonl"
    if [[ -f "${DST}" ]] && [[ $(wc -l < "${DST}") -ge ${N_EX} ]]; then
        echo "base rollouts complete ($(wc -l < "${DST}")); nothing to do."; exit 0; fi
else
    SAMPLES="0 1 2"
    ALL_OK=1
    for S in ${SAMPLES}; do
        F="${CACHE}/thinking_${SHORT}_${DATASET}_temp0.6_max${MAX_TOK}_s${S}.jsonl"
        { [[ -f "${F}" ]] && [[ $(wc -l < "${F}") -ge ${N_EX} ]]; } || { ALL_OK=0; break; }
    done
    [[ ${ALL_OK} -eq 1 ]] && { echo "think rollouts complete (s0,s1,s2); nothing to do."; exit 0; }
fi

python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" --port ${VLLM_PORT} \
    --tensor-parallel-size ${TP} --max-model-len ${MAX_MODEL_LEN} \
    --generation-config vllm \
    --gpu-memory-utilization ${GPU_MEM_UTIL:-0.90} \
    --trust-remote-code --enable-prefix-caching \
    --disable-custom-all-reduce 2>&1 &
VLLM_PID=$!
trap "kill ${VLLM_PID} 2>/dev/null || true" EXIT

READY=0; MAX_WAIT=$((TP > 1 ? 480 : 240))
for i in $(seq 1 ${MAX_WAIT}); do
    sleep 10
    curl -s "http://localhost:${VLLM_PORT}/health" >/dev/null 2>&1 && { echo "[health] ready after $((i*10))s"; READY=1; break; }
    kill -0 ${VLLM_PID} 2>/dev/null || { echo "ERROR: vLLM died"; break; }
done
# transient boot failure -> non-FATAL exit so the chain retries
[[ ${READY} -eq 1 ]] || { echo "ERR-RETRY: vLLM did not start (transient)"; exit 3; }
sleep 10

if [[ "${ROLE}" == "base" ]]; then
    echo "--- base (qa_instr) ${SHORT} ---"
    python mlp_pipeline/canonical/gen_base_qa_instr.py \
        --base_url "http://localhost:${VLLM_PORT}/v1" \
        --base_model "${MODEL}" --base_short "${SHORT}" \
        --dataset "${DATASET}" --questions_file "${GOLD}" --n_examples ${N_EX} \
        --max_tokens ${MAX_TOK} --temperature 0.0 --top_p 1.0 \
        --output_dir "${CACHE}" --max_concurrent 64 --resume
    GOT=$([[ -f "${OUT}" ]] && wc -l < "${OUT}" || echo 0)
    [[ ${GOT} -ge ${N_EX} ]] || { echo "FATAL: only ${GOT}/${N_EX} base rollouts"; exit 2; }
    ln -sfn "${OUT}" "${DST}"
    echo "== DONE base ${SHORT}: ${GOT} -> $(basename "${DST}") =="
else
    cd "${ROOT}/vllm-serve"
    for S in ${SAMPLES}; do
        F="${CACHE}/thinking_${SHORT}_${DATASET}_temp0.6_max${MAX_TOK}_s${S}.jsonl"
        E=0; [[ -f "${F}" ]] && E=$(wc -l < "${F}")
        [[ ${E} -ge ${N_EX} ]] && { echo "SKIP think s${S}: ${E}>=${N_EX}"; continue; }
        echo "--- think ${SHORT} s=${S} (have=${E}) ---"
        python generate_rollouts.py \
            --base_url "http://localhost:${VLLM_PORT}/v1" \
            --model "${MODEL}" --model_short "${SHORT}" --role thinking \
            --dataset "${DATASET}" --dataset_file "${GOLD}" --n_examples ${N_EX} \
            --max_tokens ${MAX_TOK} --temperature 0.6 --top_p 0.95 \
            --seed $((BASE_SEED + S)) --sample_idx ${S} \
            --math_directive_mode always \
            --preview_first_n 2 \
            --output_dir "${CACHE}" --max_concurrent 64 --resume
    done
    cd "${ROOT}"
    for S in ${SAMPLES}; do
        F="${CACHE}/thinking_${SHORT}_${DATASET}_temp0.6_max${MAX_TOK}_s${S}.jsonl"
        G=$([[ -f "${F}" ]] && wc -l < "${F}" || echo 0)
        [[ ${G} -ge ${N_EX} ]] || { echo "FATAL: think s${S} only ${G}/${N_EX}"; exit 2; }
    done
    echo "== DONE think ${SHORT}: s0/s1/s2 complete =="
fi
