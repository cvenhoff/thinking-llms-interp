#!/usr/bin/env bash
# eval_qa_instr_hsweep: pure eval for MLPs trained with base_prompt_style=qa_instr.
# Reads vectors from mlp_vectors_qa_instr_h${MLP_HIDDEN}/<CONFIG>/ and writes
# results to mlp_eval_qa_instr_h${MLP_HIDDEN}/<CONFIG>/.
#
# Env vars (required): CONFIG, DATASET, MLP_HIDDEN
#
#SBATCH --job-name=eval-qai-hsweep
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --cpus-per-task=8
#SBATCH --time=20:00:00
#SBATCH --output=/workspace-vast/constantinv/thinking-llms-interp/slurm_logs/final_final/%x-%j.out
#SBATCH --error=/workspace-vast/constantinv/thinking-llms-interp/slurm_logs/final_final/%x-%j.err

set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
ROOT=/workspace-vast/constantinv/thinking-llms-interp
cd "${ROOT}"
mkdir -p "${ROOT}/slurm_logs/final_final"

source .env_exports.sh 2>/dev/null || true
source .venv/bin/activate

: "${CONFIG:?CONFIG env var required}"
: "${DATASET:?DATASET env var required}"
: "${MLP_HIDDEN:?MLP_HIDDEN env var required}"
DECODE_T="${DECODE_T:-0}"
DECODE_SEED="${DECODE_SEED:-0}"
# Optional output tag to separate runs (e.g. _T06 for decode_temperature=0.6),
# so a sampled rerun does not clobber the greedy results.
TAG="${TAG:-}"

declare -A THINK_MODEL BASE_MODEL THINK_SHORT BASE_SHORT STEER_LAYER SAE_LAYER N_CLUSTERS

THINK_MODEL[orz-0.5b]="Open-Reasoner-Zero/Open-Reasoner-Zero-0.5B"
BASE_MODEL[orz-0.5b]="Qwen/Qwen2.5-0.5B"
THINK_SHORT[orz-0.5b]="open-reasoner-zero-0.5b"
BASE_SHORT[orz-0.5b]="qwen2.5-0.5b"
STEER_LAYER[orz-0.5b]=9; SAE_LAYER[orz-0.5b]=8; N_CLUSTERS[orz-0.5b]=10

THINK_MODEL[orz-1.5b]="Open-Reasoner-Zero/Open-Reasoner-Zero-1.5B"
BASE_MODEL[orz-1.5b]="Qwen/Qwen2.5-1.5B"
THINK_SHORT[orz-1.5b]="open-reasoner-zero-1.5b"
BASE_SHORT[orz-1.5b]="qwen2.5-1.5b"
STEER_LAYER[orz-1.5b]=10; SAE_LAYER[orz-1.5b]=4; N_CLUSTERS[orz-1.5b]=10

THINK_MODEL[orz-7b]="Open-Reasoner-Zero/Open-Reasoner-Zero-7B"
BASE_MODEL[orz-7b]="Qwen/Qwen2.5-7B"
THINK_SHORT[orz-7b]="open-reasoner-zero-7b"
BASE_SHORT[orz-7b]="qwen2.5-7b"
STEER_LAYER[orz-7b]=10; SAE_LAYER[orz-7b]=20; N_CLUSTERS[orz-7b]=10

THINK_MODEL[orz-32b]="Open-Reasoner-Zero/Open-Reasoner-Zero-32B"
BASE_MODEL[orz-32b]="Qwen/Qwen2.5-32B"
THINK_SHORT[orz-32b]="open-reasoner-zero-32b"
BASE_SHORT[orz-32b]="qwen2.5-32b"
STEER_LAYER[orz-32b]=24; SAE_LAYER[orz-32b]=27; N_CLUSTERS[orz-32b]=15

THINK_MODEL[r1-14b]="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"
BASE_MODEL[r1-14b]="Qwen/Qwen2.5-14B"
THINK_SHORT[r1-14b]="deepseek-r1-distill-qwen-14b"
BASE_SHORT[r1-14b]="qwen2.5-14b"
STEER_LAYER[r1-14b]=18; SAE_LAYER[r1-14b]=38; N_CLUSTERS[r1-14b]=5

THINK_MODEL[r1-llama8b]="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
BASE_MODEL[r1-llama8b]="Meta-Llama/Llama-3.1-8B"
THINK_SHORT[r1-llama8b]="deepseek-r1-distill-llama-8b"
BASE_SHORT[r1-llama8b]="llama-3.1-8b"
STEER_LAYER[r1-llama8b]=12; SAE_LAYER[r1-llama8b]=6; N_CLUSTERS[r1-llama8b]=15

THINK_MODEL[qwq-32b]="Qwen/QwQ-32B"
BASE_MODEL[qwq-32b]="Qwen/Qwen2.5-32B"
THINK_SHORT[qwq-32b]="qwq-32b"
BASE_SHORT[qwq-32b]="qwen2.5-32b"
STEER_LAYER[qwq-32b]=24; SAE_LAYER[qwq-32b]=27; N_CLUSTERS[qwq-32b]=10

THINK_MODEL[r1-32b]="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"
BASE_MODEL[r1-32b]="Qwen/Qwen2.5-32B"
THINK_SHORT[r1-32b]="deepseek-r1-distill-qwen-32b"
BASE_SHORT[r1-32b]="qwen2.5-32b"
STEER_LAYER[r1-32b]=24; SAE_LAYER[r1-32b]=27; N_CLUSTERS[r1-32b]=15

THINK_MODEL[r1-math1.5b]="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"
BASE_MODEL[r1-math1.5b]="Qwen/Qwen2.5-Math-1.5B"
THINK_SHORT[r1-math1.5b]="deepseek-r1-distill-qwen-1.5b"
BASE_SHORT[r1-math1.5b]="qwen2.5-math-1.5b"
STEER_LAYER[r1-math1.5b]=10; SAE_LAYER[r1-math1.5b]=4; N_CLUSTERS[r1-math1.5b]=15

TM="${THINK_MODEL[$CONFIG]}"
BM="${BASE_MODEL[$CONFIG]}"
TS="${THINK_SHORT[$CONFIG]}"
BS="${BASE_SHORT[$CONFIG]}"
SL="${STEER_LAYER[$CONFIG]}"
SAEL="${SAE_LAYER[$CONFIG]}"
NK="${N_CLUSTERS[$CONFIG]}"
# Optional overrides for ad-hoc layer/cluster sweeps (must match training).
# When set, VARIANT must also be set to read the matching vectors dir.
SL="${STEER_LAYER_OVR:-$SL}"
SAEL="${SAE_LAYER_OVR:-$SAEL}"
NK="${N_CLUSTERS_OVR:-$NK}"
VARIANT="${VARIANT:-}"

case "${DATASET}" in
    math500) TOTAL=500 ;;
    gsm8k)   TOTAL=1319 ;;
    *) echo "FATAL: unsupported DATASET=${DATASET}"; exit 2 ;;
esac

echo "=========================================="
echo "qa_instr-hsweep pure eval | CONFIG=${CONFIG} DATASET=${DATASET} H=${MLP_HIDDEN} decode_T=${DECODE_T} TAG=${TAG}"
echo "Job ${SLURM_JOB_ID:-local} | Node=$(hostname) | $(date -u)"
echo "Base prompt: qa_instr"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader || true
echo "=========================================="

VEC_DIR="${ROOT}/mlp_vectors_qa_instr${VARIANT}_h${MLP_HIDDEN}/${CONFIG}"
EVAL_DIR="${ROOT}/mlp_eval_qa_instr${VARIANT}_h${MLP_HIDDEN}${TAG}/${CONFIG}"
CACHE_THINK="${ROOT}/hybrid/results/response_cache_final"
CACHE_BASE_QAI="${ROOT}/hybrid/results/response_cache_base_qa_instr"
PAIR_MERGED="${ROOT}/hybrid/results/response_cache_qa_instr${VARIANT}_h${MLP_HIDDEN}${TAG}_merged/${CONFIG}"
mkdir -p "${EVAL_DIR}" "${PAIR_MERGED}"

ROLL_TEMP="0.6"
ROLL_MAX=2048
BASE_TEMP="0"
BASE_MAX=2048

echo "[merged-cache] populating ${PAIR_MERGED}"
for S in 0 1 2; do
    SRC="${CACHE_THINK}/thinking_${TS}_math500_temp${ROLL_TEMP}_max${ROLL_MAX}_s${S}.jsonl"
    ln -sfn "${SRC}" "${PAIR_MERGED}/$(basename "${SRC}")"
    SRC="${CACHE_THINK}/thinking_${TS}_gsm8k_temp${ROLL_TEMP}_max${ROLL_MAX}_s${S}.jsonl"
    ln -sfn "${SRC}" "${PAIR_MERGED}/$(basename "${SRC}")"
done
for DS in math500 gsm8k; do
    SRC="${CACHE_BASE_QAI}/base_qa_instr_${BS}_${DS}_temp${BASE_TEMP}_max${BASE_MAX}.jsonl"
    DST="${PAIR_MERGED}/base_${BS}_${DS}_temp${BASE_TEMP}_max${BASE_MAX}.jsonl"
    if [[ ! -f "${SRC}" ]]; then
        echo "FATAL: missing ${SRC}"
        exit 2
    fi
    ln -sfn "${SRC}" "${DST}"
done

[[ -f "${VEC_DIR}/cat_coef_mlp.pt" ]] || { echo "FATAL: missing ${VEC_DIR}/cat_coef_mlp.pt"; exit 2; }
[[ -f "${VEC_DIR}/mlp_config.json" ]] || { echo "FATAL: missing ${VEC_DIR}/mlp_config.json"; exit 2; }

for S in 0 1 2; do
    F="${CACHE_THINK}/thinking_${TS}_${DATASET}_temp${ROLL_TEMP}_max${ROLL_MAX}_s${S}.jsonl"
    [[ -f "${F}" ]] || { echo "FATAL: missing think rollout s${S} ${F}"; exit 2; }
done
BASE_F="${PAIR_MERGED}/base_${BS}_${DATASET}_temp${BASE_TEMP}_max${BASE_MAX}.jsonl"
[[ -f "${BASE_F}" ]] || { echo "FATAL: missing base rollout ${BASE_F}"; exit 2; }
echo "Rollouts verified."

N_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)
echo "[gpus] visible to eval: ${N_GPUS}"
TWO_GPU_FLAG=""
case "${CONFIG}" in
    *32b)        HBS=8 ;;
    *14b)        HBS=32 ;;
    *8b)         HBS=64 ;;
    *7b)         HBS=32 ;;
    *)           HBS=32 ;;
esac
if [[ ${N_GPUS} -ge 2 ]]; then
    TWO_GPU_FLAG="--two_gpu_split"
fi
HBS="${HBS_OVERRIDE:-${HBS}}"
echo "[hybrid_eval] two_gpu='${TWO_GPU_FLAG}' hybrid_gen_batch_size=${HBS}"

GPU_MON_LOG="${ROOT}/slurm_logs/final_final/gpu-mem-eval-qai-h${MLP_HIDDEN}${TAG}-${CONFIG}-${DATASET}-${SLURM_JOB_ID:-local}.log"
(while true; do
    echo "--- $(date -u) ---" >> "${GPU_MON_LOG}"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader >> "${GPU_MON_LOG}" 2>&1
    sleep 30
done) &
GPU_MON_PID=$!
trap "kill ${GPU_MON_PID} 2>/dev/null" EXIT

JUDGE_MODEL="${JUDGE_MODEL:-anthropic/claude-sonnet-4-6}"
echo "[judge] model=${JUDGE_MODEL}"

cd hybrid
python hybrid_eval.py \
    --dataset "${DATASET}" \
    --thinking_model "${TM}" \
    --base_model "${BM}" \
    --sae_layer ${SAEL} \
    --n_clusters ${NK} \
    --dom_vectors_dir "${VEC_DIR}" \
    --old_vectors_dir "${VEC_DIR}" \
    --old_vectors_layer ${SL} \
    --coef_select mlp \
    --mlp_coef_path "${VEC_DIR}/cat_coef_mlp.pt" \
    --mlp_config_path "${VEC_DIR}/mlp_config.json" \
    --max_new_tokens 2048 \
    --max_thinking_tokens 2048 \
    --temperature 0.0 \
    --decode_temperature ${DECODE_T} \
    --decode_seed ${DECODE_SEED} \
    --hybrid_gen_batch_size ${HBS} \
    --judge_repetitions 3 \
    --judge_model "${JUDGE_MODEL}" \
    --results_dir "${EVAL_DIR}" \
    --response_cache_dir "${PAIR_MERGED}" \
    --results_suffix final \
    --think_cache_temp_label "${ROLL_TEMP}" \
    --think_cache_max_tokens ${ROLL_MAX} \
    --think_cache_sample_idx 0 \
    --base_cache_temp_label "${BASE_TEMP}" \
    --base_cache_max_tokens ${BASE_MAX} \
    --base_cache_sample_idx -1 \
    --hybrid_cache_sample_idx -1 \
    --think_prompt_family auto \
    --math_directive \
    --base_prompt_style qa_instr \
    --pure_steer_base_eos \
    ${TWO_GPU_FLAG}
cd "${ROOT}"

echo ""
echo "--- judging think samples s1, s2 ---"
python mlp_pipeline/canonical/judge_extra_think_samples.py \
    --cache_dir "${CACHE_THINK}" \
    --think_short "${TS}" \
    --base_id "${BS}" \
    --dataset "${DATASET}" \
    --temp_label "${ROLL_TEMP}" \
    --max_tokens ${ROLL_MAX} \
    --sample_ids 1,2 \
    --judge_repetitions 3 \
    --judge_model "${JUDGE_MODEL}" \
    --out_dir "${EVAL_DIR}" \
    --results_suffix final

echo ""
echo "--- aggregating ---"
python mlp_pipeline/canonical/aggregate_samples_final.py \
    --eval_dir "${EVAL_DIR}" \
    --base_id "${BS}" \
    --dataset "${DATASET}" \
    --suffix final

echo ""
echo "=========================================="
echo "DONE eval-qai-hsweep h${MLP_HIDDEN} ${CONFIG} ${DATASET}  $(date -u)"
ls -lh "${EVAL_DIR}" 2>/dev/null | grep -E "(judge|hybrid_summary|rolling)" || true
echo "=========================================="
