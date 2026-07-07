#!/usr/bin/env bash
# train_qa_instr_hsweep: train MLPs with --base_prompt_style qa_instr
# (prompt: "Answer the following question:\nQ: {q}\nA:")
# Artifacts: mlp_vectors_qa_instr_h${MLP_HIDDEN}/<CONFIG>/.
#
# Env vars (required): CONFIG, MLP_HIDDEN
#
#SBATCH --job-name=train-qai-hsweep
#SBATCH --partition=general
#SBATCH --qos=high
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
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
: "${MLP_HIDDEN:?MLP_HIDDEN env var required}"

echo "=========================================="
echo "qa_instr-hsweep train | CONFIG=${CONFIG}  MLP_HIDDEN=${MLP_HIDDEN}"
echo "Job ${SLURM_JOB_ID:-local} | Node=$(hostname) | $(date -u)"
echo "Base prompt: qa_instr  ('Answer the following question:\\nQ: {q}\\nA:')"
nvidia-smi --query-gpu=index,name,memory.total --format=csv,noheader | head -1 || true
echo "=========================================="

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
# Optional overrides for ad-hoc layer/cluster sweeps. When set, VARIANT must
# also be set so artifacts land in a separate dir (do NOT clobber canonical
# vectors). Defaults preserve the per-config values above.
SL="${STEER_LAYER_OVR:-$SL}"
SAEL="${SAE_LAYER_OVR:-$SAEL}"
NK="${N_CLUSTERS_OVR:-$NK}"
VARIANT="${VARIANT:-}"

CACHE="${ROOT}/hybrid/results/response_cache_final"
TRAIN_FILE="${ROOT}/data/training_mix_v1/train.jsonl"
VAL_FILE="${ROOT}/data/training_mix_v1/val.jsonl"
SAVE_DIR="${ROOT}/mlp_vectors_qa_instr${VARIANT}_h${MLP_HIDDEN}/${CONFIG}"
# Optional: point eval (e.g. --eval_percat_only) at a different vector dir
# (backup / alt run) without clobbering the canonical path.
SAVE_DIR="${SAVE_DIR_OVR:-$SAVE_DIR}"
mkdir -p "${SAVE_DIR}"

ROLL_TEMP="0.6"
ROLL_MAX=2048
ROLL_S=0

echo "Think: ${TM} (${TS})"
echo "Base:  ${BM} (${BS})"
echo "Steer layer=${SL}  SAE: layer=${SAEL} K=${NK}"
echo "SAVE_DIR=${SAVE_DIR}"
echo ""

TRACE_FILE="${CACHE}/thinking_${TS}_trainmix_temp${ROLL_TEMP}_max${ROLL_MAX}_s${ROLL_S}.jsonl"
TOTAL_NEEDED=$(( $(wc -l < "${TRAIN_FILE}") + $(wc -l < "${VAL_FILE}") ))
[[ -f "${TRACE_FILE}" ]] || { echo "FATAL: missing ${TRACE_FILE}"; exit 2; }
TRACE_COUNT=$(wc -l < "${TRACE_FILE}")
if [[ ${TRACE_COUNT} -lt ${TOTAL_NEEDED} ]]; then
    echo "FATAL: only ${TRACE_COUNT}/${TOTAL_NEEDED} trainmix rollouts."
    exit 2
fi
echo "Trainmix rollouts verified: ${TRACE_COUNT} >= ${TOTAL_NEEDED}"

for DS in math500 gsm8k; do
    F="${CACHE}/thinking_${TS}_${DS}_temp${ROLL_TEMP}_max${ROLL_MAX}_s${ROLL_S}.jsonl"
    [[ -f "${F}" ]] || { echo "FATAL: missing ${F}"; exit 2; }
    echo "${DS} s0 rollouts: $(wc -l < "${F}") rows"
done

DISAGREE_CACHE="${SAVE_DIR}/disagree_cache.pt"
# qa_instr disagree cache is base-prompt-specific; build fresh.
TRAIN_BS="${TRAIN_BS:-4}"

N_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)
echo "[gpus] N_GPUS=${N_GPUS}  TRAIN_BS=${TRAIN_BS}"

GPU_MON_LOG="${ROOT}/slurm_logs/final_final/gpu-mem-train-qai-h${MLP_HIDDEN}-${CONFIG}-${SLURM_JOB_ID:-local}.log"
(while true; do
    echo "--- $(date -u) ---" >> "${GPU_MON_LOG}"
    nvidia-smi --query-gpu=index,memory.used,memory.total,utilization.gpu \
        --format=csv,noheader >> "${GPU_MON_LOG}" 2>&1
    sleep 30
done) &
GPU_MON_PID=$!
trap "kill ${GPU_MON_PID} 2>/dev/null" EXIT

cd train-vectors

COMMON_ARGS=(
    --base_model "${BM}"
    --thinking_model "${TM}"
    --thinking_model_short "${TS}"
    --steer_layer ${SL}
    --sae_layer ${SAEL}
    --sae_n_clusters ${NK}
    --no_bias
    --mlp_coef
    --mlp_hidden_dim ${MLP_HIDDEN}
    --mlp_lr 1e-3
    --mlp_grad_clip 1.0
    --cats_lr 1e-2
    --cats_epochs 10
    --patience 5
    --weight_decay 0.01
    --train_batch_size ${TRAIN_BS}
    --collect_batch_size 8
    --max_seq_len 4096
    --max_positions_per_example 64
    --train_data_file "${TRAIN_FILE}"
    --val_data_file "${VAL_FILE}"
    --oos_cache_dir "${CACHE}"
    --oos_math500_n 500
    --oos_gsm8k_n 500
    --rollouts_temp_label "${ROLL_TEMP}"
    --rollouts_max_tokens ${ROLL_MAX}
    --rollouts_sample_idx ${ROLL_S}
    --think_prompt_family auto
    --math_directive_mode auto
    --base_prompt_style qa_instr
    --save_dir "${SAVE_DIR}"
    --seed ${SEED_OVR:-42}
    --disagree_cache "${DISAGREE_CACHE}"
    --save_per_epoch_ckpts
)

EVAL_PERCAT="${EVAL_PERCAT:-0}"
# Idempotency guard: if a COMPLETE training run already exists (best_meta.json is
# only written after the full schedule finishes), do not retrain. This makes
# retries after a spurious non-zero srun exit (e.g. rc=127 post-completion) a
# fast no-op instead of a wasteful full retrain that would also clobber vectors
# the selection chains are already reading.
if [[ "${EVAL_PERCAT}" != "1" && -f "${SAVE_DIR}/best_meta.json" \
      && -f "${SAVE_DIR}/cat_coef_mlp.pt" && -f "${SAVE_DIR}/mlp_config.json" ]]; then
    echo "[idempotent] complete vectors already present in ${SAVE_DIR}; skipping training."
    echo "=========================================="
    echo "DONE train-qai-hsweep h${MLP_HIDDEN} ${CONFIG} (skipped, already complete) $(date -u)"
    echo "=========================================="
    exit 0
fi
if [[ "${EVAL_PERCAT}" == "1" ]]; then
    # Recompute per-category holdout CE from saved vectors. Run single-process
    # (model sharded via device_map=auto) so big models don't OOM. Needs the
    # disagree cache present (built fresh here if missing).
    echo "[eval_percat] recomputing holdout metrics (single-process)"
    # Default to trainmix_holdout ONLY (cheap vector selection); callers can
    # widen via EVAL_PERCAT_HOLDOUTS if they explicitly want the OOS sets.
    python optimize_correction_vectors.py "${COMMON_ARGS[@]}" \
        --eval_percat_only --eval_percat_out "${EVAL_PERCAT_OUT:-per_cat_ce_eval.json}" \
        --eval_percat_holdouts "${EVAL_PERCAT_HOLDOUTS:-trainmix_holdout}"
elif [[ ${N_GPUS} -ge 2 ]]; then
    if [[ ! -f "${DISAGREE_CACHE}" ]]; then
        echo "[Phase A] single-process disagreement collection ..."
        python optimize_correction_vectors.py "${COMMON_ARGS[@]}" --collect_only
    else
        echo "[Phase A] disagree cache present, skipping collection."
    fi
    : "${MASTER_PORT:=$((29500 + RANDOM % 1000))}"
    echo "[Phase D] DDP train (${N_GPUS} GPUs)  MASTER_PORT=${MASTER_PORT}"
    torchrun --nproc_per_node=${N_GPUS} \
             --master_port=${MASTER_PORT} \
             optimize_correction_vectors.py "${COMMON_ARGS[@]}" --distributed
else
    python optimize_correction_vectors.py "${COMMON_ARGS[@]}"
fi
cd "${ROOT}"

echo ""
echo "Train complete. Artifacts in ${SAVE_DIR}:"
ls -lh "${SAVE_DIR}" || true
echo ""
echo "=========================================="
echo "DONE train-qai-hsweep h${MLP_HIDDEN} ${CONFIG}  $(date -u)"
echo "=========================================="
