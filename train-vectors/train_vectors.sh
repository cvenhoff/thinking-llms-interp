#!/usr/bin/env bash
# Train one set of category correction vectors (a coefficient MLP over the
# thinking model's SAE categories) for a single model pair, from cached rollouts.
#
# Env:
#   CONFIG    one of the nine pairs (see configs.sh)
#   SAVE_DIR  where to write vectors (default artifacts/mlp_vectors_qa_instr_h512/<CONFIG>)
#   SEED      RNG seed (default 42; best-of-3 uses 42/43/44)
#
# Uses all visible GPUs (DDP via torchrun) when more than one is present.
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
ROOT="${THINKING_LLMS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
cd "${ROOT}"
source .env_exports.sh 2>/dev/null || true
source .venv/bin/activate 2>/dev/null || true
source "${ROOT}/configs.sh"

: "${CONFIG:?CONFIG required}"
cfg_load "${CONFIG}"
SEED="${SEED:-42}"
SAVE_DIR="${SAVE_DIR:-${ROOT}/artifacts/mlp_vectors_qa_instr_h${MLP_HIDDEN}/${CONFIG}}"
CACHE="${ROOT}/hybrid/results/response_cache_final"
TRAIN_FILE="${ROOT}/data/training_mix_v1/train.jsonl"
VAL_FILE="${ROOT}/data/training_mix_v1/val.jsonl"
DISAGREE_CACHE="${SAVE_DIR}/disagree_cache.pt"
mkdir -p "${SAVE_DIR}"

echo "== train-vectors | ${CONFIG} seed=${SEED} steer=${SL} sae=${SAEL} k=${NK} -> ${SAVE_DIR} | $(date -u) =="

# Idempotency: a complete run writes best_meta.json + cat_coef_mlp.pt last.
if [[ -f "${SAVE_DIR}/best_meta.json" && -f "${SAVE_DIR}/cat_coef_mlp.pt" && -f "${SAVE_DIR}/mlp_config.json" ]]; then
    echo "vectors already complete in ${SAVE_DIR}; skipping."; exit 0
fi

# Verify the cached rollouts this run reads exist.
TRACE="${CACHE}/thinking_${TS}_trainmix_temp0.6_max2048_s0.jsonl"
NEED=$(( $(wc -l < "${TRAIN_FILE}") + $(wc -l < "${VAL_FILE}") ))
[[ -f "${TRACE}" && $(wc -l < "${TRACE}") -ge ${NEED} ]] || { echo "FATAL: missing/short trainmix rollouts ${TRACE}"; exit 2; }
for DS in math500 gsm8k; do
    [[ -f "${CACHE}/thinking_${TS}_${DS}_temp0.6_max2048_s0.jsonl" ]] || { echo "FATAL: missing ${DS} rollouts"; exit 2; }
done

ARGS=(
    --base_model "${BM}" --thinking_model "${TM}" --thinking_model_short "${TS}"
    --steer_layer ${SL} --sae_layer ${SAEL} --sae_n_clusters ${NK}
    --no_bias --mlp_coef --mlp_hidden_dim ${MLP_HIDDEN} --mlp_lr 1e-3 --mlp_grad_clip 1.0
    --cats_lr 1e-2 --cats_epochs 10 --patience 5 --weight_decay 0.01
    --train_batch_size ${BS_TRAIN} --collect_batch_size 8 --max_seq_len 4096
    --max_positions_per_example 64
    --train_data_file "${TRAIN_FILE}" --val_data_file "${VAL_FILE}"
    --oos_cache_dir "${CACHE}" --oos_math500_n 500 --oos_gsm8k_n 500
    --rollouts_temp_label 0.6 --rollouts_max_tokens 2048 --rollouts_sample_idx 0
    --think_prompt_family auto --math_directive_mode auto --base_prompt_style qa_instr
    --save_dir "${SAVE_DIR}" --seed ${SEED} --disagree_cache "${DISAGREE_CACHE}"
    --save_per_epoch_ckpts
)

N_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)
cd train-vectors
if [[ ${N_GPUS} -ge 2 ]]; then
    [[ -f "${DISAGREE_CACHE}" ]] || { echo "[collect] single-process disagreement collection"; python optimize_correction_vectors.py "${ARGS[@]}" --collect_only; }
    echo "[train] DDP on ${N_GPUS} GPUs"
    torchrun --nproc_per_node=${N_GPUS} --master_port=$((29500 + RANDOM % 1000)) \
        optimize_correction_vectors.py "${ARGS[@]}" --distributed
else
    python optimize_correction_vectors.py "${ARGS[@]}"
fi
cd "${ROOT}"
echo "== DONE train-vectors ${CONFIG} seed=${SEED} $(date -u) =="
