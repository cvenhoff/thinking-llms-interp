#!/usr/bin/env bash
# Up-to-date negative-control ablations, run against the CURRENT canonical
# vectors (mlp_vectors_qa_instr_holdoutsel_h512) with the exact qa_instr eval
# setup (greedy decode, Claude judge x3, pure_steer_base_eos). Mirrors the
# original single-hybrid + 3-judge ablation protocol (base/think unchanged;
# only the hybrid arm's steering is perturbed).
#
#   ABLATION (runtime flags on the real trained vectors):
#     randcat            : --random_firing               (random SAE category)
#     randV              : --randomize_vectors            (random directions)
#     mlponly            : --randomize_vectors --random_firing (only MLP magnitude real)
#     randpos            : --random_steer_prob <p>        (random positions, p=empirical)
#
# Env (required): CONFIG (orz-1.5b|orz-32b), DATASET (math500|gsm8k), ABLATION
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
ROOT=/workspace-vast/constantinv/thinking-llms-interp
cd "${ROOT}"; mkdir -p "${ROOT}/slurm_logs/final_final"
source .env_exports.sh 2>/dev/null || true
source .venv/bin/activate

: "${CONFIG:?}"; : "${DATASET:?}"; : "${ABLATION:?}"
MLP_HIDDEN=512

declare -A THINK_MODEL BASE_MODEL THINK_SHORT BASE_SHORT STEER_LAYER SAE_LAYER N_CLUSTERS
THINK_MODEL[orz-1.5b]="Open-Reasoner-Zero/Open-Reasoner-Zero-1.5B"; BASE_MODEL[orz-1.5b]="Qwen/Qwen2.5-1.5B"; THINK_SHORT[orz-1.5b]=open-reasoner-zero-1.5b; BASE_SHORT[orz-1.5b]=qwen2.5-1.5b; STEER_LAYER[orz-1.5b]=10; SAE_LAYER[orz-1.5b]=4; N_CLUSTERS[orz-1.5b]=10
THINK_MODEL[orz-32b]="Open-Reasoner-Zero/Open-Reasoner-Zero-32B"; BASE_MODEL[orz-32b]="Qwen/Qwen2.5-32B"; THINK_SHORT[orz-32b]=open-reasoner-zero-32b; BASE_SHORT[orz-32b]=qwen2.5-32b; STEER_LAYER[orz-32b]=24; SAE_LAYER[orz-32b]=27; N_CLUSTERS[orz-32b]=15

TM="${THINK_MODEL[$CONFIG]}"; BM="${BASE_MODEL[$CONFIG]}"; TS="${THINK_SHORT[$CONFIG]}"; BS="${BASE_SHORT[$CONFIG]}"
SL="${STEER_LAYER[$CONFIG]}"; SAEL="${SAE_LAYER[$CONFIG]}"; NK="${N_CLUSTERS[$CONFIG]}"

# empirical steered-position fraction (mean of frac_steered) from the canonical
# holdoutsel run -- used as p for the random-position control.
declare -A PSTEER
PSTEER[orz-1.5b/math500]=0.0493; PSTEER[orz-1.5b/gsm8k]=0.0885
PSTEER[orz-32b/math500]=0.1225; PSTEER[orz-32b/gsm8k]=0.1151

case "${DATASET}" in math500) TOTAL=500;; gsm8k) TOTAL=1319;; *) echo "FATAL: bad DATASET"; exit 2;; esac

EXTRA=()
case "${ABLATION}" in
    randcat)             EXTRA=(--random_firing --random_seed 0) ;;
    randV)               EXTRA=(--randomize_vectors --random_seed 0) ;;
    mlponly)             EXTRA=(--randomize_vectors --random_firing --random_seed 0) ;;
    randpos)             P="${PSTEER[${CONFIG}/${DATASET}]:?no p for ${CONFIG}/${DATASET}}"; EXTRA=(--random_steer_prob "${P}" --random_seed 0) ;;
    *) echo "FATAL: unknown ABLATION=${ABLATION}"; exit 2 ;;
esac

VEC_DIR="${ROOT}/mlp_vectors_qa_instr_holdoutsel_h${MLP_HIDDEN}/${CONFIG}"
EVAL_DIR="${ROOT}/mlp_eval_qa_instr_holdoutsel_ablations/${CONFIG}-${ABLATION}/${DATASET}"
CACHE_THINK="${ROOT}/hybrid/results/response_cache_final"
CACHE_BASE_QAI="${ROOT}/hybrid/results/response_cache_base_qa_instr"
PAIR_MERGED="${ROOT}/hybrid/results/response_cache_holdoutsel_abl_merged/${CONFIG}"
mkdir -p "${EVAL_DIR}" "${PAIR_MERGED}"

ROLL_TEMP="0.6"; ROLL_MAX=2048; BASE_TEMP="0"; BASE_MAX=2048

echo "== holdoutsel ABLATION=${ABLATION} | CONFIG=${CONFIG} DATASET=${DATASET} | $(date -u) =="
echo "   VEC_DIR=${VEC_DIR}"; echo "   EXTRA=${EXTRA[*]}"

# stage caches (think s0/1/2 + base qa_instr greedy) into merged dir
for S in 0 1 2; do
    ln -sfn "${CACHE_THINK}/thinking_${TS}_${DATASET}_temp${ROLL_TEMP}_max${ROLL_MAX}_s${S}.jsonl" \
            "${PAIR_MERGED}/thinking_${TS}_${DATASET}_temp${ROLL_TEMP}_max${ROLL_MAX}_s${S}.jsonl"
done
SRC_BASE="${CACHE_BASE_QAI}/base_qa_instr_${BS}_${DATASET}_temp${BASE_TEMP}_max${BASE_MAX}.jsonl"
[[ -f "${SRC_BASE}" ]] || { echo "FATAL: missing base ${SRC_BASE}"; exit 2; }
ln -sfn "${SRC_BASE}" "${PAIR_MERGED}/base_${BS}_${DATASET}_temp${BASE_TEMP}_max${BASE_MAX}.jsonl"
[[ -f "${VEC_DIR}/cat_coef_mlp.pt" ]] || { echo "FATAL: missing ${VEC_DIR}/cat_coef_mlp.pt"; exit 2; }
for S in 0 1 2; do [[ -f "${PAIR_MERGED}/thinking_${TS}_${DATASET}_temp${ROLL_TEMP}_max${ROLL_MAX}_s${S}.jsonl" ]] || { echo "FATAL: missing think s${S}"; exit 2; }; done
echo "Rollouts verified."

N_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)
TWO=""; [[ ${N_GPUS} -ge 2 ]] && TWO="--two_gpu_split"
case "${CONFIG}" in *32b) HBS=8;; *) HBS=32;; esac
HBS="${HBS_OVERRIDE:-${HBS}}"
JUDGE_MODEL="${JUDGE_MODEL:-anthropic/claude-sonnet-4-6}"
echo "[cfg] gpus=${N_GPUS} hbs=${HBS} steer=${SL} sae=${SAEL} k=${NK}"

cd hybrid
python hybrid_eval.py \
    --dataset "${DATASET}" \
    --thinking_model "${TM}" --base_model "${BM}" \
    --sae_layer ${SAEL} --n_clusters ${NK} \
    --dom_vectors_dir "${VEC_DIR}" --old_vectors_dir "${VEC_DIR}" --old_vectors_layer ${SL} \
    --coef_select mlp --mlp_coef_path "${VEC_DIR}/cat_coef_mlp.pt" --mlp_config_path "${VEC_DIR}/mlp_config.json" \
    --max_new_tokens 2048 --max_thinking_tokens 2048 \
    --temperature 0.0 --decode_temperature 0 \
    --hybrid_gen_batch_size ${HBS} \
    --judge_repetitions 3 --judge_model "${JUDGE_MODEL}" \
    --results_dir "${EVAL_DIR}" --response_cache_dir "${PAIR_MERGED}" --results_suffix "abl_${ABLATION}" \
    --think_cache_temp_label "${ROLL_TEMP}" --think_cache_max_tokens ${ROLL_MAX} --think_cache_sample_idx 0 \
    --base_cache_temp_label "${BASE_TEMP}" --base_cache_max_tokens ${BASE_MAX} --base_cache_sample_idx -1 \
    --hybrid_cache_sample_idx -1 \
    --think_prompt_family auto --math_directive --base_prompt_style qa_instr \
    --pure_steer_base_eos "${EXTRA[@]}" ${TWO}
cd "${ROOT}"
echo "== DONE ablation ${ABLATION} ${CONFIG} ${DATASET} $(date -u) =="
ls -lh "${EVAL_DIR}" 2>/dev/null | grep -E "judge_reps" || true
