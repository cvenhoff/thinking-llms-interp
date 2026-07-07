#!/usr/bin/env bash
# Hybrid eval on the hendrycks-MATH holdout (~1000 Q, disjoint from train/val +
# math500), same recipe as the math500/gsm8k OOD evals:
#   base : qa_instr prompt, greedy temp0, max2048
#   think: chat template + math directive, temp0.6, s0/s1/s2 (3-sample mean)
#   judge: anthropic/claude-sonnet-4-6, 3 repetitions, math branch
# Reads selected vectors from mlp_vectors_qa_instr${VARIANT}_h${MLP_HIDDEN}/<CONFIG>.
#
# Env (required): CONFIG, MLP_HIDDEN ; optional VARIANT (e.g. _holdoutsel), HBS_OVERRIDE
set -euo pipefail
export PYTHONUNBUFFERED=1
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
ROOT=/workspace-vast/constantinv/thinking-llms-interp
cd "${ROOT}"; mkdir -p "${ROOT}/slurm_logs/final_final"
source .env_exports.sh 2>/dev/null || true
source .venv/bin/activate

: "${CONFIG:?CONFIG required}"; : "${MLP_HIDDEN:?MLP_HIDDEN required}"
VARIANT="${VARIANT:-}"
DATASET=hendrycks_holdout
GOLD="${ROOT}/data/hendrycks_holdout_eval/eval.jsonl"
CACHE="${ROOT}/hybrid/results/response_cache_hendrycks_holdout"

declare -A THINK_MODEL BASE_MODEL THINK_SHORT BASE_SHORT STEER_LAYER SAE_LAYER N_CLUSTERS
THINK_MODEL[orz-0.5b]="Open-Reasoner-Zero/Open-Reasoner-Zero-0.5B"; BASE_MODEL[orz-0.5b]="Qwen/Qwen2.5-0.5B"; THINK_SHORT[orz-0.5b]="open-reasoner-zero-0.5b"; BASE_SHORT[orz-0.5b]="qwen2.5-0.5b"; STEER_LAYER[orz-0.5b]=9; SAE_LAYER[orz-0.5b]=8; N_CLUSTERS[orz-0.5b]=10
THINK_MODEL[orz-1.5b]="Open-Reasoner-Zero/Open-Reasoner-Zero-1.5B"; BASE_MODEL[orz-1.5b]="Qwen/Qwen2.5-1.5B"; THINK_SHORT[orz-1.5b]="open-reasoner-zero-1.5b"; BASE_SHORT[orz-1.5b]="qwen2.5-1.5b"; STEER_LAYER[orz-1.5b]=10; SAE_LAYER[orz-1.5b]=4; N_CLUSTERS[orz-1.5b]=10
THINK_MODEL[orz-7b]="Open-Reasoner-Zero/Open-Reasoner-Zero-7B"; BASE_MODEL[orz-7b]="Qwen/Qwen2.5-7B"; THINK_SHORT[orz-7b]="open-reasoner-zero-7b"; BASE_SHORT[orz-7b]="qwen2.5-7b"; STEER_LAYER[orz-7b]=10; SAE_LAYER[orz-7b]=20; N_CLUSTERS[orz-7b]=10
THINK_MODEL[orz-32b]="Open-Reasoner-Zero/Open-Reasoner-Zero-32B"; BASE_MODEL[orz-32b]="Qwen/Qwen2.5-32B"; THINK_SHORT[orz-32b]="open-reasoner-zero-32b"; BASE_SHORT[orz-32b]="qwen2.5-32b"; STEER_LAYER[orz-32b]=24; SAE_LAYER[orz-32b]=27; N_CLUSTERS[orz-32b]=15
THINK_MODEL[r1-14b]="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"; BASE_MODEL[r1-14b]="Qwen/Qwen2.5-14B"; THINK_SHORT[r1-14b]="deepseek-r1-distill-qwen-14b"; BASE_SHORT[r1-14b]="qwen2.5-14b"; STEER_LAYER[r1-14b]=18; SAE_LAYER[r1-14b]=38; N_CLUSTERS[r1-14b]=5
THINK_MODEL[r1-llama8b]="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"; BASE_MODEL[r1-llama8b]="Meta-Llama/Llama-3.1-8B"; THINK_SHORT[r1-llama8b]="deepseek-r1-distill-llama-8b"; BASE_SHORT[r1-llama8b]="llama-3.1-8b"; STEER_LAYER[r1-llama8b]=12; SAE_LAYER[r1-llama8b]=6; N_CLUSTERS[r1-llama8b]=15
THINK_MODEL[qwq-32b]="Qwen/QwQ-32B"; BASE_MODEL[qwq-32b]="Qwen/Qwen2.5-32B"; THINK_SHORT[qwq-32b]="qwq-32b"; BASE_SHORT[qwq-32b]="qwen2.5-32b"; STEER_LAYER[qwq-32b]=24; SAE_LAYER[qwq-32b]=27; N_CLUSTERS[qwq-32b]=10
THINK_MODEL[r1-32b]="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"; BASE_MODEL[r1-32b]="Qwen/Qwen2.5-32B"; THINK_SHORT[r1-32b]="deepseek-r1-distill-qwen-32b"; BASE_SHORT[r1-32b]="qwen2.5-32b"; STEER_LAYER[r1-32b]=24; SAE_LAYER[r1-32b]=27; N_CLUSTERS[r1-32b]=15
THINK_MODEL[r1-math1.5b]="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"; BASE_MODEL[r1-math1.5b]="Qwen/Qwen2.5-Math-1.5B"; THINK_SHORT[r1-math1.5b]="deepseek-r1-distill-qwen-1.5b"; BASE_SHORT[r1-math1.5b]="qwen2.5-math-1.5b"; STEER_LAYER[r1-math1.5b]=10; SAE_LAYER[r1-math1.5b]=4; N_CLUSTERS[r1-math1.5b]=15

TM="${THINK_MODEL[$CONFIG]}"; BM="${BASE_MODEL[$CONFIG]}"; TS="${THINK_SHORT[$CONFIG]}"; BS="${BASE_SHORT[$CONFIG]}"
SL="${STEER_LAYER[$CONFIG]}"; SAEL="${SAE_LAYER[$CONFIG]}"; NK="${N_CLUSTERS[$CONFIG]}"
VEC_DIR="${ROOT}/mlp_vectors_qa_instr${VARIANT}_h${MLP_HIDDEN}/${CONFIG}"
EVAL_DIR="${ROOT}/mlp_eval_hendrycks_holdout_qa_instr${VARIANT}_h${MLP_HIDDEN}/${CONFIG}"
mkdir -p "${EVAL_DIR}"

echo "== hendrycks-holdout hybrid | CONFIG=${CONFIG} VARIANT='${VARIANT}' | $(date -u) =="
[[ -f "${VEC_DIR}/cat_coef_mlp.pt" ]] || { echo "FATAL: missing ${VEC_DIR}/cat_coef_mlp.pt"; exit 2; }
[[ -f "${GOLD}" ]] || { echo "FATAL: missing gold ${GOLD}"; exit 2; }
N_EX=$(wc -l < "${GOLD}")
for S in 0 1 2; do
    F="${CACHE}/thinking_${TS}_hendrycks_holdout_temp0.6_max2048_s${S}.jsonl"
    { [[ -f "${F}" ]] && [[ $(wc -l < "${F}") -ge ${N_EX} ]]; } || { echo "FATAL: missing/short think s${S} ${F}"; exit 2; }
done
BASE_F="${CACHE}/base_${BS}_hendrycks_holdout_temp0_max2048.jsonl"
{ [[ -f "${BASE_F}" ]] && [[ $(wc -l < "${BASE_F}") -ge ${N_EX} ]]; } || { echo "FATAL: missing/short base ${BASE_F}"; exit 2; }
echo "Rollouts verified (N=${N_EX})."

N_GPUS=$(python3 -c "import torch; print(torch.cuda.device_count())" 2>/dev/null || echo 1)
TWO_GPU_FLAG=""; [[ ${N_GPUS} -ge 2 ]] && TWO_GPU_FLAG="--two_gpu_split"
case "${CONFIG}" in *32b) HBS=8;; *14b) HBS=32;; *7b|*8b) HBS=32;; *) HBS=64;; esac
HBS="${HBS_OVERRIDE:-${HBS}}"
JUDGE_MODEL="${JUDGE_MODEL:-anthropic/claude-sonnet-4-6}"
echo "[cfg] gpus=${N_GPUS} hbs=${HBS} steer=${SL} sae=${SAEL} k=${NK}"

cd hybrid
python hybrid_eval.py \
    --dataset hendrycks_holdout --hendrycks_holdout_file "${GOLD}" \
    --thinking_model "${TM}" --base_model "${BM}" \
    --sae_layer ${SAEL} --n_clusters ${NK} \
    --dom_vectors_dir "${VEC_DIR}" --old_vectors_dir "${VEC_DIR}" --old_vectors_layer ${SL} \
    --coef_select mlp --mlp_coef_path "${VEC_DIR}/cat_coef_mlp.pt" --mlp_config_path "${VEC_DIR}/mlp_config.json" \
    --max_new_tokens 2048 --max_thinking_tokens 2048 \
    --temperature 0.0 --decode_temperature 0 \
    --hybrid_gen_batch_size ${HBS} \
    --judge_repetitions 3 --judge_model "${JUDGE_MODEL}" \
    --results_dir "${EVAL_DIR}" --response_cache_dir "${CACHE}" --results_suffix final \
    --think_cache_temp_label "0.6" --think_cache_max_tokens 2048 --think_cache_sample_idx 0 \
    --base_cache_temp_label "0" --base_cache_max_tokens 2048 --base_cache_sample_idx -1 \
    --hybrid_cache_sample_idx -1 \
    --think_prompt_family auto --math_directive --base_prompt_style qa_instr \
    --pure_steer_base_eos ${TWO_GPU_FLAG}
cd "${ROOT}"

echo ""; echo "--- judging think samples s1, s2 ---"
python mlp_pipeline/canonical/judge_extra_think_samples.py \
    --cache_dir "${CACHE}" --think_short "${TS}" --base_id "${BS}" \
    --dataset hendrycks_holdout --gold_file "${GOLD}" \
    --temp_label "0.6" --max_tokens 2048 --sample_ids 1,2 \
    --judge_repetitions 3 --judge_model "${JUDGE_MODEL}" \
    --out_dir "${EVAL_DIR}" --results_suffix final

echo ""; echo "--- aggregating ---"
python mlp_pipeline/canonical/aggregate_samples_final.py \
    --eval_dir "${EVAL_DIR}" --base_id "${BS}" \
    --dataset hendrycks_holdout --suffix final

echo ""; echo "== DONE hendrycks-holdout hybrid ${CONFIG} $(date -u) =="
ls -lh "${EVAL_DIR}" 2>/dev/null | grep -E "(judge|hybrid_summary|aggregate)" || true
