#!/bin/bash
# Single-condition eval driver for QwQ-32B SAE-cat with overridable
# selector (--coef_select), sweep, and tag.
# Usage:
#   GPUS=1,2 N_TASKS=32 SWEEP="0.1,...,1.0" SELECT=kl_top3 TAG=foo \
#       bash run_qwq32b_saecat_eval_oneshot.sh
set -uo pipefail
cd /workspace/thinking-llms-interp
source .venv/bin/activate
source .env_exports.sh

LAYER=38
SAE_LAYER=27
N_CLUSTERS=10
GPUS=${GPUS:-1,2}
N_TASKS=${N_TASKS:-32}
SWEEP=${SWEEP:-"0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0"}
SELECT=${SELECT:-pg}
TAG=${TAG:-qwq-saecat-${SELECT}-n${N_TASKS}}

SAVE1="train-vectors/results/vars/correction_vectors_qwq32b_biasfirst_stage1"
SAVE2="train-vectors/results/vars/correction_vectors_qwq32b_biasfirst_stage2_saecat"

[ -f "$SAVE1/qwen2.5-32b_bias_global.pt" ] || { echo "Bias missing"; exit 1; }
[ -f "$SAVE2/qwen2.5-32b_idx0_linear.pt" ] || { echo "Cats missing"; exit 1; }

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY=error

cd /workspace/thinking-llms-interp/hybrid
rm -f "results/rolling/rolling_qwen2.5-32b_math500_${TAG}.jsonl"
rm -f "results/summary_qwen2.5-32b_math500_${TAG}.json"
rm -f "results/judge_reps_qwen2.5-32b_math500_${TAG}.json"
echo "===== ${TAG} (GPUs ${GPUS}, n=${N_TASKS}, sweep=${SWEEP}, select=${SELECT}) ====="
CUDA_VISIBLE_DEVICES="$GPUS" python hybrid_eval.py \
    --dataset math500 --n_tasks "$N_TASKS" \
    --max_new_tokens 2000 --max_thinking_tokens 2000 \
    --batch_gen_size 4 --hybrid_gen_batch_size 4 \
    --base_model Qwen/Qwen2.5-32B \
    --thinking_model Qwen/QwQ-32B \
    --sae_layer "$SAE_LAYER" --n_clusters "$N_CLUSTERS" \
    --disable_sae_mean \
    --dom_vectors_dir ../train-vectors/results/diff_of_means \
    --dom_vectors_model_short qwen2.5-32b \
    --old_vectors_dir "../$SAVE2" \
    --old_vectors_layer "$LAYER" \
    --bias_vector_path "../$SAVE1/qwen2.5-32b_bias_global.pt" \
    --bias_layer "$LAYER" \
    --coef_sweep "$SWEEP" \
    --coef_select "$SELECT" \
    --judge_repetitions 1 \
    --results_suffix "${TAG}" \
    2>&1 | tee "/tmp/${TAG}.log"
