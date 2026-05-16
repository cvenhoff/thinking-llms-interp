#!/bin/bash
# QwQ-32B / Qwen2.5-32B bias-first recipe with PER-POSITION SAE-based
# category assignment during disagreement collection.
# Same fix that lifted ORZ-7B to 95.4% gap recovered (cats > rand by +33pp).
# Uses GPUs 1+2 only (GPU 0 reserved for ORZ-7B n=128 eval).

set -uo pipefail
cd /workspace/thinking-llms-interp
source .venv/bin/activate
source .env_exports.sh

LAYER=38                          # steering layer (canonical 32B)
SAE_LAYER=27                      # SAE classification layer (matches hybrid_eval.py)
N_CLUSTERS=10
GPUS=${GPUS:-1,2}                 # 2 GPUs => Pipeline Parallelism via device_map=auto

SAVE1_FROM_TV="results/vars/correction_vectors_qwq32b_biasfirst_stage1"   # reuse existing bias
SAVE2_FROM_TV="results/vars/correction_vectors_qwq32b_biasfirst_stage2_saecat"
SAVE1="train-vectors/$SAVE1_FROM_TV"
SAVE2="train-vectors/$SAVE2_FROM_TV"
mkdir -p "$SAVE2"

LOG2A="/tmp/qwq32b_saecat_stage2a.log"
LOG2B="/tmp/qwq32b_saecat_stage2b.log"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

BIAS_PATH_FROM_TV="$SAVE1_FROM_TV/qwen2.5-32b_bias_global.pt"
[ -f "$SAVE1/qwen2.5-32b_bias_global.pt" ] || { echo "Stage 1 bias not found: $SAVE1"; exit 1; }
echo "Reusing existing Stage 1 bias: $SAVE1/qwen2.5-32b_bias_global.pt"

# ---- Stage 2a (re-collect with per-position SAE cats, PP on 2 GPUs) ----
if [ ! -f "$SAVE2/disagreements.pt" ]; then
    echo "===== Stage 2a: re-collect under bias with SAE-cats (sae L${SAE_LAYER}) on GPUs ${GPUS} ====="
    cd /workspace/thinking-llms-interp/train-vectors
    CUDA_VISIBLE_DEVICES="$GPUS" python -u optimize_correction_vectors.py \
        --base_model "Qwen/Qwen2.5-32B" \
        --thinking_model "Qwen/QwQ-32B" \
        --thinking_model_short "qwq-32b" \
        --steer_layer "$LAYER" \
        --save_dir "$SAVE2_FROM_TV" \
        --topk 50 --train_topk 3 --kl_mode topk \
        --n_responses 20000 \
        --max_seq_len 1536 --max_positions_per_example 64 \
        --seed 42 --holdout_frac 0.1 \
        --frozen_bias_path "$BIAS_PATH_FROM_TV" --frozen_bias_layer "$LAYER" \
        --sae_classify_layer "$SAE_LAYER" --sae_n_clusters "$N_CLUSTERS" \
        --sae_disable_mean \
        --collect_only \
        2>&1 | tee "$LOG2A"
    cd /workspace/thinking-llms-interp
else
    echo "Stage 2a already done."
fi

# ---- Stage 2b (train cats with frozen bias on PP) ----
echo "===== Stage 2b: train cats on SAE-cat positions ====="
cd /workspace/thinking-llms-interp/train-vectors
CUDA_VISIBLE_DEVICES="$GPUS" python -u optimize_correction_vectors.py \
    --base_model "Qwen/Qwen2.5-32B" \
    --thinking_model "Qwen/QwQ-32B" \
    --thinking_model_short "qwq-32b" \
    --steer_layer "$LAYER" \
    --save_dir "$SAVE2_FROM_TV" \
    --topk 50 --train_topk 3 --kl_mode topk \
    --max_seq_len 1536 --max_positions_per_example 64 \
    --n_epochs 3 --example_batch_size 8 \
    --lr 0.01 --weight_decay 0.0 --max_norm 0.0 \
    --seed 42 --holdout_frac 0.1 \
    --min_disagreements 1 --min_disagreements_ratio 0.0 --min_category_share 0.0 \
    --frozen_bias_path "$BIAS_PATH_FROM_TV" --frozen_bias_layer "$LAYER" \
    --load_collected \
    2>&1 | tee "$LOG2B"
cd /workspace/thinking-llms-interp

echo "DONE QwQ training (saecat)"
ls -la "$SAVE2"

# ---- Eval (math500, 128 tasks, judge x3) ----
N_TASKS=${N_TASKS:-128}
echo "===== Eval (math500, ${N_TASKS} tasks, judge x3) ====="

run_one_pp () {
    local TAG="$1"
    local EXTRA="$2"
    cd /workspace/thinking-llms-interp/hybrid
    rm -f "results/rolling/rolling_qwen2.5-32b_math500_${TAG}.jsonl"
    rm -f "results/summary_qwen2.5-32b_math500_${TAG}.json"
    rm -f "results/judge_reps_qwen2.5-32b_math500_${TAG}.json"
    echo "===== ${TAG} (GPUs ${GPUS}) ====="
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
        --coef_sweep "0.5,1.0,1.5" \
        --judge_repetitions 3 \
        --results_suffix "${TAG}" \
        $EXTRA 2>&1 | tee "/tmp/${TAG}.log"
    cd /workspace/thinking-llms-interp
}

run_one_pp "qwq-bf-learn-saecat-${N_TASKS}"     ""
run_one_pp "qwq-bf-rand-saecat-${N_TASKS}"      "--randomize_vectors --random_seed 42"
run_one_pp "qwq-bf-biasonly-saecat-${N_TASKS}"  "--bias_only"

echo "ALL QwQ SAE-CAT EVAL DONE."
