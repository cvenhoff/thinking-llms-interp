#!/bin/bash
# QwQ-32B / Qwen2.5-32B end-to-end with LAST-TOKEN annotations
# (matches eval-time classifier).  DDP across 3 H200s for Stage 1
# and Stage 2b; collection (Stage 0/2a) is single-process.
set -uo pipefail
cd /workspace/thinking-llms-interp
source .venv/bin/activate
source .env_exports.sh

LAYER=${LAYER:-38}
NPROC=${NPROC:-3}
BS_PER_GPU=${BS_PER_GPU:-8}
N_EPOCHS=${N_EPOCHS:-2}
MAX_SEQ_LEN=${MAX_SEQ_LEN:-1536}
N_RESPONSES=${N_RESPONSES:-3000}

SAVE1_FROM_TV="results/vars/correction_vectors_qwq32b_biasfirst_stage1_lt"
SAVE2_FROM_TV="results/vars/correction_vectors_qwq32b_biasfirst_stage2_lt"
SAVE1="train-vectors/$SAVE1_FROM_TV"
SAVE2="train-vectors/$SAVE2_FROM_TV"
mkdir -p "$SAVE1" "$SAVE2"

LOG0="/tmp/qwq32b_lt_stage0.log"
LOG1="/tmp/qwq32b_lt_stage1.log"
LOG2A="/tmp/qwq32b_lt_stage2a.log"
LOG2B="/tmp/qwq32b_lt_stage2b.log"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY=error

# ---- Stage 0: collect no-bias disagreements (single-process PP, all 3 GPUs) ----
if [ ! -f "$SAVE1/disagreements.pt" ]; then
    echo "===== Stage 0 (LT): no-bias disagreement collection (n=${N_RESPONSES}) ====="
    cd /workspace/thinking-llms-interp/train-vectors
    CUDA_VISIBLE_DEVICES=0,1,2 python -u optimize_correction_vectors.py \
        --base_model "Qwen/Qwen2.5-32B" \
        --thinking_model "Qwen/QwQ-32B" \
        --thinking_model_short "qwq-32b" \
        --steer_layer "$LAYER" \
        --save_dir "$SAVE1_FROM_TV" \
        --topk 50 --train_topk 3 --kl_mode topk \
        --n_responses "$N_RESPONSES" \
        --max_seq_len "$MAX_SEQ_LEN" --max_positions_per_example 64 \
        --seed 42 --holdout_frac 0.1 \
        --collect_only \
        2>&1 | tee "$LOG0"
    cd /workspace/thinking-llms-interp
fi

# ---- Stage 1: train bias on disagreements (DDP across 3 GPUs) ----
if [ ! -f "$SAVE1/qwen2.5-32b_bias_global.pt" ]; then
    echo "===== Stage 1 (LT): bias training (DDP n=${NPROC}, BS=${BS_PER_GPU}) ====="
    cd /workspace/thinking-llms-interp/train-vectors
    torchrun --standalone --nproc_per_node="$NPROC" \
        optimize_correction_vectors.py \
        --base_model "Qwen/Qwen2.5-32B" \
        --thinking_model "Qwen/QwQ-32B" \
        --thinking_model_short "qwq-32b" \
        --steer_layer "$LAYER" \
        --save_dir "$SAVE1_FROM_TV" \
        --topk 50 --train_topk 3 --kl_mode topk \
        --max_seq_len "$MAX_SEQ_LEN" --max_positions_per_example 64 \
        --n_epochs "$N_EPOCHS" --example_batch_size "$BS_PER_GPU" \
        --lr 0.01 --weight_decay 0.0 --max_norm 0.0 \
        --seed 42 --holdout_frac 0.1 \
        --min_disagreements 1 --min_disagreements_ratio 0.0 --min_category_share 0.0 \
        --load_collected --skip_cats_phase --train_global_bias \
        2>&1 | tee "$LOG1"
    cd /workspace/thinking-llms-interp
fi

BIAS_PATH_FROM_TV="$SAVE1_FROM_TV/qwen2.5-32b_bias_global.pt"
[ -f "$SAVE1/qwen2.5-32b_bias_global.pt" ] || { echo "Stage 1 failed"; exit 1; }

# ---- Stage 2a: re-collect under bias (single-process PP) ----
if [ ! -f "$SAVE2/disagreements.pt" ]; then
    echo "===== Stage 2a (LT): re-collect under bias (n=${N_RESPONSES}) ====="
    cd /workspace/thinking-llms-interp/train-vectors
    CUDA_VISIBLE_DEVICES=0,1,2 python -u optimize_correction_vectors.py \
        --base_model "Qwen/Qwen2.5-32B" \
        --thinking_model "Qwen/QwQ-32B" \
        --thinking_model_short "qwq-32b" \
        --steer_layer "$LAYER" \
        --save_dir "$SAVE2_FROM_TV" \
        --topk 50 --train_topk 3 --kl_mode topk \
        --n_responses "$N_RESPONSES" \
        --max_seq_len "$MAX_SEQ_LEN" --max_positions_per_example 64 \
        --seed 42 --holdout_frac 0.1 \
        --frozen_bias_path "$BIAS_PATH_FROM_TV" --frozen_bias_layer "$LAYER" \
        --collect_only \
        2>&1 | tee "$LOG2A"
    cd /workspace/thinking-llms-interp
fi

# ---- Stage 2b: train cats with frozen bias (DDP across 3 GPUs) ----
echo "===== Stage 2b (LT): cats with frozen bias (DDP n=${NPROC}, BS=${BS_PER_GPU}) ====="
cd /workspace/thinking-llms-interp/train-vectors
torchrun --standalone --nproc_per_node="$NPROC" \
    optimize_correction_vectors.py \
    --base_model "Qwen/Qwen2.5-32B" \
    --thinking_model "Qwen/QwQ-32B" \
    --thinking_model_short "qwq-32b" \
    --steer_layer "$LAYER" \
    --save_dir "$SAVE2_FROM_TV" \
    --topk 50 --train_topk 3 --kl_mode topk \
    --max_seq_len "$MAX_SEQ_LEN" --max_positions_per_example 64 \
    --n_epochs "$N_EPOCHS" --example_batch_size "$BS_PER_GPU" \
    --lr 0.01 --weight_decay 0.0 --max_norm 0.0 \
    --seed 42 --holdout_frac 0.1 \
    --min_disagreements 1 --min_disagreements_ratio 0.0 --min_category_share 0.0 \
    --frozen_bias_path "$BIAS_PATH_FROM_TV" --frozen_bias_layer "$LAYER" \
    --load_collected \
    2>&1 | tee "$LOG2B"
cd /workspace/thinking-llms-interp

echo "DONE training (LT)"
ls -la "$SAVE1" "$SAVE2"

# ---- Eval ----
echo
N_TASKS=${N_TASKS:-48}
echo "===== Eval (math500, ${N_TASKS} tasks, judge x3) ====="

run_one_gpu () {
    local GPU="$1"
    local TAG="$2"
    local EXTRA="$3"
    cd /workspace/thinking-llms-interp/hybrid
    rm -f "results/rolling/rolling_qwen2.5-32b_math500_${TAG}.jsonl"
    rm -f "results/summary_qwen2.5-32b_math500_${TAG}.json"
    rm -f "results/judge_reps_qwen2.5-32b_math500_${TAG}.json"
    echo "===== ${TAG} on GPU ${GPU} ====="
    CUDA_VISIBLE_DEVICES="$GPU" python hybrid_eval.py \
        --dataset math500 --n_tasks "$N_TASKS" \
        --max_new_tokens 2000 --max_thinking_tokens 2000 \
        --batch_gen_size 8 --hybrid_gen_batch_size 8 \
        --base_model Qwen/Qwen2.5-32B \
        --thinking_model Qwen/QwQ-32B \
        --sae_layer 27 --n_clusters 10 \
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
}

cd /workspace/thinking-llms-interp
run_one_gpu 0 "qwq-bf-learn-canon-lt-${N_TASKS}"     ""                                    &
PID0=$!
run_one_gpu 1 "qwq-bf-rand-canon-lt-${N_TASKS}"      "--randomize_vectors --random_seed 42" &
PID1=$!
run_one_gpu 2 "qwq-bf-biasonly-canon-lt-${N_TASKS}"  "--bias_only"                          &
PID2=$!

wait $PID0; CODE0=$?
wait $PID1; CODE1=$?
wait $PID2; CODE2=$?
echo "Exit codes: learn=$CODE0  rand=$CODE1  biasonly=$CODE2"
