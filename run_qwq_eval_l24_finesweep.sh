#!/bin/bash
# QwQ-32B hybrid eval at LAYER 24 with fine PG coef sweep [0.1..1.0],
# 48 tasks, 3 conditions in parallel (one per H200).  Used to test
# whether retraining at layer 24 fixes the bias-first regression we saw
# at layer 38 (Hybrid << Base).
set -uo pipefail
cd /workspace/thinking-llms-interp
source .venv/bin/activate
source .env_exports.sh

LAYER=24
CATS_DIR="train-vectors/results/vars/correction_vectors_qwq32b_biasfirst_stage2_l${LAYER}"
BIAS_PATH="train-vectors/results/vars/correction_vectors_qwq32b_biasfirst_stage1_l${LAYER}/qwen2.5-32b_bias_global.pt"

if [ ! -f "$BIAS_PATH" ]; then
    echo "ERROR: bias not found at $BIAS_PATH"; exit 1
fi
if [ ! -f "$CATS_DIR/qwen2.5-32b_idx0_linear.pt" ]; then
    echo "ERROR: cats not found in $CATS_DIR"; exit 1
fi

N_TASKS=${N_TASKS:-48}
COEF_SWEEP=${COEF_SWEEP:-"0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0"}
TAG_SFX=${TAG_SFX:-"l${LAYER}-fine-${N_TASKS}"}

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
        --batch_gen_size 4 --hybrid_gen_batch_size 4 \
        --base_model Qwen/Qwen2.5-32B \
        --thinking_model Qwen/QwQ-32B \
        --sae_layer 27 --n_clusters 10 \
        --disable_sae_mean \
        --dom_vectors_dir ../train-vectors/results/diff_of_means \
        --dom_vectors_model_short qwen2.5-32b \
        --old_vectors_dir "../$CATS_DIR" \
        --old_vectors_layer "$LAYER" \
        --bias_vector_path "../$BIAS_PATH" \
        --bias_layer "$LAYER" \
        --coef_sweep "$COEF_SWEEP" \
        --judge_repetitions 3 \
        --results_suffix "${TAG}" \
        $EXTRA 2>&1 | tee "/tmp/${TAG}.log"
}

# Allow caller to limit which conditions run (default: all 3).
CONDITIONS=${CONDITIONS:-"learn rand biasonly"}

PIDS=()
for cond in $CONDITIONS; do
    case "$cond" in
        learn)    run_one_gpu 0 "qwq-bf-learn-${TAG_SFX}"     ""                                    & PIDS+=($!);;
        rand)     run_one_gpu 1 "qwq-bf-rand-${TAG_SFX}"      "--randomize_vectors --random_seed 42" & PIDS+=($!);;
        biasonly) run_one_gpu 2 "qwq-bf-biasonly-${TAG_SFX}"  "--bias_only"                          & PIDS+=($!);;
    esac
done

echo "Launched: ${PIDS[*]}"
EXIT=0
for pid in "${PIDS[@]}"; do
    wait $pid; CODE=$?; echo "pid=$pid exit=$CODE"
    [ $CODE -ne 0 ] && EXIT=$CODE
done
echo "Done.  overall exit=$EXIT"
exit $EXIT
