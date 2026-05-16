#!/bin/bash
# run_small_models_orz7b_recipe.sh
#
# Applies the ORZ-7B canonical recipe to 4 small thinking models:
#   - ORZ-0.5B  (Qwen/Qwen2.5-0.5B base,       24 layers, steer=L12, SAE=L14)
#   - ORZ-1.5B  (Qwen/Qwen2.5-1.5B base,       28 layers, steer=L14, SAE=L16)
#   - DSL-8B    (meta-llama/Llama-3.1-8B base,  32 layers, steer=L16, SAE=L18)
#   - DSQ-1.5B  (Qwen/Qwen2.5-Math-1.5B base,  28 layers, steer=L14, SAE=L16)
#
# Recipe (exact ORZ-7B canonical, only layers adapted proportionally):
#   kl_mode=topk, max_norm=0.0 (no cap), n_epochs=5, lr=0.01,
#   bs=16, max_seq_len=2048, collection_mode=disagreement (default)
#   Steer layer ~50% depth (ORZ-7B: L16/32=50%)
#   SAE   layer ~62.5% depth (ORZ-7B: L20/32=62.5%, closest available)
#
# Parallelism (all 3 H200 GPU groups run concurrently):
#   GPU 0: ORZ-0.5B then ORZ-1.5B (sequential, tiny models ~10-30 min each)
#   GPU 1: DSL-Llama-8B (~2-3h total)
#   GPU 2: DSQ-1.5B (~30-60 min total)
#
# Results land in:
#   train-vectors/results/vars/correction_vectors_<short>_orz7b_stage{1,2}/
#   hybrid/results/rolling/rolling_<base_short>_math500_<short>-orz7b-{learn,rand,biasonly}.jsonl
#
# Usage:
#   bash run_small_models_orz7b_recipe.sh
#   # or to skip training and only eval:
#   EVAL_ONLY=1 bash run_small_models_orz7b_recipe.sh

set -uo pipefail
EVAL_ONLY=${EVAL_ONLY:-0}

cd /workspace/thinking-llms-interp
source .venv/bin/activate
source .env_exports.sh 2>/dev/null || true

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY=error
export TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=86400
export TORCH_NCCL_ASYNC_ERROR_HANDLING=0
export NCCL_IB_DISABLE=1

# ---- ORZ-7B canonical recipe constants ----
N_RESPONSES=20000
N_EPOCHS=5
LR=0.01
BIAS_MAX_NORM=0.0   # no cap, matching ORZ-7B
CAT_MAX_NORM=0.0    # no cap
MAX_SEQ_LEN=2048
BS=16               # canonical batch size per process

mkdir -p /workspace/tmp

echo "============================================================"
echo "  SMALL-MODEL ORZ-7B RECIPE  ($(date '+%H:%M'))"
echo "  kl_mode=topk  max_norm=0  n_epochs=$N_EPOCHS  lr=$LR"
echo "  bs=$BS  max_seq_len=$MAX_SEQ_LEN  n_responses=$N_RESPONSES"
echo "============================================================"
echo

# ============================================================
# Core function: run full pipeline (Stage 1 + 2a + 2b + eval)
# for one thinking model on a single GPU.
#
# Args (positional):
#   $1  GPU           CUDA device index (e.g. 0)
#   $2  BASE_MODEL    HuggingFace base model ID
#   $3  THINK_MODEL   HuggingFace thinking model ID
#   $4  SHORT         short tag used in save-dir and result names
#   $5  STEER_LAYER   layer to steer in base model (~50% depth)
#   $6  SAE_LAYER     SAE eval layer in thinking model (~62.5% depth)
#   $7  N_CLUSTERS    number of SAE clusters (10 for all small models)
# ============================================================
run_orz7b_recipe () {
    local GPU="$1"
    local BASE_MODEL="$2"
    local THINK_MODEL="$3"
    local SHORT="$4"
    local STEER_LAYER="$5"
    local SAE_LAYER="$6"
    local N_CLUSTERS="$7"

    local MODEL_SHORT
    MODEL_SHORT=$(python3 -c "print('$BASE_MODEL'.split('/')[-1].lower())")
    # THINK_SHORT_FILE: matches existing responses_*.json / annotated_responses_*.json filenames
    local THINK_SHORT_FILE
    THINK_SHORT_FILE=$(python3 -c "print('$THINK_MODEL'.split('/')[-1].lower())")

    local TAG="${SHORT}"
    local SAVE1_REL="train-vectors/results/vars/correction_vectors_${TAG}_orz7b_stage1"
    local SAVE2_REL="train-vectors/results/vars/correction_vectors_${TAG}_orz7b_stage2"
    local SAVE1="/workspace/thinking-llms-interp/${SAVE1_REL}"
    local SAVE2="/workspace/thinking-llms-interp/${SAVE2_REL}"
    mkdir -p "$SAVE1" "$SAVE2"

    local LOG_PREFIX="/workspace/tmp/${TAG}"
    local BIAS_FILE="${SAVE1}/${MODEL_SHORT}_bias_global.pt"

    echo
    echo "======================================================"
    echo "  MODEL: ${SHORT} | GPU ${GPU}"
    echo "  base:    $BASE_MODEL  (model_short=$MODEL_SHORT)"
    echo "  think:   $THINK_MODEL"
    echo "  steer:   L${STEER_LAYER}  SAE: L${SAE_LAYER} K=${N_CLUSTERS}"
    echo "======================================================"

    if [ "${EVAL_ONLY}" = "0" ]; then

        # ---- Stage 1: collect disagreements + train global bias ----
        if [ -f "$BIAS_FILE" ]; then
            echo "[${SHORT}] Stage 1: already done (bias found)."
        else
            echo "[${SHORT}] Stage 1: collect ${N_RESPONSES} rollouts + train bias..."
            cd /workspace/thinking-llms-interp/train-vectors
            CUDA_VISIBLE_DEVICES="$GPU" python -u optimize_correction_vectors.py \
                --base_model   "$BASE_MODEL" \
                --thinking_model "$THINK_MODEL" \
                --thinking_model_short "$THINK_SHORT_FILE" \
                --steer_layer "$STEER_LAYER" \
                --save_dir "results/vars/correction_vectors_${TAG}_orz7b_stage1" \
                --topk 50 --train_topk 3 --kl_mode topk \
                --n_responses "$N_RESPONSES" \
                --max_seq_len "$MAX_SEQ_LEN" --max_positions_per_example 64 \
                --n_epochs "$N_EPOCHS" --example_batch_size "$BS" \
                --lr "$LR" --weight_decay 0.0 --max_norm "$BIAS_MAX_NORM" \
                --seed 42 --holdout_frac 0.1 \
                --min_disagreements 1 --min_disagreements_ratio 0.0 --min_category_share 0.0 \
                --skip_cats_phase --train_global_bias \
                2>&1 | tee "${LOG_PREFIX}_stage1.log"
            cd /workspace/thinking-llms-interp
        fi

        [ -f "$BIAS_FILE" ] || { echo "[${SHORT}] Stage 1 FAILED — $BIAS_FILE not found"; exit 1; }
        local BIAS_NORM
        BIAS_NORM=$(python3 -c \
            "import torch; v=torch.load('$BIAS_FILE',map_location='cpu',weights_only=True); \
             print(f'{v.norm().item():.3f}')" 2>/dev/null)
        echo "[${SHORT}] Stage 1 done. bias norm=${BIAS_NORM}"

        # ---- Stage 2a: re-collect under frozen bias ----
        if [ -f "$SAVE2/disagreements.pt" ]; then
            echo "[${SHORT}] Stage 2a: already done (disagreements.pt found)."
        else
            echo "[${SHORT}] Stage 2a: re-collect ${N_RESPONSES} under bias..."
            cd /workspace/thinking-llms-interp/train-vectors
            CUDA_VISIBLE_DEVICES="$GPU" python -u optimize_correction_vectors.py \
                --base_model   "$BASE_MODEL" \
                --thinking_model "$THINK_MODEL" \
                --thinking_model_short "$THINK_SHORT_FILE" \
                --steer_layer "$STEER_LAYER" \
                --save_dir "results/vars/correction_vectors_${TAG}_orz7b_stage2" \
                --topk 50 --train_topk 3 --kl_mode topk \
                --n_responses "$N_RESPONSES" \
                --max_seq_len "$MAX_SEQ_LEN" --max_positions_per_example 64 \
                --seed 43 --holdout_frac 0.1 \
                --frozen_bias_path "results/vars/correction_vectors_${TAG}_orz7b_stage1/${MODEL_SHORT}_bias_global.pt" \
                --frozen_bias_layer "$STEER_LAYER" \
                --sae_classify_layer "$SAE_LAYER" --sae_n_clusters "$N_CLUSTERS" \
                --collect_only \
                2>&1 | tee "${LOG_PREFIX}_stage2a.log"
            cd /workspace/thinking-llms-interp
        fi

        [ -f "$SAVE2/disagreements.pt" ] || { echo "[${SHORT}] Stage 2a FAILED"; exit 1; }

        # ---- Stage 2b: train category vectors ----
        local CAT0_FILE="${SAVE2}/${MODEL_SHORT}_idx0_linear.pt"
        if [ -f "$CAT0_FILE" ]; then
            echo "[${SHORT}] Stage 2b: already done (cat vectors found)."
        else
            echo "[${SHORT}] Stage 2b: train cat vectors..."
            cd /workspace/thinking-llms-interp/train-vectors
            CUDA_VISIBLE_DEVICES="$GPU" python -u optimize_correction_vectors.py \
                --base_model   "$BASE_MODEL" \
                --thinking_model "$THINK_MODEL" \
                --thinking_model_short "$THINK_SHORT_FILE" \
                --steer_layer "$STEER_LAYER" \
                --save_dir "results/vars/correction_vectors_${TAG}_orz7b_stage2" \
                --topk 50 --train_topk 3 --kl_mode topk \
                --max_seq_len "$MAX_SEQ_LEN" --max_positions_per_example 64 \
                --n_epochs "$N_EPOCHS" --example_batch_size "$BS" \
                --lr "$LR" --weight_decay 0.0 --max_norm "$CAT_MAX_NORM" \
                --seed 42 --holdout_frac 0.1 \
                --min_disagreements 1 --min_disagreements_ratio 0.0 --min_category_share 0.0 \
                --frozen_bias_path "results/vars/correction_vectors_${TAG}_orz7b_stage1/${MODEL_SHORT}_bias_global.pt" \
                --frozen_bias_layer "$STEER_LAYER" \
                --sae_classify_layer "$SAE_LAYER" --sae_n_clusters "$N_CLUSTERS" \
                --load_collected \
                2>&1 | tee "${LOG_PREFIX}_stage2b.log"
            cd /workspace/thinking-llms-interp
        fi

        [ -f "$CAT0_FILE" ] || { echo "[${SHORT}] Stage 2b FAILED"; exit 1; }
        echo "[${SHORT}] Training complete."

    fi  # EVAL_ONLY

    # ---- Eval ----
    echo
    echo "[${SHORT}] Starting eval (pg sweep, n=128, 3 judge reps)..."
    cd /workspace/thinking-llms-interp/hybrid

    run_eval_cond () {
        local COND="$1"
        local EXTRA="$2"
        local EVAL_TAG="${TAG}-orz7b-${COND}-128"
        rm -f "results/rolling/rolling_${MODEL_SHORT}_math500_${EVAL_TAG}.jsonl"
        rm -f "results/summary_${MODEL_SHORT}_math500_${EVAL_TAG}.json"
        rm -f "results/judge_reps_${MODEL_SHORT}_math500_${EVAL_TAG}.json"
        echo "  [${SHORT}/${COND}] evaluating..."
        CUDA_VISIBLE_DEVICES="$GPU" uv run python hybrid_eval.py \
            --dataset math500 --n_tasks 128 \
            --max_new_tokens 2000 --max_thinking_tokens 2000 \
            --batch_gen_size 16 --hybrid_gen_batch_size 16 \
            --base_model    "$BASE_MODEL" \
            --thinking_model "$THINK_MODEL" \
            --sae_layer "$SAE_LAYER" --n_clusters "$N_CLUSTERS" \
            --disable_sae_mean \
            --dom_vectors_dir ../train-vectors/results/diff_of_means \
            --old_vectors_dir  "$SAVE2" \
            --old_vectors_layer "$STEER_LAYER" \
            --bias_vector_path "$BIAS_FILE" \
            --bias_layer "$STEER_LAYER" \
            --coef_sweep "0.5,1.0,1.5,2.0" \
            --coef_select pg \
            --judge_repetitions 3 \
            --results_suffix "${EVAL_TAG}" \
            $EXTRA 2>&1 | tee "/workspace/tmp/${EVAL_TAG}.log"
    }

    run_eval_cond "learn"    ""
    run_eval_cond "rand"     "--randomize_vectors --random_seed 42"
    run_eval_cond "biasonly" "--bias_only"

    # ---- Print summary ----
    echo
    echo "[${SHORT}] Results:"
    python3 - <<PY
import json, os
base = '${MODEL_SHORT}'
tag  = '${TAG}'
results_dir = 'results/rolling'
for cond in ['learn', 'rand', 'biasonly']:
    f = f'{results_dir}/rolling_{base}_math500_{tag}-orz7b-{cond}-128.jsonl'
    if not os.path.exists(f):
        print(f'  {cond:<10} MISSING')
        continue
    rows = [json.loads(l) for l in open(f)]
    n = len(rows)
    aT = sum(1 for r in rows if r['judges']['thinking']['correct'])/n
    aB = sum(1 for r in rows if r['judges']['base']['correct'])/n
    aH = sum(1 for r in rows if r['judges']['hybrid']['correct'])/n
    gap = (aH-aB)/max(1e-9,aT-aB) if aT > aB else float('nan')
    print(f'  {cond:<10} T={aT:.3f}  B={aB:.3f}  H={aH:.3f}  gap={gap:+.3f}  (n={n})')
PY
    cd /workspace/thinking-llms-interp
}


# ============================================================
# GPU 0 (serial): ORZ-0.5B then ORZ-1.5B
# ============================================================
run_gpu0 () {
    run_orz7b_recipe 0 \
        "Qwen/Qwen2.5-0.5B" \
        "Open-Reasoner-Zero/Open-Reasoner-Zero-0.5B" \
        "orz-0.5b" \
        12 14 10

    run_orz7b_recipe 0 \
        "Qwen/Qwen2.5-1.5B" \
        "Open-Reasoner-Zero/Open-Reasoner-Zero-1.5B" \
        "orz-1.5b" \
        14 16 10
}

# ============================================================
# GPU 1: R1-Distill-Llama-8B
# ============================================================
run_gpu1 () {
    run_orz7b_recipe 1 \
        "meta-llama/Llama-3.1-8B" \
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
        "dsl-llama-8b" \
        16 18 10
}

# ============================================================
# GPU 2: R1-Distill-Qwen-1.5B
# ============================================================
run_gpu2 () {
    run_orz7b_recipe 2 \
        "Qwen/Qwen2.5-Math-1.5B" \
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
        "dsq-1.5b" \
        14 16 10
}

# ============================================================
# Wait for current 32B eval to free GPUs, then launch in parallel
# ============================================================
wait_for_gpus () {
    local THRESHOLD_MB=10000   # < 10GB free on ALL GPUs = still occupied
    echo "Waiting for GPUs to free (need >10GB free on all 3)..."
    while true; do
        local ALL_FREE=1
        for GPU_IDX in 0 1 2; do
            local USED_MB
            USED_MB=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits \
                       -i "$GPU_IDX" 2>/dev/null | tr -d ' ')
            local FREE_MB
            FREE_MB=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits \
                       -i "$GPU_IDX" 2>/dev/null | tr -d ' ')
            if [ "${FREE_MB:-0}" -lt "$THRESHOLD_MB" ]; then
                ALL_FREE=0
                echo "  GPU $GPU_IDX: ${USED_MB}MiB used, ${FREE_MB}MiB free — still occupied"
                break
            fi
        done
        if [ "$ALL_FREE" = "1" ]; then
            echo "  All GPUs free. Starting pipeline."
            break
        fi
        sleep 60
    done
}

# Check if GPUs are free now; if not, wait
TOTAL_USED=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits 2>/dev/null \
             | awk '{s+=$1} END {print s}')
if [ "${TOTAL_USED:-0}" -gt 30000 ]; then
    echo "GPUs busy (${TOTAL_USED} MiB used total)."
    wait_for_gpus
fi

echo
echo "Launching GPU 0 (ORZ-0.5B → ORZ-1.5B), GPU 1 (DSL-8B), GPU 2 (DSQ-1.5B) in parallel..."
echo

run_gpu0 2>&1 | tee /workspace/tmp/gpu0_small_pipeline.log &
PID0=$!

run_gpu1 2>&1 | tee /workspace/tmp/gpu1_small_pipeline.log &
PID1=$!

run_gpu2 2>&1 | tee /workspace/tmp/gpu2_small_pipeline.log &
PID2=$!

echo "PIDs: GPU0=$PID0  GPU1=$PID1  GPU2=$PID2"
echo "Logs: /workspace/tmp/gpu{0,1,2}_small_pipeline.log"
echo

wait $PID0; C0=$?
wait $PID1; C1=$?
wait $PID2; C2=$?
echo "Exit codes: GPU0=$C0  GPU1=$C1  GPU2=$C2"

# ============================================================
# Final summary across all 4 models
# ============================================================
echo
echo "============================================================"
echo "  FINAL SUMMARY — ALL SMALL MODELS (ORZ-7B recipe)"
echo "============================================================"
python3 - <<'PY'
import json, os

RESULTS_DIR = "/workspace/thinking-llms-interp/hybrid/results/rolling"
MODELS = [
    ("orz-0.5b",    "qwen2.5-0.5b",      "ORZ-0.5B  (pure RL, 0.5B)"),
    ("orz-1.5b",    "qwen2.5-1.5b",      "ORZ-1.5B  (pure RL, 1.5B)"),
    ("dsl-llama-8b","llama-3.1-8b",      "DSL-8B    (distilled, 8B)"),
    ("dsq-1.5b",    "qwen2.5-math-1.5b", "DSQ-1.5B  (distilled, 1.5B)"),
]
CONDS = ["learn", "rand", "biasonly"]

print(f"\n{'Model':<28} {'Cond':<10} {'n':>4} {'Think':>6} {'Base':>6} {'Hybrid':>6} {'Gap':>8}")
print("-" * 72)
for tag, base_short, label in MODELS:
    for cond in CONDS:
        fname = f"{RESULTS_DIR}/rolling_{base_short}_math500_{tag}-orz7b-{cond}-128.jsonl"
        if not os.path.exists(fname):
            print(f"{label:<28} {cond:<10} {'MISSING':>4}")
            continue
        rows = [json.loads(l) for l in open(fname)]
        n = len(rows)
        aT = sum(1 for r in rows if r["judges"]["thinking"]["correct"]) / n
        aB = sum(1 for r in rows if r["judges"]["base"]["correct"]) / n
        aH = sum(1 for r in rows if r["judges"]["hybrid"]["correct"]) / n
        gap = (aH - aB) / max(1e-9, aT - aB) if aT > aB else float("nan")
        print(f"{label:<28} {cond:<10} {n:>4} {aT:>6.3f} {aB:>6.3f} {aH:>6.3f} {gap:>+8.3f}")
    print()
PY

echo "============================================================"
echo "  DONE  ($(date '+%H:%M'))"
echo "============================================================"
