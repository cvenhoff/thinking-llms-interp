#!/usr/bin/env bash
# ============================================================
# Joint bias+cats training and full math500 evaluation
# for 4 small model pairs using the corrected pipeline.
#
# Fixes vs. previous runs:
#   1. Joint bias+cats training (--joint_cats_and_bias):
#      at every disagreement position V[cat] + b are applied
#      together and both updated in one Adam step — the
#      optimiser routes category-agnostic signal into b and
#      category-specific signal into V[cat], matching inference.
#      No more 3-stage bias-first pipeline.
#   2. Correct SAE layer + K per model (from original paper
#      run scripts, NOT the paper appendix).
#   3. Full math500 evaluation (--n_tasks 0, sequential from
#      idx 0) so all three conditions (learn/biasonly/rand)
#      always run on the exact same 500 problems.
#
# GPU layout:
#   GPU 0 : ORZ-0.5B (train+eval) → ORZ-1.5B (train+eval)
#   GPU 1 : DSL-8B   (train+eval)
#   GPU 2 : DSQ-Math-1.5B (train+eval)
#
# Model pair configs (from original run scripts @ bf9df36):
#   ORZ-0.5B      steer=9  sae_layer=8  K=10
#   ORZ-1.5B      steer=10 sae_layer=8  K=5
#   DSL-8B        steer=12 sae_layer=6  K=15
#   DSQ-Math-1.5B steer=10 sae_layer=4  K=15
# ============================================================
set -uo pipefail
cd /workspace/thinking-llms-interp
source .venv/bin/activate
source .env_exports.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY=error

RESP_DIR="../generate-responses/results/vars"
SAVE_ROOT="train-vectors/results/vars"
COEF_SWEEP="0.1,0.25,0.5,1.0,1.5"

# ============================================================
# Helper: train (joint bias+cats) then evaluate (3 conditions)
# Usage: run_pair GPU BASE THINK THINK_SHORT TAG STEER SAE_L K
#        GEN_BS  HYBRID_BS   (generation batch sizes for eval)
# ============================================================
run_pair() {
    local GPU="$1"
    local BASE="$2"
    local THINK="$3"
    local THINK_SHORT="$4"
    local TAG="$5"
    local STEER="$6"
    local SAE_L="$7"
    local K="$8"
    local GEN_BS="${9:-64}"       # base+thinking standalone gen batch
    local HYBRID_BS="${10:-64}"   # hybrid decoding batch

    local BASE_SHORT
    BASE_SHORT=$(python3 -c "print('$BASE'.split('/')[-1].lower())")
    local SAVE_DIR="${SAVE_ROOT}/correction_vectors_${TAG}_joint"
    local BIAS_PATH="${SAVE_DIR}/${BASE_SHORT}_bias_global.pt"
    local LOG_PREFIX="/tmp/${TAG}"

    echo ""
    echo "======================================================"
    echo "  Model : ${TAG}"
    echo "  Base  : ${BASE} (short: ${BASE_SHORT})"
    echo "  Think : ${THINK}"
    echo "  Steer : L${STEER}  SAE: L${SAE_L}  K=${K}"
    echo "======================================================"

    # ---- Training: joint bias+cats in one pass ----
    if [ -f "${SAVE_DIR}/${BASE_SHORT}_idx0_linear.pt" ]; then
        echo "[${TAG}] Train: already done — skipping."
    else
        echo "[${TAG}] Train: joint bias+cats..."
        mkdir -p "${SAVE_DIR}"
        cd train-vectors
        CUDA_VISIBLE_DEVICES=$GPU python -u optimize_correction_vectors.py \
            --base_model        "$BASE" \
            --thinking_model    "$THINK" \
            --thinking_model_short "$THINK_SHORT" \
            --steer_layer       "$STEER" \
            --sae_classify_layer "$SAE_L" \
            --sae_n_clusters    "$K" \
            --joint_cats_and_bias \
            --kl_mode topk --topk 50 --train_topk 3 \
            --n_epochs 5 --lr 0.01 --max_norm 0.0 \
            --example_batch_size 64 \
            --max_seq_len 2048 --max_positions_per_example 64 \
            --holdout_frac 0.1 \
            --n_responses 20000 \
            --collection_mode disagreement \
            --responses_dir "$RESP_DIR" \
            --save_dir "results/vars/correction_vectors_${TAG}_joint" \
            --seed 42 \
            2>&1 | tee "${LOG_PREFIX}_train.log"
        cd /workspace/thinking-llms-interp

        # Verify outputs
        [ -f "$BIAS_PATH" ] || { echo "ERROR: bias not saved at $BIAS_PATH"; exit 1; }
        [ -f "${SAVE_DIR}/${BASE_SHORT}_idx0_linear.pt" ] || { echo "ERROR: cat vectors not saved"; exit 1; }
        echo "[${TAG}] Train done."
    fi

    # ---- Evaluation: 3 conditions, full math500 ----
    run_eval() {
        local COND="$1"   # learn | biasonly | rand
        local EXTRA="$2"
        local SUFFIX="${TAG}-joint-${COND}-500"
        local ROLLING="hybrid/results/rolling/rolling_${BASE_SHORT}_math500_${SUFFIX}.jsonl"

        if [ -f "$ROLLING" ] && [ "$(python3 -c "
import json
rows=[json.loads(l) for l in open('$ROLLING')]
print(len(rows))")" -ge 500 ]; then
            echo "[${TAG}] Eval ${COND}: already done — skipping."
            return
        fi

        echo "[${TAG}] Eval: ${COND}..."
        rm -f "$ROLLING"
        cd hybrid
        CUDA_VISIBLE_DEVICES=$GPU python -u hybrid_eval.py \
            --dataset math500 \
            --n_tasks 0 \
            --eval_start_idx 0 \
            --max_new_tokens 2000 \
            --max_thinking_tokens 2000 \
            --batch_gen_size "$GEN_BS" \
            --hybrid_gen_batch_size "$HYBRID_BS" \
            --base_model    "$BASE" \
            --thinking_model "$THINK" \
            --sae_layer     "$SAE_L" \
            --n_clusters    "$K" \
            --old_vectors_dir   "../${SAVE_DIR}" \
            --old_vectors_layer "$STEER" \
            --bias_vector_path  "../${BIAS_PATH}" \
            --bias_layer        "$STEER" \
            --coef_sweep "$COEF_SWEEP" \
            --coef_select pg \
            --judge_repetitions 3 \
            --results_suffix "$SUFFIX" \
            $EXTRA \
            2>&1 | tee "${LOG_PREFIX}_eval_${COND}.log"
        cd /workspace/thinking-llms-interp
        echo "[${TAG}] Eval ${COND} done."
    }

    # For small models (<=1.5B) run learn+biasonly in parallel
    # (both load tiny models, H200 has plenty of room).
    # For 8B models run sequentially to avoid memory pressure.
    local MODEL_PARAMS
    MODEL_PARAMS=$(python3 -c "
b='$BASE'.lower()
if '8b' in b or '7b' in b: print('large')
else: print('small')
")
    if [ "$MODEL_PARAMS" = "small" ]; then
        run_eval "learn"    "" &
        EPID_LEARN=$!
        run_eval "biasonly" "--bias_only" &
        EPID_BIAS=$!
        wait $EPID_LEARN; wait $EPID_BIAS
    else
        run_eval "learn"    ""
        run_eval "biasonly" "--bias_only"
    fi
    run_eval "rand" "--randomize_vectors --random_seed 42"

    echo "[${TAG}] ALL DONE."
}

# ============================================================
# GPU 1: DSL-8B  (Llama-8B + R1-Distill-Llama-8B, ~32GB total)
#   gen_bs=64  hybrid_bs=64  (both 8B models, generous headroom)
# ============================================================
run_gpu1() {
    run_pair 1 \
        "meta-llama/Llama-3.1-8B" \
        "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
        "deepseek-r1-distill-llama-8b" \
        "dsl-8b" \
        12 6 15 \
        64 64
}

# ============================================================
# GPU 2: DSQ-Math-1.5B (two 1.5B models, tiny footprint)
#   gen_bs=256  hybrid_bs=128
# ============================================================
run_gpu2() {
    run_pair 2 \
        "Qwen/Qwen2.5-Math-1.5B" \
        "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B" \
        "deepseek-r1-distill-qwen-1.5b" \
        "dsq-math-1.5b" \
        10 4 15 \
        256 128
}

# ============================================================
# GPU 0: ORZ-0.5B → ORZ-1.5B (sub-2B models, minimal VRAM)
#   gen_bs=512  hybrid_bs=256 for 0.5B
#   gen_bs=256  hybrid_bs=128 for 1.5B
# ============================================================
run_gpu0() {
    run_pair 0 \
        "Qwen/Qwen2.5-0.5B" \
        "Open-Reasoner-Zero/Open-Reasoner-Zero-0.5B" \
        "open-reasoner-zero-0.5b" \
        "orz-0.5b" \
        9 8 10 \
        512 256

    run_pair 0 \
        "Qwen/Qwen2.5-1.5B" \
        "Open-Reasoner-Zero/Open-Reasoner-Zero-1.5B" \
        "open-reasoner-zero-1.5b" \
        "orz-1.5b" \
        10 8 5 \
        256 128
}

# ============================================================
# Launch all 3 GPU lanes in parallel
# ============================================================
echo "Launching GPU 0 (ORZ-0.5B → ORZ-1.5B), GPU 1 (DSL-8B), GPU 2 (DSQ-Math-1.5B)..."

run_gpu0 2>&1 | tee /tmp/gpu0_joint.log &
PID0=$!
run_gpu1 2>&1 | tee /tmp/gpu1_joint.log &
PID1=$!
run_gpu2 2>&1 | tee /tmp/gpu2_joint.log &
PID2=$!

wait $PID0; STATUS0=$?
wait $PID1; STATUS1=$?
wait $PID2; STATUS2=$?

echo ""
echo "======================================================"
echo "ALL GPU LANES FINISHED"
echo "  GPU 0 (ORZ-0.5B, ORZ-1.5B) : exit=$STATUS0"
echo "  GPU 1 (DSL-8B)              : exit=$STATUS1"
echo "  GPU 2 (DSQ-Math-1.5B)       : exit=$STATUS2"
echo "======================================================"

# ============================================================
# Print final summary
# ============================================================
python3 - << 'PY'
import json, os, glob

def summary(base_short, tag, label):
    out = []
    for cond in ["learn", "biasonly", "rand"]:
        p = f"hybrid/results/rolling/rolling_{base_short}_math500_{tag}-joint-{cond}-500.jsonl"
        if not os.path.exists(p):
            out.append(f"  {label} {cond:8s}: not found")
            continue
        rows = [json.loads(l) for l in open(p)]
        n = len(rows)
        if n == 0:
            out.append(f"  {label} {cond:8s}: 0 rows")
            continue
        b  = sum(r["judges"]["base"]["correct"]     for r in rows) / n
        t  = sum(r["judges"]["thinking"]["correct"] for r in rows) / n
        h  = sum(r["judges"]["hybrid"]["correct"]   for r in rows) / n
        gap = t - b
        rec = (h - b) / gap * 100 if gap > 0 else float("nan")
        status = "[DONE]" if n >= 500 else f"({n}/500)"
        out.append(f"  {label} {cond:8s}: base={b:.3f} think={t:.3f} hybrid={h:.3f}  rec={rec:.1f}%  {status}")
    return out

print("\n=== RESULTS ===")
for base_short, tag, label in [
    ("qwen2.5-0.5b",      "orz-0.5b",      "ORZ-0.5B     "),
    ("qwen2.5-1.5b",      "orz-1.5b",      "ORZ-1.5B     "),
    ("llama-3.1-8b",      "dsl-8b",        "DSL-8B       "),
    ("qwen2.5-math-1.5b", "dsq-math-1.5b", "DSQ-Math-1.5B"),
]:
    print(f"\n{label}:")
    for line in summary(base_short, tag, label):
        print(line)
PY
