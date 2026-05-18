#!/usr/bin/env bash
# ============================================================
# Cats-only ablation for ORZ-0.5B on GPU 2.
# Trains category vectors WITHOUT a global bias vector, then
# evaluates hybrid generation using only the cat vectors.
# This isolates whether category-specific vectors alone can
# recover part of the thinking-base gap.
#
# Reuses the disagreements.pt cached from the joint run
# (same model pair / SAE config) via --load_collected.
#
# Conditions evaluated:
#   catsonly  — hybrid with cat vector applied at disagreements
#   rand      — random vectors (control)
# ============================================================
set -uo pipefail
cd /workspace/thinking-llms-interp
source .venv/bin/activate
source .env_exports.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY=error

GPU=2
BASE="Qwen/Qwen2.5-0.5B"
THINK="Open-Reasoner-Zero/Open-Reasoner-Zero-0.5B"
THINK_SHORT="open-reasoner-zero-0.5b"
TAG="orz-0.5b"
STEER=9
SAE_L=8
K=10

BASE_SHORT="qwen2.5-0.5b"
JOINT_DIR="train-vectors/results/vars/correction_vectors_${TAG}_joint"
SAVE_DIR="train-vectors/results/vars/correction_vectors_${TAG}_catsonly"
LOG_PREFIX="/tmp/${TAG}_catsonly"
COEF_SWEEP="0.1,0.25,0.5,1.0,1.5"

echo ""
echo "======================================================"
echo "  Cats-only ablation: ${TAG}"
echo "  Base  : ${BASE}"
echo "  Think : ${THINK}"
echo "  Steer : L${STEER}  SAE: L${SAE_L}  K=${K}"
echo "  Save  : ${SAVE_DIR}"
echo "======================================================"

# ---- Training: cats only, no bias ----
if ls "${SAVE_DIR}/${BASE_SHORT}_idx"*"_linear.pt" &>/dev/null; then
    echo "[${TAG}-catsonly] Train: already done — skipping."
else
    echo "[${TAG}-catsonly] Train: cats only (no bias)..."
    mkdir -p "${SAVE_DIR}"
    # Copy disagreements from the joint run before cd-ing into train-vectors.
    if [ -f "${JOINT_DIR}/disagreements.pt" ] && [ ! -f "${SAVE_DIR}/disagreements.pt" ]; then
        echo "  Copying disagreements.pt from joint run..."
        cp "${JOINT_DIR}/disagreements.pt" "${SAVE_DIR}/disagreements.pt"
    fi

    cd train-vectors

    CUDA_VISIBLE_DEVICES=$GPU python -u optimize_correction_vectors.py \
        --base_model        "$BASE" \
        --thinking_model    "$THINK" \
        --thinking_model_short "$THINK_SHORT" \
        --steer_layer       "$STEER" \
        --sae_classify_layer "$SAE_L" \
        --sae_n_clusters    "$K" \
        --kl_mode topk --topk 50 --train_topk 3 \
        --n_epochs 5 --lr 0.01 \
        --example_batch_size 16 \
        --max_seq_len 2048 --max_positions_per_example 64 \
        --holdout_frac 0.1 \
        --n_responses 20000 \
        --collection_mode disagreement \
        --responses_dir "../generate-responses/results/vars" \
        --save_dir "results/vars/correction_vectors_${TAG}_catsonly" \
        --seed 42 \
        --load_collected \
        2>&1 | tee "${LOG_PREFIX}_train.log"
    cd /workspace/thinking-llms-interp

    NCAT=$(ls "${SAVE_DIR}/${BASE_SHORT}_idx"*"_linear.pt" 2>/dev/null | wc -l)
    [ "$NCAT" -gt 0 ] || { echo "ERROR: no cat vectors saved in ${SAVE_DIR}"; exit 1; }
    echo "[${TAG}-catsonly] Train done. ${NCAT} cat vectors saved."
fi

# ---- Helper: run one eval condition ----
run_eval() {
    local COND="$1"
    local EXTRA="$2"
    local SUFFIX="${TAG}-catsonly-${COND}-500"
    local ROLLING="hybrid/results/rolling/rolling_${BASE_SHORT}_math500_${SUFFIX}.jsonl"

    if [ -f "$ROLLING" ] && [ "$(python3 -c "
import json; rows=[json.loads(l) for l in open('$ROLLING')]; print(len(rows))")" -ge 500 ]; then
        echo "[${TAG}-catsonly] Eval ${COND}: already done — skipping."
        return
    fi

    echo "[${TAG}-catsonly] Eval: ${COND}..."
    rm -f "$ROLLING"
    cd hybrid
    CUDA_VISIBLE_DEVICES=$GPU python -u hybrid_eval.py \
        --dataset math500 \
        --n_tasks 0 \
        --eval_start_idx 0 \
        --max_new_tokens 2000 \
        --max_thinking_tokens 2000 \
        --batch_gen_size 128 \
        --hybrid_gen_batch_size 128 \
        --base_model    "$BASE" \
        --thinking_model "$THINK" \
        --sae_layer     "$SAE_L" \
        --n_clusters    "$K" \
        --dom_vectors_dir   "../${SAVE_DIR}" \
        --old_vectors_dir   "../${SAVE_DIR}" \
        --old_vectors_layer "$STEER" \
        --coef_sweep "$COEF_SWEEP" \
        --coef_select pg \
        --judge_repetitions 3 \
        --results_suffix "$SUFFIX" \
        $EXTRA \
        2>&1 | tee "${LOG_PREFIX}_eval_${COND}.log"
    cd /workspace/thinking-llms-interp
    echo "[${TAG}-catsonly] Eval ${COND} done."
}

# Run catsonly and rand in parallel (tiny models, fits easily)
run_eval "catsonly" "" &
EPID_CATS=$!
run_eval "rand"     "--randomize_vectors --random_seed 42" &
EPID_RAND=$!
wait $EPID_CATS; wait $EPID_RAND

echo "[${TAG}-catsonly] ALL DONE."
