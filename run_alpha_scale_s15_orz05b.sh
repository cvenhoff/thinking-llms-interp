#!/usr/bin/env bash
# ============================================================
# Stage-1.5 + per_cat_bias_scale recipe for ORZ-0.5B.
#
# Pipeline:
#   Stage 1  — train global bias on 0.5B disagreements
#              (reuses disagreements.pt from joint run).
#   Stage 1.5 — filter disagreements NOT resolved by bias.
#   Stage 2  — train cat vectors with per-category bias scale
#              (alpha_i), bias frozen. Saves bias_alpha.json.
#
# Eval:
#   learn:    bias + alpha_i-scaled bias + cat vector
#   biasonly: bias only (no cat vectors)
# ============================================================
set -uo pipefail
cd /workspace/thinking-llms-interp
source .venv/bin/activate
source .env_exports.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY=error

BASE="Qwen/Qwen2.5-0.5B"
THINK="Open-Reasoner-Zero/Open-Reasoner-Zero-0.5B"
THINK_SHORT="open-reasoner-zero-0.5b"
BASE_SHORT="qwen2.5-0.5b"
TAG="orz-0.5b"
STEER=9
SAE_L=8
K=10
COEF=1.0

JOINT_DIR="train-vectors/results/vars/correction_vectors_${TAG}_joint"
SAVE1="train-vectors/results/vars/correction_vectors_${TAG}_s15_stage1"
SAVE2A="train-vectors/results/vars/correction_vectors_${TAG}_s15_stage2_alphascale"
LOG_PREFIX="/tmp/${TAG}_alpha_scale"

echo ""
echo "======================================================"
echo "  Stage-1.5 + per_cat_bias_scale recipe: ${TAG}"
echo "  Base  : ${BASE}"
echo "  Think : ${THINK}"
echo "  Steer : L${STEER}  SAE: L${SAE_L}  K=${K}"
echo "  GPUs  : 0,1,2 (DDP)  Coef: ${COEF}"
echo "======================================================"

mkdir -p "${SAVE1}" "${SAVE2A}"

# Copy pre-collected disagreements from joint run into stage1 dir
if [ -f "${JOINT_DIR}/disagreements.pt" ] && [ ! -f "${SAVE1}/disagreements.pt" ]; then
    echo "[s15] Copying disagreements.pt from joint run..."
    cp "${JOINT_DIR}/disagreements.pt" "${SAVE1}/disagreements.pt"
fi

# ---- Stage 1: train global bias ----
if [ -f "${SAVE1}/${BASE_SHORT}_bias_global.pt" ]; then
    echo "[s15] Stage 1: already done — skipping."
else
    echo "[s15] Stage 1: train global bias (3-GPU DDP)..."
    cd train-vectors
    CUDA_VISIBLE_DEVICES=0,1,2 torchrun \
        --standalone --nproc_per_node=3 \
        optimize_correction_vectors.py \
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
        --save_dir "results/vars/correction_vectors_${TAG}_s15_stage1" \
        --seed 42 \
        --load_collected \
        --skip_cats_phase \
        --train_global_bias \
        2>&1 | tee "${LOG_PREFIX}_stage1.log"
    cd /workspace/thinking-llms-interp
    [ -f "${SAVE1}/${BASE_SHORT}_bias_global.pt" ] \
        || { echo "ERROR: Stage 1 failed (no bias saved)"; exit 1; }
    echo "[s15] Stage 1 done."
fi

BIAS_PATH_FROM_TV="results/vars/correction_vectors_${TAG}_s15_stage1/${BASE_SHORT}_bias_global.pt"

# Ensure bias_layer.json exists
BIAS_LAYER_JSON="${SAVE1}/bias_layer.json"
[ -f "$BIAS_LAYER_JSON" ] || echo "{\"layer\": $STEER}" > "$BIAS_LAYER_JSON"

# ---- Stage 2 (alpha-scale): filter residual disagreements + train cats ----
if ls "${SAVE2A}/${BASE_SHORT}_idx"*"_linear.pt" &>/dev/null 2>&1; then
    echo "[alpha] Stage 2 alpha-scale: already done — skipping."
else
    echo "[alpha] Stage 1.5 + Stage 2 (per_cat_bias_scale, 3-GPU DDP)..."
    if [ -f "${SAVE1}/disagreements.pt" ] && [ ! -f "${SAVE2A}/disagreements.pt" ]; then
        echo "  Copying disagreements.pt from stage 1..."
        cp "${SAVE1}/disagreements.pt" "${SAVE2A}/disagreements.pt"
    fi
    cd train-vectors
    CUDA_VISIBLE_DEVICES=0,1,2 torchrun \
        --standalone --nproc_per_node=3 \
        optimize_correction_vectors.py \
        --base_model        "$BASE" \
        --thinking_model    "$THINK" \
        --thinking_model_short "$THINK_SHORT" \
        --steer_layer       "$STEER" \
        --sae_classify_layer "$SAE_L" \
        --sae_n_clusters    "$K" \
        --kl_mode topk --topk 50 --train_topk 3 \
        --n_epochs 5 --lr 0.01 \
        --example_batch_size 32 \
        --max_seq_len 2048 --max_positions_per_example 64 \
        --max_positions_per_cat 1000 \
        --per_cat_loss \
        --per_cat_bias_scale \
        --collect_batch_size 16 \
        --holdout_frac 0.1 \
        --n_responses 20000 \
        --collection_mode disagreement \
        --responses_dir "../generate-responses/results/vars" \
        --save_dir "results/vars/correction_vectors_${TAG}_s15_stage2_alphascale" \
        --seed 42 \
        --load_collected \
        --filter_by_bias \
        --frozen_bias_path "$BIAS_PATH_FROM_TV" \
        --frozen_bias_layer "$STEER" \
        2>&1 | tee "${LOG_PREFIX}_stage2.log"
    cd /workspace/thinking-llms-interp
    NCAT=$(ls "${SAVE2A}/${BASE_SHORT}_idx"*"_linear.pt" 2>/dev/null | wc -l)
    [ "$NCAT" -gt 0 ] || { echo "ERROR: Stage 2 alpha-scale failed (no cat vectors saved)"; exit 1; }
    echo "[alpha] Stage 2 done. ${NCAT} cat vectors saved."
fi

# ---- Eval ----
run_eval() {
    local COND="$1"
    local EXTRA="$2"
    local SUFFIX="${TAG}-alpha-${COND}-500"
    local ROLLING="hybrid/results/rolling/rolling_${BASE_SHORT}_math500_${SUFFIX}.jsonl"

    if [ -f "$ROLLING" ] && [ "$(python3 -c "
import json; rows=[json.loads(l) for l in open('$ROLLING')]
print(len([r for r in rows if 'judges' in r]))" 2>/dev/null)" -ge 490 ]; then
        echo "[alpha] Eval ${COND}: already done — skipping."
        return
    fi

    echo "[alpha] Eval: ${COND}..."
    rm -f "$ROLLING"
    cd hybrid
    CUDA_VISIBLE_DEVICES=0 python -u hybrid_eval.py \
        --dataset math500 \
        --n_tasks 0 \
        --eval_start_idx 0 \
        --max_new_tokens 2000 \
        --max_thinking_tokens 2000 \
        --batch_gen_size 512 \
        --hybrid_gen_batch_size 512 \
        --base_model    "$BASE" \
        --thinking_model "$THINK" \
        --sae_layer     "$SAE_L" \
        --n_clusters    "$K" \
        --dom_vectors_dir   "../${SAVE2A}" \
        --old_vectors_dir   "../${SAVE2A}" \
        --old_vectors_layer "$STEER" \
        --bias_vector_path  "../${SAVE1}/${BASE_SHORT}_bias_global.pt" \
        --bias_layer        "$STEER" \
        --fixed_coef "$COEF" \
        --judge_repetitions 3 \
        --results_suffix "$SUFFIX" \
        $EXTRA \
        2>&1 | tee "${LOG_PREFIX}_eval_${COND}.log"
    cd /workspace/thinking-llms-interp
    echo "[alpha] Eval ${COND} done."
}

run_eval "learn"    ""
run_eval "biasonly" "--bias_only"

echo ""
echo "[alpha] ALL DONE for ${TAG} alpha-scale experiment."

python3 -c "
import json, os

tag  = '${TAG}'
base = '${BASE_SHORT}'

def parse(fpath):
    with open(fpath) as f:
        data = json.load(f)
    pr = data['per_rep']
    think  = pr['thinking']['mean_pct']
    base_  = pr['base']['mean_pct']
    hybrid = pr['hybrid']['mean_pct']
    gap    = think - base_
    rec    = (hybrid - base_) / max(gap, 0.01) * 100
    return think, base_, hybrid, rec

print(f'\n--- Results: {tag} alpha-scale ---')
for cond in ['learn', 'biasonly']:
    fpath = f'hybrid/results/judge_reps_{base}_math500_{tag}-alpha-{cond}-500.json'
    try:
        T, B, H, rec = parse(fpath)
        print(f'  {cond:10s}: think={T:.1f}%  base={B:.1f}%  hybrid={H:.1f}%  gap_recovery={rec:.0f}%')
    except Exception as e:
        print(f'  {cond:10s}: {e}')
" 2>/dev/null || true
