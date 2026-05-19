#!/usr/bin/env bash
# ============================================================
# Stage 2 with --per_cat_bias_scale for ORZ-1.5B.
# Uses the Stage 1 bias from the standard s15 run (already
# trained), re-runs ONLY Stage 2 cats with each category
# free to learn its own bias scale alpha_i (init 1.0), then
# evaluates learn / biasonly on MATH-500.
#
# The alpha_i values are folded into the saved cat vectors
# so hybrid_eval.py needs zero changes:
#   saved_V[i] = V[i] + (alpha[i]-1)*bias
#   => at eval:  saved_V[i] + bias = V[i] + alpha[i]*bias
# ============================================================
set -uo pipefail
cd /workspace/thinking-llms-interp
source .venv/bin/activate
source .env_exports.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY=error

BASE="Qwen/Qwen2.5-1.5B"
THINK="Open-Reasoner-Zero/Open-Reasoner-Zero-1.5B"
THINK_SHORT="open-reasoner-zero-1.5b"
BASE_SHORT="qwen2.5-1.5b"
TAG="orz-1.5b"
STEER=14
SAE_L=16
K=10
COEF=1.0

# Reuse Stage 1 bias from the standard s15 run
SAVE1="train-vectors/results/vars/correction_vectors_${TAG}_s15_stage1"
# New save dir for the alpha-scale Stage 2
SAVE2A="train-vectors/results/vars/correction_vectors_${TAG}_s15_stage2_alphascale"
LOG_PREFIX="/tmp/${TAG}_alpha_scale"

echo ""
echo "======================================================"
echo "  Stage-1.5 + per_cat_bias_scale recipe: ${TAG}"
echo "  Stage 1 bias: ${SAVE1} (pre-trained, reused)"
echo "  Stage 2 save: ${SAVE2A}"
echo "  GPUs: 0,1,2 (DDP)  COEF: ${COEF}"
echo "======================================================"

# Sanity: Stage 1 bias must exist
[ -f "${SAVE1}/${BASE_SHORT}_bias_global.pt" ] \
    || { echo "ERROR: Stage 1 bias not found at ${SAVE1}/${BASE_SHORT}_bias_global.pt"; exit 1; }

mkdir -p "${SAVE2A}"

BIAS_PATH_FROM_TV="results/vars/correction_vectors_${TAG}_s15_stage1/${BASE_SHORT}_bias_global.pt"

# ---- Stage 2 (alpha-scale): filter residual disagreements + train cats ----
if ls "${SAVE2A}/${BASE_SHORT}_idx"*"_linear.pt" &>/dev/null 2>&1; then
    echo "[alpha] Stage 2 alpha-scale: already done — skipping."
else
    echo "[alpha] Stage 1.5 + Stage 2 (per_cat_bias_scale, 3-GPU DDP)..."
    # Copy disagreements.pt from Stage 1 so we skip collection
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
        --batch_gen_size 128 \
        --hybrid_gen_batch_size 128 \
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
import json

tag  = '${TAG}'
base = '${BASE_SHORT}'

def parse(fpath):
    rows = [json.loads(l) for l in open(fpath)]
    tc, tn, bc, bn, hc, hn = 0,0,0,0,0,0
    for row in rows:
        j = row.get('judges', {})
        if 'thinking' in j:
            tn += 1
            if j['thinking'].get('correct'): tc += 1
        if 'base' in j:
            bn += 1
            if j['base'].get('correct'): bc += 1
        if 'hybrid' in j:
            hn += 1
            if j['hybrid'].get('correct'): hc += 1
    if hn == 0: return None
    T = tc/tn*100; B = bc/bn*100; H = hc/hn*100
    gap = (H-B)/max(T-B,0.01)*100
    return T, B, H, gap

print(f'\n--- Results: {tag} alpha-scale stage-2 ---')
print('  Reference (standard s15):')
print('    biasonly: ~58%  learn: ~61% gap recovery')
for cond in ['learn', 'biasonly']:
    fpath = f'hybrid/results/rolling/rolling_{base}_math500_{tag}-alpha-{cond}-500.jsonl'
    try:
        r = parse(fpath)
        if r:
            T,B,H,gap = r
            print(f'  {cond:10s}: think={T:.1f}%  base={B:.1f}%  hybrid={H:.1f}%  gap_recovery={gap:.0f}%')
    except Exception as e:
        print(f'  {cond:10s}: {e}')
" 2>/dev/null || true
