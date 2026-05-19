#!/usr/bin/env bash
# ============================================================
# Unified-norm single-stage training for ORZ-1.5B.
#
# Single joint optimisation of:
#   - shared unit-norm bias direction b
#   - per-cat unit-norm directions V[i]
#   - per-cat bias scales alpha[i]   (init ~||bias|| = 10)
#   - per-cat cat scales  beta[i]    (init 1.0)
# Steering for cat i = alpha[i]*normalize(b) + beta[i]*normalize(V[i])
#
# Saves:
#   bias_global.pt          - normalize(b)  (norm = 1)
#   idx*_linear.pt          - beta[i]*normalize(V[i])
#   bias_alpha.json         - per-cat alpha values
#
# hybrid_eval.py loads these unchanged: cat_vec[k] + alpha[k]*bias_vec
#   = beta[k]*V_hat[k] + alpha[k]*b_hat  (correct)
#
# No frozen bias, no filter_by_bias, no Stage 1/2 split.
# Cap raised to 3k positions per category for more signal.
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

SAVE="train-vectors/results/vars/correction_vectors_${TAG}_unified"
LOG_PREFIX="/tmp/${TAG}_unified"

echo ""
echo "======================================================"
echo "  Unified-norm single-stage training: ${TAG}"
echo "  Base  : ${BASE}"
echo "  Think : ${THINK}"
echo "  Steer : L${STEER}  SAE: L${SAE_L}  K=${K}"
echo "  GPUs  : 0,1,2 (DDP)  Coef: ${COEF}"
echo "  Cap   : 3000 positions/cat  (up from 1000)"
echo "======================================================"

mkdir -p "${SAVE}"

# ---- Single-stage training (3-GPU DDP) ----
if ls "${SAVE}/${BASE_SHORT}_idx"*"_linear.pt" &>/dev/null 2>&1; then
    echo "[unified] Training: already done — skipping."
else
    echo "[unified] Collecting disagreements + training (3-GPU DDP)..."
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
        --max_positions_per_cat 3000 \
        --per_cat_loss \
        --unified_norm \
        --init_alpha 10.0 \
        --init_beta  1.0 \
        --collect_batch_size 16 \
        --holdout_frac 0.1 \
        --n_responses 20000 \
        --collection_mode disagreement \
        --responses_dir "../generate-responses/results/vars" \
        --save_dir "results/vars/correction_vectors_${TAG}_unified" \
        --seed 42 \
        2>&1 | tee "${LOG_PREFIX}_train.log"
    cd /workspace/thinking-llms-interp
    NCAT=$(ls "${SAVE}/${BASE_SHORT}_idx"*"_linear.pt" 2>/dev/null | wc -l)
    [ "$NCAT" -gt 0 ] || { echo "ERROR: Training failed (no cat vectors saved)"; exit 1; }
    echo "[unified] Training done. ${NCAT} cat vectors saved."
    # Show what alpha and beta learned
    python3 -c "
import json, os
alpha_path = '${SAVE}/bias_alpha.json'
if os.path.exists(alpha_path):
    d = json.load(open(alpha_path))
    vals = list(d.values())
    print(f'  alpha: min={min(vals):.2f}  max={max(vals):.2f}  mean={sum(vals)/len(vals):.2f}')
    for k,v in sorted(d.items()):
        print(f'    {k}: {v:.3f}')
" 2>/dev/null || true
fi

# ---- Eval ----
run_eval() {
    local COND="$1"
    local EXTRA="$2"
    local SUFFIX="${TAG}-unified-${COND}-500"
    local ROLLING="hybrid/results/rolling/rolling_${BASE_SHORT}_math500_${SUFFIX}.jsonl"

    if [ -f "$ROLLING" ] && [ "$(python3 -c "
import json; rows=[json.loads(l) for l in open('$ROLLING')]
print(len([r for r in rows if 'judges' in r]))" 2>/dev/null)" -ge 490 ]; then
        echo "[unified] Eval ${COND}: already done — skipping."
        return
    fi

    echo "[unified] Eval: ${COND}..."
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
        --dom_vectors_dir   "../${SAVE}" \
        --old_vectors_dir   "../${SAVE}" \
        --old_vectors_layer "$STEER" \
        --bias_vector_path  "../${SAVE}/${BASE_SHORT}_bias_global.pt" \
        --bias_layer        "$STEER" \
        --fixed_coef "$COEF" \
        --judge_repetitions 3 \
        --results_suffix "$SUFFIX" \
        $EXTRA \
        2>&1 | tee "${LOG_PREFIX}_eval_${COND}.log"
    cd /workspace/thinking-llms-interp
    echo "[unified] Eval ${COND} done."
}

run_eval "learn"    ""
run_eval "biasonly" "--bias_only"

echo ""
echo "[unified] ALL DONE for ${TAG}."

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

print(f'\n--- Results: {tag} unified-norm ---')
print('  Reference (standard s15): biasonly~58%  learn~61%')
print('  Reference (alpha-scale):  learn~66%')
for cond in ['learn', 'biasonly']:
    fpath = f'hybrid/results/rolling/rolling_{base}_math500_{tag}-unified-{cond}-500.jsonl'
    try:
        r = parse(fpath)
        if r:
            T,B,H,gap = r
            print(f'  {cond:10s}: think={T:.1f}%  base={B:.1f}%  hybrid={H:.1f}%  gap_recovery={gap:.0f}%')
    except Exception as e:
        print(f'  {cond:10s}: {e}')
" 2>/dev/null || true
