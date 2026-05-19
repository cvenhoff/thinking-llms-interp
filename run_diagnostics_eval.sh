#!/usr/bin/env bash
# Diagnostic evaluations to understand cat vector failure modes
# Tests: norm-matched cats, Stage2 cats at reduced scale, cats only (no bias)
set -uo pipefail
cd /workspace/thinking-llms-interp
source .venv/bin/activate
source .env_exports.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY=error

BASE="Qwen/Qwen2.5-1.5B"
THINK="Open-Reasoner-Zero/Open-Reasoner-Zero-1.5B"
BASE_SHORT="qwen2.5-1.5b"
TAG="orz-1.5b"
SAE_L=16
K=10
STEER=14
COEF=1.0

SAVE1="train-vectors/results/vars/correction_vectors_${TAG}_s15_stage1"
SAVE2="train-vectors/results/vars/correction_vectors_${TAG}_s15_stage2"
SAVE_NORMED="train-vectors/results/vars/correction_vectors_${TAG}_expA_normed"

GPU="$1"

run_eval() {
    local COND="$1"
    local VEC_DIR="$2"
    local EXTRA="$3"
    local SUFFIX="${TAG}-diag-${COND}-500"
    local ROLLING="hybrid/results/rolling/rolling_${BASE_SHORT}_math500_${SUFFIX}.jsonl"

    if [ -f "$ROLLING" ] && [ "$(python3 -c "
import json; rows=[json.loads(l) for l in open('$ROLLING')]
print(len([r for r in rows if 'judges' in r]))" 2>/dev/null)" -ge 490 ]; then
        echo "[diag eval] ${COND}: already done — skipping."
        return
    fi

    echo "[diag eval] Running: ${COND}..."
    rm -f "$ROLLING"
    cd hybrid
    CUDA_VISIBLE_DEVICES="${GPU}" python -u hybrid_eval.py \
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
        --dom_vectors_dir   "../${VEC_DIR}" \
        --old_vectors_dir   "../${VEC_DIR}" \
        --old_vectors_layer "$STEER" \
        --bias_vector_path  "../${SAVE1}/${BASE_SHORT}_bias_global.pt" \
        --bias_layer        "$STEER" \
        --fixed_coef "$COEF" \
        --judge_repetitions 3 \
        --results_suffix "$SUFFIX" \
        $EXTRA \
        2>&1 | tee "/workspace/tmp/diag_${COND}.log"
    cd /workspace/thinking-llms-interp
    echo "[diag eval] ${COND} done."
}

echo "======================================================"
echo "  Diagnostic Evals on GPU ${GPU}"
echo "======================================================"

# Test 1: Norm-matched cats (same direction as expA cats, but |combined| = |bias|)
run_eval "expAnormed"  "$SAVE_NORMED"  ""

# Test 2: Stage2 cats (the ones that gave 59%) for comparison
run_eval "stage2cats"  "$SAVE2"       ""

# Print summary
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

print('\n--- Diagnostic Results ---')
print('  biasonly    (s15): think=57.6%  base=44.4%  hybrid=52.0%  gap_recovery=58%')
print('  learn       (s15): think=57.8%  base=44.6%  hybrid=52.4%  gap_recovery=59%')
print('  learn       (expA): think=57.8%  base=44.6%  hybrid=44.4%  gap_recovery=-2%')
for cond in ['expAnormed', 'stage2cats']:
    fpath = f'hybrid/results/rolling/rolling_{base}_math500_{tag}-diag-{cond}-500.jsonl'
    try:
        r = parse(fpath)
        if r:
            T,B,H,gap = r
            print(f'  {cond:15s}: think={T:.1f}%  base={B:.1f}%  hybrid={H:.1f}%  gap_recovery={gap:.0f}%')
    except Exception as e:
        print(f'  {cond}: {e}')
" 2>/dev/null || true
