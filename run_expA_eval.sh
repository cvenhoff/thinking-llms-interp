#!/usr/bin/env bash
# Eval script for ExpA cats + bias
# Runs after run_expA_cats_allpos.sh completes
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
SAVE_A="train-vectors/results/vars/correction_vectors_${TAG}_expA_allpos"

run_eval() {
    local COND="$1"
    local EXTRA="$2"
    local SUFFIX="${TAG}-expA-${COND}-500"
    local ROLLING="hybrid/results/rolling/rolling_${BASE_SHORT}_math500_${SUFFIX}.jsonl"

    if [ -f "$ROLLING" ] && [ "$(python3 -c "
import json; rows=[json.loads(l) for l in open('$ROLLING')]
print(len([r for r in rows if 'judges' in r]))" 2>/dev/null)" -ge 500 ]; then
        echo "[expA eval] ${COND}: already done — skipping."
        return
    fi

    echo "[expA eval] Running: ${COND}..."
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
        --dom_vectors_dir   "../${SAVE_A}" \
        --old_vectors_dir   "../${SAVE_A}" \
        --old_vectors_layer "$STEER" \
        --bias_vector_path  "../${SAVE1}/${BASE_SHORT}_bias_global.pt" \
        --bias_layer        "$STEER" \
        --fixed_coef "$COEF" \
        --judge_repetitions 3 \
        --results_suffix "$SUFFIX" \
        $EXTRA \
        2>&1 | tee "/workspace/tmp/expA_eval_${COND}.log"
    cd /workspace/thinking-llms-interp
    echo "[expA eval] ${COND} done."
}

echo "======================================================"
echo "  ExpA Eval: bias + all-position cats"
echo "  (testing if removing stage1.5 filter fixes cats)"
echo "======================================================"

# First run learn (most important: is bias+expA_cats > biasonly?)
run_eval "learn"    ""
# Then biasonly for direct comparison
run_eval "biasonly" "--bias_only"

# Print summary
python3 -c "
import json, glob

tag  = '${TAG}'
base = '${BASE_SHORT}'

def parse(fpath):
    rows = [json.loads(l) for l in open(fpath)]
    counts = {}
    for row in rows:
        j = row.get('judges', {})
        jlist = j if isinstance(j, list) else [j]
        for item in jlist:
            for cond in ['thinking', 'base', 'hybrid']:
                if cond in item:
                    counts.setdefault(cond, {'c': 0, 'n': 0})
                    counts[cond]['n'] += 1
                    if item[cond].get('correct'):
                        counts[cond]['c'] += 1
    return counts

print(f'\n--- ExpA Results vs Stage2 baseline ---')
print(f'  biasonly (s15): think=57.8%  base=44.2%  hybrid=52.8%  gap_recovery=63%')
print(f'  learn    (s15): think=57.8%  base=44.6%  hybrid=52.4%  gap_recovery=59%')
print()
for cond in ['learn', 'biasonly']:
    fpath = f'hybrid/results/rolling/rolling_{base}_math500_{tag}-expA-{cond}-500.jsonl'
    try:
        counts = parse(fpath)
        think_acc = counts['thinking']['c'] / counts['thinking']['n'] * 100
        base_acc  = counts['base']['c']     / counts['base']['n']     * 100
        hyb_acc   = counts['hybrid']['c']   / counts['hybrid']['n']   * 100
        gap  = think_acc - base_acc
        rec  = (hyb_acc - base_acc) / max(gap, 0.01) * 100
        print(f'  {cond:10s} (expA): think={think_acc:.1f}%  base={base_acc:.1f}%  hybrid={hyb_acc:.1f}%  gap_recovery={rec:.0f}%')
    except Exception as e:
        print(f'  {cond:10s}: {e}')
" 2>/dev/null || true
