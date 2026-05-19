#!/bin/bash
# Quick coeff sensitivity test with current stage2 cats (on GPU 0 when free)
set -uo pipefail
cd /workspace/thinking-llms-interp
source .venv/bin/activate; source .env_exports.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True PYTHONUNBUFFERED=1 TRANSFORMERS_VERBOSITY=error

BASE="Qwen/Qwen2.5-1.5B"; THINK="Open-Reasoner-Zero/Open-Reasoner-Zero-1.5B"
BASE_SHORT="qwen2.5-1.5b"; TAG="orz-1.5b"; SAE_L=16; K=10; STEER=14
SAVE1="train-vectors/results/vars/correction_vectors_${TAG}_s15_stage1"
SAVE2="train-vectors/results/vars/correction_vectors_${TAG}_s15_stage2"

for COEF in 0.3 0.5; do
    SUFFIX="${TAG}-s15-learnc${COEF/./_}-500"
    ROLLING="hybrid/results/rolling/rolling_${BASE_SHORT}_math500_${SUFFIX}.jsonl"
    [ -f "$ROLLING" ] && python3 -c "
import json; rows=[json.loads(l) for l in open('$ROLLING')]
n=len([r for r in rows if 'judges' in r]); print(n)" 2>/dev/null | grep -q "^500$" && { echo "coef=$COEF: already done"; continue; }
    rm -f "$ROLLING"
    echo "=== Testing coef=$COEF ==="
    cd hybrid
    CUDA_VISIBLE_DEVICES=0 python -u hybrid_eval.py \
        --dataset math500 --n_tasks 0 --eval_start_idx 0 \
        --max_new_tokens 2000 --max_thinking_tokens 2000 \
        --batch_gen_size 128 --hybrid_gen_batch_size 128 \
        --base_model "$BASE" --thinking_model "$THINK" \
        --sae_layer "$SAE_L" --n_clusters "$K" \
        --dom_vectors_dir   "../${SAVE2}" \
        --old_vectors_dir   "../${SAVE2}" \
        --old_vectors_layer "$STEER" \
        --bias_vector_path  "../${SAVE1}/${BASE_SHORT}_bias_global.pt" \
        --bias_layer        "$STEER" \
        --fixed_coef "$COEF" \
        --judge_repetitions 3 \
        --results_suffix "$SUFFIX" \
        2>&1 | tee "/workspace/tmp/s15_coef${COEF/./_}.log"
    cd /workspace/thinking-llms-interp
done
echo "Coeff sensitivity done"
