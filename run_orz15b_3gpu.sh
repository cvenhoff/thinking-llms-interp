#!/usr/bin/env bash
# ============================================================
# Stage-1.5 recipe for ORZ-1.5B using all 3 GPUs via DDP.
# Stage 1 and Stage 2 training both use torchrun --nproc_per_node=3.
# Eval runs all 3 conditions sequentially on GPU 0.
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

SAVE1="train-vectors/results/vars/correction_vectors_${TAG}_s15_stage1"
SAVE2="train-vectors/results/vars/correction_vectors_${TAG}_s15_stage2"
LOG_PREFIX="/tmp/${TAG}_s15"

echo ""
echo "======================================================"
echo "  Stage-1.5 recipe (3-GPU DDP): ${TAG}"
echo "  Base  : ${BASE}"
echo "  Think : ${THINK}"
echo "  Steer : L${STEER}  SAE: L${SAE_L}  K=${K}"
echo "  GPUs  : 0,1,2 (DDP)  Coef: ${COEF}"
echo "======================================================"

mkdir -p "${SAVE1}" "${SAVE2}"

# ---- Stage 1: collect + train bias (3-GPU DDP) ----
if [ -f "${SAVE1}/${BASE_SHORT}_bias_global.pt" ]; then
    echo "[s15] Stage 1: already done — skipping."
else
    echo "[s15] Stage 1: collect disagreements + train global bias (3-GPU DDP)..."
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
        --collect_batch_size 16 \
        --holdout_frac 0.1 \
        --n_responses 20000 \
        --collection_mode disagreement \
        --responses_dir "../generate-responses/results/vars" \
        --save_dir "results/vars/correction_vectors_${TAG}_s15_stage1" \
        --seed 42 \
        --skip_cats_phase \
        --train_global_bias \
        2>&1 | tee "${LOG_PREFIX}_stage1.log"
    cd /workspace/thinking-llms-interp
    [ -f "${SAVE1}/${BASE_SHORT}_bias_global.pt" ] \
        || { echo "ERROR: Stage 1 failed (no bias saved)"; exit 1; }
    echo "[s15] Stage 1 done."
fi

BIAS_PATH_FROM_TV="results/vars/correction_vectors_${TAG}_s15_stage1/${BASE_SHORT}_bias_global.pt"
BIAS_LAYER_JSON="${SAVE1}/bias_layer.json"
if [ ! -f "$BIAS_LAYER_JSON" ]; then
    echo "{\"layer\": $STEER}" > "$BIAS_LAYER_JSON"
fi

# ---- Stage 1.5 + Stage 2: filter residual disagreements + train cats (3-GPU DDP) ----
if ls "${SAVE2}/${BASE_SHORT}_idx"*"_linear.pt" &>/dev/null 2>&1; then
    echo "[s15] Stage 2: already done — skipping."
else
    echo "[s15] Stage 1.5 + Stage 2: filter residual disagrs + train cats (3-GPU DDP)..."
    if [ -f "${SAVE1}/disagreements.pt" ] && [ ! -f "${SAVE2}/disagreements.pt" ]; then
        echo "  Copying disagreements.pt from stage 1..."
        cp "${SAVE1}/disagreements.pt" "${SAVE2}/disagreements.pt"
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
        --collect_batch_size 16 \
        --holdout_frac 0.1 \
        --n_responses 20000 \
        --collection_mode disagreement \
        --responses_dir "../generate-responses/results/vars" \
        --save_dir "results/vars/correction_vectors_${TAG}_s15_stage2" \
        --seed 42 \
        --load_collected \
        --filter_by_bias \
        --frozen_bias_path "$BIAS_PATH_FROM_TV" \
        --frozen_bias_layer "$STEER" \
        2>&1 | tee "${LOG_PREFIX}_stage2.log"
    cd /workspace/thinking-llms-interp
    NCAT=$(ls "${SAVE2}/${BASE_SHORT}_idx"*"_linear.pt" 2>/dev/null | wc -l)
    [ "$NCAT" -gt 0 ] || { echo "ERROR: Stage 2 failed (no cat vectors saved)"; exit 1; }
    echo "[s15] Stage 2 done. ${NCAT} cat vectors saved."
fi

# ---- Eval: run 3 conditions sequentially on GPU 0 ----
run_eval() {
    local COND="$1"
    local EXTRA="$2"
    local SUFFIX="${TAG}-s15-${COND}-500"
    local ROLLING="hybrid/results/rolling/rolling_${BASE_SHORT}_math500_${SUFFIX}.jsonl"

    if [ -f "$ROLLING" ] && [ "$(python3 -c "
import json; rows=[json.loads(l) for l in open('$ROLLING')]
print(len([r for r in rows if 'judges' in r]))" 2>/dev/null)" -ge 500 ]; then
        echo "[s15] Eval ${COND}: already done — skipping."
        return
    fi

    echo "[s15] Eval: ${COND}..."
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
        --dom_vectors_dir   "../${SAVE2}" \
        --old_vectors_dir   "../${SAVE2}" \
        --old_vectors_layer "$STEER" \
        --bias_vector_path  "../${SAVE1}/${BASE_SHORT}_bias_global.pt" \
        --bias_layer        "$STEER" \
        --fixed_coef "$COEF" \
        --judge_repetitions 3 \
        --results_suffix "$SUFFIX" \
        $EXTRA \
        2>&1 | tee "${LOG_PREFIX}_eval_${COND}.log"
    cd /workspace/thinking-llms-interp
    echo "[s15] Eval ${COND} done."
}

run_eval "learn"    ""
run_eval "biasonly" "--bias_only"
run_eval "rand"     "--randomize_vectors --random_seed 42"

echo ""
echo "[s15] ALL DONE for ${TAG}."

python3 -c "
import json, glob, sys

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

print(f'\n--- Results: {tag} stage-1.5 ---')
for cond in ['learn', 'biasonly', 'rand']:
    fpath = f'hybrid/results/rolling/rolling_{base}_math500_{tag}-s15-{cond}-500.jsonl'
    try:
        counts = parse(fpath)
        think_acc = counts['thinking']['c'] / counts['thinking']['n'] * 100
        base_acc  = counts['base']['c']     / counts['base']['n']     * 100
        hyb_acc   = counts['hybrid']['c']   / counts['hybrid']['n']   * 100
        gap  = think_acc - base_acc
        rec  = (hyb_acc - base_acc) / max(gap, 0.01) * 100
        print(f'  {cond:10s}: think={think_acc:.1f}% base={base_acc:.1f}% hybrid={hyb_acc:.1f}%  gap_recovery={rec:.0f}%')
    except Exception as e:
        print(f'  {cond:10s}: {e}')
" 2>/dev/null || true
