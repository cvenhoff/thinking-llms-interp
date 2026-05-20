#!/usr/bin/env bash
# No-bias probe: train a single vector per category directly, no Stage 1, no filter.
set -uo pipefail
cd /workspace/thinking-llms-interp
source .venv/bin/activate
source .env_exports.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY=error

LABEL="${TAG}_nobias_steer${STEER}"
SAVE="train-vectors/results/vars/correction_vectors_${LABEL}"
LOG="/tmp/${LABEL}"

echo ""
echo "======================================================"
echo "  NO-BIAS PROBE: ${TAG}  SAE=L${SAE_L}  STEER=L${STEER}  GPU=${GPU}"
echo "======================================================"

mkdir -p "${SAVE}"

if [ -n "${DISAGR_SRC:-}" ] && [ -f "${DISAGR_SRC}" ] && [ ! -f "${SAVE}/disagreements.pt" ]; then
    cp "${DISAGR_SRC}" "${SAVE}/disagreements.pt"
fi

LAUNCHER="CUDA_VISIBLE_DEVICES=${GPU} python -u"

cd train-vectors
eval ${LAUNCHER} optimize_correction_vectors.py \
    --base_model        "$BASE" \
    --thinking_model    "$THINK" \
    --thinking_model_short "$THINK_SHORT" \
    --steer_layer       "$STEER" \
    --sae_classify_layer "$SAE_L" \
    --sae_n_clusters    "$K" \
    --kl_mode topk --topk 50 --train_topk 3 \
    --n_epochs 5 \
    --lr 0.01 \
    --example_batch_size 32 \
    --max_seq_len 2048 --max_positions_per_example 64 \
    --max_positions_per_cat 500 \
    --per_cat_loss \
    --collect_batch_size 32 \
    --holdout_frac 0.1 \
    --n_responses 20000 \
    --collection_mode disagreement \
    --responses_dir "../generate-responses/results/vars" \
    --save_dir "results/vars/correction_vectors_${LABEL}" \
    --seed 42 \
    --load_collected \
    2>&1 | tee "${LOG}.log"
cd /workspace/thinking-llms-interp

echo ""
echo "======================================================"
echo "  RESULTS: ${TAG}  NO-BIAS  STEER=L${STEER}"
echo "======================================================"
python3 - <<PYEOF
import re, sys
log = "${LOG}.log"
try:
    content = open(log).read()
except:
    print("  (log not found)"); sys.exit()
lines = content.split('\n')
best_kl, best_percat = None, None
for i, l in enumerate(lines):
    if 'new best' in l and 'per-cat' not in l:
        m = re.search(r'holdout_kl=(\S+)', l)
        if m: best_kl = float(m.group(1))
        for j in range(max(0,i-3), min(len(lines), i+3)):
            if 'per-cat holdout_kl' in lines[j]:
                best_percat = lines[j]
if best_percat is None:
    for l in lines:
        if 'per-cat holdout_kl' in l: best_percat = l
print(f"  Best holdout KL: {best_kl}")
if best_percat:
    cats = re.findall(r'idx(\d+)=([0-9.]+)', best_percat)
    helpful = sum(1 for _, kl in cats if float(kl) < 1.0)
    print(f"  Per-category ({helpful}/{len(cats)} helpful):")
    for cat, kl in cats:
        v = float(kl)
        tag = 'HELPFUL' if v < 1.0 else ('neutral' if v < 1.01 else 'HARMFUL')
        print(f"    idx{cat}: {v:.3f}  {tag}")
PYEOF
