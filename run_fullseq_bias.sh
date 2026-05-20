#!/usr/bin/env bash
# Full-seq bias recipe:
#   Stage 1: train bias on ALL reasoning positions (full next-token KL)
#   Stage 2: collect disagreements, apply frozen bias, train cat vectors
# Compares against no-bias (run_nobias_probe.sh) at same steer layer.
set -uo pipefail
cd /workspace/thinking-llms-interp
source .venv/bin/activate
source .env_exports.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY=error

K="${K:-10}"
LABEL="${TAG}_fullseq_steer${STEER}"
SAVE1="train-vectors/results/vars/correction_vectors_${LABEL}_stage1"
SAVE2="train-vectors/results/vars/correction_vectors_${LABEL}_stage2"
LOG="/tmp/${LABEL}"

echo ""
echo "======================================================"
echo "  FULL-SEQ BIAS RECIPE: ${TAG}  SAE=L${SAE_L}  STEER=L${STEER}  GPU=${GPU}"
echo "======================================================"

mkdir -p "${SAVE1}" "${SAVE2}"

LAUNCHER="CUDA_VISIBLE_DEVICES=${GPU} python -u"

# ---- Stage 1: bias on ALL reasoning positions ----
if [ -f "${SAVE1}/${BASE_SHORT}_bias_global.pt" ]; then
    echo "[stage1] already done — skipping."
else
    echo "[stage1] training bias on full-sequence KL (all positions)..."
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
        --holdout_frac 0.1 \
        --n_responses 2048 \
        --collection_mode disagreement \
        --responses_dir "../generate-responses/results/vars" \
        --save_dir "results/vars/correction_vectors_${LABEL}_stage1" \
        --seed 42 \
        --skip_cats_phase \
        --train_global_bias \
        --full_seq_bias \
        2>&1 | tee "${LOG}_stage1.log"
    cd /workspace/thinking-llms-interp
    [ -f "${SAVE1}/${BASE_SHORT}_bias_global.pt" ] \
        || { echo "ERROR: Stage 1 failed"; exit 1; }
    echo "[stage1] done."
fi

BIAS_PATH_FROM_TV="results/vars/correction_vectors_${LABEL}_stage1/${BASE_SHORT}_bias_global.pt"

# Copy disagreements.pt for stage 2
if [ -n "${DISAGR_SRC:-}" ] && [ -f "${DISAGR_SRC}" ] && [ ! -f "${SAVE2}/disagreements.pt" ]; then
    echo "  Copying disagreements.pt from ${DISAGR_SRC}..."
    cp "${DISAGR_SRC}" "${SAVE2}/disagreements.pt"
fi

# ---- Stage 2: cat vectors on residual disagreements ----
if ls "${SAVE2}/${BASE_SHORT}_idx"*"_linear.pt" &>/dev/null 2>&1; then
    echo "[stage2] already done — skipping."
else
    echo "[stage2] training cat vectors on residual disagreements..."
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
        --save_dir "results/vars/correction_vectors_${LABEL}_stage2" \
        --seed 42 \
        --load_collected \
        --filter_by_bias \
        --frozen_bias_path "$BIAS_PATH_FROM_TV" \
        --frozen_bias_layer "$STEER" \
        2>&1 | tee "${LOG}_stage2.log"
    cd /workspace/thinking-llms-interp
fi

# ---- Report per-cat holdout KL ----
echo ""
echo "======================================================"
echo "  RESULTS: ${TAG}  FULL-SEQ BIAS  STEER=L${STEER}"
echo "======================================================"
python3 - <<PYEOF
import re, sys
log = "${LOG}_stage2.log"
try:
    content = open(log).read()
except:
    print("  (log not found)"); sys.exit()
lines = content.split('\n')

# Print full KL curve
print("  KL curve (new bests):")
for l in lines:
    if 'new best holdout_kl' in l:
        m = re.search(r'new best holdout_kl=(\S+) at (\S+)', l)
        if m: print(f"    {m.group(2)}: {m.group(1)}")

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
print(f"\n  Best holdout KL: {best_kl}")
if best_percat:
    cats = re.findall(r'idx(\d+)=([0-9.]+)', best_percat)
    helpful = sum(1 for _, kl in cats if float(kl) < 1.0)
    print(f"  Per-category ({helpful}/{len(cats)} helpful):")
    for cat, kl in cats:
        v = float(kl)
        tag = 'HELPFUL' if v < 1.0 else ('neutral' if v < 1.01 else 'HARMFUL')
        print(f"    idx{cat}: {v:.3f}  {tag}")
PYEOF
