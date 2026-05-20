#!/usr/bin/env bash
# Quick steer-layer probe: 1 epoch, small cap — just get holdout KL signal.
set -uo pipefail
cd /workspace/thinking-llms-interp
source .venv/bin/activate
source .env_exports.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY=error

K="${K:-10}"
NPROC="${NPROC:-1}"
LABEL="${TAG}_sae${SAE_L}_steer${STEER}"
SAVE1="train-vectors/results/vars/correction_vectors_${LABEL}_stage1"
SAVE2="train-vectors/results/vars/correction_vectors_${LABEL}_stage2"
LOG="/tmp/${LABEL}"

echo ""
echo "======================================================"
echo "  STEER-LAYER PROBE: ${TAG}  SAE=L${SAE_L}  STEER=L${STEER}  GPU=${GPU}"
echo "======================================================"

mkdir -p "${SAVE1}" "${SAVE2}"

if [ -n "${DISAGR_SRC:-}" ] && [ -f "${DISAGR_SRC}" ] && [ ! -f "${SAVE1}/disagreements.pt" ]; then
    cp "${DISAGR_SRC}" "${SAVE1}/disagreements.pt"
fi

LAUNCHER="CUDA_VISIBLE_DEVICES=${GPU} python -u"
[ "${NPROC:-1}" -gt 1 ] && \
    LAUNCHER="CUDA_VISIBLE_DEVICES=${GPU} torchrun --standalone --nproc_per_node=${NPROC}"

# ---- Stage 1: bias ----
if [ -f "${SAVE1}/${BASE_SHORT}_bias_global.pt" ]; then
    echo "[stage1] already done — skipping."
else
    echo "[stage1] training bias at layer ${STEER}..."
    cd train-vectors
    eval ${LAUNCHER} optimize_correction_vectors.py \
        --base_model        "$BASE" \
        --thinking_model    "$THINK" \
        --thinking_model_short "$THINK_SHORT" \
        --steer_layer       "$STEER" \
        --sae_classify_layer "$SAE_L" \
        --sae_n_clusters    "$K" \
        --kl_mode topk --topk 50 --train_topk 3 \
        --n_epochs 1 \
        --lr 0.01 \
        --example_batch_size 32 \
        --max_seq_len 2048 --max_positions_per_example 64 \
        --max_positions_per_cat 500 \
        --holdout_frac 0.1 \
        --n_responses 20000 \
        --collection_mode disagreement \
        --responses_dir "../generate-responses/results/vars" \
        --save_dir "results/vars/correction_vectors_${LABEL}_stage1" \
        --seed 42 \
        --load_collected \
        --skip_cats_phase \
        --train_global_bias \
        2>&1 | tee "${LOG}_stage1.log"
    cd /workspace/thinking-llms-interp
    [ -f "${SAVE1}/${BASE_SHORT}_bias_global.pt" ] \
        || { echo "ERROR: Stage 1 failed"; exit 1; }
fi

BIAS_PATH_FROM_TV="results/vars/correction_vectors_${LABEL}_stage1/${BASE_SHORT}_bias_global.pt"

[ -f "${SAVE1}/disagreements.pt" ] && [ ! -f "${SAVE2}/disagreements.pt" ] \
    && cp "${SAVE1}/disagreements.pt" "${SAVE2}/disagreements.pt"

# ---- Stage 2: cat vectors ----
if ls "${SAVE2}/${BASE_SHORT}_idx"*"_linear.pt" &>/dev/null 2>&1; then
    echo "[stage2] already done — skipping."
else
    echo "[stage2] training cat vectors at layer ${STEER}..."
    cd train-vectors
    eval ${LAUNCHER} optimize_correction_vectors.py \
        --base_model        "$BASE" \
        --thinking_model    "$THINK" \
        --thinking_model_short "$THINK_SHORT" \
        --steer_layer       "$STEER" \
        --sae_classify_layer "$SAE_L" \
        --sae_n_clusters    "$K" \
        --kl_mode topk --topk 50 --train_topk 3 \
        --n_epochs 1 \
        --lr 0.01 \
        --example_batch_size 32 \
        --max_seq_len 2048 --max_positions_per_example 64 \
        --max_positions_per_cat 500 \
        --per_cat_loss \
        --per_cat_bias_scale \
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

# ---- Extract per-cat holdout KL ----
echo ""
echo "======================================================"
echo "  RESULTS: ${TAG}  SAE=L${SAE_L}  STEER=L${STEER}"
echo "======================================================"
python3 - <<PYEOF
import re, sys

log = "${LOG}_stage2.log"
try:
    content = open(log).read()
except:
    print("  (log not found)"); sys.exit()

lines = content.split('\n')

# Collect all per-cat holdout_kl lines and find the best-checkpoint one
best_kl, best_percat = None, None
for i, l in enumerate(lines):
    if 'new best' in l and 'per-cat' not in l:
        m = re.search(r'holdout_kl=(\S+)', l)
        if m:
            best_kl = float(m.group(1))
        # look for per-cat line nearby
        for j in range(max(0,i-3), min(len(lines), i+3)):
            if 'per-cat holdout_kl' in lines[j]:
                best_percat = lines[j]

if best_percat is None:
    # fallback: last per-cat line
    for l in lines:
        if 'per-cat holdout_kl' in l:
            best_percat = l

print(f"  Best holdout KL: {best_kl}")
if best_percat:
    cats = re.findall(r'idx(\d+)=([0-9.]+)', best_percat)
    helpful = sum(1 for _, kl in cats if float(kl) < 1.0)
    print(f"  Per-category ({helpful}/{len(cats)} helpful):")
    for cat, kl in cats:
        v = float(kl)
        tag = 'HELPFUL' if v < 1.0 else ('neutral' if v < 1.01 else 'HARMFUL')
        print(f"    idx{cat}: {v:.3f}  {tag}")
else:
    print("  (no per-cat KL found yet)")
PYEOF
