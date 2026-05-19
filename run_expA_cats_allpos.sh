#!/usr/bin/env bash
# =============================================================
# Experiment A: Train cat vectors on ALL disagreements (no stage-1.5 filter)
# with frozen bias. This tests whether removing the residual-only filter
# fixes the anti-alignment between cats and bias.
#
# Key difference from stage2: --filter_by_bias is REMOVED.
# Cats train on ALL 251k positions (capped 1k/cat) with frozen bias.
# =============================================================
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
SAVE_A="train-vectors/results/vars/correction_vectors_${TAG}_expA_allpos"
LOG_A="/workspace/tmp/expA_allpos"

mkdir -p "${SAVE_A}" /workspace/tmp

echo ""
echo "======================================================"
echo "  Experiment A: cats on ALL positions, frozen bias"
echo "  (no stage-1.5 filter — fixing anti-alignment)"
echo "  GPUs: 1,2  Epochs: 10  Cap: 1000/cat"
echo "======================================================"

# Copy collected disagreements if not already present
if [ ! -f "${SAVE_A}/disagreements.pt" ]; then
    echo "[ExpA] Copying disagreements.pt from stage1..."
    cp "${SAVE1}/disagreements.pt" "${SAVE_A}/disagreements.pt"
fi

BIAS_PATH_FROM_TV="results/vars/correction_vectors_${TAG}_s15_stage1/${BASE_SHORT}_bias_global.pt"

echo "[ExpA] Training cat vectors on ALL disagreements with frozen bias..."
cd train-vectors
CUDA_VISIBLE_DEVICES=1,2 torchrun \
    --standalone --nproc_per_node=2 \
    optimize_correction_vectors.py \
    --base_model        "$BASE" \
    --thinking_model    "$THINK" \
    --thinking_model_short "$THINK_SHORT" \
    --steer_layer       "$STEER" \
    --sae_classify_layer "$SAE_L" \
    --sae_n_clusters    "$K" \
    --kl_mode topk --topk 50 --train_topk 3 \
    --n_epochs 10 --lr 0.01 \
    --example_batch_size 32 \
    --max_seq_len 2048 --max_positions_per_example 64 \
    --max_positions_per_cat 1000 \
    --per_cat_loss \
    --collect_batch_size 16 \
    --holdout_frac 0.1 \
    --seed 42 \
    --load_collected \
    --frozen_bias_path  "$BIAS_PATH_FROM_TV" \
    --frozen_bias_layer "$STEER" \
    --save_dir "results/vars/correction_vectors_${TAG}_expA_allpos" \
    2>&1 | tee "${LOG_A}.log"
cd /workspace/thinking-llms-interp

NCAT=$(ls "${SAVE_A}/${BASE_SHORT}_idx"*"_linear.pt" 2>/dev/null | wc -l)
[ "$NCAT" -gt 0 ] || { echo "ERROR: ExpA failed (no cat vectors saved)"; exit 1; }
echo "[ExpA] Training done. ${NCAT} cat vectors saved to ${SAVE_A}"

# Quick cosine analysis
echo "[ExpA] Checking alignment with bias..."
python3 -c "
import torch, glob, os

save_a = '${SAVE_A}'
save1  = '${SAVE1}'

bias_obj = torch.load(f'{save1}/${BASE_SHORT}_bias_global.pt', map_location='cpu', weights_only=False)
bias = (bias_obj.get('bias', next(iter(bias_obj.values()))) if isinstance(bias_obj,dict) else bias_obj).float().squeeze()
bias_n = bias / bias.norm()
print(f'Bias norm: {bias.norm().item():.3f}')

print('ExpA cat stats:')
for f in sorted(glob.glob(f'{save_a}/${BASE_SHORT}_idx*_linear.pt')):
    key = os.path.basename(f).split('_')[2]
    obj = torch.load(f, map_location='cpu', weights_only=False)
    v = (obj[key] if isinstance(obj,dict) and key in obj else next(iter(obj.values())) if isinstance(obj,dict) else obj).float().squeeze()
    vn = v / v.norm().clamp_min(1e-8)
    print(f'  {key}: norm={v.norm().item():.2f}  cos_bias={(vn*bias_n).sum().item():+.3f}')
" 2>/dev/null

echo "[ExpA] Done."
