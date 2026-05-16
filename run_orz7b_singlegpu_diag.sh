#!/bin/bash
# Diagnostic: rerun ORZ-7B Stage 2b cat-vector training on a SINGLE GPU
# (no device_map sharding) using the EXACT same canonical data /
# hyperparameters as run_orz7b_canon.sh.  We reuse:
#   - the Stage 1 bias  from  correction_vectors_orz7b_biasfirst_stage1_canon/
#   - the Stage 2a disagreements from correction_vectors_orz7b_biasfirst_stage2_canon/
#
# Hypothesis: the multi-GPU pipeline-parallel adaptations added to
# optimize_correction_vectors.py since commit b078d01 introduce a
# regression on stage-2b only (cat vectors learn a direction that
# anti-correlates with the frozen bias).  Stage 1 (bias only) and
# stage 2a (collect_only) don't allreduce/cross-device anything that
# would break.
#
# Pass criterion: trained cats end with cosine(V[k], bias) >= 0 for
# the dominant categories (idx8, idx9), comparable to the >+0.3 we
# saw in the smaller sanity run on the same machine.

set -uo pipefail
cd /workspace/thinking-llms-interp
source .venv/bin/activate
source .env_exports.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1

LAYER=16
SAVE2_FROM_TV="results/vars/correction_vectors_orz7b_biasfirst_stage2_singlegpu_diag"
SAVE2="train-vectors/$SAVE2_FROM_TV"

CANON_S1_FROM_TV="results/vars/correction_vectors_orz7b_biasfirst_stage1_canon"
CANON_S2_FROM_TV="results/vars/correction_vectors_orz7b_biasfirst_stage2_canon"
CANON_S1="train-vectors/$CANON_S1_FROM_TV"
CANON_S2="train-vectors/$CANON_S2_FROM_TV"

[ -f "$CANON_S1/qwen2.5-7b_bias_global.pt" ] || { echo "Missing bias"; exit 1; }
[ -f "$CANON_S2/disagreements.pt" ] || { echo "Missing disagreements"; exit 1; }

mkdir -p "$SAVE2"
# Reuse the canon disagreements (stage 2a output) verbatim - that file
# was a no-grad collection, so even if PP affected it, comparing
# 1-GPU vs 3-GPU stage-2b training on identical inputs is the right
# diagnostic.
cp -f "$CANON_S2/disagreements.pt" "$SAVE2/disagreements.pt"

LOG="/tmp/orz7b_singlegpu_diag.log"

# Train on a single GPU (CUDA_VISIBLE_DEVICES=0 forces everything onto
# GPU 0; device_map="auto" then can't shard - all submodules end up on
# the one visible device, matching the historical 1-GPU run).
echo "===== Stage 2b single-GPU diagnostic: 5 epochs, BS=16, L${LAYER} ====="
cd /workspace/thinking-llms-interp/train-vectors
CUDA_VISIBLE_DEVICES=0 python -u optimize_correction_vectors.py \
    --base_model "Qwen/Qwen2.5-7B" \
    --thinking_model "Open-Reasoner-Zero/Open-Reasoner-Zero-7B" \
    --thinking_model_short "open-reasoner-zero-7b" \
    --steer_layer "$LAYER" \
    --save_dir "$SAVE2_FROM_TV" \
    --topk 50 --train_topk 3 --kl_mode topk \
    --max_seq_len 2048 --max_positions_per_example 64 \
    --n_epochs 5 --example_batch_size 16 \
    --lr 0.01 --weight_decay 0.0 --max_norm 0.0 \
    --seed 42 --holdout_frac 0.1 \
    --min_disagreements 1 --min_disagreements_ratio 0.0 --min_category_share 0.0 \
    --frozen_bias_path "../$CANON_S1/qwen2.5-7b_bias_global.pt" --frozen_bias_layer "$LAYER" \
    --load_collected \
    2>&1 | tee "$LOG"

cd /workspace/thinking-llms-interp
echo
echo "DONE training. Cat-vs-bias cosine analysis:"
python3 -c "
import torch, glob, os
b = torch.load('$CANON_S1/qwen2.5-7b_bias_global.pt', map_location='cpu', weights_only=False)
bias = (b['bias'] if isinstance(b, dict) else b).float().squeeze()
print(f'bias norm = {float(bias.norm()):.3f}')
print()
print(f'{\"cat\":>5}  {\"||V||\":>8}  {\"cos(V,bias)\":>12}')
for p in sorted(glob.glob('$SAVE2/qwen2.5-7b_idx*_linear.pt'), key=lambda s: int(s.split('idx')[1].split('_')[0])):
    o = torch.load(p, map_location='cpu', weights_only=False)
    k = list(o.keys())[0]
    v = o[k].float()
    cos = float((v*bias).sum() / (v.norm()*bias.norm()+1e-8))
    print(f'  {k:>5}  {float(v.norm()):8.3f}  {cos:+12.3f}')
"
