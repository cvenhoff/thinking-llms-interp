#!/bin/bash
# Patch DeepSeek-R1-Distill-Llama-8B SAE checkpoints with missing activation_mean.
#
# Only patches layer 6 (for the L6 K30 taxonomy we selected).
# At the end, annotates 5 thinking traces with and without mean centering
# for sanity checking.

set -e

cd /workspace/thinking-llms-interp/train-saes
source ../.venv/bin/activate

echo "=============================================="
echo "Patching activation_mean for DeepSeek-R1-Distill-Llama-8B (layer 6)"
echo "Then validating with K=30 SAE on 5 traces"
echo "=============================================="

python -u patch_sae_activation_mean.py \
  --model "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
  --layers 6 \
  --n_examples 100000 \
  --validate_n_clusters 30 \
  --validate_n_traces 5 \
  2>&1

echo ""
echo "=============================================="
echo "DONE — check results/vars/sanity_deepseek-r1-distill-llama-8b_layer6_k30_*.json"
echo "=============================================="
