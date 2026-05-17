#!/usr/bin/env bash
# Paper Table 1 — Qwen2.5-32B + DeepSeek-R1-Distill-Qwen-32B
# Steer layer: 24 | SAE layer: 27 | K: 15
# Original script: hybrid/run_qwen_32b_on_deepseek.sh @ commit bf9df36
set -euo pipefail
cd /workspace/thinking-llms-interp/hybrid

python hybrid_token.py \
    --dataset gsm8k \
    --thinking_model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
    --base_model Qwen/Qwen2.5-32B \
    --steering_layer 24 \
    --sae_layer 27 --n_clusters 15 \
    --max_new_tokens 2000 --max_thinking_tokens 2000

python hybrid_token.py \
    --dataset math500 \
    --thinking_model deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
    --base_model Qwen/Qwen2.5-32B \
    --steering_layer 24 \
    --sae_layer 27 --n_clusters 15 \
    --max_new_tokens 2000 --max_thinking_tokens 2000
