#!/usr/bin/env bash
# Paper Table 1 — Qwen2.5-32B + QwQ-32B
# Steer layer: 24 | SAE layer: 27 | K: 10
# Original script: hybrid/run_qwen_32b_on_qwq.sh @ commit bf9df36
set -euo pipefail
cd /workspace/thinking-llms-interp/hybrid

python hybrid_token.py \
    --dataset gsm8k \
    --thinking_model Qwen/QwQ-32B \
    --base_model Qwen/Qwen2.5-32B \
    --steering_layer 24 \
    --sae_layer 27 --n_clusters 10 \
    --max_new_tokens 2000 --max_thinking_tokens 2000

python hybrid_token.py \
    --dataset math500 \
    --thinking_model Qwen/QwQ-32B \
    --base_model Qwen/Qwen2.5-32B \
    --steering_layer 24 \
    --sae_layer 27 --n_clusters 10 \
    --max_new_tokens 2000 --max_thinking_tokens 2000
