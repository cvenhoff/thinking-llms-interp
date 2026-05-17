#!/usr/bin/env bash
# Paper Table 1 — Qwen2.5-32B + Open-Reasoner-Zero-32B
# Steer layer: 24 | SAE layer: 27 | K: 15
# Original script: hybrid/run_qwen_32b_on_orz.sh @ commit 6bb44d8
set -euo pipefail
cd /workspace/thinking-llms-interp/hybrid

python hybrid_token.py \
    --dataset gsm8k \
    --thinking_model Open-Reasoner-Zero/Open-Reasoner-Zero-32B \
    --base_model Qwen/Qwen2.5-32B \
    --steering_layer 24 \
    --sae_layer 27 --n_clusters 15 \
    --max_new_tokens 2000 --max_thinking_tokens 2000

python hybrid_token.py \
    --dataset math500 \
    --thinking_model Open-Reasoner-Zero/Open-Reasoner-Zero-32B \
    --base_model Qwen/Qwen2.5-32B \
    --steering_layer 24 \
    --sae_layer 27 --n_clusters 15 \
    --max_new_tokens 2000 --max_thinking_tokens 2000
