#!/usr/bin/env bash
# Paper Table 1 — Qwen2.5-0.5B + Open-Reasoner-Zero-0.5B
# Steer layer: 9 | SAE layer: 8 | K: 10
# Original script: hybrid/run_qwen_0.5b.sh @ commit bf9df36
set -euo pipefail
cd /workspace/thinking-llms-interp/hybrid

python hybrid_token.py \
    --dataset gsm8k \
    --thinking_model Open-Reasoner-Zero/Open-Reasoner-Zero-0.5B \
    --base_model Qwen/Qwen2.5-0.5B \
    --steering_layer 9 \
    --sae_layer 8 --n_clusters 10 \
    --max_new_tokens 2000 --max_thinking_tokens 2000 \
    --coefficients 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --token_windows 0 -1 -15 -50 -100

python hybrid_token.py \
    --dataset math500 \
    --thinking_model Open-Reasoner-Zero/Open-Reasoner-Zero-0.5B \
    --base_model Qwen/Qwen2.5-0.5B \
    --steering_layer 9 \
    --sae_layer 8 --n_clusters 10 \
    --max_new_tokens 2000 --max_thinking_tokens 2000 \
    --coefficients 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --token_windows 0 -1 -15 -50 -100
