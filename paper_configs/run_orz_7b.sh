#!/usr/bin/env bash
# Paper Table 1 — Qwen2.5-7B + Open-Reasoner-Zero-7B
# Steer layer: 10 | SAE layer: 20 | K: 10
# Original script: hybrid/run_qwen_7b.sh @ commit bf9df36
set -euo pipefail
cd /workspace/thinking-llms-interp/hybrid

python hybrid_token.py \
    --dataset gsm8k \
    --thinking_model Open-Reasoner-Zero/Open-Reasoner-Zero-7B \
    --base_model Qwen/Qwen2.5-7B \
    --steering_layer 10 \
    --sae_layer 20 --n_clusters 10 \
    --max_new_tokens 2000 --max_thinking_tokens 2000 \
    --coefficients 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --token_windows 0 -1 -15 -50 -100

python hybrid_token.py \
    --dataset math500 \
    --thinking_model Open-Reasoner-Zero/Open-Reasoner-Zero-7B \
    --base_model Qwen/Qwen2.5-7B \
    --steering_layer 10 \
    --sae_layer 20 --n_clusters 10 \
    --max_new_tokens 2000 --max_thinking_tokens 2000 \
    --coefficients 0.1 0.2 0.3 0.4 0.5 0.6 0.7 0.8 0.9 1.0 \
    --token_windows 0 -1 -15 -50 -100
