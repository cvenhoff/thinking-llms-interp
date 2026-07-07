#!/usr/bin/env bash
# Shared configuration for the category-vector hybrid-steering pipeline.
#
# Source this from any stage script to get the per-pair settings used for the
# paper. Every stage (rollouts, vector training, hybrid eval, ablations) reads
# the same table so a pair is described in exactly one place.
#
#   source "${ROOT}/configs.sh"
#   cfg_load orz-32b        # populates TM/BM/TS/BS/SL/SAEL/NK/FMT/TP/BS_TRAIN
#
# The nine canonical pairs (thinking model steered onto its base model):
CONFIGS=(orz-0.5b orz-1.5b orz-7b orz-32b r1-14b r1-32b qwq-32b r1-llama8b r1-math1.5b)

declare -A THINK_MODEL BASE_MODEL THINK_SHORT BASE_SHORT
declare -A STEER_LAYER SAE_LAYER N_CLUSTERS THINK_FMT

# --- ORZ family (RL-only reasoning models) ---
THINK_MODEL[orz-0.5b]="Open-Reasoner-Zero/Open-Reasoner-Zero-0.5B"; BASE_MODEL[orz-0.5b]="Qwen/Qwen2.5-0.5B"
THINK_SHORT[orz-0.5b]="open-reasoner-zero-0.5b"; BASE_SHORT[orz-0.5b]="qwen2.5-0.5b"
STEER_LAYER[orz-0.5b]=9;  SAE_LAYER[orz-0.5b]=8;  N_CLUSTERS[orz-0.5b]=10; THINK_FMT[orz-0.5b]=orz

THINK_MODEL[orz-1.5b]="Open-Reasoner-Zero/Open-Reasoner-Zero-1.5B"; BASE_MODEL[orz-1.5b]="Qwen/Qwen2.5-1.5B"
THINK_SHORT[orz-1.5b]="open-reasoner-zero-1.5b"; BASE_SHORT[orz-1.5b]="qwen2.5-1.5b"
STEER_LAYER[orz-1.5b]=10; SAE_LAYER[orz-1.5b]=4;  N_CLUSTERS[orz-1.5b]=10; THINK_FMT[orz-1.5b]=orz

THINK_MODEL[orz-7b]="Open-Reasoner-Zero/Open-Reasoner-Zero-7B"; BASE_MODEL[orz-7b]="Qwen/Qwen2.5-7B"
THINK_SHORT[orz-7b]="open-reasoner-zero-7b"; BASE_SHORT[orz-7b]="qwen2.5-7b"
STEER_LAYER[orz-7b]=10; SAE_LAYER[orz-7b]=20; N_CLUSTERS[orz-7b]=10; THINK_FMT[orz-7b]=orz

THINK_MODEL[orz-32b]="Open-Reasoner-Zero/Open-Reasoner-Zero-32B"; BASE_MODEL[orz-32b]="Qwen/Qwen2.5-32B"
THINK_SHORT[orz-32b]="open-reasoner-zero-32b"; BASE_SHORT[orz-32b]="qwen2.5-32b"
STEER_LAYER[orz-32b]=24; SAE_LAYER[orz-32b]=27; N_CLUSTERS[orz-32b]=15; THINK_FMT[orz-32b]=orz

# --- R1-distilled / QwQ family (SFT and/or SFT+RL reasoning models) ---
THINK_MODEL[r1-14b]="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"; BASE_MODEL[r1-14b]="Qwen/Qwen2.5-14B"
THINK_SHORT[r1-14b]="deepseek-r1-distill-qwen-14b"; BASE_SHORT[r1-14b]="qwen2.5-14b"
STEER_LAYER[r1-14b]=18; SAE_LAYER[r1-14b]=38; N_CLUSTERS[r1-14b]=5; THINK_FMT[r1-14b]=r1

THINK_MODEL[r1-32b]="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"; BASE_MODEL[r1-32b]="Qwen/Qwen2.5-32B"
THINK_SHORT[r1-32b]="deepseek-r1-distill-qwen-32b"; BASE_SHORT[r1-32b]="qwen2.5-32b"
STEER_LAYER[r1-32b]=24; SAE_LAYER[r1-32b]=27; N_CLUSTERS[r1-32b]=15; THINK_FMT[r1-32b]=r1

THINK_MODEL[qwq-32b]="Qwen/QwQ-32B"; BASE_MODEL[qwq-32b]="Qwen/Qwen2.5-32B"
THINK_SHORT[qwq-32b]="qwq-32b"; BASE_SHORT[qwq-32b]="qwen2.5-32b"
STEER_LAYER[qwq-32b]=24; SAE_LAYER[qwq-32b]=27; N_CLUSTERS[qwq-32b]=10; THINK_FMT[qwq-32b]=qwq

THINK_MODEL[r1-llama8b]="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"; BASE_MODEL[r1-llama8b]="Meta-Llama/Llama-3.1-8B"
THINK_SHORT[r1-llama8b]="deepseek-r1-distill-llama-8b"; BASE_SHORT[r1-llama8b]="llama-3.1-8b"
STEER_LAYER[r1-llama8b]=12; SAE_LAYER[r1-llama8b]=6; N_CLUSTERS[r1-llama8b]=15; THINK_FMT[r1-llama8b]=r1

THINK_MODEL[r1-math1.5b]="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"; BASE_MODEL[r1-math1.5b]="Qwen/Qwen2.5-Math-1.5B"
THINK_SHORT[r1-math1.5b]="deepseek-r1-distill-qwen-1.5b"; BASE_SHORT[r1-math1.5b]="qwen2.5-math-1.5b"
STEER_LAYER[r1-math1.5b]=10; SAE_LAYER[r1-math1.5b]=4; N_CLUSTERS[r1-math1.5b]=15; THINK_FMT[r1-math1.5b]=r1

# MLP coefficient network hidden width (paper value).
MLP_HIDDEN=512

# Per-pair tensor-parallel size for vLLM rollout generation.
tp_for() { case "$1" in *32b|orz-7b|r1-14b|r1-llama8b) echo 2 ;; *) echo 1 ;; esac; }
# Per-pair training micro-batch size (larger models need smaller batches).
train_bs_for() { case "$1" in *32b) echo 1 ;; *14b) echo 2 ;; orz-7b|r1-llama8b) echo 4 ;; *) echo 8 ;; esac; }
# Per-pair hybrid-eval generation batch size.
hybrid_bs_for() { case "$1" in *32b) echo 8 ;; *) echo 32 ;; esac; }

# Populate TM/BM/TS/BS/SL/SAEL/NK/FMT/TP/BS_TRAIN/BS_HYBRID for a config.
cfg_load() {
    local c="$1"
    [[ -n "${THINK_MODEL[$c]:-}" ]] || { echo "unknown config: $c (choices: ${CONFIGS[*]})" >&2; return 1; }
    TM="${THINK_MODEL[$c]}"; BM="${BASE_MODEL[$c]}"
    TS="${THINK_SHORT[$c]}"; BS="${BASE_SHORT[$c]}"
    SL="${STEER_LAYER[$c]}"; SAEL="${SAE_LAYER[$c]}"; NK="${N_CLUSTERS[$c]}"
    FMT="${THINK_FMT[$c]}"; TP="$(tp_for "$c")"
    BS_TRAIN="$(train_bs_for "$c")"; BS_HYBRID="$(hybrid_bs_for "$c")"
}
