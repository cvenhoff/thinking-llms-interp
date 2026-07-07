#!/usr/bin/env bash
# Stage 1: generate every rollout the pipeline needs, for one pair or all nine.
#
#   bash vllm-serve/run_rollouts.sh            # all configs
#   bash vllm-serve/run_rollouts.sh orz-32b    # one config
#
# Produces base + thinking rollouts for the training mix, MATH500, GSM8K, the
# best-of-3 selection holdout, and the Hendrycks-MATH holdout. Each generator is
# idempotent (skips caches that are already complete), so reruns are cheap.
set -euo pipefail
ROOT="${THINKING_LLMS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${ROOT}/configs.sh"
SDIR="${ROOT}/vllm-serve"

TARGETS=("$@"); [[ ${#TARGETS[@]} -eq 0 ]] && TARGETS=("${CONFIGS[@]}")
for CONFIG in "${TARGETS[@]}"; do
    cfg_load "${CONFIG}"
    echo "########## rollouts: ${CONFIG} ##########"
    MODEL="${TM}" SHORT="${TS}" FMT="${FMT}" TP="${TP}" VLLM_PORT=8000 bash "${SDIR}/gen_think.sh"
    ROLE=think MODEL="${TM}" SHORT="${TS}" TP="${TP}" VLLM_PORT=9700 bash "${SDIR}/gen_hendrycks.sh"
    BASE_MODEL="${BM}" BASE_SHORT="${BS}" TP="${TP}" VLLM_PORT=9100 bash "${SDIR}/gen_base.sh"
    ROLE=base MODEL="${BM}" SHORT="${BS}" TP="${TP}" VLLM_PORT=9700 bash "${SDIR}/gen_hendrycks.sh"
done
echo "########## rollouts complete ##########"
