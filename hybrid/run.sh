#!/usr/bin/env bash
# Stage 3: best-of-3 selection + hybrid evaluation for one pair or all nine.
# For each pair: pick the best vectors on the holdout, then evaluate the hybrid
# model on MATH500, GSM8K and the Hendrycks-MATH holdout.
#
#   bash hybrid/run.sh            # all configs
#   bash hybrid/run.sh orz-32b    # one config
set -euo pipefail
ROOT="${THINKING_LLMS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${ROOT}/configs.sh"
SDIR="${ROOT}/hybrid"

TARGETS=("$@"); [[ ${#TARGETS[@]} -eq 0 ]] && TARGETS=("${CONFIGS[@]}")
for CONFIG in "${TARGETS[@]}"; do
    echo "########## hybrid eval: ${CONFIG} ##########"
    bash "${SDIR}/select_best_of_3.sh" "${CONFIG}"
    for DS in math500 gsm8k; do
        CONFIG="${CONFIG}" DATASET="${DS}" bash "${SDIR}/eval_ood.sh"
    done
    CONFIG="${CONFIG}" bash "${SDIR}/eval_hendrycks.sh"
done
echo "########## hybrid eval complete ##########"
