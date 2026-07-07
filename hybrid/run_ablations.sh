#!/usr/bin/env bash
# Stage 4: negative-control ablations (paper Fig. ablations). Run on one small
# and one large pair (orz-1.5b, orz-32b) across the four controls and both math
# benchmarks, against the canonical selected vectors.
#
#   bash hybrid/run_ablations.sh
set -euo pipefail
ROOT="${THINKING_LLMS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
SDIR="${ROOT}/hybrid"
CONFIGS_ABL=(orz-1.5b orz-32b)
ABLATIONS=(randcat randV mlponly randpos)

for CONFIG in "${CONFIGS_ABL[@]}"; do
    for ABLATION in "${ABLATIONS[@]}"; do
        for DS in math500 gsm8k; do
            echo "########## ablation: ${CONFIG} ${ABLATION} ${DS} ##########"
            CONFIG="${CONFIG}" DATASET="${DS}" ABLATION="${ABLATION}" bash "${SDIR}/eval_ablation.sh"
        done
    done
done
echo "########## ablations complete ##########"
