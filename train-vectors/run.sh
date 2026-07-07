#!/usr/bin/env bash
# Stage 2: train the best-of-3 vector sets for one pair or all nine.
#
#   bash train-vectors/run.sh            # all configs
#   bash train-vectors/run.sh orz-32b    # one config
#
# Three independent runs per pair (seeds 42/43/44): run1 under
# mlp_vectors_qa_instr_h512/<cfg>, run2/run3 under the _bo3 tree. These are
# intermediate candidates that only feed the best-of-3 selection in stage 3, which
# promotes the winner to mlp_vectors_qa_instr_holdoutsel_h512/<cfg>. Only that
# winner is shipped; rerun this stage to regenerate the candidates.
set -euo pipefail
ROOT="${THINKING_LLMS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${ROOT}/configs.sh"
SDIR="${ROOT}/train-vectors"
BO3="${ROOT}/artifacts/mlp_vectors_qa_instr_h${MLP_HIDDEN}_bo3"

TARGETS=("$@"); [[ ${#TARGETS[@]} -eq 0 ]] && TARGETS=("${CONFIGS[@]}")
for CONFIG in "${TARGETS[@]}"; do
    echo "########## train-vectors: ${CONFIG} ##########"
    CONFIG="${CONFIG}" SEED=42 SAVE_DIR="${ROOT}/artifacts/mlp_vectors_qa_instr_h${MLP_HIDDEN}/${CONFIG}" bash "${SDIR}/train_vectors.sh"
    CONFIG="${CONFIG}" SEED=43 SAVE_DIR="${BO3}/${CONFIG}/run2" bash "${SDIR}/train_vectors.sh"
    CONFIG="${CONFIG}" SEED=44 SAVE_DIR="${BO3}/${CONFIG}/run3" bash "${SDIR}/train_vectors.sh"
done
echo "########## train-vectors complete ##########"
