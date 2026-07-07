#!/usr/bin/env bash
# Stage 5: render every paper figure and table from the eval artifacts into
# figures/figs/.
#
#   bash figures/run.sh
set -euo pipefail
ROOT="${THINKING_LLMS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
export THINKING_LLMS_ROOT="${ROOT}"
cd "${ROOT}"
source .env_exports.sh 2>/dev/null || true
source .venv/bin/activate 2>/dev/null || true
SDIR="${ROOT}/figures"

python "${SDIR}/render_result_tables.py"          # main-results + gap-recovered + train-mix tables
python "${SDIR}/plot_ablation_bars.py"            # ablation bar plot
python "${SDIR}/render_loss_curves_qa_instr_h512.py"  # vector training loss curves
python "${SDIR}/make_hybrid_example_figure_orz32b.py" # ORZ-32B hybrid rollout example
echo "figures written to ${SDIR}/figs/"
