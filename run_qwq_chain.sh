#!/bin/bash
# QwQ-32B / Qwen2.5-32B bias-first chain: Stage 1 -> Stage 2 (canonical recipe).
# Stage 0 should already have produced disagreements.pt under stage1 dir.
set -euo pipefail
cd /workspace/thinking-llms-interp
source .venv/bin/activate
source .env_exports.sh

STAGE1_DIR="train-vectors/results/vars/correction_vectors_qwq32b_biasfirst_stage1"
if [ ! -f "$STAGE1_DIR/disagreements.pt" ]; then
    echo "ERROR: $STAGE1_DIR/disagreements.pt missing (run stage 0 first)"; exit 1
fi

echo "===== QwQ Stage 1 (bias) ====="
bash train-vectors/run_qwq32b_biasfirst_stage1.sh

echo "===== QwQ Stage 2 (collect under bias + cats) ====="
bash train-vectors/run_qwq32b_biasfirst_stage2.sh

echo "DONE QwQ chain."
