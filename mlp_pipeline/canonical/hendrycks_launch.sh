#!/usr/bin/env bash
# Launch all 9 hendrycks-MATH holdout chains as independent background drivers.
# 32B configs first (longest), then 14b/8b/7b, then small. Each chain is
# idempotent and self-healing; safe to re-run.
set -uo pipefail
ROOT=/workspace-vast/constantinv/thinking-llms-interp
SDIR="${ROOT}/mlp_pipeline/canonical"
ORCH="${ROOT}/slurm_logs/final_final/orch_hendrycks_chain"
mkdir -p "${ORCH}"
CONFIGS=(orz-32b qwq-32b r1-32b r1-14b r1-llama8b orz-7b orz-1.5b r1-math1.5b orz-0.5b)
for c in "${CONFIGS[@]}"; do
    if pgrep -fa "hendrycks_chain.sh ${c}$" >/dev/null 2>&1; then
        echo "[skip] chain already running: ${c}"; continue; fi
    setsid nohup bash "${SDIR}/hendrycks_chain.sh" "${c}" \
        > "${ORCH}/driver_${c}.log" 2>&1 &
    echo "[launch] ${c} (driver pid $!)"
    sleep 3
done
echo "all chains launched. drivers: ${ORCH}/driver_*.log"
