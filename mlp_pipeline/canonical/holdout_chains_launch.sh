#!/usr/bin/env bash
# Launch all 9 independent holdout-gap selection chains. 32B first so they enter
# the QOS queue earliest. Each chain self-heals and is idempotent.
ROOT=/workspace-vast/constantinv/thinking-llms-interp
SDIR="${ROOT}/mlp_pipeline/canonical"
ORCH="${ROOT}/slurm_logs/final_final/orch_holdout_chain"
mkdir -p "${ORCH}"; cd "${ROOT}"
CONFIGS=(orz-32b qwq-32b r1-32b r1-14b orz-7b r1-llama8b orz-1.5b orz-0.5b r1-math1.5b)
for cfg in "${CONFIGS[@]}"; do
    if pgrep -fa "holdout_chain.sh ${cfg}$" >/dev/null 2>&1; then
        echo "chain ${cfg} already running, skip"; continue; fi
    setsid nohup bash "${SDIR}/holdout_chain.sh" "${cfg}" \
        > "${ORCH}/${cfg}.nohup.log" 2>&1 &
    echo "launched chain ${cfg} (pid $!)"
    sleep 3
done
echo "all holdout chains launched $(date -u)"
