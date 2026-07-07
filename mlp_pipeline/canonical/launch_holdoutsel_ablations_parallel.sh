#!/usr/bin/env bash
# Fire EVERY remaining negative-control ablation as an independent, concurrent
# srun (no per-config serialization). Each job is idempotent (skips if its
# judge_reps already exist) and reads shared cached rollouts + vectors, so all
# 24 combos are fully independent. Low priority (--nice) => backfill behind the
# hendrycks 32B hybrids; slurm runs as many as fit the QOS at once.
set -uo pipefail
ROOT=/workspace-vast/constantinv/thinking-llms-interp
SDIR="${ROOT}/mlp_pipeline/canonical"
ORCH="${ROOT}/slurm_logs/final_final/orch_holdoutsel_abl"; mkdir -p "${ORCH}"
cd "${ROOT}"
PARTITION="${PARTITION:-general}"; QOS="${QOS:-high}"; NICE="${NICE:-220}"; RETRY="${RETRY:-8}"
ABLS=(randcat randV mlponly randpos)
gres(){ case "$1" in *32b) echo gpu:2;; *) echo gpu:1;; esac; }
mem(){  case "$1" in *32b) echo 384G;; *) echo 64G;; esac; }
bshort(){ case "$1" in orz-1.5b) echo qwen2.5-1.5b;; orz-32b) echo qwen2.5-32b;; esac; }

one(){ # cfg abl ds  -- retrying srun, backgrounded by caller
    local cfg=$1 abl=$2 ds=$3 bs; bs="$(bshort "$cfg")"
    local out="${ROOT}/mlp_eval_qa_instr_holdoutsel_ablations/${cfg}-${abl}/${ds}"
    [[ -f "${out}/judge_reps_${bs}_${ds}_abl_${abl}.json" ]] && { echo "[skip] ${cfg}/${abl}/${ds}"; return; }
    local logf="${ORCH}/${cfg}_${abl}_${ds}.log" a rc
    for (( a=1; a<=RETRY; a++ )); do
        env CONFIG="${cfg}" DATASET="${ds}" ABLATION="${abl}" HBS_OVERRIDE=8 \
            srun --partition="${PARTITION}" --qos="${QOS}" --export=ALL --ntasks=1 \
            --nice="${NICE}" --job-name="abl-${cfg}-${abl}-${ds}" --gres="$(gres "$cfg")" \
            --mem="$(mem "$cfg")" --cpus-per-task=8 --time="8:00:00" \
            stdbuf -oL -eL bash "${SDIR}/eval_qa_instr_holdoutsel_ablation.sh" > "${logf}" 2>&1
        rc=$?
        [[ ${rc} -eq 0 ]] && { echo "[ok] ${cfg}/${abl}/${ds}"; return; }
        grep -q "^FATAL:" "${logf}" 2>/dev/null && { echo "[FATAL] ${cfg}/${abl}/${ds}"; return; }
        echo "[retry] ${cfg}/${abl}/${ds} rc=${rc} a=${a}"; sleep 45
    done
    echo "[gaveup] ${cfg}/${abl}/${ds}"
}

pids=()
for cfg in orz-32b orz-1.5b; do
    for abl in "${ABLS[@]}"; do
        for ds in math500 gsm8k; do
            one "${cfg}" "${abl}" "${ds}" &
            pids+=($!); sleep 1
        done
    done
done
echo "dispatched ${#pids[@]} concurrent ablation jobs; waiting..."
for p in "${pids[@]}"; do wait "$p"; done
echo "[ALL DONE] holdoutsel ablations $(date -u)"
