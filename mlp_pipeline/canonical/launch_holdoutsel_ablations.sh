#!/usr/bin/env bash
# Submit the up-to-date negative-control ablations for orz-1.5b + orz-32b on
# math500 + gsm8k, against the current holdout-selected vectors. Low priority
# (high --nice) so they backfill behind the hendrycks-holdout campaign.
#
#   bash launch_holdoutsel_ablations.sh            # spawns one driver per config
#   bash launch_holdoutsel_ablations.sh <config>   # (internal) drives one config
set -uo pipefail
ROOT=/workspace-vast/constantinv/thinking-llms-interp
SDIR="${ROOT}/mlp_pipeline/canonical"
ORCH="${ROOT}/slurm_logs/final_final/orch_holdoutsel_abl"; mkdir -p "${ORCH}"
cd "${ROOT}"
PARTITION="${PARTITION:-general}"; QOS="${QOS:-high}"; NICE="${NICE:-220}"; RETRY="${RETRY:-6}"
ABLS=(randcat randV mlponly randpos)

gres(){ case "$1" in *32b) echo gpu:2;; *) echo gpu:1;; esac; }
mem(){  case "$1" in *32b) echo 384G;; *) echo 64G;; esac; }
bshort(){ case "$1" in orz-1.5b) echo qwen2.5-1.5b;; orz-32b) echo qwen2.5-32b;; esac; }

# ---- dispatch: no arg => spawn a driver per config ----
if [[ $# -eq 0 ]]; then
    for cfg in orz-32b orz-1.5b; do
        setsid nohup bash "$0" "$cfg" > "${ORCH}/driver_${cfg}.log" 2>&1 &
        echo "[launch] ${cfg} ablation driver (pid $!)"; sleep 2
    done
    echo "launched. logs: ${ORCH}/"
    exit 0
fi

# ---- per-config driver ----
CFG="$1"; BS="$(bshort "$CFG")"
for abl in "${ABLS[@]}"; do
    for ds in math500 gsm8k; do
        out="${ROOT}/mlp_eval_qa_instr_holdoutsel_ablations/${CFG}-${abl}/${ds}"
        if [[ -f "${out}/judge_reps_${BS}_${ds}_abl_${abl}.json" ]]; then
            echo "[skip] ${CFG}/${ds}/${abl} done"; continue; fi
        logf="${ORCH}/${CFG}_${abl}_${ds}.log"
        ok=0
        for (( a=1; a<=RETRY; a++ )); do
            env CONFIG="${CFG}" DATASET="${ds}" ABLATION="${abl}" HBS_OVERRIDE=8 \
                srun --partition="${PARTITION}" --qos="${QOS}" --export=ALL --ntasks=1 \
                --nice="${NICE}" --job-name="abl-${CFG}-${abl}-${ds}" --gres="$(gres "$CFG")" \
                --mem="$(mem "$CFG")" --cpus-per-task=8 --time="8:00:00" \
                stdbuf -oL -eL bash "${SDIR}/eval_qa_instr_holdoutsel_ablation.sh" > "${logf}" 2>&1
            rc=$?
            [[ ${rc} -eq 0 ]] && { echo "[ok] ${CFG}/${ds}/${abl}"; ok=1; break; }
            grep -q "^FATAL:" "${logf}" 2>/dev/null && { echo "[FATAL] ${CFG}/${ds}/${abl} (see ${logf})"; break; }
            echo "[retry] ${CFG}/${ds}/${abl} rc=${rc} (attempt ${a})"; sleep 40
        done
        [[ ${ok} -eq 1 ]] || echo "[incomplete] ${CFG}/${ds}/${abl}"
    done
done
echo "[done] ${CFG} all ablations"
