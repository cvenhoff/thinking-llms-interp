#!/usr/bin/env bash
# best-of-3 (bo3) TRAINING orchestrator.
# Goal: every one of the 9 canonical model pairs ends up with 3 independent
# qa_instr h512 vector runs so we can pick the best via cheap trainmix-holdout
# steering accuracy before the expensive hybrid eval.
#
#   run1 = existing canonical vectors at mlp_vectors_qa_instr_h512/<cfg>
#          (already on disk from the main training; NOT retrained here)
#   run2, run3 trained here land in mlp_vectors_qa_instr_h512_bo3/<cfg>/run<N>
#
# Each new run REUSES the config's existing disagree_cache.pt (symlinked in
# already) so Phase-A collection is skipped -- only the DDP optimisation runs,
# with a distinct seed per run for an independent draw.
#
# Run detached:
#   setsid nohup bash mlp_pipeline/canonical/orchestrate_bo3_train.sh >boot_bo3_train 2>&1 &

set -uo pipefail
ROOT=/workspace-vast/constantinv/thinking-llms-interp
SDIR="${ROOT}/mlp_pipeline/canonical"
ORCH="${ROOT}/slurm_logs/final_final/orch_bo3_train"
mkdir -p "${ORCH}"
cd "${ROOT}"

MASTER_LOG="${ORCH}/orchestrator.log"
PARTITION="${PARTITION:-general}"
QOS="${QOS:-high}"
MAX_RETRY="${MAX_RETRY:-4}"
MLP_HIDDEN=512
MAX_CONCURRENT="${MAX_CONCURRENT:-8}"
BO3="${ROOT}/mlp_vectors_qa_instr_h${MLP_HIDDEN}_bo3"

log() { echo "[$(date -u '+%F %T')] $*" >> "${MASTER_LOG}"; }

# per-config resources
gres_for()  { case "$1" in *32b) echo "gpu:2";; *14b) echo "gpu:2";; orz-7b|r1-llama8b) echo "gpu:2";; *) echo "gpu:1";; esac; }
mem_for()   { case "$1" in *32b) echo "384G";; *14b) echo "256G";; orz-7b|r1-llama8b) echo "160G";; *) echo "64G";; esac; }
bs_for()    { case "$1" in *32b) echo 1;; *14b) echo 2;; orz-7b|r1-llama8b) echo 4;; *) echo 8;; esac; }
time_for()  { case "$1" in *32b) echo "24:00:00";; *14b) echo "16:00:00";; orz-7b|r1-llama8b) echo "12:00:00";; *) echo "8:00:00";; esac; }

vectors_ok() { [[ -f "$1/cat_coef_mlp.pt" && -f "$1/mlp_config.json" ]]; }

train_run() {
    local cfg=$1 runN=$2 seed=$3
    local save="${BO3}/${cfg}/run${runN}"
    local logf="${ORCH}/train_${cfg}_run${runN}.log"
    local label="bo3-${cfg}-r${runN}"
    if vectors_ok "${save}"; then log "${label}: vectors present, skip"; return 0; fi
    mkdir -p "${save}"
    # ensure disagree cache symlink exists (idempotent)
    [[ -e "${save}/disagree_cache.pt" ]] || \
        ln -sfn "${ROOT}/mlp_vectors_qa_instr_h${MLP_HIDDEN}/${cfg}/disagree_cache.pt" "${save}/disagree_cache.pt"
    local attempt rc
    for (( attempt=1; attempt<=MAX_RETRY; attempt++ )); do
        log "${label}: attempt ${attempt}/${MAX_RETRY} seed=${seed} -> ${logf}"
        env "CONFIG=${cfg}" "MLP_HIDDEN=${MLP_HIDDEN}" "TRAIN_BS=$(bs_for "${cfg}")" \
            "SEED_OVR=${seed}" "SAVE_DIR_OVR=${save}" \
            srun --partition="${PARTITION}" --qos="${QOS}" --export=ALL --ntasks=1 \
                 --job-name="${label}" \
                 --gres="$(gres_for "${cfg}")" --mem="$(mem_for "${cfg}")" \
                 --cpus-per-task=8 --time="$(time_for "${cfg}")" \
                 stdbuf -oL -eL bash "${SDIR}/train_qa_instr_hsweep.sh" > "${logf}" 2>&1
        rc=$?
        if [[ ${rc} -eq 0 ]] && vectors_ok "${save}"; then
            log "${label}: SUCCESS (attempt ${attempt})"; return 0; fi
        if grep -q "^FATAL:" "${logf}" 2>/dev/null; then
            log "${label}: FATAL (rc=${rc}); not retrying."; return 2; fi
        log "${label}: failed rc=${rc}; retry after 30s"; sleep 30
    done
    log "${label}: GAVE UP"; return 1
}

# throttle to MAX_CONCURRENT background jobs
throttle() { while (( $(jobs -rp | wc -l) >= MAX_CONCURRENT )); do sleep 20; done; }

log "##### bo3 TRAIN boot (pid $$) MAX_CONCURRENT=${MAX_CONCURRENT} #####"

# manifest: cfg:runN:seed  (run1 = existing canonical, not listed)
RUNS=(
  "orz-32b:2:43"    "orz-32b:3:44"
  "orz-0.5b:2:43"   "orz-0.5b:3:44"
  "orz-1.5b:2:43"   "orz-1.5b:3:44"
  "orz-7b:2:43"     "orz-7b:3:44"
  "r1-14b:2:43"     "r1-14b:3:44"
  "r1-llama8b:2:43" "r1-llama8b:3:44"
  "qwq-32b:2:43"    "qwq-32b:3:44"
  "r1-32b:2:43"     "r1-32b:3:44"
  "r1-math1.5b:2:43" "r1-math1.5b:3:44"
)

for spec in "${RUNS[@]}"; do
    IFS=: read -r cfg runN seed <<< "${spec}"
    throttle
    train_run "${cfg}" "${runN}" "${seed}" &
    sleep 3
done
wait
log "##### bo3 TRAIN all runs complete #####"

# summary of which run dirs now have vectors
{
  echo "==== bo3 train summary $(date -u) ===="
  for spec in "${RUNS[@]}"; do
    IFS=: read -r cfg runN seed <<< "${spec}"
    d="${BO3}/${cfg}/run${runN}"
    if vectors_ok "${d}"; then echo "OK   ${cfg} run${runN}"; else echo "MISS ${cfg} run${runN}"; fi
  done
} | tee -a "${MASTER_LOG}"
