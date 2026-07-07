#!/usr/bin/env bash
# Independent per-config chain using the HOLDOUT-GAP selection method:
#   0) ensure vLLM base rollouts for the 788-Q trainmix holdout
#   1) SELECT: real hybrid gap-recovery on the holdout for each of the 3 vector
#      runs (base greedy + think s0 cached; only hybrid is generated)
#   2) pick the run with highest holdout gap -> promote to _holdoutsel
#   3) HYBRID math500 + gsm8k (VARIANT=_holdoutsel)
# Idempotent; eval jobs run at --nice so they backfill politely.
#
#   bash mlp_pipeline/canonical/holdout_chain.sh <config>
set -uo pipefail
CONFIG="${1:?usage: holdout_chain.sh <config>}"
ROOT=/workspace-vast/constantinv/thinking-llms-interp
SDIR="${ROOT}/mlp_pipeline/canonical"
MLP_HIDDEN=512
SELVEC="${ROOT}/mlp_vectors_qa_instr_holdoutsel_h${MLP_HIDDEN}/${CONFIG}"
ORCH="${ROOT}/slurm_logs/final_final/orch_holdout_chain"
mkdir -p "${ORCH}"; LOG="${ORCH}/${CONFIG}.log"
cd "${ROOT}"; source .venv/bin/activate 2>/dev/null || true
PARTITION="${PARTITION:-general}"; QOS="${QOS:-high}"; NICE="${NICE:-120}"; RETRY="${RETRY:-6}"
log(){ echo "[$(date -u '+%F %T')] [${CONFIG}] $*" | tee -a "${LOG}"; }

declare -A BASE_MODEL BASE_SHORT
BASE_MODEL[orz-0.5b]="Qwen/Qwen2.5-0.5B";  BASE_SHORT[orz-0.5b]=qwen2.5-0.5b
BASE_MODEL[orz-1.5b]="Qwen/Qwen2.5-1.5B";  BASE_SHORT[orz-1.5b]=qwen2.5-1.5b
BASE_MODEL[orz-7b]="Qwen/Qwen2.5-7B";      BASE_SHORT[orz-7b]=qwen2.5-7b
BASE_MODEL[orz-32b]="Qwen/Qwen2.5-32B";    BASE_SHORT[orz-32b]=qwen2.5-32b
BASE_MODEL[r1-14b]="Qwen/Qwen2.5-14B";     BASE_SHORT[r1-14b]=qwen2.5-14b
BASE_MODEL[r1-llama8b]="Meta-Llama/Llama-3.1-8B"; BASE_SHORT[r1-llama8b]=llama-3.1-8b
BASE_MODEL[qwq-32b]="Qwen/Qwen2.5-32B";    BASE_SHORT[qwq-32b]=qwen2.5-32b
BASE_MODEL[r1-32b]="Qwen/Qwen2.5-32B";     BASE_SHORT[r1-32b]=qwen2.5-32b
BASE_MODEL[r1-math1.5b]="Qwen/Qwen2.5-Math-1.5B"; BASE_SHORT[r1-math1.5b]=qwen2.5-math-1.5b
BM="${BASE_MODEL[$CONFIG]}"; BS="${BASE_SHORT[$CONFIG]}"

gres_hy(){ case "$1" in *32b|*14b|orz-7b|r1-llama8b) echo gpu:2;; *) echo gpu:1;; esac; }
mem_hy(){ case "$1" in *32b) echo 384G;; *14b) echo 256G;; orz-7b|r1-llama8b) echo 160G;; *) echo 64G;; esac; }
gres_bg(){ case "$1" in *32b) echo gpu:2;; *) echo gpu:1;; esac; }
tp_bg(){ case "$1" in *32b) echo 2;; *) echo 1;; esac; }
port_bg(){ echo $(( 9300 + (RANDOM % 400) )); }

# ---- 3 vector run dirs for this config (uniform layout for all configs) ----
DIRS=("${ROOT}/mlp_vectors_qa_instr_h${MLP_HIDDEN}/${CONFIG}" \
      "${ROOT}/mlp_vectors_qa_instr_h${MLP_HIDDEN}_bo3/${CONFIG}/run2" \
      "${ROOT}/mlp_vectors_qa_instr_h${MLP_HIDDEN}_bo3/${CONFIG}/run3")

srun_retry(){ # jobname logf gres mem timelimit -- ENV... -- script
    local jn=$1 logf=$2 gres=$3 mem=$4 tlim=$5; shift 5; shift
    local -a env_a=(); while [[ "$1" != "--" ]]; do env_a+=("$1"); shift; done; shift
    local script=$1 a rc
    for (( a=1; a<=RETRY; a++ )); do
        log "${jn}: attempt ${a}/${RETRY} -> $(basename "${logf}")"
        env "${env_a[@]}" srun --partition="${PARTITION}" --qos="${QOS}" --export=ALL \
            --ntasks=1 --nice="${NICE}" --job-name="${jn}" --gres="${gres}" \
            --mem="${mem}" --cpus-per-task=8 --time="${tlim}" \
            stdbuf -oL -eL bash "${script}" > "${logf}" 2>&1
        rc=$?
        [[ ${rc} -eq 0 ]] && { log "${jn}: SUCCESS"; return 0; }
        grep -q "^FATAL:" "${logf}" 2>/dev/null && { log "${jn}: FATAL rc=${rc}"; return 2; }
        log "${jn}: failed rc=${rc}; retry in 40s"; sleep 40
    done
    log "${jn}: GAVE UP"; return 1
}

log "##### holdout chain start (pid $$) nice=${NICE} #####"

# ---- STAGE 0: ensure vLLM base rollouts for the holdout ----
BASE_DST="${ROOT}/hybrid/results/response_cache_final/base_${BS}_holdoutmix_temp0_max2048.jsonl"
NEED=$(wc -l < "${ROOT}/data/trainmix_holdout_eval/eval.jsonl")
if [[ -f "${BASE_DST}" ]] && [[ $(wc -l < "${BASE_DST}") -ge ${NEED} ]]; then
    log "base holdout rollouts present ($(wc -l < "${BASE_DST}"))"
else
    srun_retry "hbase-${CONFIG}" "${ORCH}/base_${CONFIG}.log" "$(gres_bg "$CONFIG")" \
        "$(mem_hy "$CONFIG")" "4:00:00" -- \
        "BASE_MODEL=${BM}" "BASE_SHORT=${BS}" "TP=$(tp_bg "$CONFIG")" "VLLM_PORT=$(port_bg)" \
        -- "${SDIR}/gen_base_holdoutmix.sh" || { log "base gen failed; abort"; exit 1; }
fi

# ---- STAGE 1: selection sweep over 3 runs (base pre-generated -> run in parallel) ----
sel_pids=()
for i in 0 1 2; do
    d="${DIRS[$i]}"; tag="run$((i+1))"
    if [[ -f "${d}/cat_coef_mlp.pt" ]]; then :; else log "sel ${tag}: MISSING vectors ${d}"; continue; fi
    if ls "${ROOT}/mlp_eval_holdoutmix/${CONFIG}/${tag}"/judge_reps_*_holdoutmix_final.json >/dev/null 2>&1; then
        log "sel ${tag}: done, skip"; continue; fi
    ( srun_retry "hsel-${CONFIG}-${tag}" "${ORCH}/sel_${CONFIG}_${tag}.log" \
        "$(gres_hy "$CONFIG")" "$(mem_hy "$CONFIG")" "8:00:00" -- \
        "CONFIG=${CONFIG}" "VEC_DIR=${d}" "RUNTAG=${tag}" -- \
        "${SDIR}/eval_holdoutmix_run.sh" ) &
    sel_pids+=($!); sleep 8
done
for p in "${sel_pids[@]}"; do wait "$p"; done
log "selection sweep done"

# ---- STAGE 2: pick best run by holdout gap, promote ----
best=$(python - "$CONFIG" "$BS" "${DIRS[@]}" <<'PY'
import json, sys, os
cfg, bs = sys.argv[1], sys.argv[2]; dirs = sys.argv[3:]
ROOT="/workspace-vast/constantinv/thinking-llms-interp"
best=None; bg=-1e9; rank=[]
for i,d in enumerate(dirs):
    f=f"{ROOT}/mlp_eval_holdoutmix/{cfg}/run{i+1}/judge_reps_{bs}_holdoutmix_final.json"
    if not os.path.exists(f): continue
    pr=json.load(open(f))["per_rep"]
    b=pr["base"]["mean_pct"]; t=pr["thinking"]["mean_pct"]; h=pr["hybrid"]["mean_pct"]
    g=(h-b)/(t-b)*100 if (t-b)!=0 else -1e9
    rank.append((f"run{i+1}", g)); 
    if g>bg: bg=g; best=d
sys.stderr.write("holdout gaps: "+", ".join(f"{r}={g:.1f}" for r,g in rank)+"\n")
print(best or ""); print(f"{bg:.2f}")
PY
)
best_dir=$(echo "$best" | sed -n '1p'); best_gap=$(echo "$best" | sed -n '2p')
if [[ -z "${best_dir}" || ! -f "${best_dir}/cat_coef_mlp.pt" ]]; then
    log "SELECTION FAILED (no judge_reps); abort"; exit 1; fi
log "best=${best_dir} holdout_gap=${best_gap}"
mkdir -p "${SELVEC}"
cp -f "${best_dir}/cat_coef_mlp.pt" "${best_dir}/mlp_config.json" "${SELVEC}/" 2>/dev/null
cp -f "${best_dir}/layer_map.json" "${SELVEC}/" 2>/dev/null || true
cp -f "${best_dir}"/*_linear.pt "${SELVEC}/" 2>/dev/null || true
cp -f "${best_dir}"/*_correction_meta.json "${SELVEC}/" 2>/dev/null || true
cp -f "${best_dir}"/disagree_cache.pt "${SELVEC}/" 2>/dev/null || \
    ln -sfn "${ROOT}/mlp_vectors_qa_instr_h${MLP_HIDDEN}/${CONFIG}/disagree_cache.pt" "${SELVEC}/disagree_cache.pt" 2>/dev/null || true
echo "${best_dir}" > "${SELVEC}/.selected_from"

# ---- STAGE 3: OOD hybrids -- math500 + gsm8k (independent, run concurrently) ----
hy_ok(){ [[ -f "${ROOT}/mlp_eval_qa_instr_holdoutsel_h${MLP_HIDDEN}/${CONFIG}/hybrid_summary_${BS}_$1_final.json" ]]; }
ood_pids=()
for ds in math500 gsm8k; do
    if hy_ok "${ds}"; then log "hybrid ${ds}: done, skip"; continue; fi
    ( srun_retry "holdhy-${CONFIG}-${ds}" "${ORCH}/hy_${CONFIG}_${ds}.log" \
        "$(gres_hy "$CONFIG")" "$(mem_hy "$CONFIG")" "20:00:00" -- \
        "CONFIG=${CONFIG}" "DATASET=${ds}" "MLP_HIDDEN=${MLP_HIDDEN}" "VARIANT=_holdoutsel" \
        "HBS_OVERRIDE=8" "DECODE_T=0" -- "${SDIR}/eval_qa_instr_hsweep.sh" ) &
    ood_pids+=($!); sleep 8
done
for p in "${ood_pids[@]}"; do wait "$p"; done
log "all OOD hybrids done"

# ---- report ----
{
  echo "==== holdout chain ${CONFIG} FINAL $(date -u)  (best holdout_gap=${best_gap}) ===="
  for ds in math500 gsm8k; do
    f="${ROOT}/mlp_eval_qa_instr_holdoutsel_h${MLP_HIDDEN}/${CONFIG}/hybrid_summary_${BS}_${ds}_final.json"
    [[ -f "$f" ]] && python3 -c "import json;d=json.load(open('$f'))['headline'];print('%-9s base=%.1f think=%.1f hybrid=%.1f gap=%.1f'%('$ds',d['base_mean_pct'],d['thinking_mean_pct'],d['hybrid_mean_pct'],d['gap_recovered_pct']))" || echo "${ds}: NO SUMMARY"
  done
} | tee -a "${LOG}"
log "##### holdout chain done #####"
