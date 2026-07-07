#!/usr/bin/env bash
# Independent per-config chain for the hendrycks-MATH holdout (~1000 Q):
#   0) vLLM base (qa_instr, greedy) + think (temp0.6 s0/s1/s2) rollouts, in
#      PARALLEL (independent models); each self-skips if already complete
#   1) hybrid eval on holdout-selected vectors (VARIANT=_holdoutsel), judge x3
# Idempotent; jobs run at --nice so they backfill politely.
#
#   bash mlp_pipeline/canonical/hendrycks_chain.sh <config>
set -uo pipefail
CONFIG="${1:?usage: hendrycks_chain.sh <config>}"
ROOT=/workspace-vast/constantinv/thinking-llms-interp
SDIR="${ROOT}/mlp_pipeline/canonical"
MLP_HIDDEN=512
ORCH="${ROOT}/slurm_logs/final_final/orch_hendrycks_chain"
mkdir -p "${ORCH}"; LOG="${ORCH}/${CONFIG}.log"
cd "${ROOT}"; source .venv/bin/activate 2>/dev/null || true
PARTITION="${PARTITION:-general}"; QOS="${QOS:-high}"; NICE="${NICE:-120}"; RETRY="${RETRY:-8}"
log(){ echo "[$(date -u '+%F %T')] [${CONFIG}] $*" | tee -a "${LOG}"; }

declare -A THINK_MODEL BASE_MODEL THINK_SHORT BASE_SHORT
THINK_MODEL[orz-0.5b]="Open-Reasoner-Zero/Open-Reasoner-Zero-0.5B"; BASE_MODEL[orz-0.5b]="Qwen/Qwen2.5-0.5B"; THINK_SHORT[orz-0.5b]=open-reasoner-zero-0.5b; BASE_SHORT[orz-0.5b]=qwen2.5-0.5b
THINK_MODEL[orz-1.5b]="Open-Reasoner-Zero/Open-Reasoner-Zero-1.5B"; BASE_MODEL[orz-1.5b]="Qwen/Qwen2.5-1.5B"; THINK_SHORT[orz-1.5b]=open-reasoner-zero-1.5b; BASE_SHORT[orz-1.5b]=qwen2.5-1.5b
THINK_MODEL[orz-7b]="Open-Reasoner-Zero/Open-Reasoner-Zero-7B"; BASE_MODEL[orz-7b]="Qwen/Qwen2.5-7B"; THINK_SHORT[orz-7b]=open-reasoner-zero-7b; BASE_SHORT[orz-7b]=qwen2.5-7b
THINK_MODEL[orz-32b]="Open-Reasoner-Zero/Open-Reasoner-Zero-32B"; BASE_MODEL[orz-32b]="Qwen/Qwen2.5-32B"; THINK_SHORT[orz-32b]=open-reasoner-zero-32b; BASE_SHORT[orz-32b]=qwen2.5-32b
THINK_MODEL[r1-14b]="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B"; BASE_MODEL[r1-14b]="Qwen/Qwen2.5-14B"; THINK_SHORT[r1-14b]=deepseek-r1-distill-qwen-14b; BASE_SHORT[r1-14b]=qwen2.5-14b
THINK_MODEL[r1-llama8b]="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"; BASE_MODEL[r1-llama8b]="Meta-Llama/Llama-3.1-8B"; THINK_SHORT[r1-llama8b]=deepseek-r1-distill-llama-8b; BASE_SHORT[r1-llama8b]=llama-3.1-8b
THINK_MODEL[qwq-32b]="Qwen/QwQ-32B"; BASE_MODEL[qwq-32b]="Qwen/Qwen2.5-32B"; THINK_SHORT[qwq-32b]=qwq-32b; BASE_SHORT[qwq-32b]=qwen2.5-32b
THINK_MODEL[r1-32b]="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B"; BASE_MODEL[r1-32b]="Qwen/Qwen2.5-32B"; THINK_SHORT[r1-32b]=deepseek-r1-distill-qwen-32b; BASE_SHORT[r1-32b]=qwen2.5-32b
THINK_MODEL[r1-math1.5b]="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B"; BASE_MODEL[r1-math1.5b]="Qwen/Qwen2.5-Math-1.5B"; THINK_SHORT[r1-math1.5b]=deepseek-r1-distill-qwen-1.5b; BASE_SHORT[r1-math1.5b]=qwen2.5-math-1.5b

TM="${THINK_MODEL[$CONFIG]}"; BM="${BASE_MODEL[$CONFIG]}"; TS="${THINK_SHORT[$CONFIG]}"; BS="${BASE_SHORT[$CONFIG]}"
CACHE="${ROOT}/hybrid/results/response_cache_hendrycks_holdout"
GOLD="${ROOT}/data/hendrycks_holdout_eval/eval.jsonl"
NEED=$(wc -l < "${GOLD}")

gres_gen(){ case "$1" in *32b|*14b|orz-7b|r1-llama8b) echo gpu:2;; *) echo gpu:1;; esac; }
tp_gen(){   case "$1" in *32b|*14b|orz-7b|r1-llama8b) echo 2;;     *) echo 1;;      esac; }
mem_gen(){  case "$1" in *32b) echo 192G;; *14b) echo 160G;; orz-7b|r1-llama8b) echo 128G;; *) echo 64G;; esac; }
mem_hy(){   case "$1" in *32b) echo 384G;; *14b) echo 256G;; orz-7b|r1-llama8b) echo 160G;; *) echo 64G;; esac; }
port(){ echo $(( 9600 + (RANDOM % 400) )); }

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
        log "${jn}: failed rc=${rc}; retry in 45s"; sleep 45
    done
    log "${jn}: GAVE UP"; return 1
}

think_ok(){ for S in 0 1 2; do f="${CACHE}/thinking_${TS}_hendrycks_holdout_temp0.6_max2048_s${S}.jsonl"; { [[ -f "$f" ]] && [[ $(wc -l <"$f") -ge ${NEED} ]]; } || return 1; done; }
base_ok(){ f="${CACHE}/base_${BS}_hendrycks_holdout_temp0_max2048.jsonl"; [[ -f "$f" ]] && [[ $(wc -l <"$f") -ge ${NEED} ]]; }
hy_ok(){ [[ -f "${ROOT}/mlp_eval_hendrycks_holdout_qa_instr_holdoutsel_h${MLP_HIDDEN}/${CONFIG}/hybrid_summary_${BS}_hendrycks_holdout_final.json" ]]; }

log "##### hendrycks chain start (pid $$) nice=${NICE} NEED=${NEED} #####"

# ---- STAGE 0: base + think rollouts (parallel, independent models) ----
gen_pids=()
if think_ok; then log "think rollouts present, skip"; else
    ( srun_retry "hmk-th-${CONFIG}" "${ORCH}/think_${CONFIG}.log" "$(gres_gen "$CONFIG")" \
        "$(mem_gen "$CONFIG")" "16:00:00" -- \
        "ROLE=think" "MODEL=${TM}" "SHORT=${TS}" "TP=$(tp_gen "$CONFIG")" "VLLM_PORT=$(port)" \
        -- "${SDIR}/gen_hendrycks_holdout.sh" ) &
    gen_pids+=($!); sleep 5
fi
if base_ok; then log "base rollouts present, skip"; else
    ( srun_retry "hmk-ba-${CONFIG}" "${ORCH}/base_${CONFIG}.log" "$(gres_gen "$CONFIG")" \
        "$(mem_gen "$CONFIG")" "8:00:00" -- \
        "ROLE=base" "MODEL=${BM}" "SHORT=${BS}" "TP=$(tp_gen "$CONFIG")" "VLLM_PORT=$(port)" \
        -- "${SDIR}/gen_hendrycks_holdout.sh" ) &
    gen_pids+=($!); sleep 5
fi
gfail=0; for p in "${gen_pids[@]}"; do wait "$p" || gfail=1; done
if ! think_ok || ! base_ok; then log "ROLLOUT GEN INCOMPLETE (gfail=${gfail}); abort"; exit 1; fi
log "rollouts ready (base + think s0/s1/s2)"

# ---- STAGE 1: hybrid eval ----
if hy_ok; then log "hybrid: done, skip"; else
    srun_retry "hmk-hy-${CONFIG}" "${ORCH}/hy_${CONFIG}.log" "$(gres_gen "$CONFIG")" \
        "$(mem_hy "$CONFIG")" "24:00:00" -- \
        "CONFIG=${CONFIG}" "MLP_HIDDEN=${MLP_HIDDEN}" "VARIANT=_holdoutsel" "HBS_OVERRIDE=8" \
        -- "${SDIR}/eval_hendrycks_holdout_hybrid.sh" || { log "hybrid failed; abort"; exit 1; }
fi

# ---- report ----
f="${ROOT}/mlp_eval_hendrycks_holdout_qa_instr_holdoutsel_h${MLP_HIDDEN}/${CONFIG}/hybrid_summary_${BS}_hendrycks_holdout_final.json"
[[ -f "$f" ]] && python3 -c "import json;d=json.load(open('$f'))['headline'];print('[${CONFIG}] hendrycks_holdout base=%.1f think=%.1f hybrid=%.1f gap=%.1f'%(d['base_mean_pct'],d['thinking_mean_pct'],d['hybrid_mean_pct'],d['gap_recovered_pct']))" | tee -a "${LOG}"
log "##### hendrycks chain done #####"
