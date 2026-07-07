#!/usr/bin/env bash
# Best-of-3 vector selection for one pair. Runs the holdout gap-recovery eval on
# each of the three trained runs, then promotes the run with the highest holdout
# gap to artifacts/mlp_vectors_qa_instr_holdoutsel_h512/<CONFIG>/ (the vectors every OOD /
# Hendrycks / ablation eval then reads).
#
#   bash hybrid/select_best_of_3.sh orz-32b
set -euo pipefail
ROOT="${THINKING_LLMS_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
source "${ROOT}/configs.sh"
CONFIG="${1:?usage: select_best_of_3.sh <config>}"
cfg_load "${CONFIG}"
H="${MLP_HIDDEN}"
DIRS=(
    "${ROOT}/artifacts/mlp_vectors_qa_instr_h${H}/${CONFIG}"
    "${ROOT}/artifacts/mlp_vectors_qa_instr_h${H}_bo3/${CONFIG}/run2"
    "${ROOT}/artifacts/mlp_vectors_qa_instr_h${H}_bo3/${CONFIG}/run3"
)
SELVEC="${ROOT}/artifacts/mlp_vectors_qa_instr_holdoutsel_h${H}/${CONFIG}"

# ---- run the holdout eval for each run (idempotent) ----
for i in 0 1 2; do
    tag="run$((i+1))"; d="${DIRS[$i]}"
    [[ -f "${d}/cat_coef_mlp.pt" ]] || { echo "MISSING vectors for ${tag}: ${d}"; continue; }
    if ls "${ROOT}/artifacts/mlp_eval_holdoutmix/${CONFIG}/${tag}"/judge_reps_*_holdoutmix_final.json >/dev/null 2>&1; then
        echo "${tag}: holdout eval done, skip"; continue
    fi
    CONFIG="${CONFIG}" VEC_DIR="${d}" RUNTAG="${tag}" bash "${ROOT}/hybrid/eval_holdoutmix.sh"
done

# ---- pick the run with the highest holdout gap ----
read -r best_dir best_gap < <(python - "${ROOT}" "${CONFIG}" "${BS}" "${DIRS[@]}" <<'PY'
import json, sys, os
root, cfg, bs = sys.argv[1], sys.argv[2], sys.argv[3]; dirs = sys.argv[4:]
best, bg = "", -1e9; rank = []
for i, d in enumerate(dirs):
    f = os.path.join(root, "artifacts", "mlp_eval_holdoutmix", cfg, f"run{i+1}", f"judge_reps_{bs}_holdoutmix_final.json")
    if not os.path.exists(f):
        continue
    pr = json.load(open(f))["per_rep"]
    b, t, h = pr["base"]["mean_pct"], pr["thinking"]["mean_pct"], pr["hybrid"]["mean_pct"]
    g = (h - b) / (t - b) * 100 if (t - b) != 0 else -1e9
    rank.append(f"run{i+1}={g:.1f}")
    if g > bg:
        bg, best = g, d
sys.stderr.write("holdout gaps: " + ", ".join(rank) + "\n")
print(best, f"{bg:.2f}")
PY
)
[[ -n "${best_dir}" && -f "${best_dir}/cat_coef_mlp.pt" ]] || { echo "FATAL: selection failed (no holdout results)"; exit 1; }
echo "best=${best_dir}  holdout_gap=${best_gap}"

mkdir -p "${SELVEC}"
cp -f "${best_dir}/cat_coef_mlp.pt" "${best_dir}/mlp_config.json" "${SELVEC}/"
cp -f "${best_dir}/layer_map.json" "${SELVEC}/" 2>/dev/null || true
cp -f "${best_dir}"/*_linear.pt "${SELVEC}/" 2>/dev/null || true
cp -f "${best_dir}"/*_correction_meta.json "${SELVEC}/" 2>/dev/null || true
cp -f "${best_dir}/disagree_cache.pt" "${SELVEC}/" 2>/dev/null || \
    ln -sfn "${ROOT}/artifacts/mlp_vectors_qa_instr_h${H}/${CONFIG}/disagree_cache.pt" "${SELVEC}/disagree_cache.pt" 2>/dev/null || true
echo "${best_dir}" > "${SELVEC}/.selected_from"
echo "== promoted ${best_dir} -> ${SELVEC} =="
