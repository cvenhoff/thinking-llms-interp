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

for i in 0 1 2; do
    tag="run$((i+1))"; d="${DIRS[$i]}"
    if ls "${ROOT}/artifacts/mlp_eval_holdoutmix/${CONFIG}/${tag}"/judge_reps_*_holdoutmix_final.json >/dev/null 2>&1; then
        echo "${tag}: holdout result present, skip"; continue
    fi
    [[ -f "${d}/cat_coef_mlp.pt" ]] || { echo "${tag}: no candidate vectors (run train-vectors/run.sh to regenerate), skip"; continue; }
    CONFIG="${CONFIG}" VEC_DIR="${d}" RUNTAG="${tag}" bash "${ROOT}/hybrid/eval_holdoutmix.sh"
done

mkdir -p "${SELVEC}"
read -r best_idx best_gap < <(python - "${ROOT}" "${CONFIG}" "${BS}" "${SELVEC}" "${DIRS[@]}" <<'PY'
import json, sys, os
root, cfg, bs, selvec = sys.argv[1:5]; dirs = sys.argv[5:]
best_i, bg, best_tag = -1, -1e9, ""; gaps = {}
for i, d in enumerate(dirs):
    tag = f"run{i+1}"
    f = os.path.join(root, "artifacts", "mlp_eval_holdoutmix", cfg, tag, f"judge_reps_{bs}_holdoutmix_final.json")
    if not os.path.exists(f):
        continue
    pr = json.load(open(f))["per_rep"]
    b, t, h = pr["base"]["mean_pct"], pr["thinking"]["mean_pct"], pr["hybrid"]["mean_pct"]
    g = (h - b) / (t - b) * 100 if (t - b) != 0 else -1e9
    gaps[tag] = round(g, 2)
    if g > bg:
        bg, best_i, best_tag = g, i, tag
sel = {"config": cfg, "holdout_gap_recovered_pct": gaps,
       "selected_run": best_tag, "selected_gap_recovered_pct": round(bg, 2)}
with open(os.path.join(selvec, "selection.json"), "w") as fh:
    json.dump(sel, fh, indent=2); fh.write("\n")
sys.stderr.write("holdout gaps: " + ", ".join(f"{k}={v}" for k, v in gaps.items()) + "\n")
print(best_i, f"{bg:.2f}")
PY
)
[[ "${best_idx}" -ge 0 ]] && best_dir="${DIRS[$best_idx]}" || best_dir=""
if [[ -n "${best_dir}" && -f "${best_dir}/cat_coef_mlp.pt" ]]; then
    echo "best=${best_dir}  holdout_gap=${best_gap}"
    cp -f "${best_dir}/cat_coef_mlp.pt" "${best_dir}/mlp_config.json" "${SELVEC}/"
    cp -f "${best_dir}/layer_map.json" "${SELVEC}/" 2>/dev/null || true
    cp -f "${best_dir}"/*_linear.pt "${SELVEC}/" 2>/dev/null || true
    cp -f "${best_dir}"/*_correction_meta.json "${SELVEC}/" 2>/dev/null || true
    echo "== promoted ${best_dir} -> ${SELVEC} (selection.json written) =="
elif [[ -f "${SELVEC}/cat_coef_mlp.pt" ]]; then
    # Candidate vector sets are not shipped; the winner is already promoted here.
    # selection.json is refreshed above. Rerun train-vectors/run.sh to re-promote.
    echo "== winner already promoted in ${SELVEC} (holdout_gap=${best_gap}); candidates not present, kept as-is =="
else
    echo "FATAL: no candidate vectors and no promoted vectors in ${SELVEC}"; exit 1
fi
