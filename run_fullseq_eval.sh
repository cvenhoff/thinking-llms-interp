#!/usr/bin/env bash
# Eval for full-seq bias recipe: bias always-on + cat vectors at disagreements.
set -uo pipefail
cd /workspace/thinking-llms-interp/hybrid
source /workspace/thinking-llms-interp/.venv/bin/activate
source /workspace/thinking-llms-interp/.env_exports.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY=error

CUDA_VISIBLE_DEVICES=${GPU} python -u hybrid_eval.py \
    --dataset math500 \
    --n_tasks 0 \
    --max_new_tokens 2000 \
    --max_thinking_tokens 2000 \
    --batch_gen_size 512 \
    --hybrid_gen_batch_size 512 \
    --base_model    "$BASE" \
    --thinking_model "$THINK" \
    --sae_layer     "$SAE_L" \
    --n_clusters    "$K" \
    --dom_vectors_dir   "../${VECS_DIR}" \
    --old_vectors_dir   "../${VECS_DIR}" \
    --old_vectors_layer "$STEER" \
    --bias_vector_path  "../${BIAS_PATH}" \
    --bias_layer        "$STEER" \
    --bias_always_on \
    --fixed_coef 1.0 \
    --judge_repetitions 3 \
    --results_suffix "${SUFFIX}" \
    2>&1 | tee "/tmp/eval_${SUFFIX}.log"
echo "EVAL ${SUFFIX} DONE"

# Quick gap report
python3 - <<PYEOF
import json, glob
results_dir = "/workspace/thinking-llms-interp/hybrid/results"
flist = glob.glob(f"{results_dir}/judge_reps_*${SUFFIX}*.json")
if not flist:
    print("no results found"); exit()
data = json.load(open(sorted(flist)[-1]))
reps = data["per_rep"]
think = reps["thinking"]["mean_pct"] / 100
base  = reps["base"]["mean_pct"] / 100
hyb   = reps["hybrid"]["mean_pct"] / 100
gap   = think - base
rec   = (hyb - base) / gap if gap else float("nan")
print(f"\n=== ${SUFFIX} ===")
print(f"  think={think:.3f}  base={base:.3f}  hybrid={hyb:.3f}")
print(f"  gap={gap:.3f}  recovered={rec:.1%}")
PYEOF
