#!/bin/bash
# Master script: ORZ-recipe comparison across ORZ-32B, QwQ-32B, R1-Distill-32B.
# Hypothesis: pure-RL models (ORZ) should show higher gap recovery and lower
# random-vector confound than SFT+RL (QwQ) or distilled (R1-Distill) models,
# because the base (Qwen2.5-32B) already has the SFT "scaffold" that ORZ builds on.
#
# Runs sequentially (each needs all 3 GPUs).
# Results land in:
#   train-vectors/results/vars/correction_vectors_<model>_canon_stage{1,2}/
#   hybrid/results/rolling/rolling_qwen2.5-32b_math500_<model>-pg-{learn,rand,biasonly}-128.jsonl
set -uo pipefail
cd /workspace/thinking-llms-interp
source .venv/bin/activate
source .env_exports.sh

echo "=========================================================="
echo "  3-MODEL ORZ-RECIPE COMPARISON"
echo "  Recipe: disagree-only, CE loss, max_norm 25/12, layer 38"
echo "  Eval: pg rule, n=128, math500, trained+random+biasonly"
echo "  Order: QwQ-32B (reuse stage0) → R1-Distill → ORZ-32B"
echo "=========================================================="
echo

# ---- 1. QwQ-32B (reuse stage0 disagreements, skip full collection) ----
echo "### MODEL 1/3: QwQ-32B ###"
THINKING_MODEL="Qwen/QwQ-32B" \
THINKING_SHORT="qwq-32b" \
SKIP_STAGE0=1 \
STAGE0_DISAGREE_PT="train-vectors/results/vars/correction_vectors_qwq32b_biasfirst_disagree_ce_cap25/disagreements.pt" \
bash run_model_pipeline.sh
echo

# QwQ Stage1 may have failed on first attempt (NCCL timeout); pipeline
# continues and QwQ will be retried at the end after other models finish.

# ---- 2. DeepSeek-R1-Distill-Qwen-32B ----
echo "### MODEL 2/3: R1-Distill-Qwen-32B ###"
THINKING_MODEL="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" \
THINKING_SHORT="deepseek-r1-distill-qwen-32b" \
bash run_model_pipeline.sh
echo

# ---- 3. Open-Reasoner-Zero-32B ----
echo "### MODEL 3/3: ORZ-32B ###"
THINKING_MODEL="Open-Reasoner-Zero/Open-Reasoner-Zero-32B" \
THINKING_SHORT="open-reasoner-zero-32b" \
bash run_model_pipeline.sh
echo

# ---- QwQ-32B retry (if stage1 failed earlier due to NCCL timeout) ----
QWQ_BIAS="train-vectors/results/vars/correction_vectors_qwq-32b_canon_stage1/qwen2.5-32b_bias_global.pt"
QWQ_CAT="train-vectors/results/vars/correction_vectors_qwq-32b_canon_stage2/qwen2.5-32b_idx0_linear.pt"
if [ ! -f "$QWQ_BIAS" ] || [ ! -f "$QWQ_CAT" ]; then
  echo "### QwQ-32B RETRY (stages incomplete) ###"
  THINKING_MODEL="Qwen/QwQ-32B" \
  THINKING_SHORT="qwq-32b" \
  SKIP_STAGE0=1 \
  STAGE0_DISAGREE_PT="train-vectors/results/vars/correction_vectors_qwq32b_biasfirst_disagree_ce_cap25/disagreements.pt" \
  bash run_model_pipeline.sh
  echo
fi

# ---- Summary ----
echo "=========================================================="
echo "  FINAL COMPARISON SUMMARY"
echo "=========================================================="
cd /workspace/thinking-llms-interp/hybrid
python3 - <<'PY'
import json, glob, os

models = [
    ("qwq-32b",                     "QwQ-32B (SFT+RL)"),
    ("deepseek-r1-distill-qwen-32b","R1-Distill (distilled)"),
    ("open-reasoner-zero-32b",      "ORZ-32B (pure RL)"),
]
conds = ["learn", "rand", "biasonly"]

print(f"\n{'model':<22} {'condition':<10} {'n':>4} {'T':>6} {'B':>6} {'H':>6} {'gap':>7}")
print("-" * 70)
for tag, label in models:
    for cond in conds:
        f = f"results/rolling/rolling_qwen2.5-32b_math500_{tag}-pg-{cond}-128.jsonl"
        if not os.path.exists(f):
            print(f"{label:<22} {cond:<10} missing")
            continue
        rows = [json.loads(l) for l in open(f)]
        n = len(rows)
        aT = sum(1 for r in rows if r["judges"]["thinking"]["correct"])/n
        aB = sum(1 for r in rows if r["judges"]["base"]["correct"])/n
        aH = sum(1 for r in rows if r["judges"]["hybrid"]["correct"])/n
        gap = (aH-aB)/max(1e-9,aT-aB) if aT > aB else float("nan")
        print(f"{label:<22} {cond:<10} {n:>4} {aT:>6.3f} {aB:>6.3f} {aH:>6.3f} {gap:>+7.3f}")
    print()
PY
