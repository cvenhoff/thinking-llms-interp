#!/bin/bash
# Fixed coef=1 evaluation for all 3 models × 3 conditions (trained/random/biasonly).
# No pg sweep — uses kl_topk=1 with coef_sweep="1.0" so every position is steered
# at exactly coef=1.0 (no pg perplexity guardrail, no cross-coef selection).
# Base and thinking responses are cached from the pg run; only hybrid is regenerated.
set -uo pipefail
cd /workspace/thinking-llms-interp/hybrid
source ../.venv/bin/activate
source ../.env_exports.sh

BASE_MODEL="Qwen/Qwen2.5-32B"
SAE_LAYER=27
N_CLUSTERS=10
STEER_LAYER=38

run_eval_coef1 () {
    local TAG="$1"          # model short name
    local THINKING_MODEL="$2"
    local COND="$3"         # learn / rand / biasonly
    local EXTRA="${4:-}"

    local SAVE1="correction_vectors_${TAG}_canon_stage1"
    local SAVE2="correction_vectors_${TAG}_canon_stage2"
    local CATS_DIR_ABS="/workspace/thinking-llms-interp/train-vectors/results/vars/${SAVE2}"
    local BIAS_PATH_ABS="/workspace/thinking-llms-interp/train-vectors/results/vars/${SAVE1}/qwen2.5-32b_bias_global.pt"
    local EVAL_TAG="${TAG}-coef1-${COND}-128"

    rm -f "results/rolling/rolling_qwen2.5-32b_math500_${EVAL_TAG}.jsonl"
    rm -f "results/summary_qwen2.5-32b_math500_${EVAL_TAG}.json"
    rm -f "results/judge_reps_qwen2.5-32b_math500_${EVAL_TAG}.json"
    echo "  --- ${EVAL_TAG} ---"
    uv run python hybrid_eval.py \
        --dataset math500 --n_tasks 128 \
        --max_new_tokens 2000 --max_thinking_tokens 2000 \
        --batch_gen_size 4 --hybrid_gen_batch_size 4 \
        --base_model "$BASE_MODEL" \
        --thinking_model "$THINKING_MODEL" \
        --sae_layer "$SAE_LAYER" --n_clusters "$N_CLUSTERS" \
        --disable_sae_mean \
        --dom_vectors_dir ../train-vectors/results/diff_of_means \
        --dom_vectors_model_short qwen2.5-32b \
        --old_vectors_dir "$CATS_DIR_ABS" \
        --old_vectors_layer "$STEER_LAYER" \
        --bias_vector_path "$BIAS_PATH_ABS" \
        --bias_layer "$STEER_LAYER" \
        --coef_sweep "1.0" \
        --coef_select kl_topk \
        --kl_topk 1 \
        --judge_repetitions 3 \
        --results_suffix "${EVAL_TAG}" \
        $EXTRA 2>&1 | tee "/workspace/tmp/${EVAL_TAG}.log"
}

echo "=========================================================="
echo "  COEF=1 FIXED EVALUATION (3 models × 3 conditions)"
echo "  coef_sweep=1.0  coef_select=kl_topk(K=1)"
echo "=========================================================="
echo

echo "### MODEL 1/3: QwQ-32B ###"
run_eval_coef1 "qwq-32b" "Qwen/QwQ-32B" "learn" ""
run_eval_coef1 "qwq-32b" "Qwen/QwQ-32B" "rand"  "--randomize_vectors --random_seed 42"
run_eval_coef1 "qwq-32b" "Qwen/QwQ-32B" "biasonly" "--bias_only"
echo

echo "### MODEL 2/3: R1-Distill-Qwen-32B ###"
run_eval_coef1 "deepseek-r1-distill-qwen-32b" "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" "learn" ""
run_eval_coef1 "deepseek-r1-distill-qwen-32b" "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" "rand"  "--randomize_vectors --random_seed 42"
run_eval_coef1 "deepseek-r1-distill-qwen-32b" "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B" "biasonly" "--bias_only"
echo

echo "### MODEL 3/3: ORZ-32B ###"
run_eval_coef1 "open-reasoner-zero-32b" "Open-Reasoner-Zero/Open-Reasoner-Zero-32B" "learn" ""
run_eval_coef1 "open-reasoner-zero-32b" "Open-Reasoner-Zero/Open-Reasoner-Zero-32B" "rand"  "--randomize_vectors --random_seed 42"
run_eval_coef1 "open-reasoner-zero-32b" "Open-Reasoner-Zero/Open-Reasoner-Zero-32B" "biasonly" "--bias_only"
echo

echo "=========================================================="
echo "  FINAL COEF=1 COMPARISON SUMMARY"
echo "=========================================================="
python3 - <<'PY'
import json, glob, os

models = [
    ("qwq-32b",                     "QwQ-32B (SFT+RL)"),
    ("deepseek-r1-distill-qwen-32b","R1-Distill (distilled)"),
    ("open-reasoner-zero-32b",      "ORZ-32B (pure RL)"),
]
conds = ["learn", "rand", "biasonly"]

print(f"\n{'model':<25} {'cond':<10} {'n':>4} {'T':>6} {'B':>6} {'H':>6} {'gap':>7}")
print("-" * 70)
for tag, label in models:
    for cond in conds:
        f = f"results/rolling/rolling_qwen2.5-32b_math500_{tag}-coef1-{cond}-128.jsonl"
        if not os.path.exists(f):
            print(f"{label:<25} {cond:<10} missing")
            continue
        rows = [json.loads(l) for l in open(f)]
        n = len(rows)
        aT = sum(1 for r in rows if r["judges"]["thinking"]["correct"])/n
        aB = sum(1 for r in rows if r["judges"]["base"]["correct"])/n
        aH = sum(1 for r in rows if r["judges"]["hybrid"]["correct"])/n
        gap = (aH-aB)/max(1e-9,aT-aB) if aT > aB else float("nan")
        print(f"{label:<25} {cond:<10} {n:>4} {aT:>6.3f} {aB:>6.3f} {aH:>6.3f} {gap:>+7.3f}")
    print()
PY
