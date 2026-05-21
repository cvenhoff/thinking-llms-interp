#!/usr/bin/env bash
# ============================================================
# Hybrid eval for legacy CE-trained vectors: clean recipe
#   - Fixed coefficient (no per-token guardrail sweep)
#   - Bias folded into cat vectors (applied at same layer, same coef)
#   - SAE disagreement gate
#   - steer_all_positions_full  (matches legacy token_window=0)
#   - No adaptive decisions at inference time
#
# EVAL_TYPE controls the ablation:
#   full      – bias + SAE-selected cat vector (default)
#   bias_only – only the bias vector, no cat selection
#   rand_cats – bias + random cat vector (random_firing ablation)
#
# Usage examples:
#   # ORZ-1.5B full hybrid
#   CUDA_VISIBLE_DEVICES=0 MODEL_SIZE=1.5b bash run_legacy_eval.sh
#   # ORZ-0.5B bias-only ablation
#   CUDA_VISIBLE_DEVICES=1 MODEL_SIZE=0.5b EVAL_TYPE=bias_only bash run_legacy_eval.sh
#   # ORZ-1.5B random cats ablation
#   CUDA_VISIBLE_DEVICES=0 MODEL_SIZE=1.5b EVAL_TYPE=rand_cats bash run_legacy_eval.sh
#   # override coef
#   CUDA_VISIBLE_DEVICES=0 MODEL_SIZE=1.5b FIXED_COEF=0.5 bash run_legacy_eval.sh
# ============================================================
set -e

MODEL_SIZE=${MODEL_SIZE:-1.5b}   # 0.5b | 1.5b | 7b
FIXED_COEF=${FIXED_COEF:-}       # empty = adaptive perplexity guardrail; set to e.g. 0.5 for fixed
COEF_SWEEP=${COEF_SWEEP:-}       # override candidate set e.g. "0.1,0.5,1.0"; empty = default in hybrid_eval.py
PG_BIAS_CAT_SWEEP=${PG_BIAS_CAT_SWEEP:-0}  # 1 -> cartesian (bias_coef,cat_coef) PG
PG_BIAS_COEFS=${PG_BIAS_COEFS:-0.0,0.5,1.0}
PG_CAT_COEFS=${PG_CAT_COEFS:-0.0,0.5,1.0}
TOKEN_WINDOW=${TOKEN_WINDOW:-0}  # 0 = legacy; >0 = static last-N window (forces full-seq forward)
DATASET=${DATASET:-math500}
N_TASKS=${N_TASKS:-0}    # 0 = all
BATCH_SIZE=${BATCH_SIZE:-500}            # thinking/base pre-gen batch
HYBRID_BATCH=${HYBRID_BATCH:-$BATCH_SIZE}  # hybrid gen batch (lower for large models)
MAX_NEW_TOKENS=${MAX_NEW_TOKENS:-2000}
MAX_THINKING_TOKENS=${MAX_THINKING_TOKENS:-2000}
# STEER_MODE: steer_all_positions (O(1), KV-cache, all positions),
#             steer_all_positions_full (O(N²), all positions),
#             last_token (legacy --token_windows -1: steer only last position; no flag passed)
STEER_MODE=${STEER_MODE:-steer_all_positions_full}
RESULTS_SUFFIX=${RESULTS_SUFFIX:-legacy_ce}
EVAL_TYPE=${EVAL_TYPE:-full}  # full | bias_only | rand_cats

# ---- Model-specific config ----
if [[ "$MODEL_SIZE" == "1.5b" ]]; then
    BASE_MODEL="Qwen/Qwen2.5-1.5B"
    THINK_MODEL="Open-Reasoner-Zero/Open-Reasoner-Zero-1.5B"
    SAE_LAYER=8
    N_CLUSTERS=5
    OLD_VECTORS_LAYER=10
    VECTORS_DIR="train-vectors/results/vars/optimized_vectors_legacy_ce"
    BIAS_PATH="${VECTORS_DIR}/qwen2.5-1.5b_bias_linear.pt"
elif [[ "$MODEL_SIZE" == "0.5b" ]]; then
    BASE_MODEL="Qwen/Qwen2.5-0.5B"
    THINK_MODEL="Open-Reasoner-Zero/Open-Reasoner-Zero-0.5B"
    SAE_LAYER=8
    N_CLUSTERS=10
    OLD_VECTORS_LAYER=9
    VECTORS_DIR="train-vectors/results/vars/optimized_vectors_legacy_ce"
    BIAS_PATH="${VECTORS_DIR}/qwen2.5-0.5b_bias_linear.pt"
elif [[ "$MODEL_SIZE" == "7b" ]]; then
    BASE_MODEL="Qwen/Qwen2.5-7B"
    THINK_MODEL="Open-Reasoner-Zero/Open-Reasoner-Zero-7B"
    SAE_LAYER=20
    N_CLUSTERS=10
    OLD_VECTORS_LAYER=16
    VECTORS_DIR="train-vectors/results/vars/optimized_vectors_legacy_ce"
    BIAS_PATH="${VECTORS_DIR}/qwen2.5-7b_bias_linear.pt"
else
    echo "ERROR: MODEL_SIZE must be 1.5b, 0.5b, or 7b"
    exit 1
fi

# Allow env override of vectors dir (e.g. to use paper-main vectors)
if [[ -n "$VECTORS_DIR_OVERRIDE" ]]; then
    VECTORS_DIR="$VECTORS_DIR_OVERRIDE"
    BIAS_PATH="${VECTORS_DIR}/qwen2.5-${MODEL_SIZE}_bias_linear.pt"
fi

# ---- Ablation flags ----
EXTRA_FLAGS=""
SUFFIX_TAG=""
if [[ "$EVAL_TYPE" == "bias_only" ]]; then
    EXTRA_FLAGS="--bias_only"
    SUFFIX_TAG="_biasonly"
elif [[ "$EVAL_TYPE" == "rand_cats" ]]; then
    EXTRA_FLAGS="--random_firing"
    SUFFIX_TAG="_randcats"
elif [[ "$EVAL_TYPE" != "full" ]]; then
    echo "ERROR: EVAL_TYPE must be full, bias_only, or rand_cats"
    exit 1
fi

# Sanity: verify required files exist (skip for bias_only — cat vectors not needed)
echo "=== Checking required files ==="
if [[ "$EVAL_TYPE" != "bias_only" ]]; then
    for i in $(seq 0 $((N_CLUSTERS - 1))); do
        MODEL_SHORT="${BASE_MODEL##*/}"
        MODEL_SHORT="${MODEL_SHORT,,}"
        f="${VECTORS_DIR}/${MODEL_SHORT}_idx${i}_linear.pt"
        if [[ ! -f "$f" ]]; then
            echo "ERROR: Missing cat vector: $f"
            exit 1
        fi
    done
fi
if [[ ! -f "$BIAS_PATH" ]]; then
    echo "ERROR: Missing bias vector: $BIAS_PATH"
    exit 1
fi
echo "  All vector files present ✓"

echo ""
echo "======================================"
echo " Hybrid eval: ORZ-${MODEL_SIZE}  type=${EVAL_TYPE}"
echo "   coef=${FIXED_COEF}  dataset=${DATASET}"
echo "   max_tokens=${MAX_NEW_TOKENS}  hybrid_batch=${HYBRID_BATCH}"
echo "   STEER_MODE=${STEER_MODE}"
echo "   No guardrail. Disagreement gate."
echo "======================================"

cd hybrid

# shellcheck disable=SC2086
python hybrid_eval.py \
    --base_model "$BASE_MODEL" \
    --thinking_model "$THINK_MODEL" \
    --dataset "$DATASET" \
    --sae_layer "$SAE_LAYER" \
    --n_clusters "$N_CLUSTERS" \
    --n_tasks "$N_TASKS" \
    --batch_gen_size "$BATCH_SIZE" \
    --hybrid_gen_batch_size "$HYBRID_BATCH" \
    --dom_vectors_dir "../${VECTORS_DIR}" \
    --old_vectors_dir "../${VECTORS_DIR}" \
    --old_vectors_layer "$OLD_VECTORS_LAYER" \
    --bias_vector_path "../${BIAS_PATH}" \
    ${FIXED_COEF:+--fixed_coef "$FIXED_COEF"} \
    ${FIXED_COEF:+--coef_select fixed} \
    ${COEF_SWEEP:+--coef_sweep "$COEF_SWEEP"} \
    $([ "$PG_BIAS_CAT_SWEEP" = "1" ] && echo "--pg_bias_cat_sweep --pg_bias_coefs $PG_BIAS_COEFS --pg_cat_coefs $PG_CAT_COEFS") \
    $([ "$TOKEN_WINDOW" != "0" ] && echo "--token_window $TOKEN_WINDOW") \
    $([ "$STEER_MODE" != "last_token" ] && echo "--${STEER_MODE}") \
    --max_new_tokens "$MAX_NEW_TOKENS" \
    --max_thinking_tokens "$MAX_THINKING_TOKENS" \
    --results_suffix "${RESULTS_SUFFIX}${FIXED_COEF:+_coef${FIXED_COEF}}${SUFFIX_TAG}" \
    $EXTRA_FLAGS \
    "$@"

echo ""
echo "======================================"
echo " Eval DONE: ORZ-${MODEL_SIZE}  type=${EVAL_TYPE}  coef=${FIXED_COEF:-adaptive}"
echo "======================================"
