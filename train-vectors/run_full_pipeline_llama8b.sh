#!/bin/bash
# Full pipeline for DeepSeek-R1-Distill-Llama-8B (thinking) + Llama-3.1-8B (base)
# Same batch sizes as Qwen/ORZ run, except attribution uses half.
set -e

cd /workspace/thinking-llms-interp/train-vectors
source ../.venv/bin/activate
export $(cat ../.env | xargs)

# ---- Model config ----
BASE_MODEL="meta-llama/Llama-3.1-8B"
MODEL_SHORT="llama-3.1-8b"
THINKING_MODEL="deepseek-ai/DeepSeek-R1-Distill-Llama-8B"
SAE_LAYER=6
N_CLUSTERS=15
N_TRAIN=1000
N_EVAL=100
N_ATTRIBUTION=50

# ---- Paths ----
TRAIN_PAIRS="results/synthetic_pairs/synthetic_pairs_${MODEL_SHORT}_${N_CLUSTERS}clusters.json"
EVAL_PAIRS="results/synthetic_pairs/synthetic_pairs_${MODEL_SHORT}_${N_CLUSTERS}clusters_eval.json"
SAVE_DIR="results/diff_of_means"
DOM_DIR="/workspace/thinking-llms-interp/train-vectors/results/diff_of_means"

mkdir -p results/synthetic_pairs results/diff_of_means/figures

# ==============================================================================
# STEP 1a: Generate TRAINING synthetic pairs
# ==============================================================================
if [ -f "$TRAIN_PAIRS" ]; then
  echo "=============================================="
  echo "STEP 1a: SKIPPED – training pairs already exist: $TRAIN_PAIRS"
  echo "=============================================="
else
  echo "=============================================="
  echo "STEP 1a: Generate ${N_TRAIN} TRAINING synthetic pairs (layer ${SAE_LAYER}, K=${N_CLUSTERS})"
  echo "=============================================="
  python -u generate_synthetic_pairs.py \
    --model "$BASE_MODEL" \
    --thinking_model "$THINKING_MODEL" \
    --layer "$SAE_LAYER" \
    --n_clusters "$N_CLUSTERS" \
    --n_questions "$N_TRAIN" \
    --question_offset 0 \
    --max_new_tokens 512 \
    --batch_size 32 \
    --dataset "TIGER-Lab/MMLU-Pro" \
    2>&1
fi

# ==============================================================================
# STEP 1b: Generate EVAL synthetic pairs (disjoint questions)
# ==============================================================================
echo ""
if [ -f "$EVAL_PAIRS" ]; then
  echo "=============================================="
  echo "STEP 1b: SKIPPED – eval pairs already exist: $EVAL_PAIRS"
  echo "=============================================="
else
  echo "=============================================="
  echo "STEP 1b: Generate ${N_EVAL} EVAL synthetic pairs (disjoint questions)"
  echo "=============================================="
  python -u generate_synthetic_pairs.py \
    --model "$BASE_MODEL" \
    --thinking_model "$THINKING_MODEL" \
    --layer "$SAE_LAYER" \
    --n_clusters "$N_CLUSTERS" \
    --n_questions "$N_EVAL" \
    --question_offset "$N_TRAIN" \
    --max_new_tokens 512 \
    --batch_size 32 \
    --dataset "TIGER-Lab/MMLU-Pro" \
    --output_suffix "_eval" \
    2>&1
fi

# ==============================================================================
# STEP 2: Compute diff-of-means vectors + attribution
# ==============================================================================
echo ""
echo "=============================================="
echo "STEP 2: Compute diff-of-means vectors + attribution (${N_ATTRIBUTION} examples)"
echo "=============================================="
python -u compute_diff_of_means.py \
  --model "$BASE_MODEL" \
  --pairs_file "$TRAIN_PAIRS" \
  --save_dir "$SAVE_DIR" \
  --batch_size 12 \
  --n_train_pairs "$N_TRAIN" \
  --n_attribution_examples "$N_ATTRIBUTION" \
  --n_eval_questions 0 \
  --min_layer_frac 0.2 \
  2>&1

# ==============================================================================
# STEP 3: Multi-coefficient eval sweep on UNSEEN eval pairs (32-token window)
# ==============================================================================
echo ""
echo "=============================================="
echo "STEP 3: Multi-coefficient eval sweep on UNSEEN eval pairs (32-token window)"
echo "=============================================="
python -u compute_diff_of_means.py \
  --model "$BASE_MODEL" \
  --pairs_file "$TRAIN_PAIRS" \
  --eval_pairs_file "$EVAL_PAIRS" \
  --save_dir "$SAVE_DIR" \
  --batch_size 24 \
  --skip_vectors \
  --skip_attribution \
  --use_raw_norm \
  --steer_coeffs "0.5,1.0,1.5,2.0,2.5,3.0" \
  --eval_max_tokens 32 \
  --n_eval_questions "$N_EVAL" \
  --gen_batch_size 128 \
  --min_layer_frac 0.2 \
  2>&1

# ==============================================================================
# STEP 4: Print best coefficients
# ==============================================================================
echo ""
echo "=============================================="
echo "STEP 4: Print best coefficients"
echo "=============================================="
python3 -c "
import json
with open('${SAVE_DIR}/dom_best_coeffs_${MODEL_SHORT}.json') as f:
    data = json.load(f)
for cid in sorted(data.keys(), key=int):
    bc = data[cid]
    best = bc['best_coeff']
    s = bc['scores_by_coeff'][str(best)]
    print(f'Cat {cid}: best_coeff={best} (beh_Δ={s[\"beh_delta\"]:+.2f}, qual_Δ={s[\"qual_delta\"]:+.2f}, rep={s[\"rep_pct\"]:.0f}%)')
"

# ==============================================================================
# STEP 5: Hybrid model on MATH500 (filter: base wrong & thinking right)
# ==============================================================================
echo ""
echo "=============================================="
echo "STEP 5: Hybrid model on MATH500 (skip if base correct, disagreement-only steering)"
echo "=============================================="
cd /workspace/thinking-llms-interp/hybrid
PYTHONPATH=/workspace/thinking-llms-interp:$PYTHONPATH python -u hybrid_token.py \
  --dataset math500 \
  --thinking_model "$THINKING_MODEL" \
  --base_model "$BASE_MODEL" \
  --steering_layer 6 \
  --sae_layer "$SAE_LAYER" \
  --n_clusters "$N_CLUSTERS" \
  --max_new_tokens 2000 \
  --max_thinking_tokens 2000 \
  --coefficients 1.0 \
  --token_windows 0 \
  --no-guardrail \
  --n_tasks 500 \
  --n_cold_start_tokens 0 \
  --dom-vectors-dir "$DOM_DIR" \
  --dom-vectors-model-short "$MODEL_SHORT" \
  --dom-raw-norm \
  --disable-sae-mean \
  --skip-if-base-correct \
  --filter-base-wrong-thinking-right \
  --results-suffix dom-vectors-best-coeff \
  --show_progress \
  2>&1

echo ""
echo "=============================================="
echo "PIPELINE COMPLETE (Llama 8B)"
echo "=============================================="

