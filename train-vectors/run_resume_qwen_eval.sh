#!/bin/bash
# Resume Qwen/ORZ pipeline from Step 2 (attribution) onwards
# NOW USING Layer 8, K=5 taxonomy
set -e

cd /workspace/thinking-llms-interp/train-vectors
source ../.venv/bin/activate
export $(cat ../.env | xargs)

MODEL="Qwen/Qwen2.5-7B"
MODEL_SHORT="qwen2.5-7b"
SAE_LAYER=8
N_CLUSTERS=5
N_TRAIN=1000
N_EVAL=100
N_ATTRIBUTION=50

TRAIN_PAIRS="results/synthetic_pairs/synthetic_pairs_${MODEL_SHORT}_${N_CLUSTERS}clusters.json"
EVAL_PAIRS="results/synthetic_pairs/synthetic_pairs_${MODEL_SHORT}_${N_CLUSTERS}clusters_eval.json"
SAVE_DIR="results/diff_of_means"
DOM_DIR="/workspace/thinking-llms-interp/train-vectors/results/diff_of_means"

# ==============================================================================
# STEP 2: Attribution only (vectors already exist)
# ==============================================================================
echo "=============================================="
echo "STEP 2: Attribution (skip vectors, batch_size=24)"
echo "=============================================="
python -u compute_diff_of_means.py \
  --model "$MODEL" \
  --pairs_file "$TRAIN_PAIRS" \
  --save_dir "$SAVE_DIR" \
  --batch_size 24 \
  --n_train_pairs "$N_TRAIN" \
  --n_attribution_examples "$N_ATTRIBUTION" \
  --n_eval_questions 0 \
  --min_layer_frac 0.2 \
  --skip_vectors \
  2>&1

# ==============================================================================
# STEP 3: Multi-coefficient eval sweep (extended: 0.5–3.0)
# ==============================================================================
echo "=============================================="
echo "STEP 3: Multi-coefficient eval sweep (coeffs 0.5,1.0,1.5,2.0,2.5,3.0)"
echo "=============================================="

python -u compute_diff_of_means.py \
  --model "$MODEL" \
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
echo "STEP 5: Hybrid model on MATH500 (filter: base wrong & thinking right)"
echo "=============================================="
cd /workspace/thinking-llms-interp/hybrid
PYTHONPATH=/workspace/thinking-llms-interp:$PYTHONPATH python -u hybrid_token.py \
  --dataset math500 \
  --thinking_model Open-Reasoner-Zero/Open-Reasoner-Zero-7B \
  --base_model "$MODEL" \
  --steering_layer 8 \
  --sae_layer "$SAE_LAYER" \
  --n_clusters "$N_CLUSTERS" \
  --max_new_tokens 2000 \
  --max_thinking_tokens 2000 \
  --coefficients 1.0 \
  --token_windows 0 \
  --no-guardrail \
  --disable_disagreement_only \
  --n_tasks 500 \
  --n_cold_start_tokens 0 \
  --dom-vectors-dir "$DOM_DIR" \
  --dom-vectors-model-short "$MODEL_SHORT" \
  --dom-raw-norm \
  --filter-base-wrong-thinking-right \
  --results-suffix dom-vectors-best-coeff \
  --show_progress \
  2>&1

echo ""
echo "=============================================="
echo "PIPELINE RESUME COMPLETE (Qwen)"
echo "=============================================="
