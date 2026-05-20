#!/usr/bin/env bash
# ============================================================
# Parallel full training for ORZ-1.5B and ORZ-0.5B
#
# Multi-vector cat training: all N cat vectors trained in one
# forward/backward loop — mathematically equivalent to sequential
# but N_cats × faster (shared model forward/backward cost).
#
# GPU allocation:
#   GPU 0: ORZ-1.5B  (bias, then 5 cats in parallel via multi-vector)
#   GPU 1: ORZ-0.5B  (bias, then 10 cats in parallel via multi-vector)
#   GPU 2: free (for eval after training, or leave idle)
#
# Usage:
#   bash launch_full_training.sh
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_DIR="$SCRIPT_DIR/train-vectors"
LOG_DIR="$SCRIPT_DIR/logs/launch_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

N_TRAIN=2048
N_EVAL=512
MAX_ITERS=50
LR="1e-2"

echo "======================================"
echo " Full training launch (multi-vector cats)"
echo "   N_TRAIN=$N_TRAIN  MAX_ITERS=$MAX_ITERS"
echo "   Logs: $LOG_DIR"
echo "======================================"

# ---------------------------------------------------------------------------
# GPU 0: ORZ-1.5B – bias then all 5 cats simultaneously
# ---------------------------------------------------------------------------
(
  source "$SCRIPT_DIR/.venv/bin/activate"
  cd "$TRAIN_DIR"
  SAVE_DIR="results/vars/optimized_vectors_legacy_ce"
  MODEL="Qwen/Qwen2.5-1.5B"
  LAYER=10
  MINI_BS=8    # 8 examples/step mixed across 5 cats → ~1-2 per cat per step

  echo "[GPU0] 1.5B bias training" | tee "$LOG_DIR/gpu0.log"
  CUDA_VISIBLE_DEVICES=0 python optimize_steering_vectors.py \
      --model "$MODEL" --max_iters "$MAX_ITERS" \
      --n_training_examples "$N_TRAIN" --n_eval_examples "$N_EVAL" \
      --optim_minibatch_size "$MINI_BS" --layer "$LAYER" \
      --steering_vector_idx -1 --lr "$LR" --save_path "$SAVE_DIR" \
      >> "$LOG_DIR/gpu0.log" 2>&1
  echo "[GPU0] 1.5B bias done" | tee -a "$LOG_DIR/gpu0.log"

  echo "[GPU0] 1.5B cats (multi-vector, 5 cats simultaneously)" | tee -a "$LOG_DIR/gpu0.log"
  CUDA_VISIBLE_DEVICES=0 python optimize_cat_vectors_multi.py \
      --model "$MODEL" --max_iters "$MAX_ITERS" \
      --n_training_examples "$N_TRAIN" --n_eval_examples "$N_EVAL" \
      --optim_minibatch_size "$MINI_BS" --layer "$LAYER" \
      --lr "$LR" --use_activation_perplexity_selection \
      --save_path "$SAVE_DIR" \
      >> "$LOG_DIR/gpu0.log" 2>&1
  echo "[GPU0] 1.5B cats done" | tee -a "$LOG_DIR/gpu0.log"

  python write_layer_map.py --save_dir "$SAVE_DIR" --layer "$LAYER" --n_cats 5 \
      >> "$LOG_DIR/gpu0.log" 2>&1
  echo "[GPU0] === 1.5B TRAINING COMPLETE ===" | tee -a "$LOG_DIR/gpu0.log"
  touch "$LOG_DIR/1.5b_training_done"
) &
GPU0_PID=$!

# ---------------------------------------------------------------------------
# GPU 1: ORZ-0.5B – bias then all 10 cats simultaneously
# ---------------------------------------------------------------------------
(
  source "$SCRIPT_DIR/.venv/bin/activate"
  cd "$TRAIN_DIR"
  SAVE_DIR="results/vars/optimized_vectors_legacy_ce"
  MODEL="Qwen/Qwen2.5-0.5B"
  LAYER=9
  MINI_BS=8    # 8 examples/step mixed across 10 cats → ~1 per cat per step

  echo "[GPU1] 0.5B bias training" | tee "$LOG_DIR/gpu1.log"
  CUDA_VISIBLE_DEVICES=1 python optimize_steering_vectors.py \
      --model "$MODEL" --max_iters "$MAX_ITERS" \
      --n_training_examples "$N_TRAIN" --n_eval_examples "$N_EVAL" \
      --optim_minibatch_size "$MINI_BS" --layer "$LAYER" \
      --steering_vector_idx -1 --lr "$LR" --save_path "$SAVE_DIR" \
      >> "$LOG_DIR/gpu1.log" 2>&1
  echo "[GPU1] 0.5B bias done" | tee -a "$LOG_DIR/gpu1.log"

  echo "[GPU1] 0.5B cats (multi-vector, 10 cats simultaneously)" | tee -a "$LOG_DIR/gpu1.log"
  CUDA_VISIBLE_DEVICES=1 python optimize_cat_vectors_multi.py \
      --model "$MODEL" --max_iters "$MAX_ITERS" \
      --n_training_examples "$N_TRAIN" --n_eval_examples "$N_EVAL" \
      --optim_minibatch_size "$MINI_BS" --layer "$LAYER" \
      --lr "$LR" --use_activation_perplexity_selection \
      --save_path "$SAVE_DIR" \
      >> "$LOG_DIR/gpu1.log" 2>&1
  echo "[GPU1] 0.5B cats done" | tee -a "$LOG_DIR/gpu1.log"

  python write_layer_map.py --save_dir "$SAVE_DIR" --layer "$LAYER" --n_cats 10 \
      >> "$LOG_DIR/gpu1.log" 2>&1
  echo "[GPU1] === 0.5B TRAINING COMPLETE ===" | tee -a "$LOG_DIR/gpu1.log"
  touch "$LOG_DIR/0.5b_training_done"
) &
GPU1_PID=$!

echo "PIDs: GPU0=$GPU0_PID  GPU1=$GPU1_PID"
echo "Log dir: $LOG_DIR"
echo ""
echo "Monitor with:"
echo "  tail -f $LOG_DIR/gpu0.log"
echo "  tail -f $LOG_DIR/gpu1.log"
echo ""

# ---------------------------------------------------------------------------
# Wait for both and launch evals
# ---------------------------------------------------------------------------
wait $GPU0_PID && echo "GPU0 (1.5B) finished" || echo "GPU0 (1.5B) FAILED"
wait $GPU1_PID && echo "GPU1 (0.5B) finished" || echo "GPU1 (0.5B) FAILED"

echo ""
echo "======================================"
echo " ALL TRAINING COMPLETE"
echo " Launching evals on GPU 0 and GPU 1..."
echo "======================================"

cd "$SCRIPT_DIR"

(
  source .venv/bin/activate
  CUDA_VISIBLE_DEVICES=0 MODEL_SIZE=1.5b bash run_legacy_eval.sh 2>&1 \
    | tee "$LOG_DIR/eval_1.5b.log"
) &
EVAL0_PID=$!

(
  source .venv/bin/activate
  CUDA_VISIBLE_DEVICES=1 MODEL_SIZE=0.5b bash run_legacy_eval.sh 2>&1 \
    | tee "$LOG_DIR/eval_0.5b.log"
) &
EVAL1_PID=$!

wait $EVAL0_PID && echo "1.5B eval done" || echo "1.5B eval FAILED"
wait $EVAL1_PID && echo "0.5B eval done" || echo "0.5B eval FAILED"

echo ""
echo "======================================"
echo " ALL DONE"
echo " Logs: $LOG_DIR"
echo "======================================"
