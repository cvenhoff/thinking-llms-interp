#!/usr/bin/env bash
# ============================================================
# Parallel full training for ORZ-1.5B and ORZ-0.5B
#
# GPU allocation strategy:
#   GPU 0: 1.5B bias  →  1.5B cats 0..4 (sequential on same GPU)
#   GPU 1: 0.5B bias  →  0.5B cats 0..4
#   GPU 2: wait for 0.5B bias  →  0.5B cats 5..9
#
# After training completes, evaluations run from GPU 0 and GPU 1.
#
# Usage:
#   bash launch_full_training.sh
# ============================================================
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TRAIN_DIR="$SCRIPT_DIR/train-vectors"
LOG_DIR="$SCRIPT_DIR/logs/launch_$(date +%Y%m%d_%H%M%S)"
mkdir -p "$LOG_DIR"

# ---- Shared training params ----
N_TRAIN=2048
N_EVAL=512
MAX_ITERS=50
LR="1e-2"

echo "======================================"
echo " Full training launch"
echo "   N_TRAIN=$N_TRAIN  MAX_ITERS=$MAX_ITERS"
echo "   Logs: $LOG_DIR"
echo "======================================"

# ---------------------------------------------------------------------------
# GPU 0: 1.5B – bias, then cats 0..4
# ---------------------------------------------------------------------------
(
  source "$SCRIPT_DIR/.venv/bin/activate"
  cd "$TRAIN_DIR"
  SAVE_DIR="results/vars/optimized_vectors"
  MODEL="Qwen/Qwen2.5-1.5B"
  LAYER=10
  MINI_BS=8

  echo "[GPU0] 1.5B bias training" | tee "$LOG_DIR/gpu0.log"
  CUDA_VISIBLE_DEVICES=0 python optimize_steering_vectors.py \
      --model "$MODEL" --max_iters "$MAX_ITERS" \
      --n_training_examples "$N_TRAIN" --n_eval_examples "$N_EVAL" \
      --optim_minibatch_size "$MINI_BS" --layer "$LAYER" \
      --steering_vector_idx -1 --lr "$LR" --save_path "$SAVE_DIR" \
      >> "$LOG_DIR/gpu0.log" 2>&1
  echo "[GPU0] 1.5B bias done" | tee -a "$LOG_DIR/gpu0.log"

  for CLUSTER in 0 1 2 3 4; do
    echo "[GPU0] 1.5B cat idx${CLUSTER}" | tee -a "$LOG_DIR/gpu0.log"
    CUDA_VISIBLE_DEVICES=0 python optimize_steering_vectors.py \
        --model "$MODEL" --max_iters "$MAX_ITERS" \
        --n_training_examples "$N_TRAIN" --n_eval_examples "$N_EVAL" \
        --optim_minibatch_size "$MINI_BS" --layer "$LAYER" \
        --steering_vector_idx "$CLUSTER" --lr "$LR" \
        --use_activation_perplexity_selection \
        --save_path "$SAVE_DIR" \
        >> "$LOG_DIR/gpu0.log" 2>&1
    echo "[GPU0] 1.5B cat idx${CLUSTER} done" | tee -a "$LOG_DIR/gpu0.log"
  done

  python write_layer_map.py --save_dir "$SAVE_DIR" --layer "$LAYER" --n_cats 5 \
      >> "$LOG_DIR/gpu0.log" 2>&1
  echo "[GPU0] === 1.5B TRAINING COMPLETE ===" | tee -a "$LOG_DIR/gpu0.log"
  # Signal completion
  touch "$LOG_DIR/1.5b_training_done"
) &
GPU0_PID=$!

# ---------------------------------------------------------------------------
# GPU 1: 0.5B – bias, then cats 0..4
# ---------------------------------------------------------------------------
(
  source "$SCRIPT_DIR/.venv/bin/activate"
  cd "$TRAIN_DIR"
  SAVE_DIR="results/vars/optimized_vectors"
  MODEL="Qwen/Qwen2.5-0.5B"
  LAYER=9
  MINI_BS=4

  echo "[GPU1] 0.5B bias training" | tee "$LOG_DIR/gpu1.log"
  CUDA_VISIBLE_DEVICES=1 python optimize_steering_vectors.py \
      --model "$MODEL" --max_iters "$MAX_ITERS" \
      --n_training_examples "$N_TRAIN" --n_eval_examples "$N_EVAL" \
      --optim_minibatch_size "$MINI_BS" --layer "$LAYER" \
      --steering_vector_idx -1 --lr "$LR" --save_path "$SAVE_DIR" \
      >> "$LOG_DIR/gpu1.log" 2>&1
  echo "[GPU1] 0.5B bias done" | tee -a "$LOG_DIR/gpu1.log"
  touch "$LOG_DIR/0.5b_bias_done"  # signal GPU2 to start

  for CLUSTER in 0 1 2 3 4; do
    echo "[GPU1] 0.5B cat idx${CLUSTER}" | tee -a "$LOG_DIR/gpu1.log"
    CUDA_VISIBLE_DEVICES=1 python optimize_steering_vectors.py \
        --model "$MODEL" --max_iters "$MAX_ITERS" \
        --n_training_examples "$N_TRAIN" --n_eval_examples "$N_EVAL" \
        --optim_minibatch_size "$MINI_BS" --layer "$LAYER" \
        --steering_vector_idx "$CLUSTER" --lr "$LR" \
        --use_activation_perplexity_selection \
        --save_path "$SAVE_DIR" \
        >> "$LOG_DIR/gpu1.log" 2>&1
    echo "[GPU1] 0.5B cat idx${CLUSTER} done" | tee -a "$LOG_DIR/gpu1.log"
  done
  echo "[GPU1] cats 0-4 done" | tee -a "$LOG_DIR/gpu1.log"
) &
GPU1_PID=$!

# ---------------------------------------------------------------------------
# GPU 2: 0.5B – wait for bias, then cats 5..9
# ---------------------------------------------------------------------------
(
  source "$SCRIPT_DIR/.venv/bin/activate"
  cd "$TRAIN_DIR"
  SAVE_DIR="results/vars/optimized_vectors"
  MODEL="Qwen/Qwen2.5-0.5B"
  LAYER=9
  MINI_BS=4

  echo "[GPU2] Waiting for 0.5B bias to complete..." | tee "$LOG_DIR/gpu2.log"
  while [[ ! -f "$LOG_DIR/0.5b_bias_done" ]]; do sleep 10; done
  echo "[GPU2] 0.5B bias ready, starting cats 5..9" | tee -a "$LOG_DIR/gpu2.log"

  for CLUSTER in 5 6 7 8 9; do
    echo "[GPU2] 0.5B cat idx${CLUSTER}" | tee -a "$LOG_DIR/gpu2.log"
    CUDA_VISIBLE_DEVICES=2 python optimize_steering_vectors.py \
        --model "$MODEL" --max_iters "$MAX_ITERS" \
        --n_training_examples "$N_TRAIN" --n_eval_examples "$N_EVAL" \
        --optim_minibatch_size "$MINI_BS" --layer "$LAYER" \
        --steering_vector_idx "$CLUSTER" --lr "$LR" \
        --use_activation_perplexity_selection \
        --save_path "$SAVE_DIR" \
        >> "$LOG_DIR/gpu2.log" 2>&1
    echo "[GPU2] 0.5B cat idx${CLUSTER} done" | tee -a "$LOG_DIR/gpu2.log"
  done
  echo "[GPU2] cats 5-9 done" | tee -a "$LOG_DIR/gpu2.log"
  touch "$LOG_DIR/0.5b_cats_gpu2_done"
) &
GPU2_PID=$!

echo "PIDs: GPU0=$GPU0_PID  GPU1=$GPU1_PID  GPU2=$GPU2_PID"
echo "Log dir: $LOG_DIR"
echo ""
echo "Monitor with:"
echo "  tail -f $LOG_DIR/gpu0.log"
echo "  tail -f $LOG_DIR/gpu1.log"
echo "  tail -f $LOG_DIR/gpu2.log"
echo ""

# ---------------------------------------------------------------------------
# Wait for all jobs and write layer_map for 0.5B when both GPU1/GPU2 finish
# ---------------------------------------------------------------------------
wait $GPU1_PID $GPU2_PID
echo "======================================"
echo " 0.5B cat training complete. Writing layer_map.json..."
cd "$TRAIN_DIR"
source "$SCRIPT_DIR/.venv/bin/activate"
python write_layer_map.py --save_dir "results/vars/optimized_vectors" --layer 9 --n_cats 10 \
    >> "$LOG_DIR/gpu1.log" 2>&1
touch "$LOG_DIR/0.5b_training_done"
echo " 0.5B layer_map.json written ✓"

wait $GPU0_PID
echo "======================================"
echo " ALL TRAINING COMPLETE"
echo "======================================"
echo ""
echo "Next: run evals:"
echo "  CUDA_VISIBLE_DEVICES=0 MODEL_SIZE=1.5b bash run_legacy_eval.sh"
echo "  CUDA_VISIBLE_DEVICES=1 MODEL_SIZE=0.5b bash run_legacy_eval.sh"
