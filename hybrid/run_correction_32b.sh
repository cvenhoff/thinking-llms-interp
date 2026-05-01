#!/usr/bin/env bash
# =============================================================================
# 32B correction-vector pipeline for BOTH
#   * Qwen2.5-32B  <-  QwQ-32B
#   * Qwen2.5-32B  <-  Open-Reasoner-Zero-32B
#
# Phase 1  (per pair):   train per-category correction vectors + a single
#                        global-bias vector on 1500 annotated thinking
#                        responses.  Steering layer is HARD-CODED to
#                        num_layers // 2 + num_layers // 10  =  64/2 + 64/10
#                        =  32 + 6  =  38  (Qwen2.5-32B has 64 layers).
#
# Phase 2  (per pair):   hybrid_eval on MATH500, 500 tasks, using the newly
#                        trained per-category vectors at disagreement gates.
#                        NO bias is folded in -- the bias is a separate
#                        baseline below.
#
# Phase 3  (per pair):   hybrid_eval bias-only ablation.  Replaces every
#                        per-category vector with the single global-bias
#                        vector, applied at its own chosen layer (same
#                        layer 38 here since no layer sweep is done).
#                        Tells us whether category-specific steering does
#                        meaningful work beyond a single global direction.
#
# 32B MEMORY CHECK  (H200 = 141 GB)
#   - Collection phase:  base (~66 GB bf16) + thinking (~66 GB bf16)
#                        = ~132 GB on GPU.  Forward on one 2048-token seq
#                        at a time (bs=1) peaks at <3 GB activations.
#                        Tight but fits; thinking model is freed before
#                        training.
#   - Training phase:    base only (~66 GB), activations for layers
#                        38..63 at bs=16,len=2048,H=5120,bf16 = 8.7 GB.
#                        Plenty of headroom.
#   - Hybrid eval:       hybrid_gen_batch_size=4 as in run_all_pairs.
# =============================================================================
set -u
cd "$(dirname "$0")"

export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export TRANSFORMERS_VERBOSITY=error
export PYTHONUNBUFFERED=1

TRAIN_SCRIPT=../train-vectors/optimize_correction_vectors.py
OUTDIR_QWQ=../train-vectors/results/vars/correction_vectors_qwq32b
OUTDIR_ORZ=../train-vectors/results/vars/correction_vectors_orz32b
OUTDIR_DSQWEN=../train-vectors/results/vars/correction_vectors_dsqwen32b
DOM_DIR=../train-vectors/results/diff_of_means   # kept for --dom_* argparse compat
LOG_ROOT=results/benchmark_logs_correction_32b
mkdir -p "$OUTDIR_QWQ" "$OUTDIR_ORZ" "$OUTDIR_DSQWEN" "$LOG_ROOT"

STEER_LAYER=38          # = 64//2 + 64//10 for Qwen2.5-32B (64 layers)
N_RESPONSES=2000        # ~2000 of 12K annotated responses used per pair
N_EPOCHS=5
# Training runs in a SEPARATE process from data collection (--collect_only
# then --load_collected) so only ONE 32B model is ever resident at a time:
#   * collect  : 2x 32B bf16 (~124 GB)       -- no grads
#   * train    : 1x 32B bf16  (~62 GB) + gradient checkpointing for
#                activations (~10-15 GB) + vec grads
# Gradient checkpointing re-computes layer activations on backward, so
# activation memory scales with only a handful of layers' worth rather
# than 26 full bf16 tensors.  EX_BATCH=4 keeps peak activation memory
# below 15 GB on the longest 2048-token bucket.
EX_BATCH=4
MAX_POS_PER_EX=64
LR=1e-2
TOPK=50
SEED=42
MAX_SEQ_LEN=2048

# -----------------------------------------------------------------------------
# TRAIN -----------------------------------------------------------------------
train_pair() {
  local tag="$1"              # e.g. qwq-32b / orz-32b
  local thinker="$2"          # HF id
  local short="$3"            # annotated_responses_<short>.json
  local save_dir="$4"

  local tlog="$save_dir/train.log"
  echo ""
  echo "=========================================================="
  echo "[$(date +%H:%M:%S)] TRAIN  pair=$tag"
  echo "  thinker: $thinker"
  echo "  save:    $save_dir"
  echo "  log:     $tlog"
  echo "=========================================================="

  # Wipe previous *trained vector* artifacts so we always train fresh, but
  # KEEP an existing disagreements.pt around: collection takes 15 min and
  # is deterministic given (thinker, n_responses, seed), so it's safe (and
  # time-saving) to resume from it across train-only failures.
  rm -f "$save_dir"/*_linear.pt "$save_dir"/*_bias_global.pt \
        "$save_dir"/layer_map.json "$save_dir"/bias_layer.json \
        "$save_dir"/*_correction_meta.json "$save_dir"/train.log

  # ---- Phase 1a: COLLECT (loads BOTH base + thinking; writes
  # disagreements.pt; exits so GPU mem is truly freed by the OS). ----
  if [[ -s "$save_dir/disagreements.pt" ]]; then
    echo "[$(date +%H:%M:%S)] COLLECT  pair=$tag  SKIPPED "\
"(disagreements.pt already present, $(du -h "$save_dir/disagreements.pt" | cut -f1))"
    echo "[collect-cached] $save_dir/disagreements.pt" > "$tlog"
  else
    echo "[$(date +%H:%M:%S)] COLLECT  pair=$tag  (base + thinking on GPU)"
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    uv run python "$TRAIN_SCRIPT" \
        --base_model Qwen/Qwen2.5-32B \
        --thinking_model "$thinker" \
        --thinking_model_short "$short" \
        --steer_layer $STEER_LAYER \
        --save_dir "$save_dir" \
        --n_responses $N_RESPONSES \
        --max_seq_len $MAX_SEQ_LEN \
        --topk $TOPK --seed $SEED \
        --collect_only \
        2>&1 | tee "$tlog"
    local rc_c="${PIPESTATUS[0]}"
    echo "[$(date +%H:%M:%S)] COLLECT  pair=$tag  exit=$rc_c"
    if [[ "$rc_c" != "0" ]]; then
      echo "!!! COLLECT failed for $tag; skipping training." >&2
      return 1
    fi
  fi

  # ---- Phase 1b: TRAIN (loads ONLY base; reads disagreements.pt). ----
  echo "[$(date +%H:%M:%S)] TRAIN    pair=$tag  (base only)"
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  uv run python "$TRAIN_SCRIPT" \
      --base_model Qwen/Qwen2.5-32B \
      --thinking_model "$thinker" \
      --thinking_model_short "$short" \
      --steer_layer $STEER_LAYER \
      --train_global_bias \
      --save_dir "$save_dir" \
      --max_positions_per_example $MAX_POS_PER_EX \
      --n_epochs $N_EPOCHS \
      --example_batch_size $EX_BATCH \
      --lr $LR --topk $TOPK --seed $SEED \
      --holdout_frac 0.0 \
      --min_disagreements_ratio 0.1 \
      --load_collected \
      2>&1 | tee -a "$tlog"
  local rc_t="${PIPESTATUS[0]}"
  echo "[$(date +%H:%M:%S)] TRAIN    pair=$tag  exit=$rc_t"
  if [[ "$rc_t" != "0" ]]; then
    echo "!!! TRAIN failed for $tag; aborting pipeline." >&2
    return 2
  fi
  return 0
}

# -----------------------------------------------------------------------------
# HYBRID EVAL -----------------------------------------------------------------
# Shared knobs for all 32B hybrid_eval runs.
SHARED_HYBRID=(
  --dataset math500 --n_tasks 500
  --max_new_tokens 2000 --max_thinking_tokens 2000
  --base_model Qwen/Qwen2.5-32B
  --sae_layer 27
  --hybrid_gen_batch_size 4
  --batch_gen_size 4          # reference thinking/base response gen:
                              # single model at a time but with the OTHER
                              # 32B model also resident, headroom is only
                              # ~17 GB so default (32) OOMs.  Matches the
                              # 32B pairs in run_all_pairs_benchmark.sh.
  --coef_sweep 1.0
  --dom_vectors_dir "$DOM_DIR"
  --dom_vectors_model_short qwen2.5-32b
)

eval_pair() {
  local tag="$1"              # qwq-32b / orz-32b
  local thinker="$2"
  local n_clusters="$3"
  local corr_dir="$4"
  shift 4
  local extra=("$@")          # condition-specific flags (e.g. --disable_sae_mean)

  local std_tag="corr-${tag}-standard-500"
  local bias_tag="corr-${tag}-globalbias-500"

  # Clear stale caches for these two tags.
  rm -f "results/rolling/rolling_qwen2.5-32b_math500_${std_tag}.jsonl" \
        "results/rolling/rolling_qwen2.5-32b_math500_${bias_tag}.jsonl" \
        "results/summary_qwen2.5-32b_math500_${std_tag}.json" \
        "results/summary_qwen2.5-32b_math500_${bias_tag}.json" 2>/dev/null

  # ---- STANDARD: per-category correction vectors, NO bias ----
  local slog="$LOG_ROOT/${tag}-standard.log"
  echo ""
  echo "=========================================================="
  echo "[$(date +%H:%M:%S)] EVAL  $tag/standard"
  echo "  log: $slog"
  echo "=========================================================="
  uv run python hybrid_eval.py \
      "${SHARED_HYBRID[@]}" \
      --thinking_model "$thinker" \
      --n_clusters "$n_clusters" \
      --old_vectors_dir "$corr_dir" \
      --old_vectors_layer $STEER_LAYER \
      "${extra[@]}" \
      --results_suffix "$std_tag" 2>&1 | tee "$slog"
  echo "[$(date +%H:%M:%S)] EVAL  $tag/standard  exit=${PIPESTATUS[0]}"

  # ---- GLOBAL-BIAS-ONLY ablation ----
  local blog="$LOG_ROOT/${tag}-globalbias.log"
  echo ""
  echo "=========================================================="
  echo "[$(date +%H:%M:%S)] EVAL  $tag/globalbias"
  echo "  log: $blog"
  echo "=========================================================="
  uv run python hybrid_eval.py \
      "${SHARED_HYBRID[@]}" \
      --thinking_model "$thinker" \
      --n_clusters "$n_clusters" \
      --old_vectors_dir "$corr_dir" \
      --old_vectors_layer $STEER_LAYER \
      --bias_vector_path "$corr_dir/qwen2.5-32b_bias_global.pt" \
      --bias_only \
      "${extra[@]}" \
      --results_suffix "$bias_tag" 2>&1 | tee "$blog"
  echo "[$(date +%H:%M:%S)] EVAL  $tag/globalbias  exit=${PIPESTATUS[0]}"
}

# =============================================================================
# Entry: optional positional args select which phase/pair to run.
#   e.g. `run_correction_32b.sh train_qwq` or `eval_orz`
# Default: full pipeline (train both, eval both).
# =============================================================================
STAGES=("$@")
if [[ ${#STAGES[@]} -eq 0 ]]; then
  STAGES=(train_qwq train_orz train_dsqwen eval_qwq eval_orz eval_dsqwen)
fi

for stage in "${STAGES[@]}"; do
  case "$stage" in
    train_qwq)
      train_pair qwq-32b Qwen/QwQ-32B qwq-32b "$OUTDIR_QWQ" \
        || { echo "train_qwq failed; aborting."; exit 1; }
      ;;
    train_orz)
      train_pair orz-32b Open-Reasoner-Zero/Open-Reasoner-Zero-32B \
                 open-reasoner-zero-32b "$OUTDIR_ORZ" \
        || { echo "train_orz failed; aborting."; exit 1; }
      ;;
    train_dsqwen)
      train_pair dsqwen-32b deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
                 deepseek-r1-distill-qwen-32b "$OUTDIR_DSQWEN" \
        || { echo "train_dsqwen failed; aborting."; exit 1; }
      ;;
    eval_qwq)
      eval_pair qwq-32b Qwen/QwQ-32B 10 "$OUTDIR_QWQ" --disable_sae_mean
      ;;
    eval_orz)
      eval_pair orz-32b Open-Reasoner-Zero/Open-Reasoner-Zero-32B \
                15 "$OUTDIR_ORZ"
      ;;
    eval_dsqwen)
      eval_pair dsqwen-32b deepseek-ai/DeepSeek-R1-Distill-Qwen-32B \
                15 "$OUTDIR_DSQWEN"
      ;;
    *)
      echo "Unknown stage: $stage" >&2
      exit 2
      ;;
  esac
done

echo ""
echo "[$(date +%H:%M:%S)] 32B correction pipeline complete"
