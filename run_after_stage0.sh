#!/bin/bash
# Master orchestrator: bias-first pipeline for both 32B pairs, DDP-parallel.
#
# Training: each stage uses torchrun --nproc_per_node=3 with each rank
# holding its own copy of the 64 GB base model on a dedicated H200.  V/b
# grads are all-reduced after every backward, giving an effective batch
# of BS_PER_GPU * 3 with single-GPU latency per step (~3-4 s).
#
# Recipe deltas vs canonical (to fit ~5 h training budget across both pairs):
#   * n_epochs   2  (was 5)
#   * BS_PER_GPU 8  (was 16, OOM-safe on H200 + frozen 64 GB base)
#   * stage 2a re-collect n_responses 5000 (was 20000) - pipeline-parallel
#     inference, no DDP needed
# Everything else (lr 0.01, max_positions_per_example 64, kl topk-3, etc.)
# matches the proven bias-first recipe.
set -uo pipefail
cd /workspace/thinking-llms-interp
source .venv/bin/activate
source .env_exports.sh

export N_EPOCHS=${N_EPOCHS:-2}
export BS_PER_GPU=${BS_PER_GPU:-8}
export NPROC=${NPROC:-3}
export N_RECOLLECT=${N_RECOLLECT:-5000}

QWQ_S0_PT=train-vectors/results/vars/correction_vectors_qwq32b_biasfirst_stage1/disagreements.pt
DSQ_S0_PT=train-vectors/results/vars/correction_vectors_dsqwen32b_biasfirst_stage1/disagreements.pt

echo "[after_stage0] config: N_EPOCHS=$N_EPOCHS  BS_PER_GPU=$BS_PER_GPU  NPROC=$NPROC  N_RECOLLECT=$N_RECOLLECT"
echo "[after_stage0] checking stage 0 dumps..."
if [ ! -f "$QWQ_S0_PT" ] || [ ! -f "$DSQ_S0_PT" ]; then
    echo "ERROR: stage 0 dumps missing; cannot proceed"
    ls -la "$QWQ_S0_PT" "$DSQ_S0_PT" 2>&1
    exit 1
fi

T_START=$(date +%s)
echo "===== QwQ training chain (DDP x$NPROC, bs=$BS_PER_GPU, n_epochs=$N_EPOCHS) ====="
bash run_qwq_chain.sh 2>&1 | tee /tmp/qwq_chain.log
QWQ_STATUS=${PIPESTATUS[0]}
T_QWQ=$(($(date +%s) - T_START))
echo "[after_stage0] QwQ chain exit=$QWQ_STATUS  elapsed=${T_QWQ}s"

if [ "$QWQ_STATUS" -ne 0 ]; then
    echo "[after_stage0] QwQ failed; aborting"
    exit 1
fi

T_DSQ_START=$(date +%s)
echo "===== DSQ training chain (DDP x$NPROC, bs=$BS_PER_GPU, n_epochs=$N_EPOCHS) ====="
bash run_dsq_chain.sh 2>&1 | tee /tmp/dsq_chain.log
DSQ_STATUS=${PIPESTATUS[0]}
T_DSQ=$(($(date +%s) - T_DSQ_START))
echo "[after_stage0] DSQ chain exit=$DSQ_STATUS  elapsed=${T_DSQ}s"

if [ "$DSQ_STATUS" -ne 0 ]; then
    echo "[after_stage0] DSQ failed; aborting eval"
    exit 1
fi

T_TRAIN_TOTAL=$(($(date +%s) - T_START))
echo "[after_stage0] TRAINING ALL DONE.  elapsed=${T_TRAIN_TOTAL}s ($((T_TRAIN_TOTAL/60)) min)"

# Eval phase: 3 conditions per pair, 1 GPU each, in parallel.
echo "===== QwQ eval (3 conditions in parallel) ====="
bash run_qwq_eval_parallel.sh 2>&1 | tee /tmp/qwq_eval_parallel.log

echo "===== DSQ eval (3 conditions in parallel) ====="
bash run_dsq_eval_parallel.sh 2>&1 | tee /tmp/dsq_eval_parallel.log

T_TOTAL=$(($(date +%s) - T_START))
echo
echo "ALL DONE.  total elapsed=${T_TOTAL}s ($((T_TOTAL/60)) min)"
