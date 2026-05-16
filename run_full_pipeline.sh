#!/bin/bash
# Full bias-first pipeline for both 32B pairs (DDP-parallel) + eval.
# Runs end-to-end: QwQ S1 -> QwQ S2 -> DSQ S1 -> DSQ S2 -> evals.
set -uo pipefail
cd /workspace/thinking-llms-interp
source .venv/bin/activate
source .env_exports.sh

export N_EPOCHS=${N_EPOCHS:-2}
export BS_PER_GPU=${BS_PER_GPU:-6}
export NPROC=${NPROC:-3}
export N_RECOLLECT=${N_RECOLLECT:-5000}

QWQ_BIAS=train-vectors/results/vars/correction_vectors_qwq32b_biasfirst_stage1/qwen2.5-32b_bias_global.pt
DSQ_BIAS=train-vectors/results/vars/correction_vectors_dsqwen32b_biasfirst_stage1/qwen2.5-32b_bias_global.pt
QWQ_CATS=train-vectors/results/vars/correction_vectors_qwq32b_biasfirst_stage2/qwen2.5-32b_idx0_linear.pt
DSQ_CATS=train-vectors/results/vars/correction_vectors_dsqwen32b_biasfirst_stage2/qwen2.5-32b_idx0_linear.pt

T_START=$(date +%s)
echo "[pipeline] config: N_EPOCHS=$N_EPOCHS  BS_PER_GPU=$BS_PER_GPU  NPROC=$NPROC  N_RECOLLECT=$N_RECOLLECT"

promote_best_checkpoint () {
    # If the trainer crashed near the end (e.g. NCCL timeout during the
    # final eval) the canonical {prefix}_global.pt / *_linear.pt files
    # may not exist, but the per-epoch crash-resilient {prefix}_best.pt
    # snapshot will.  Promote the latter when needed.  Returns 0 if a
    # promotion happened (=> safe to skip the stage), 1 otherwise.
    local SAVE_DIR="$1"
    local KIND="$2"             # "bias" or "cats"
    local MODEL_SHORT="$3"
    if [ "$KIND" = "bias" ]; then
        local CANON="$SAVE_DIR/${MODEL_SHORT}_bias_global.pt"
        local BEST="$SAVE_DIR/${MODEL_SHORT}_bias_best.pt"
        [ -f "$CANON" ] && return 0
        [ -f "$BEST" ] || return 1
        python -c "
import torch, json, os, sys
ckpt = torch.load('$BEST', map_location='cpu')
V = ckpt['V']  # (1, hidden) for bias training
bv = V[0]
torch.save({'bias': bv}, '$CANON')
with open(os.path.join('$SAVE_DIR', 'bias_layer.json'), 'w') as f:
    json.dump({'layer': 38, 'norm': float(bv.norm().item())}, f, indent=2)
print(f'[recover] promoted $BEST (epoch={ckpt.get(\"epoch\")}, '
      f'kl={ckpt.get(\"holdout_kl\"):.4f}, norm={bv.norm():.3f}) '
      f'-> $CANON')
" || return 1
        return 0
    elif [ "$KIND" = "cats" ]; then
        local CANON="$SAVE_DIR/${MODEL_SHORT}_idx0_linear.pt"
        [ -f "$CANON" ] && return 0
        # Find any cats *_best.pt in the save dir (seed agnostic).
        local BEST
        BEST=$(ls -1t "$SAVE_DIR"/${MODEL_SHORT}_cats_seed*_best.pt 2>/dev/null | head -n 1)
        [ -z "$BEST" ] && return 1
        [ -f "$BEST" ] || return 1
        python -c "
import torch, json, os
ckpt = torch.load('$BEST', map_location='cpu')
V = ckpt['V']  # (n_cats, hidden)
sd = '$SAVE_DIR'
ms = '$MODEL_SHORT'
n_cats = int(V.shape[0])
for c in range(n_cats):
    out = os.path.join(sd, f'{ms}_idx{c}_linear.pt')
    torch.save({'V': V[c]}, out)
    print(f'[recover] wrote {out} (norm={float(V[c].norm()):.3f})')
with open(os.path.join(sd, 'layer_map.json'), 'w') as f:
    json.dump({f'idx{c}': 38 for c in range(n_cats)}, f, indent=2)
print(f'[recover] promoted {n_cats} cats from $BEST '
      f'(epoch={ckpt.get(\"epoch\")}, kl={ckpt.get(\"holdout_kl\"):.4f})')
" || return 1
        return 0
    fi
    return 1
}

run_or_skip () {
    # Skip a script when its expected output already exists, or when a
    # crash-resilient checkpoint can be promoted to that output.
    local TAG="$1"
    local OUTFILE="$2"
    local SCRIPT="$3"
    local KIND="${4:-}"
    local SAVE_DIR="${5:-}"
    local MODEL_SHORT="${6:-}"
    if [ -f "$OUTFILE" ]; then
        echo "[pipeline] SKIP $TAG  (output $OUTFILE already present)"
        return 0
    fi
    if [ -n "$KIND" ] && [ -n "$SAVE_DIR" ] && [ -n "$MODEL_SHORT" ]; then
        if promote_best_checkpoint "$SAVE_DIR" "$KIND" "$MODEL_SHORT"; then
            if [ -f "$OUTFILE" ]; then
                echo "[pipeline] RECOVERED $TAG from crash checkpoint"
                return 0
            fi
        fi
    fi
    local T0=$(date +%s)
    echo "[pipeline] === $TAG ==="
    bash "$SCRIPT" 2>&1 | tee "/tmp/${TAG}.log"
    local STATUS=${PIPESTATUS[0]}
    local DT=$(($(date +%s) - T0))
    echo "[pipeline] $TAG done exit=$STATUS elapsed=${DT}s"
    while pgrep -f "optimize_correction_vectors.py" >/dev/null; do sleep 5; done
    sleep 8  # GPU memory release grace
    if [ "$STATUS" -ne 0 ]; then
        # Even if the trainer crashed, see if we can salvage a best
        # checkpoint.  If yes, promote and continue; otherwise abort.
        if [ -n "$KIND" ] && [ -n "$SAVE_DIR" ] && [ -n "$MODEL_SHORT" ] \
                && promote_best_checkpoint "$SAVE_DIR" "$KIND" "$MODEL_SHORT" \
                && [ -f "$OUTFILE" ]; then
            echo "[pipeline] $TAG crashed but RECOVERED from checkpoint"
            return 0
        fi
        echo "[pipeline] $TAG FAILED; aborting"
        exit 1
    fi
}

QWQ_S1_DIR="train-vectors/results/vars/correction_vectors_qwq32b_biasfirst_stage1"
QWQ_S2_DIR="train-vectors/results/vars/correction_vectors_qwq32b_biasfirst_stage2"
DSQ_S1_DIR="train-vectors/results/vars/correction_vectors_dsqwen32b_biasfirst_stage1"
DSQ_S2_DIR="train-vectors/results/vars/correction_vectors_dsqwen32b_biasfirst_stage2"

run_or_skip "qwq_s1" "$QWQ_BIAS" "train-vectors/run_qwq32b_biasfirst_stage1.sh" \
    "bias" "$QWQ_S1_DIR" "qwen2.5-32b"
run_or_skip "qwq_s2" "$QWQ_CATS" "train-vectors/run_qwq32b_biasfirst_stage2.sh" \
    "cats" "$QWQ_S2_DIR" "qwen2.5-32b"
run_or_skip "dsq_s1" "$DSQ_BIAS" "train-vectors/run_dsqwen32b_biasfirst_stage1.sh" \
    "bias" "$DSQ_S1_DIR" "qwen2.5-32b"
run_or_skip "dsq_s2" "$DSQ_CATS" "train-vectors/run_dsqwen32b_biasfirst_stage2.sh" \
    "cats" "$DSQ_S2_DIR" "qwen2.5-32b"

T_TRAIN_END=$(date +%s)
echo "[pipeline] TRAINING DONE.  total=$(((T_TRAIN_END - T_START)/60)) min"

T_EVAL=$(date +%s)
echo "===== QwQ eval (3 conditions in parallel) ====="
bash run_qwq_eval_parallel.sh 2>&1 | tee /tmp/qwq_eval_parallel.log

echo "===== DSQ eval (3 conditions in parallel) ====="
bash run_dsq_eval_parallel.sh 2>&1 | tee /tmp/dsq_eval_parallel.log
echo "[pipeline] eval elapsed $(($(date +%s) - T_EVAL))s"

T_ALL=$(($(date +%s) - T_START))
echo
echo "[pipeline] ALL DONE total=${T_ALL}s ($((T_ALL/60)) min)"
