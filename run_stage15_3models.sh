#!/usr/bin/env bash
# Launch stage-1.5 recipe for ORZ-1.5B (GPU0), ORZ-7B (GPU1), DSL-Llama-8B (GPU2)
set -uo pipefail
cd /workspace/thinking-llms-interp
source .venv/bin/activate
source .env_exports.sh
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY=error
mkdir -p /workspace/tmp

run_model () {
GPU=$1 BASE=$2 THINK=$3 THINK_SHORT=$4 BASE_SHORT=$5 TAG=$6 STEER=$7 SAE_L=$8 K=$9
export GPU BASE THINK THINK_SHORT BASE_SHORT TAG STEER SAE_L K
bash /workspace/thinking-llms-interp/run_stage15_recipe.sh \
  2>&1 | tee /workspace/tmp/${TAG}_s15.log
}

run_model 0 \
  "Qwen/Qwen2.5-1.5B" \
  "Open-Reasoner-Zero/Open-Reasoner-Zero-1.5B" \
  "open-reasoner-zero-1.5b" \
  "qwen2.5-1.5b" \
  "orz-1.5b" \
  14 16 10 &
PID0=$!

run_model 1 \
  "Qwen/Qwen2.5-7B" \
  "Open-Reasoner-Zero/Open-Reasoner-Zero-7B" \
  "open-reasoner-zero-7b" \
  "qwen2.5-7b" \
  "orz-7b" \
  16 20 10 &
PID1=$!

run_model 2 \
  "meta-llama/Llama-3.1-8B" \
  "deepseek-ai/DeepSeek-R1-Distill-Llama-8B" \
  "deepseek-r1-distill-llama-8b" \
  "llama-3.1-8b" \
  "dsl-llama-8b" \
  16 18 10 &
PID2=$!

echo "Launched: ORZ-1.5B (GPU0, PID $PID0)  ORZ-7B (GPU1, PID $PID1)  DSL-Llama-8B (GPU2, PID $PID2)"
echo "Logs: /workspace/tmp/{orz-1.5b,orz-7b,dsl-llama-8b}_s15.log"
wait $PID0; C0=$?
wait $PID1; C1=$?
wait $PID2; C2=$?
echo "Exit codes: orz-1.5b=$C0  orz-7b=$C1  dsl-llama-8b=$C2"
