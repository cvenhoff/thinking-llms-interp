#!/usr/bin/env bash
# Generic vLLM model server for SLURM.
# Usage:  sbatch --export=MODEL=Qwen/QwQ-32B,NGPU=2 serve_model.sh
#   or:   sbatch --export=MODEL=Qwen/QwQ-32B,NGPU=2,PORT=8001 serve_model.sh
#
# The server is OpenAI-compatible at http://<node>:<PORT>/v1
# A "ready" sentinel file is written to /tmp/vllm_ready_<PORT> when the
# server is listening, so dependent jobs or scripts can poll for it.

#SBATCH --job-name=vllm-serve
#SBATCH --partition=general
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
#SBATCH --mem=200G
#SBATCH --time=24:00:00
#SBATCH --output=/workspace-vast/constantinv/thinking-llms-interp/slurm_logs/%x-%j.out
#SBATCH --error=/workspace-vast/constantinv/thinking-llms-interp/slurm_logs/%x-%j.err

set -euo pipefail

cd /workspace-vast/constantinv/thinking-llms-interp
source .env_exports.sh
source .venv_vllm/bin/activate

: "${MODEL:?Must set MODEL (e.g. Qwen/QwQ-32B)}"
: "${NGPU:=2}"
: "${PORT:=8000}"
: "${MAX_MODEL_LEN:=8192}"
: "${GPU_MEMORY_UTILIZATION:=0.90}"

READY_FILE="/tmp/vllm_ready_${PORT}"
rm -f "${READY_FILE}"

echo "===================================="
echo "vLLM serve  model=${MODEL}  ngpu=${NGPU}  port=${PORT}"
echo "Job ${SLURM_JOB_NAME} ${SLURM_JOB_ID}  node=$(hostname)  start=$(date -u)"
nvidia-smi --query-gpu=index,name,memory.used,memory.total --format=csv
echo "===================================="

# Write connection info so clients can discover the endpoint
INFO_FILE="/workspace-vast/constantinv/thinking-llms-interp/vllm-serve/active_servers.txt"
echo "${SLURM_JOB_ID}  $(hostname):${PORT}  ${MODEL}  ngpu=${NGPU}" >> "${INFO_FILE}"

# Background a watcher that creates the ready sentinel once the server is up
(while true; do
    if curl -s "http://localhost:${PORT}/health" > /dev/null 2>&1; then
        touch "${READY_FILE}"
        echo "[watcher] vLLM ready at $(date -u)"
        break
    fi
    sleep 5
done) &

python -m vllm.entrypoints.openai.api_server \
    --model "${MODEL}" \
    --tensor-parallel-size "${NGPU}" \
    --port "${PORT}" \
    --max-model-len "${MAX_MODEL_LEN}" \
    --gpu-memory-utilization "${GPU_MEMORY_UTILIZATION}" \
    --trust-remote-code \
    --disable-log-requests \
    --disable-custom-all-reduce \
    --enable-prefix-caching
