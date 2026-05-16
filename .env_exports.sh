export HF_HOME=/workspace/hf_cache/huggingface
export HF_HUB_CACHE=/workspace/hf_cache/huggingface/hub
export TRANSFORMERS_CACHE=/workspace/hf_cache/huggingface/hub
export HF_HUB_ENABLE_HF_TRANSFER=1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONUNBUFFERED=1
export TRANSFORMERS_VERBOSITY=error
# Load API keys (OPENAI_API_KEY, ANTHROPIC_API_KEY, OPENROUTER_API_KEY, HF_TOKEN).
if [ -f /workspace/thinking-llms-interp/.env ]; then
    set -a
    . /workspace/thinking-llms-interp/.env
    set +a
fi
