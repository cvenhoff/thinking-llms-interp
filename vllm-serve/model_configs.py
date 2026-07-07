"""Model configurations for vLLM serving and response formatting.

Each entry maps a short model name to its HuggingFace ID, GPU
requirements, and response format.  vLLM is used exclusively for
generating thinking-model rollouts (MMLU training data); base models
are irrelevant here.
"""

MODEL_CONFIGS = {
    # ── ORZ family ─────────────────────────────────────────────
    "orz-0.5b": {
        "model_id": "Open-Reasoner-Zero/Open-Reasoner-Zero-0.5B",
        "ngpu": 1,
        "format": "orz",
        "model_short": "open-reasoner-zero-0.5b",
    },
    "orz-1.5b": {
        "model_id": "Open-Reasoner-Zero/Open-Reasoner-Zero-1.5B",
        "ngpu": 1,
        "format": "orz",
        "model_short": "open-reasoner-zero-1.5b",
    },
    "orz-7b": {
        "model_id": "Open-Reasoner-Zero/Open-Reasoner-Zero-7B",
        "ngpu": 1,
        "format": "orz",
        "model_short": "open-reasoner-zero-7b",
    },
    "orz-32b": {
        "model_id": "Open-Reasoner-Zero/Open-Reasoner-Zero-32B",
        "ngpu": 2,
        "format": "orz",
        "model_short": "open-reasoner-zero-32b",
    },
    # ── R1-Distill family ──────────────────────────────────────
    "r1-distill-qwen-1.5b": {
        "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        "ngpu": 1,
        "format": "r1",
        "model_short": "deepseek-r1-distill-qwen-1.5b",
    },
    "r1-distill-llama-8b": {
        "model_id": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "ngpu": 1,
        "format": "r1",
        "model_short": "deepseek-r1-distill-llama-8b",
    },
    "r1-distill-qwen-14b": {
        "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
        "ngpu": 1,
        "format": "r1",
        "model_short": "deepseek-r1-distill-qwen-14b",
    },
    "r1-distill-qwen-32b": {
        "model_id": "deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
        "ngpu": 2,
        "format": "r1",
        "model_short": "deepseek-r1-distill-qwen-32b",
    },
    # ── QwQ ────────────────────────────────────────────────────
    "qwq-32b": {
        "model_id": "Qwen/QwQ-32B",
        "ngpu": 2,
        "format": "qwq",
        "model_short": "qwq-32b",
    },
}

# Models that need 2 GPUs (tensor parallel)
LARGE_MODELS = {k for k, v in MODEL_CONFIGS.items() if v["ngpu"] == 2}

# Models that fit on 1 GPU
SMALL_MODELS = {k for k, v in MODEL_CONFIGS.items() if v["ngpu"] == 1}
