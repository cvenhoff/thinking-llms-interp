"""FSDP utilities for distributed training of steering vectors."""

import os
import datetime
import functools

import torch
import torch.distributed as dist
from torch.distributed.fsdp import (
    FullyShardedDataParallel as FSDP,
    MixedPrecision,
    ShardingStrategy,
)
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy


# ---------------------------------------------------------------------------
# Distributed helpers
# ---------------------------------------------------------------------------

def init_distributed():
    """Initialize NCCL process group and set local device."""
    if not dist.is_initialized():
        timeout_sec = int(os.environ.get("NCCL_TIMEOUT", 1800))
        dist.init_process_group(
            "nccl",
            timeout=datetime.timedelta(seconds=timeout_sec),
        )
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    return local_rank


def get_rank():
    return dist.get_rank() if dist.is_initialized() else 0


def get_world_size():
    return dist.get_world_size() if dist.is_initialized() else 1


def is_main():
    return get_rank() == 0


# ---------------------------------------------------------------------------
# FSDP model wrapping
# ---------------------------------------------------------------------------

_DECODER_LAYER_MAP = {
    "qwen2": "transformers.models.qwen2.modeling_qwen2.Qwen2DecoderLayer",
    "qwen3": "transformers.models.qwen3.modeling_qwen3.Qwen3DecoderLayer",
    "llama": "transformers.models.llama.modeling_llama.LlamaDecoderLayer",
}


def _resolve_class(dotted: str):
    parts = dotted.rsplit(".", 1)
    mod = __import__(parts[0], fromlist=[parts[1]])
    return getattr(mod, parts[1])


def get_decoder_layer_class(model_config):
    mt = getattr(model_config, "model_type", "").lower()
    for key, cls_path in _DECODER_LAYER_MAP.items():
        if key in mt:
            return _resolve_class(cls_path)
    raise ValueError(
        f"Unsupported model_type={mt} for FSDP. "
        f"Add to _DECODER_LAYER_MAP in fsdp_utils.py")


def wrap_model_fsdp(model, local_rank):
    """Wrap a CausalLM with FSDP (transformer-level auto-wrap)."""
    layer_cls = get_decoder_layer_class(model.config)
    policy = functools.partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={layer_cls},
    )
    mp = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.bfloat16,
    )
    model = FSDP(
        model,
        auto_wrap_policy=policy,
        mixed_precision=mp,
        sharding_strategy=ShardingStrategy.FULL_SHARD,
        device_id=local_rank,
        use_orig_params=True,
    )
    return model


# ---------------------------------------------------------------------------
# Gradient synchronisation for non-FSDP parameters (V, MLP)
# ---------------------------------------------------------------------------

def sync_gradients(*params_or_modules):
    """All-reduce gradients for standalone Parameters / Modules."""
    ws = get_world_size()
    if ws <= 1:
        return
    tensors = []
    for obj in params_or_modules:
        if isinstance(obj, torch.nn.Module):
            for p in obj.parameters():
                if p.grad is not None:
                    tensors.append(p.grad)
        elif isinstance(obj, torch.Tensor) and obj.grad is not None:
            tensors.append(obj.grad)
    for t in tensors:
        dist.all_reduce(t, op=dist.ReduceOp.SUM)
        t.div_(ws)
