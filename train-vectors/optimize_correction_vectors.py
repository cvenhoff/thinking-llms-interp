"""Train steering vectors that directly *correct* the base model's
disagreements with the thinking-model rollout.

Pipeline (per model pair, e.g. ORZ-0.5B -> Qwen2.5-0.5B):

1.  Load annotated thinking-model responses (MMLU-Pro, same data used by
    optimize_steering_vectors.py).
2.  For each response, build the hybrid-eval base prompt
        "Answer the question below. Explain your reasoning step by step.\n\n
         Question:\n{q}\n\nStep by step answer:\n"
    concatenate the thinking rollout, tokenise with the BASE tokenizer.
3.  Run the base model once over the full sequence and identify every
    token position i in [prompt_len-1, L-2] where
       argmax(base_logits[i]) != thinking_ids[i+1]
    i.e. the base model would have mispredicted what the thinking model
    emitted as token i+1.
4.  Look up the SAE category label for the mispredicted token (position
    i+1) using the response's ``annotated_thinking``.  If token i+1 does
    not fall inside a labelled span we drop it.
    The resulting training records are tuples
       (example_idx, token_pos, category, target_token).
5.  For each category c in turn, optimise a single residual-stream vector
    v_c of size (hidden,) by running the base model over each training
    example with a forward (POST) hook that ADDS v_c to the *output* of
    the steering layer only at the disagreement positions labelled c
    (this matches hybrid_eval.py's eval-time steering hook exactly).
    The training objective at each position is the top-K truncated
       KL( thinking_model_distribution || base_model_steered_distribution )
    i.e. a *hard* argmax gate for selecting WHICH positions to steer,
    plus a *soft* KL target that regresses onto the thinking model's
    full next-token distribution at those positions (cached during
    collection as the top-K log-probs + token ids, K=50 covers ~>99% of
    the probability mass for Qwen-style models).
6.  Categories with no training examples are left out (we save nothing;
    hybrid_eval.py already prints a WARNING and treats missing keys as
    "no steering").  This matches the user's requirement:
        "If any vector gets no example we don't steer it, even if during
         hybrid inference the SAE says we should."

No bias vector is trained.  The resulting ``*_idx{N}_linear.pt`` files are
drop-in compatible with hybrid_eval.py's ``--old_vectors_dir`` loader.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import sys
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch import nn
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import dotenv
dotenv.load_dotenv("../.env")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import utils  # noqa: E402
from utils.responses import extract_thinking_process  # noqa: E402


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------
# We data-parallelize over `example_batch_size`-sized minibatches: each rank
# holds a full copy of the (frozen) base model on its own GPU, processes its
# shard of training examples, and we all-reduce V.grad / b.grad after each
# backward pass.  This works because V/b are tiny (a few KB - a few MB) so
# the gradient sync is essentially free, and each per-rank step is much
# faster than the equivalent pipeline-parallel step at the same effective
# batch size (no inter-GPU activation forwarding latency).

def _is_ddp() -> bool:
    return "RANK" in os.environ and "WORLD_SIZE" in os.environ \
        and int(os.environ.get("WORLD_SIZE", "1")) > 1


def _ddp_setup() -> Tuple[int, int, int]:
    """Initialise torch.distributed if launched via torchrun.  Returns
    (rank, world_size, local_rank).  Sets the active CUDA device to
    ``local_rank`` so subsequent ``device_map={"": rank}`` model loads
    place the full base model on this rank's GPU."""
    if not _is_ddp():
        return 0, 1, 0
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ.get("LOCAL_RANK", str(rank)))
    if not dist.is_initialized():
        # Collective timeout: configurable via NCCL_COLLECTIVE_TIMEOUT_SEC,
        # default 86400 s (24 h).  H200 + two 32 B models can hit CUDA-
        # allocator hiccups that stall a rank for hours; we rely on
        # the tqdm wall-clock to detect genuine hangs rather than an
        # auto-kill that loses the whole run.
        import datetime as _dt
        _nccl_timeout = int(os.environ.get("NCCL_COLLECTIVE_TIMEOUT_SEC", "86400"))
        dist.init_process_group(
            backend="nccl",
            timeout=_dt.timedelta(seconds=_nccl_timeout))
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank


def _ddp_cleanup() -> None:
    if dist.is_initialized():
        dist.destroy_process_group()


def _is_rank_zero() -> bool:
    return (not _is_ddp()) or (int(os.environ.get("RANK", "0")) == 0)


def _ddp_print(*args, **kwargs) -> None:
    """Print only from rank 0 (avoid log spam from all ranks)."""
    if _is_rank_zero():
        print(*args, **kwargs)


def _ddp_allreduce_(t: torch.Tensor, op=dist.ReduceOp.SUM) -> torch.Tensor:
    """In-place all-reduce.  No-op when not DDP."""
    if dist.is_initialized():
        dist.all_reduce(t, op=op)
    return t


# ---------------------------------------------------------------------------
# Annotation -> token-range parsing  (mirrors optimize_steering_vectors.py but
# keeps ALL matches, not ``matches[:-1]``, and returns a per-character label
# array so we can map arbitrary token offsets to their category).
# ---------------------------------------------------------------------------

ANNOTATION_PATTERN = re.compile(
    r'\["([\d.]+):(\S+?)"\](.*?)\["end-section"\]', re.DOTALL)


def _aligned_tokenize_pair(
    base_tokenizer, base_prompt: str,
    think_tokenizer, think_prompt: str,
    thinking: str,
):
    """Tokenize ``base_prompt + thinking`` and ``think_prompt + thinking`` once
    on each side WITH offset-mapping, and find a shared character-boundary
    anchor inside the thinking text.

    Why this is needed
    ------------------
    BPE can merge the last prompt character with the first thinking character
    differently between the two sides (e.g. ORZ's think prompt ends with
    ``'>'`` which merges into ``'>To'`` for the full text, while the base
    prompt ends with ``':'`` which stays as a separate token).  A naive
    ``offset = len(tok(think_prompt)) - len(tok(base_prompt))`` is therefore
    off-by-one for many examples and the position-by-position rollout
    target check ``think_ids[i_t+1] == base_ids[i+1]`` fails everywhere.

    Robust alignment
    ----------------
    For each side, take the set of character positions (relative to the
    start of ``thinking``) at which a token *ends* and that lies entirely
    past the prompt.  The smallest character position present in BOTH sets
    is the first place at which the two tokenisations re-synchronise on a
    shared text boundary.  Because we use the same tokenizer family inside
    a pair and the remaining text is identical, both sides produce
    *identical* tokens past that boundary (BPE is deterministic on the
    remaining suffix).

    Returns
    -------
    dict with keys:
      ``b_ids``, ``b_offsets``        -- base tokenisation of full text
      ``t_ids``, ``t_offsets``        -- think tokenisation of full text
      ``b_anchor``, ``t_anchor``      -- first aligned indices (so that
                                         ``b_ids[b_anchor:]`` and
                                         ``t_ids[t_anchor:]`` are identical
                                         token sequences).  -1 if no shared
                                         boundary was found.
      ``anchor_c``                    -- the shared character boundary in
                                         the thinking text, -1 if missing.
    """
    base_full = base_prompt + thinking
    think_full = think_prompt + thinking
    # ``return_offsets_mapping`` requires a fast tokenizer; we don't fall
    # back here because all 9 model pairs we use have fast tokenizers.
    enc_b = base_tokenizer(base_full, return_offsets_mapping=True,
                           truncation=False)
    enc_t = think_tokenizer(think_full, return_offsets_mapping=True,
                            truncation=False)
    bp = len(base_prompt)
    tp = len(think_prompt)
    # ``c`` = end-char-in-thinking; only tokens that lie *entirely* past the
    # prompt boundary participate (a token that straddles the boundary
    # cannot serve as an anchor because part of its content is in the
    # prompt region).
    b_ends: Dict[int, int] = {}
    for i, (s, e) in enumerate(enc_b["offset_mapping"]):
        if s >= bp and e > bp:
            c = e - bp
            b_ends.setdefault(c, i)
    t_ends: Dict[int, int] = {}
    for i, (s, e) in enumerate(enc_t["offset_mapping"]):
        if s >= tp and e > tp:
            c = e - tp
            t_ends.setdefault(c, i)
    common = set(b_ends.keys()) & set(t_ends.keys())
    if not common:
        b_anchor = t_anchor = anchor_c = -1
    else:
        anchor_c = min(common)
        # The anchor token *ends* at this char boundary; the FIRST aligned
        # token (start of the shared suffix) is the next one.
        b_anchor = b_ends[anchor_c] + 1
        t_anchor = t_ends[anchor_c] + 1
    return {
        "b_ids": enc_b["input_ids"],
        "b_offsets": enc_b["offset_mapping"],
        "t_ids": enc_t["input_ids"],
        "t_offsets": enc_t["offset_mapping"],
        "b_anchor": b_anchor,
        "t_anchor": t_anchor,
        "anchor_c": anchor_c,
    }


def _token_category_labels(annotated_thinking: str, full_text: str,
                           tokenizer) -> Dict[int, str]:
    """Return a dict {token_idx_in_full_text: category} covering every
    token that falls inside *some* ``["act:category"]...["end-section"]``
    span in the response.

    ``full_text`` must be ``prompt + thinking_process`` (the exact string
    that will be tokenised and fed to the base model) so that the token
    offsets returned line up with the model's forward pass.
    """
    token_cat: Dict[int, str] = {}
    char_to_token = utils.get_char_to_token_map(full_text, tokenizer)

    for m in ANNOTATION_PATTERN.finditer(annotated_thinking):
        try:
            _ = float(m.group(1).strip())
        except ValueError:
            continue
        label = m.group(2).strip()
        text = m.group(3).strip()
        if not text:
            continue

        # Prefer the original sentence-boundary anchor (keeps labels clean
        # when a sub-sentence appears inside a longer sentence), but fall
        # back to a plain substring match so that the FIRST sentence of
        # the rollout -- which has no prior sentence-ender in `full_text`
        # (it comes right after "Step by step answer:\n" or a chat prompt
        # end marker) -- still gets labelled instead of silently dropped.
        pat = r'(?:[.?!;\n]|\n\n)\s*(' + re.escape(text) + ')'
        mm = re.search(pat, full_text)
        if mm is not None:
            text_pos = mm.start(1)
        else:
            text_pos = full_text.find(text)
        if text_pos < 0:
            continue

        tok_start = char_to_token.get(text_pos)
        tok_end_incl = char_to_token.get(text_pos + len(text) - 1)
        if tok_start is None or tok_end_incl is None:
            continue
        tok_end = tok_end_incl + 1
        if tok_start >= tok_end:
            continue

        for ti in range(tok_start, tok_end):
            # first label wins (shouldn't usually overlap, but be safe)
            token_cat.setdefault(ti, label)
    return token_cat


# ---------------------------------------------------------------------------
# Base-model forward with optional residual-stream injection
# ---------------------------------------------------------------------------

class _InjectHook:
    """Forward (POST) hook that adds `v * mask` to the *output* of the
    specified transformer decoder layer, i.e. the residual STREAM leaving
    that layer and entering the next one.

    This matches hybrid_eval.py's steering hook semantics exactly
    (see hybrid_eval.py `hook(mod, inp, out)` where
    `h[mask, -1:, :] += delta` is applied to the *output* tuple of the
    layer).

    When backprop is enabled, ``v`` must be a leaf tensor with
    ``requires_grad=True``.
    """

    def __init__(self, v: torch.Tensor, pos_mask: torch.Tensor):
        self.v = v            # (hidden,)
        self.mask = pos_mask  # (1, L, 1) or (B, L, 1)

    def __call__(self, _module, _inp, out):
        # A decoder layer returns either a Tensor or a tuple whose first
        # element is the hidden state.  Only mutate the hidden state.
        h = out[0] if isinstance(out, tuple) else out
        shifted = h + self.mask.to(h.device, h.dtype) \
            * self.v.to(h.device, h.dtype).view(1, 1, -1)
        return (shifted,) + out[1:] if isinstance(out, tuple) else shifted


class _InjectMultiHook:
    """Like ``_InjectHook`` but with a PER-POSITION category lookup into a
    matrix of trainable vectors ``V``.  Used for joint training: at every
    disagreement position we add ``V[cat_idx_of(b, p)]`` to the residual
    stream, so a single forward/backward pass trains all category vectors
    simultaneously (gradients for position p only flow into V[cat(p)],
    not other rows).

    If ``b`` is provided (a trainable (hidden,) vector), every disagreement
    position additionally receives ``b`` on top of its category vector,
    enabling joint cats+bias training where the optimiser can route
    category-agnostic corrections into ``b`` and category-specific
    corrections into ``V[k]``.
    """

    def __init__(self, V: torch.Tensor,
                 pos_bids: torch.Tensor, pos_tids: torch.Tensor,
                 pos_cats: torch.Tensor,
                 b: Optional[torch.Tensor] = None,
                 bias_frozen: Optional[torch.Tensor] = None):
        self.V = V                    # (n_cats, hidden), trainable
        self.pos_bids = pos_bids      # (N,) long
        self.pos_tids = pos_tids      # (N,) long
        self.pos_cats = pos_cats      # (N,) long in [0, n_cats)
        self.b = b                    # (hidden,) or None, trainable
        # Frozen-bias term that is added at disagreement positions
        # ALONGSIDE V[cat] (and the trainable b if set).  No grad flows
        # into bias_frozen.  Used when training V on top of a
        # previously-trained global bias: the hook applies
        # ``bias_frozen + V[cat[p]]`` so the optimizer learns the
        # category-specific RESIDUAL on top of the static bias.
        self.bias_frozen = bias_frozen   # (hidden,) detached or None

    def __call__(self, _module, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        # Build an additive update tensor zero everywhere except at
        # disagreement positions, where it equals V[cat] (+ b if set,
        # + bias_frozen if set).  Using a zeros_like(h) + index_copy
        # keeps the op autograd-friendly and avoids any in-place
        # mutation on `h` itself.
        # ----- Multi-GPU safety -----
        # ``device_map="auto"`` may place the steer layer on a different
        # device than the trainable params (V/b) and the precomputed
        # index tensors.  Move everything to ``h``'s device so the
        # in-place index assign and the residual add are well-defined.
        # Cross-device ``.to`` is differentiable, so gradients still
        # flow back into V (and b/bias_frozen if trainable).
        h_dev = h.device
        update = torch.zeros_like(h)
        cats = self.pos_cats.to(h_dev)
        per_pos = self.V.to(h_dev)[cats]                      # (N, hidden) f32
        if self.b is not None:
            per_pos = per_pos + self.b.to(h_dev).unsqueeze(0)  # broadcast bias
        if self.bias_frozen is not None:
            per_pos = per_pos + self.bias_frozen.to(h_dev).unsqueeze(0)
        bids = self.pos_bids.to(h_dev)
        tids = self.pos_tids.to(h_dev)
        update[bids, tids, :] = per_pos.to(h.dtype)
        shifted = h + update
        return (shifted,) + out[1:] if isinstance(out, tuple) else shifted


@contextmanager
def _inject_at_layer(model, layer_idx: int, hook: _InjectHook):
    h = model.model.layers[layer_idx].register_forward_hook(hook)
    try:
        yield
    finally:
        h.remove()


class _BiasOnlyHook:
    """Add a fixed (frozen) bias vector to ALL positions of the residual
    stream at a single layer.  Used during disagreement re-collection to
    simulate inference under a previously-trained global bias steering
    vector: we want to find positions where ``base + bias`` still
    disagrees with the thinking model.
    """

    def __init__(self, bias: torch.Tensor):
        self.bias = bias  # (hidden,) detached, on the model's device

    def __call__(self, _module, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        shifted = h + self.bias.to(h.device, h.dtype).view(1, 1, -1)
        return (shifted,) + out[1:] if isinstance(out, tuple) else shifted


@contextmanager
def _bias_hook_at_layer(model, layer_idx: int, bias: Optional[torch.Tensor]):
    """Apply ``bias`` (frozen, all-positions) at ``layer_idx`` for the
    duration of the context.  No-op if ``bias`` is None."""
    if bias is None:
        yield
        return
    handle = model.model.layers[layer_idx].register_forward_hook(
        _BiasOnlyHook(bias))
    try:
        yield
    finally:
        handle.remove()


def compute_mean_activation_magnitude(
    base_model,
    per_example: List[dict],
    steer_layer: int,
    pad_token_id: int,
    n_examples: int = 16,
    max_positions_per_ex: int = 64,
    seed: int = 0,
) -> float:
    """Mean L2 norm of the residual stream at ``steer_layer``, measured
    across ``n_examples`` examples (skipping pad positions, randomly
    subsampling at most ``max_positions_per_ex`` positions per example).

    Used as the target magnitude for random-direction initialization of
    correction vectors -- starting at the same scale as the activations
    they will be added to gives the optimizer a better-conditioned
    starting point than pure zeros.
    """
    device = next(base_model.parameters()).device
    layer_module = base_model.model.layers[steer_layer]
    captured: List[torch.Tensor] = []

    def hook(_mod, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captured.append(h.detach())

    handle = layer_module.register_forward_hook(hook)
    rng = torch.Generator().manual_seed(int(seed))
    n_norm_sum = 0.0
    n_count = 0
    try:
        with torch.no_grad():
            picks = per_example[:n_examples]
            for ex in picks:
                captured.clear()
                ids = ex["ids"].unsqueeze(0).to(device)
                attn = (ids != pad_token_id).long()
                base_model(input_ids=ids, attention_mask=attn,
                           use_cache=False)
                if not captured:
                    continue
                h = captured[-1].squeeze(0)              # (T, H)
                mask = attn.squeeze(0).bool().cpu()
                h = h[mask.to(h.device)]
                if h.shape[0] > max_positions_per_ex:
                    idx = torch.randperm(
                        h.shape[0], generator=rng)[:max_positions_per_ex]
                    h = h[idx.to(h.device)]
                norms = h.float().norm(dim=-1)
                n_norm_sum += float(norms.sum().item())
                n_count += int(norms.numel())
    finally:
        handle.remove()
    return float(n_norm_sum / max(n_count, 1))


# ---------------------------------------------------------------------------
# Data collection
# ---------------------------------------------------------------------------

def _build_base_prompt(question: str) -> str:
    return f"User: {question}\nAssistant:"


@torch.no_grad()
def collect_disagreements(
    base_model,
    thinking_model,
    base_tokenizer,
    responses: List[dict],
    *,
    max_seq_len: int,
    max_examples: int,
    topk: int,
    thinking_tokenizer=None,
    progress: bool = True,
    collection_mode: str = "disagreement",
    entropy_threshold: float = 1.0,
    frozen_bias: Optional[torch.Tensor] = None,
    frozen_bias_layer: Optional[int] = None,
    sae_classifier=None,
    sae_classify_layer: Optional[int] = None,
    sae_n_clusters: Optional[int] = None,
) -> Tuple[List[dict], Dict[str, List[Tuple[int, int, torch.Tensor, torch.Tensor]]]]:
    """Return ``(per_example, per_category)``.

    Semantics (must match ``hybrid_eval.py`` exactly at steering time):
      - Base model sees ``_build_base_prompt(question) + rollout`` tokenised
        with the base tokenizer.  Steering at eval time is applied at the
        last position of this base-prompt sequence as it grows, so base
        positions are the ones we train for.
      - Thinking model, for gating & top-K targets, sees the *same rollout
        text* but prepended with the thinking model's own chat template
        (``apply_chat_template([{role:user, content:question}], add_generation_prompt=True)``),
        tokenised with the thinking tokenizer.  This matches
        ``hybrid_eval.py``'s thinking rollout context.

    Position selection (``collection_mode``):
      - ``'disagreement'`` (default, legacy): hard-argmax disagreement
        between base prediction and the actual rollout token at base
        position ``i``.
      - ``'entropy'``: include every position where the THINKING model's
        next-token entropy at the corresponding thinking position is
        ``>= entropy_threshold`` (in nats), regardless of whether the
        base model's top-1 agrees with the rollout token.  Motivation:
        when the thinking model is uncertain, its top-K distribution
        carries useful soft-target signal even if base/thinking happen
        to agree on top-1.

    The training target at every retained position is the thinking
    model's top-K next-token distribution at the *corresponding*
    thinking position, found by offsetting through the prompt-length
    difference and verifying the rollout tokens match 1:1.

    per_example[i] = {"ids": LongTensor(L,), "prompt_len": int}
    per_category[cat] = list of
        (example_idx, token_pos_base, topk_logprobs (K,), topk_ids (K,))
    """
    per_example: List[dict] = []
    per_category: Dict[str, List[Tuple[int, int, torch.Tensor, torch.Tensor]]] \
        = defaultdict(list)

    think_tokenizer = thinking_tokenizer if thinking_tokenizer is not None \
        else base_tokenizer

    if collection_mode not in ("disagreement", "entropy", "union"):
        raise ValueError(
            f"collection_mode must be 'disagreement', 'entropy' or "
            f"'union', got {collection_mode!r}")

    base_device = next(base_model.parameters()).device
    think_device = next(thinking_model.parameters()).device

    # If an SAE classifier is provided, register a forward hook on the
    # thinking model at ``sae_classify_layer`` so we can classify EACH
    # disagreement position with the SAE on its OWN per-token activation
    # -- exactly as ``hybrid_eval.py`` does at generation time.  This
    # bypasses the (sentence-level) ``annotated_thinking`` labels for
    # category assignment, eliminating the train/eval semantic mismatch.
    _sae_state: Dict[str, Optional[torch.Tensor]] = {"acts": None}
    _sae_hook_handle = None
    if sae_classifier is not None:
        if sae_classify_layer is None:
            raise ValueError("sae_classify_layer must be set when "
                             "sae_classifier is provided.")
        try:
            _sae_target = thinking_model.model.layers[sae_classify_layer]
        except Exception:
            _sae_target = thinking_model.module.model.layers[sae_classify_layer]

        def _sae_capture_hook(_mod, _inp, out):
            h = out[0] if isinstance(out, tuple) else out
            _sae_state["acts"] = h.detach()
        _sae_hook_handle = _sae_target.register_forward_hook(_sae_capture_hook)
        if _is_rank_zero():
            print(f"  [sae-cat] classifying each disagreement position with "
                  f"SAE on thinking-model activation @ layer "
                  f"{sae_classify_layer}; bypassing sentence annotations.",
                  flush=True)

    it = responses[:max_examples] if max_examples else responses
    if collection_mode == "disagreement":
        desc = "Collecting disagreements"
    elif collection_mode == "entropy":
        desc = f"Collecting hi-entropy positions (>= {entropy_threshold})"
    else:
        desc = (f"Collecting union(disagree, entropy >= "
                f"{entropy_threshold}) positions")
    it_iter = tqdm(it, desc=desc, disable=not progress)

    n_too_long = n_no_labels = n_no_disagree = n_align_skip = 0
    n_align_partial = 0
    n_no_anchor = 0
    n_pos_considered = 0
    n_pos_dropped_low_entropy = 0
    entropy_sum = 0.0
    entropy_n = 0

    for resp in it_iter:
        ann = resp.get("annotated_thinking")
        if not ann:
            continue
        question = resp["original_message"]["content"]
        thinking = extract_thinking_process(resp["full_response"])
        if not thinking or not thinking.strip():
            continue

        # --- Build the two prompt strings.
        base_prompt = _build_base_prompt(question)
        try:
            think_prompt_text = think_tokenizer.apply_chat_template(
                [{"role": "user", "content": question}],
                tokenize=False, add_generation_prompt=True)
        except Exception:
            think_prompt_text = base_prompt  # graceful fallback

        # --- Tokenize ONCE with offset-mapping on each side and find the
        # first shared char-boundary anchor inside the thinking text.  This
        # is the only correct way to align two tokenisations that may merge
        # the prompt/rollout boundary differently (e.g. ORZ's ``>`` merges
        # into ``>To`` while the base ``:`` stays as a separate token).
        base_full_text = base_prompt + thinking
        align = _aligned_tokenize_pair(
            base_tokenizer, base_prompt,
            think_tokenizer, think_prompt_text,
            thinking,
        )
        base_ids = torch.tensor(align["b_ids"], dtype=torch.long)
        think_ids = torch.tensor(align["t_ids"], dtype=torch.long)
        Lb = base_ids.shape[0]
        Lt = think_ids.shape[0]
        if Lb < 8:
            continue
        if Lb > max_seq_len or Lt > max_seq_len:
            n_too_long += 1
            continue
        b_anchor = align["b_anchor"]
        t_anchor = align["t_anchor"]
        if b_anchor < 0:
            n_no_anchor += 1
            continue

        if sae_classifier is not None:
            # Cats come from per-token SAE classification of thinking
            # activations (computed below); skip the (sentence-level)
            # annotation parser so we don't gate on it.
            tok_cat: Dict[int, str] = {}
        else:
            tok_cat = _token_category_labels(ann, base_full_text, base_tokenizer)
            if not tok_cat:
                n_no_labels += 1
                continue

        # --- Base logits (for disagreement detection).  Only needed when
        # the disagreement gate is part of the collection rule.
        # When ``frozen_bias`` is provided, the base forward is hooked to
        # add ``bias`` to every residual-stream position at
        # ``frozen_bias_layer``, so we collect disagreements UNDER a
        # bias-steered base model -- positions where bias alone is not
        # enough.
        if collection_mode in ("disagreement", "union"):
            ids_gpu_b = base_ids.unsqueeze(0).to(base_device)
            with _bias_hook_at_layer(base_model,
                                     frozen_bias_layer
                                     if frozen_bias is not None else 0,
                                     frozen_bias):
                out_b = base_model(ids_gpu_b, use_cache=False)
            pred_b = out_b.logits[0].argmax(dim=-1).cpu()  # (Lb,)
            del out_b
        else:
            pred_b = None

        # --- Thinking logits on ITS OWN templated sequence (top-K targets).
        ids_gpu_t = think_ids.unsqueeze(0).to(think_device)
        out_t = thinking_model(ids_gpu_t, use_cache=False)
        t_logits = out_t.logits[0]  # (Lt, vocab), bf16
        del out_t

        # If SAE-based per-position classification is enabled, classify
        # every thinking position once (vectorised) using the activations
        # captured via the forward hook above.  This mirrors hybrid_eval's
        # last-token classification on a per-token basis -- exactly the
        # signal the steering hook will see at generation time.
        sae_cat_per_pos: Optional[List[str]] = None
        if sae_classifier is not None:
            acts_t = _sae_state.get("acts")
            if acts_t is None:
                raise RuntimeError(
                    "SAE classify hook did not fire on the thinking model.")
            # acts_t: (1, Lt, hidden)
            cat_ids = sae_classifier(acts_t[0])  # (Lt,) int
            sae_cat_per_pos = [f"idx{int(c)}" for c in cat_ids.tolist()]
            _sae_state["acts"] = None
        t_logprobs = torch.log_softmax(t_logits.float(), dim=-1)
        del t_logits
        # Compute next-token entropy at every thinking position from the
        # FULL distribution.  -sum_i p_i * log p_i, in nats.
        if collection_mode in ("entropy", "union"):
            t_entropy = -(t_logprobs.exp() * t_logprobs).sum(dim=-1).cpu()
        else:
            t_entropy = None
        topk_lp, topk_idx = t_logprobs.topk(k=topk, dim=-1)  # (Lt, K)
        del t_logprobs
        topk_lp = topk_lp.cpu()
        topk_idx = topk_idx.cpu()

        # ``offset`` is exact past the anchor: every base position
        # ``i >= b_anchor - 1`` predicts a token in the aligned region,
        # and the corresponding think position is ``i_t = i + offset``.
        offset = t_anchor - b_anchor
        ex_idx = len(per_example)
        found_any = False
        n_ex_miss = 0
        for i in range(max(b_anchor - 1, 0), Lb - 1):
            target = int(base_ids[i + 1].item())
            if sae_cat_per_pos is not None:
                # Classify the THINKING activation at the position that
                # produces this base position's prediction (i.e. the
                # thinking position aligned with base position i).  We do
                # the alignment check below; defer the cat lookup until
                # after we know i_t.
                cat = None
            else:
                cat = tok_cat.get(i + 1)
                if cat is None:
                    continue
            n_pos_considered += 1
            if collection_mode == "disagreement":
                if int(pred_b[i].item()) == target:
                    continue
            i_t = i + offset
            if i_t < 0 or i_t + 1 >= Lt:
                n_ex_miss += 1
                continue
            # Defensive check: the anchor guarantees identical tokens past
            # the anchor index for both sides (same tokenizer, identical
            # remaining text, deterministic BPE).  Keep the guard so a
            # tokenizer surprise degrades gracefully instead of producing
            # off-by-one training targets.
            if int(think_ids[i_t + 1].item()) != target:
                n_ex_miss += 1
                continue
            # Now that i_t is validated, look up the SAE-classified cat
            # at the THINKING position whose activation drives this base
            # position's prediction.  This matches eval semantics: at
            # generation step t the SAE sees the thinking-model's
            # activation at position t-1.
            if sae_cat_per_pos is not None:
                if i_t < 0 or i_t >= len(sae_cat_per_pos):
                    continue
                cat = sae_cat_per_pos[i_t]
            if collection_mode == "entropy":
                ent = float(t_entropy[i_t].item())
                entropy_sum += ent
                entropy_n += 1
                if ent < entropy_threshold:
                    n_pos_dropped_low_entropy += 1
                    continue
            elif collection_mode == "union":
                # Keep position iff base disagrees with target OR
                # thinking-model entropy is above threshold.
                disag = int(pred_b[i].item()) != target
                ent = float(t_entropy[i_t].item())
                entropy_sum += ent
                entropy_n += 1
                if not disag and ent < entropy_threshold:
                    n_pos_dropped_low_entropy += 1
                    continue
            per_category[cat].append(
                (ex_idx, i, topk_lp[i_t].clone(), topk_idx[i_t].clone()))
            found_any = True

        if n_ex_miss > 0:
            n_align_partial += 1
        if found_any:
            per_example.append({"ids": base_ids.cpu(),
                                "prompt_len": int(b_anchor)})
        else:
            if n_ex_miss > 0:
                n_align_skip += 1
            else:
                n_no_disagree += 1

    it_iter.close() if hasattr(it_iter, "close") else None
    print(f"  collection_mode={collection_mode}"
          + (f" entropy_threshold={entropy_threshold}"
             if collection_mode == "entropy" else ""),
          flush=True)
    print(f"  responses processed: {len(per_example)} retained, "
          f"{n_too_long} too long, {n_no_labels} without labels, "
          f"{n_no_disagree} without retained positions, "
          f"{n_no_anchor} without shared char anchor, "
          f"{n_align_skip} fully mis-aligned, "
          f"{n_align_partial} with some skipped positions",
          flush=True)
    if collection_mode == "entropy" and entropy_n > 0:
        avg_ent = entropy_sum / entropy_n
        keep_frac = (entropy_n - n_pos_dropped_low_entropy) \
            / max(entropy_n, 1)
        print(f"  entropy stats over {entropy_n} candidate positions: "
              f"mean={avg_ent:.3f} nats, "
              f"kept={entropy_n - n_pos_dropped_low_entropy} "
              f"({100*keep_frac:.1f}%) at thr={entropy_threshold}",
              flush=True)
    if collection_mode == "disagreement":
        label = "disagreement positions"
    elif collection_mode == "entropy":
        label = "high-entropy positions"
    else:
        label = "union(disagree|hi-entropy) positions"
    for cat in sorted(per_category.keys(),
                      key=lambda k: int(k[3:]) if k.startswith("idx")
                      and k[3:].isdigit() else 0):
        print(f"  {cat}: {len(per_category[cat])} {label}",
              flush=True)
    if _sae_hook_handle is not None:
        try:
            _sae_hook_handle.remove()
        except Exception:
            pass
    return per_example, per_category


# ---------------------------------------------------------------------------
# Per-category optimisation + holdout evaluation
# ---------------------------------------------------------------------------

def _group_positions_by_example(
    positions: List[Tuple[int, int, torch.Tensor, torch.Tensor]],
) -> Dict[int, List[Tuple[int, torch.Tensor, torch.Tensor]]]:
    by_ex: Dict[int, List[Tuple[int, torch.Tensor, torch.Tensor]]] = \
        defaultdict(list)
    for ex_idx, pos, topk_lp, topk_idx in positions:
        by_ex[ex_idx].append((pos, topk_lp, topk_idx))
    return by_ex


def _compute_batch_kl_loss(
    base_model,
    mb: List[int],
    per_example: List[dict],
    by_example: Dict[int, List[Tuple[int, torch.Tensor, torch.Tensor]]],
    v: torch.Tensor,
    steer_layer: int,
    pad_token_id: int,
) -> Tuple[torch.Tensor, int]:
    """Forward a padded minibatch through base_model with v injected at
    the disagreement positions of every example, and return
    (summed_top_K_KL_over_positions, n_positions).

    Works for both training (v.requires_grad=True) and holdout eval
    (wrap the caller in torch.no_grad()).
    """
    device = next(base_model.parameters()).device
    B = len(mb)
    Lmax = max(per_example[e]["ids"].shape[0] for e in mb)
    ids_batch = torch.full((B, Lmax), pad_token_id,
                           device=device, dtype=torch.long)
    attn = torch.zeros((B, Lmax), device=device, dtype=torch.long)
    pos_mask = torch.zeros((B, Lmax, 1), device=device, dtype=torch.float32)
    pos_bids: List[int] = []
    pos_tids: List[int] = []
    topk_lps: List[torch.Tensor] = []
    topk_ixs: List[torch.Tensor] = []
    for bi, ex_idx in enumerate(mb):
        ex_ids = per_example[ex_idx]["ids"]
        L = ex_ids.shape[0]
        ids_batch[bi, :L] = ex_ids.to(device, non_blocking=True)
        attn[bi, :L] = 1
        for p, tlp, tix in by_example[ex_idx]:
            pos_mask[bi, p, 0] = 1.0
            pos_bids.append(bi)
            pos_tids.append(p)
            topk_lps.append(tlp)
            topk_ixs.append(tix)
    if not pos_bids:
        return torch.zeros((), device=device), 0

    pos_bids_t = torch.tensor(pos_bids, device=device, dtype=torch.long)
    pos_tids_t = torch.tensor(pos_tids, device=device, dtype=torch.long)
    topk_lp_t = torch.stack(topk_lps).to(device, torch.float32)
    topk_idx_t = torch.stack(topk_ixs).to(device, torch.long)
    topk_probs = topk_lp_t.exp()

    hook = _InjectHook(v, pos_mask)
    with _inject_at_layer(base_model, steer_layer, hook):
        body_out = base_model.model(
            input_ids=ids_batch, attention_mask=attn, use_cache=False)
    hidden = body_out.last_hidden_state
    selected = hidden[pos_bids_t, pos_tids_t, :]
    logits = base_model.lm_head(selected).float()
    base_lp = torch.log_softmax(logits, dim=-1)
    base_lp_at_topk = base_lp.gather(-1, topk_idx_t)
    per_pos = -(topk_probs * base_lp_at_topk).sum(dim=-1)
    return per_pos.sum(), per_pos.shape[0]


@torch.no_grad()
def evaluate_kl(
    base_model,
    per_example: List[dict],
    positions: List[Tuple[int, int, torch.Tensor, torch.Tensor]],
    v: torch.Tensor,
    *,
    steer_layer: int,
    example_batch_size: int,
    pad_token_id: int,
) -> Tuple[float, int]:
    """Return (mean_top_K_KL_loss, n_positions) on the given positions
    using the provided vector v (no gradient).  Length-bucketed.
    """
    by_example = _group_positions_by_example(positions)
    if not by_example:
        return float("nan"), 0
    ex_ids = sorted(by_example.keys(),
                    key=lambda e: per_example[e]["ids"].shape[0])
    loss_sum = 0.0
    n_total = 0
    v_eval = v.to(next(base_model.parameters()).device).detach()
    for i in range(0, len(ex_ids), example_batch_size):
        mb = ex_ids[i:i + example_batch_size]
        s, n = _compute_batch_kl_loss(
            base_model, mb, per_example, by_example,
            v_eval, steer_layer, pad_token_id)
        loss_sum += float(s.item())
        n_total += n
    return (loss_sum / max(n_total, 1)), n_total


def train_category_vector(
    base_model,
    per_example: List[dict],
    positions: List[Tuple[int, int, torch.Tensor, torch.Tensor]],
    *,
    steer_layer: int,
    hidden_size: int,
    lr: float,
    n_epochs: int,
    example_batch_size: int,
    max_positions_per_example: int,
    seed: int,
    weight_decay: float,
    pad_token_id: int,
) -> Tuple[torch.Tensor, float]:
    """Optimise a single category's steering vector so that the base
    model's next-token distribution at every disagreement position in
    this category MOVES TOWARDS the thinking-model distribution cached
    during collection.

    Objective at each position i (top-K truncated KL, equivalent to
    cross-entropy of steered-base against the truncated thinking dist):
        loss_i = - sum_k  p_think[i,k] * log p_base_steered[i, idx_k]
    where (p_think[i], idx[i]) is the cached top-K thinking
    distribution.  Minimising this is equivalent to minimising
    KL(thinking_topk || base_steered) up to a v-independent constant.

    Returns (v, final_per_position_loss).
    """
    device = next(base_model.parameters()).device

    by_example = _group_positions_by_example(positions)
    example_ids = sorted(by_example.keys())

    rng = random.Random(seed)
    if max_positions_per_example > 0:
        capped: Dict[int, List[Tuple[int, torch.Tensor, torch.Tensor]]] = {}
        for k, pts in by_example.items():
            if len(pts) <= max_positions_per_example:
                capped[k] = pts
            else:
                capped[k] = rng.sample(pts, max_positions_per_example)
        by_example = capped

    v = torch.zeros(hidden_size, device=device, dtype=torch.float32,
                    requires_grad=True)
    opt = torch.optim.Adam([v], lr=lr, weight_decay=weight_decay)

    total_positions = sum(len(v_) for v_ in by_example.values())
    print(f"    Optimising over {total_positions} positions in "
          f"{len(example_ids)} examples for {n_epochs} epochs "
          f"(batch={example_batch_size}, lr={lr}, layer={steer_layer})",
          flush=True)

    example_ids_sorted = sorted(
        example_ids, key=lambda e: per_example[e]["ids"].shape[0])

    def build_batches(epoch_rng: random.Random) -> List[List[int]]:
        batches = [example_ids_sorted[i:i + example_batch_size]
                   for i in range(0, len(example_ids_sorted),
                                  example_batch_size)]
        epoch_rng.shuffle(batches)
        return batches

    last_loss = float("nan")
    n_batches = (len(example_ids) + example_batch_size - 1) // example_batch_size
    pbar = tqdm(total=n_epochs * n_batches, desc="    training", leave=False)
    for epoch in range(n_epochs):
        batches = build_batches(rng)
        ep_loss_sum = 0.0
        ep_positions = 0
        for mb in batches:
            opt.zero_grad()
            loss_sum, n_pts = _compute_batch_kl_loss(
                base_model, mb, per_example, by_example,
                v, steer_layer, pad_token_id)
            if n_pts == 0:
                pbar.update(1)
                continue
            (loss_sum / n_pts).backward()
            opt.step()

            ep_loss_sum += float(loss_sum.detach().item())
            ep_positions += n_pts
            running = ep_loss_sum / max(ep_positions, 1)
            pbar.set_postfix(ep=f"{epoch+1}/{n_epochs}",
                             kl=f"{running:.3f}",
                             norm=f"{v.detach().norm().item():.2f}")
            pbar.update(1)

        last_loss = ep_loss_sum / max(ep_positions, 1)
        print(f"    epoch {epoch + 1}/{n_epochs}  "
              f"avg topK-CE(think‖base) = {last_loss:.4f}  "
              f"positions = {ep_positions}  "
              f"norm = {v.detach().norm().item():.3f}", flush=True)
    pbar.close()

    return v.detach().float().cpu(), last_loss


# ---------------------------------------------------------------------------
# Joint multi-vector training
# ---------------------------------------------------------------------------

def _group_positions_by_example_with_cat(
    positions: List[Tuple[int, int, int, torch.Tensor, torch.Tensor]],
) -> Dict[int, List[Tuple[int, int, torch.Tensor, torch.Tensor]]]:
    """Positions carry a cat_idx.  Group by example_idx."""
    by_ex: Dict[int, List[Tuple[int, int, torch.Tensor, torch.Tensor]]] \
        = defaultdict(list)
    for ex_idx, pos, cat_idx, topk_lp, topk_idx in positions:
        by_ex[ex_idx].append((pos, cat_idx, topk_lp, topk_idx))
    return by_ex


def train_vectors_joint(
    base_model,
    per_example: List[dict],
    positions_with_cat: List[
        Tuple[int, int, int, torch.Tensor, torch.Tensor]],
    n_cats: int,
    *,
    steer_layer: int,
    hidden_size: int,
    lr: float,
    n_epochs: int,
    example_batch_size: int,
    max_positions_per_example: int,
    seed: int,
    weight_decay: float,
    pad_token_id: int,
    desc: str = "joint",
    metrics_path: Optional[str] = None,
    cat_key_lookup: Optional[List[str]] = None,
    kl_mode: str = "topk",
    train_topk: int = 3,
    holdout_positions_with_cat: Optional[List[
        Tuple[int, int, int, torch.Tensor, torch.Tensor]]] = None,
    train_bias: bool = False,
    max_norm: float = 0.0,
    select_best_holdout: bool = True,
    per_example_loss: bool = False,
    cap_resample_each_epoch: bool = False,
    init_v_norm: float = 0.0,
    bias_frozen: Optional[torch.Tensor] = None,
    # Crash-resilient checkpointing of the best-holdout snapshot to disk.
    # When provided (and we're on rank 0), every time we improve the
    # holdout KL we additionally write the snapshot to
    # ``{checkpoint_dir}/{checkpoint_prefix}_best.pt`` so that an NCCL
    # timeout near the end of training doesn't lose the entire run.
    checkpoint_dir: Optional[str] = None,
    checkpoint_prefix: Optional[str] = None,
    # How many times PER EPOCH to run holdout + train-mean eval.  4 means
    # at 25/50/75/100 % of each epoch; 1 keeps the historical (epoch-end
    # only) cadence.  Mid-epoch chunk evals feed the best-holdout
    # snapshot selection too.
    eval_chunks_per_epoch: int = 4,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], List[dict]]:
    """Train ``n_cats`` category vectors simultaneously in one sweep.

    positions_with_cat[k] = (ex_idx, token_pos, cat_idx, topk_lp, topk_idx)

    Per-position gradients for the KL loss only flow into V[cat_idx]
    (other rows receive zero gradient at that position), so jointly
    optimising is mathematically equivalent to running n_cats independent
    optimisations at the same (steer_layer, data split), just ~n_cats times
    faster because we do ONE forward/backward per batch instead of n_cats.

    If ``train_bias`` is True, a single shared bias vector ``b`` is
    co-trained alongside V: at every disagreement position we apply
    ``V[cat[p]] + b`` and update both via the same Adam step.  The
    optimiser is free to route category-agnostic corrections into ``b``
    and category-specific corrections into ``V[k]``, which is the
    parametrisation the inference-time ``cat + bias`` composition
    expects (so there is no double-counting of the shared correction).

    Returns ``(V, b_or_none, metrics)`` where V is (n_cats, hidden) on
    CPU/float32 and b is (hidden,) on CPU/float32 (or None if
    ``train_bias=False``).
    """
    device = next(base_model.parameters()).device
    rng = random.Random(seed)

    # FULL by-example map (uncapped). The position cap is applied
    # per-EPOCH below with fresh randomness so that, across epochs,
    # every disagreement position has a chance to contribute even if
    # individual examples are too long for one epoch's cap.
    by_example_full: Dict[
        int, List[Tuple[int, int, torch.Tensor, torch.Tensor]]] \
        = _group_positions_by_example_with_cat(positions_with_cat)

    def _cap_per_epoch(
        rng_epoch: random.Random,
    ) -> Dict[int, List[Tuple[int, int, torch.Tensor, torch.Tensor]]]:
        """Subsample (per epoch) to at most max_positions_per_example
        positions per (example, category). With cap=0 returns the full
        map unchanged."""
        if not max_positions_per_example or max_positions_per_example <= 0:
            return by_example_full
        capped: Dict[
            int, List[Tuple[int, int, torch.Tensor, torch.Tensor]]] = {}
        for ex_idx, recs in by_example_full.items():
            by_cat: Dict[
                int, List[Tuple[int, int, torch.Tensor, torch.Tensor]]] \
                = defaultdict(list)
            for p, c, lp, ix in recs:
                by_cat[c].append((p, c, lp, ix))
            kept: List[Tuple[int, int, torch.Tensor, torch.Tensor]] = []
            for c, recs_c in by_cat.items():
                if len(recs_c) > max_positions_per_example:
                    recs_c = rng_epoch.sample(
                        recs_c, max_positions_per_example)
                kept.extend(recs_c)
            capped[ex_idx] = kept
        return capped

    # When NOT resampling per epoch, build the cap once with deterministic
    # per-(ex,cat) seeds (matches the historical pipeline behavior).
    if not cap_resample_each_epoch:
        if max_positions_per_example and max_positions_per_example > 0:
            fixed_capped: Dict[
                int, List[Tuple[int, int, torch.Tensor, torch.Tensor]]] = {}
            for ex_idx, recs in by_example_full.items():
                by_cat: Dict[
                    int, List[Tuple[int, int, torch.Tensor, torch.Tensor]]] \
                    = defaultdict(list)
                for p, c, lp, ix in recs:
                    by_cat[c].append((p, c, lp, ix))
                kept: List[Tuple[int, int, torch.Tensor, torch.Tensor]] = []
                for c, recs_c in by_cat.items():
                    if len(recs_c) > max_positions_per_example:
                        rng_local = random.Random(f"{seed}-{ex_idx}-{c}")
                        recs_c = rng_local.sample(
                            recs_c, max_positions_per_example)
                    kept.extend(recs_c)
                fixed_capped[ex_idx] = kept
            by_example_fixed = fixed_capped
        else:
            by_example_fixed = by_example_full
        by_example = by_example_fixed
    else:
        by_example = by_example_full

    ex_ids_with_data = sorted(by_example.keys())
    # Length-bucket so padded minibatches waste as little compute as
    # possible.  Same strategy as train_category_vector.
    ex_ids_sorted = sorted(ex_ids_with_data,
                           key=lambda e: per_example[e]["ids"].shape[0])

    if init_v_norm and init_v_norm > 0:
        # Random unit-norm direction per category, scaled to ``init_v_norm``
        # so the starting magnitude matches the residual-stream activations.
        # Seeded by the seed argument so different seeds get genuinely
        # different starting directions (not just different shuffle orders).
        gen = torch.Generator().manual_seed(int(seed))
        V_init = torch.randn(n_cats, hidden_size, generator=gen,
                             dtype=torch.float32)
        V_init = (V_init
                  / V_init.norm(dim=-1, keepdim=True).clamp_min(1e-8)
                  * float(init_v_norm))
        V = V_init.to(device).detach().clone().requires_grad_(True)
        print(f"    [joint] init V from random direction at "
              f"norm={float(init_v_norm):.3f} (seed={seed})", flush=True)
    else:
        V = torch.zeros((n_cats, hidden_size), device=device,
                        dtype=torch.float32, requires_grad=True)
    b = (torch.zeros(hidden_size, device=device, dtype=torch.float32,
                     requires_grad=True)
         if train_bias else None)
    opt_params = [V] + ([b] if b is not None else [])
    opt = torch.optim.Adam(opt_params, lr=lr, weight_decay=weight_decay)

    n_pos_total = sum(len(v) for v in by_example.values())
    print(f"    [joint] {n_cats} vectors"
          f"{' + bias' if train_bias else ''}, "
          f"{len(ex_ids_sorted)} examples, "
          f"{n_pos_total} positions total, layer={steer_layer}, "
          f"bs={example_batch_size}, lr={lr}, epochs={n_epochs}  "
          f"(kl_mode={kl_mode}, train_topk={train_topk})",
          flush=True)

    # Holdout split: precompute length-sorted indices + by_example for
    # per-epoch no-grad eval with current V.
    holdout_by_example: Optional[Dict[
        int, List[Tuple[int, int, torch.Tensor, torch.Tensor]]]] = None
    holdout_ex_ids_sorted: Optional[List[int]] = None
    holdout_n_pos_total = 0
    if holdout_positions_with_cat:
        holdout_by_example = _group_positions_by_example_with_cat(
            holdout_positions_with_cat)
        holdout_ex_ids_sorted = sorted(
            holdout_by_example.keys(),
            key=lambda e: per_example[e]["ids"].shape[0])
        holdout_n_pos_total = sum(len(v) for v in holdout_by_example.values())
        print(f"    [joint] holdout: {len(holdout_ex_ids_sorted)} examples, "
              f"{holdout_n_pos_total} positions", flush=True)

    steps_per_epoch = math.ceil(len(ex_ids_sorted) / example_batch_size)
    total_steps = n_epochs * steps_per_epoch
    # Rank-zero tqdm; other ranks get a no-op pbar (still iterates fine).
    pbar = tqdm(total=total_steps, desc=desc, mininterval=1.0,
                disable=(not _is_rank_zero()))
    last_loss = float("nan")

    # Structured-metrics log: one JSONL record per training step + one per
    # epoch marker.  Makes downstream plotting and regressions trivial
    # (no tqdm-line regex needed).
    metrics_records: List[dict] = []
    # Only rank 0 writes the metrics file (avoid racing/clobbering).
    metrics_fh = (open(metrics_path, "w")
                  if metrics_path and _is_rank_zero() else None)
    def _emit(rec: dict) -> None:
        metrics_records.append(rec)
        if metrics_fh is not None:
            metrics_fh.write(json.dumps(rec) + "\n")
            metrics_fh.flush()

    _emit({
        "phase": desc, "event": "start",
        "n_cats": int(n_cats),
        "n_train_examples": int(len(ex_ids_sorted)),
        "n_positions_total": int(n_pos_total),
        "steer_layer": int(steer_layer),
        "example_batch_size": int(example_batch_size),
        "lr": float(lr),
        "n_epochs": int(n_epochs),
        "steps_per_epoch": int(steps_per_epoch),
        "total_steps": int(total_steps),
        "cat_key_lookup": list(cat_key_lookup) if cat_key_lookup else None,
    })

    global_step = 0
    best_holdout_kl = float("inf")
    best_V_cpu: Optional[torch.Tensor] = None
    best_b_cpu: Optional[torch.Tensor] = None
    best_epoch: int = 0
    # Mid-epoch eval cadence: we run a quick holdout eval ``_ec`` times
    # per epoch so the train/holdout curves are denser than the
    # historical 1/epoch.  The exact step cadence is computed per epoch
    # because the rank-local batch count can drift slightly when the
    # DDP trimming kicks in.
    _ec = max(1, int(eval_chunks_per_epoch))
    if _is_rank_zero():
        print(f"    [joint] eval cadence: {_ec} chunks/epoch "
              f"(~{max(1, math.ceil(steps_per_epoch / _ec))} train "
              "steps per chunk)", flush=True)

    def _run_holdout_eval(epoch_idx: int, chunk_idx: int, total_chunks: int,
                          train_kl_local: float, train_pos_local: int) -> None:
        """Run a no-grad holdout eval with current V/b and emit metrics.

        ``train_kl_local`` and ``train_pos_local`` are this rank's running
        epoch loss accumulators; we all-reduce them inside so the
        printed/recorded train_kl reflects the full DDP shard.  Updates
        ``best_holdout_kl`` and the on-disk best checkpoint.
        """
        nonlocal best_holdout_kl, best_V_cpu, best_b_cpu, best_epoch
        if holdout_by_example is None or not holdout_ex_ids_sorted:
            return
        # All-reduce the running train numbers for the print line.
        if dist.is_initialized():
            _t = torch.tensor(
                [float(train_kl_local), float(train_pos_local)],
                device=device, dtype=torch.float64)
            _ddp_allreduce_(_t)
            tr_sum = float(_t[0].item())
            tr_cnt = int(_t[1].item())
        else:
            tr_sum = float(train_kl_local)
            tr_cnt = int(train_pos_local)
        train_kl = tr_sum / max(tr_cnt, 1)
        with torch.no_grad():
            ec_sum = torch.zeros(n_cats, device=device, dtype=torch.float64)
            ec_cnt = torch.zeros(n_cats, device=device, dtype=torch.long)
            _e_starts = list(range(0, len(holdout_ex_ids_sorted),
                                   example_batch_size))
            _ddp_w = (int(os.environ["WORLD_SIZE"])
                      if _is_ddp() else 1)
            _ddp_r = (int(os.environ["RANK"])
                      if _is_ddp() else 0)
            if _ddp_w > 1:
                _trim = len(_e_starts) - (len(_e_starts) % _ddp_w)
                _e_starts = _e_starts[:_trim]
            _e_starts = _e_starts[_ddp_r::_ddp_w]
            for _i in _e_starts:
                mb_eval = holdout_ex_ids_sorted[_i:_i + example_batch_size]
                per_pos_e, pos_cats_e = _compute_batch_kl_loss_joint(
                    base_model, mb_eval, per_example, holdout_by_example,
                    V.detach(), steer_layer, pad_token_id,
                    kl_mode=kl_mode, train_topk=train_topk,
                    b=(b.detach() if b is not None else None),
                    bias_frozen=bias_frozen)
                if per_pos_e.numel() == 0:
                    continue
                pp_e = per_pos_e.detach().double()
                ec_sum.scatter_add_(0, pos_cats_e, pp_e)
                ec_cnt.scatter_add_(0, pos_cats_e,
                                    torch.ones_like(pos_cats_e,
                                                    dtype=torch.long))
            if dist.is_initialized():
                _ddp_allreduce_(ec_sum)
                _ddp_allreduce_(ec_cnt)
            total_cnt = int(ec_cnt.sum().item())
            total_kl = float(ec_sum.sum().item()) / max(total_cnt, 1)
            per_cat_kl_eval = [
                (float((ec_sum[i] / ec_cnt[i]).item())
                 if int(ec_cnt[i].item()) > 0 else None)
                for i in range(n_cats)]
        print(f"    [joint] ep{epoch_idx+1}/{n_epochs} "
              f"chunk {chunk_idx}/{total_chunks}: "
              f"train_kl={train_kl:.4f} (n={tr_cnt})  "
              f"holdout_kl={total_kl:.4f} (n={total_cnt})", flush=True)
        _emit({
            "phase": desc, "event": "chunk_eval",
            "epoch": int(epoch_idx + 1),
            "chunk": int(chunk_idx),
            "total_chunks": int(total_chunks),
            "step": int(global_step),
            "split": "holdout",
            "train_running_kl": float(train_kl),
            "train_n_positions": int(tr_cnt),
            "avg_kl": float(total_kl),
            "n_positions": int(total_cnt),
            "per_cat_avg_kl": per_cat_kl_eval,
            "per_cat_positions": [int(x) for x in ec_cnt.tolist()],
        })
        # Update best-holdout snapshot (mirrors the historical
        # end-of-epoch logic, just runs at chunk cadence now).
        if (select_best_holdout and total_cnt > 0
                and total_kl < best_holdout_kl):
            best_holdout_kl = float(total_kl)
            best_V_cpu = V.detach().float().cpu().clone()
            best_b_cpu = (b.detach().float().cpu().clone()
                          if b is not None else None)
            best_epoch = int(epoch_idx + 1)
            print(f"    [joint] new best holdout_kl="
                  f"{total_kl:.4f} at ep{epoch_idx+1} "
                  f"chunk {chunk_idx}/{total_chunks} -> snapshot saved",
                  flush=True)
            if (checkpoint_dir is not None
                    and checkpoint_prefix is not None
                    and _is_rank_zero()):
                try:
                    os.makedirs(checkpoint_dir, exist_ok=True)
                    ckpt_path = os.path.join(
                        checkpoint_dir,
                        f"{checkpoint_prefix}_best.pt")
                    torch.save({
                        "V": best_V_cpu,
                        "b": best_b_cpu,
                        "epoch": int(best_epoch),
                        "chunk": int(chunk_idx),
                        "holdout_kl": float(best_holdout_kl),
                    }, ckpt_path)
                    print(f"    [joint] checkpointed best to "
                          f"{ckpt_path}", flush=True)
                except Exception as exc:
                    print(f"    [joint] WARN: checkpoint write "
                          f"failed: {exc}", flush=True)
    # Periodic step-based fallback checkpointing: write the *current*
    # V/b to disk every CKPT_EVERY_STEPS so that even if we crash
    # *before* the first holdout-eval (e.g. NCCL timeout right at the
    # epoch boundary), downstream stages can still recover something
    # reasonable.  Best-holdout snapshot, when available, is always
    # preferred over this; we only write the step ckpt when a best
    # checkpoint does not yet exist for this run.  We also force a
    # checkpoint at the end of every training epoch (before the
    # epoch-end barrier) so a crash in the post-loop allreduce / eval
    # never wastes a full epoch of work.
    ckpt_every_steps = 50

    def _maybe_write_step_ckpt(epoch_idx: int, force: bool = False) -> None:
        # rank 0 only; never overwrites a real best-holdout snapshot
        if (checkpoint_dir is None or checkpoint_prefix is None
                or not _is_rank_zero()):
            return
        if best_V_cpu is not None:
            return  # a real best snapshot is already on disk
        if not force and (global_step % ckpt_every_steps != 0):
            return
        try:
            os.makedirs(checkpoint_dir, exist_ok=True)
            ckpt_path = os.path.join(
                checkpoint_dir, f"{checkpoint_prefix}_best.pt")
            torch.save({
                "V": V.detach().float().cpu().clone(),
                "b": (b.detach().float().cpu().clone()
                      if b is not None else None),
                "epoch": int(epoch_idx + 1),
                "step": int(global_step),
                "holdout_kl": float("nan"),
                "kind": "step_fallback",
            }, ckpt_path)
        except Exception as exc:
            print(f"    [joint] WARN: step ckpt write failed: {exc}",
                  flush=True)
    for epoch in range(n_epochs):
        if cap_resample_each_epoch:
            # Re-cap positions for this epoch with fresh randomness so
            # all disagreement positions get a chance to contribute over
            # time.
            rng_epoch = random.Random(f"{seed}-epoch-{epoch}")
            by_example = _cap_per_epoch(rng_epoch)
            # Length-bucket again (cap may shrink some examples).
            ex_ids_sorted = sorted(
                sorted(by_example.keys()),
                key=lambda e: per_example[e]["ids"].shape[0])
        # Shuffle *buckets* of similar-length examples to preserve
        # length-bucketing benefits while keeping some stochasticity.
        bucket_starts = list(range(0, len(ex_ids_sorted),
                                   example_batch_size))
        # All ranks shuffle identically (same `rng` seeded by `seed`),
        # then take every `world_size`-th bucket starting at `rank`.
        rng.shuffle(bucket_starts)
        ddp_world = (int(os.environ["WORLD_SIZE"])
                     if _is_ddp() else 1)
        ddp_rank = (int(os.environ["RANK"])
                    if _is_ddp() else 0)
        if ddp_world > 1:
            # Trim to a multiple of world_size so every rank processes
            # exactly the same number of batches.  Without this, rank 0
            # gets ceil(n/W) while others get floor(n/W); the extra batch
            # causes rank 0 to arrive at the epoch-end barrier late, which
            # triggers a 25+ min NCCL barrier spin (100% SM, 0% mem).
            # We lose at most world_size-1 batches (<1% of epoch data).
            trim = len(bucket_starts) - (len(bucket_starts) % ddp_world)
            bucket_starts = bucket_starts[:trim]
        bucket_starts = bucket_starts[ddp_rank::ddp_world]

        ep_loss_sum = 0.0
        ep_positions = 0
        # Per-category running accumulators for this epoch.
        ep_cat_sum = torch.zeros(n_cats, device=device, dtype=torch.float64)
        ep_cat_cnt = torch.zeros(n_cats, device=device, dtype=torch.long)
        # Rank-local step counter used to schedule mid-epoch chunk evals.
        # Using len(bucket_starts) (this rank's batch count) keeps the
        # cadence the same in single-rank and DDP runs.
        step_in_epoch_local = 0
        n_steps_this_epoch = len(bucket_starts)
        eval_step_period_local = max(
            1, math.ceil(n_steps_this_epoch / _ec))
        for start in bucket_starts:
            mb = ex_ids_sorted[start:start + example_batch_size]
            opt.zero_grad(set_to_none=True)
            per_pos, pos_cats_batch = _compute_batch_kl_loss_joint(
                base_model, mb, per_example, by_example,
                V, steer_layer, pad_token_id,
                kl_mode=kl_mode, train_topk=train_topk, b=b,
                bias_frozen=bias_frozen)
            n_pts = int(per_pos.numel())
            if n_pts == 0:
                pbar.update(1)
                global_step += 1
                step_in_epoch_local += 1
                continue
            loss_sum = per_pos.sum()
            if per_example_loss:
                # Per-example mean -> mean over examples. Reconstruct
                # the within-batch example index of each position from
                # mb's iteration order (matches
                # _compute_batch_kl_loss_joint).
                B_mb = len(mb)
                pos_bids_list: List[int] = []
                for bi, ex_idx in enumerate(mb):
                    pos_bids_list.extend([bi] * len(by_example[ex_idx]))
                pos_bids_t = torch.tensor(pos_bids_list, device=device,
                                          dtype=torch.long)
                ex_loss_sum = torch.zeros(B_mb, device=device,
                                          dtype=per_pos.dtype)
                ex_count = torch.zeros(B_mb, device=device,
                                       dtype=per_pos.dtype)
                ex_loss_sum.scatter_add_(0, pos_bids_t, per_pos)
                ex_count.scatter_add_(0, pos_bids_t,
                                      torch.ones_like(per_pos))
                mask = ex_count > 0
                ex_mean = ex_loss_sum[mask] / ex_count[mask]
                loss = ex_mean.mean()
            else:
                # Default: mean over all positions in the batch
                # (matches the historical aggregation).
                loss = loss_sum / n_pts
            loss.backward()
            # ---- DDP gradient sync ----
            # V/b are tiny (a few KB - a few MB), so an all-reduce is
            # essentially free.  We average the per-rank grads so the
            # optimiser step is the SAME on every rank (deterministic
            # with the same V/b state, lr, weight_decay).
            if dist.is_initialized():
                if V.grad is not None:
                    dist.all_reduce(V.grad, op=dist.ReduceOp.SUM)
                    V.grad.div_(float(dist.get_world_size()))
                if b is not None and b.grad is not None:
                    dist.all_reduce(b.grad, op=dist.ReduceOp.SUM)
                    b.grad.div_(float(dist.get_world_size()))
            opt.step()
            # Optional row-wise norm clip on V (and b if present).
            if max_norm and max_norm > 0:
                with torch.no_grad():
                    row_norms = V.detach().norm(dim=-1, keepdim=True)
                    scale = torch.clamp(max_norm / row_norms.clamp(min=1e-8),
                                        max=1.0)
                    V.mul_(scale)
                    if b is not None:
                        bn = b.detach().norm()
                        if bn > max_norm:
                            b.mul_(max_norm / bn)
            # Per-cat stats (no grad).
            with torch.no_grad():
                pp = per_pos.detach().double()
                ep_cat_sum.scatter_add_(0, pos_cats_batch, pp)
                ep_cat_cnt.scatter_add_(
                    0, pos_cats_batch,
                    torch.ones_like(pos_cats_batch, dtype=torch.long))
                batch_sum = float(loss_sum.detach().item())
                # Batch-level per-cat KL (mean within this step).
                batch_cat_sum = torch.zeros(n_cats, device=device,
                                            dtype=torch.float64)
                batch_cat_cnt = torch.zeros(n_cats, device=device,
                                            dtype=torch.long)
                batch_cat_sum.scatter_add_(0, pos_cats_batch, pp)
                batch_cat_cnt.scatter_add_(
                    0, pos_cats_batch,
                    torch.ones_like(pos_cats_batch, dtype=torch.long))
                norms = V.detach().norm(dim=-1)
                bias_norm = (float(b.detach().norm().item())
                             if b is not None else None)
            ep_loss_sum += batch_sum
            ep_positions += n_pts
            running = ep_loss_sum / max(ep_positions, 1)
            postfix = {
                "ep": f"{epoch+1}/{n_epochs}",
                "kl": f"{running:.4f}",
                "norms": f"[{norms.mean().item():.1f}]",
            }
            if bias_norm is not None:
                postfix["bias"] = f"{bias_norm:.1f}"
            pbar.set_postfix(**postfix)
            pbar.update(1)
            last_loss = running
            _emit({
                "phase": desc, "event": "step",
                "epoch": int(epoch + 1), "step": int(global_step),
                "n_positions": int(n_pts),
                "batch_avg_kl": float(batch_sum / max(n_pts, 1)),
                "running_avg_kl": float(running),
                "norm_mean": float(norms.mean().item()),
                "norm_max": float(norms.max().item()),
                "norm_min": float(norms.min().item()),
                "norms_per_cat": [float(x) for x in norms.tolist()],
                "bias_norm": bias_norm,
                "batch_per_cat_kl": [
                    (float((batch_cat_sum[i] /
                            batch_cat_cnt[i]).item())
                     if int(batch_cat_cnt[i].item()) > 0 else None)
                    for i in range(n_cats)],
                "batch_per_cat_count": [
                    int(x) for x in batch_cat_cnt.tolist()],
            })
            global_step += 1
            step_in_epoch_local += 1
            _maybe_write_step_ckpt(epoch)
            # Periodic Python GC to prevent reference-cycle accumulation
            # across 700-900 steps building up a huge deferred-free list
            # that triggers a 25+ min GPU memory compaction at epoch end.
            if global_step % 50 == 0:
                import gc as _gc
                _gc.collect()
            # Mid-epoch holdout eval: every ``eval_step_period_local``
            # rank-local steps we run a quick no-grad holdout pass so
            # the train/holdout curves are sampled ``_ec`` times per
            # epoch (and the best-holdout snapshot can capture an
            # intra-epoch minimum).  Skip the boundary that coincides
            # with the end of the epoch -- the existing end-of-epoch
            # eval handles that.
            at_chunk_boundary = (
                step_in_epoch_local % eval_step_period_local == 0
                and step_in_epoch_local < n_steps_this_epoch)
            if at_chunk_boundary:
                chunk_idx = step_in_epoch_local // eval_step_period_local
                _run_holdout_eval(epoch, chunk_idx, _ec,
                                  ep_loss_sum, ep_positions)
        # End-of-training-loop ckpt (force write even if not at the
        # CKPT_EVERY_STEPS boundary): this is the snapshot that
        # crash-recovery downstream will use if the upcoming holdout
        # eval / allreduce times out.
        _maybe_write_step_ckpt(epoch, force=True)
        # End-of-epoch summary.
        # Ensure all ranks have finished the training loop before
        # issuing the small allreduces below.  We intentionally skip
        # empty_cache() here: on H200 with ~100 GB allocated and many
        # fragmented free blocks, empty_cache() triggers a GPU-side
        # memory compaction that can take 30+ minutes (100% SM, 0% mem
        # bandwidth), which outlasts even a 24-hour NCCL timeout in the
        # worst case.  The 24-h collective timeout set at init_process_group
        # and TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC=86400 give us enough
        # headroom for any legitimate tail latency without the compaction.
        try:
            torch.cuda.synchronize()
        except Exception:
            pass
        if dist.is_initialized():
            try:
                dist.barrier()
            except Exception as _exc:
                print(f"    [joint] WARN: epoch-end barrier failed "
                      f"({_exc}); proceeding", flush=True)
        # ---- DDP: aggregate per-cat sums/counts and total loss across
        # ranks so the printed/logged numbers reflect the FULL epoch
        # (each rank only saw its 1/world_size shard).
        if dist.is_initialized():
            _ddp_allreduce_(ep_cat_sum)
            _ddp_allreduce_(ep_cat_cnt)
            _ep_loss_t = torch.tensor(
                [ep_loss_sum, float(ep_positions)],
                device=device, dtype=torch.float64)
            _ddp_allreduce_(_ep_loss_t)
            ep_loss_sum = float(_ep_loss_t[0].item())
            ep_positions = int(_ep_loss_t[1].item())
        with torch.no_grad():
            norms = V.detach().norm(dim=-1)
            per_cat_avg_kl = [
                (float((ep_cat_sum[i] / ep_cat_cnt[i]).item())
                 if int(ep_cat_cnt[i].item()) > 0 else None)
                for i in range(n_cats)]
        print(f"    [joint] epoch {epoch+1}/{n_epochs}: "
              f"kl = {ep_loss_sum / max(ep_positions,1):.4f}  "
              f"positions = {ep_positions}  "
              f"mean_norm = {norms.mean().item():.3f}  "
              f"max_norm = {norms.max().item():.3f}",
              flush=True)
        _emit({
            "phase": desc, "event": "epoch_end",
            "epoch": int(epoch + 1),
            "avg_kl": float(ep_loss_sum / max(ep_positions, 1)),
            "n_positions": int(ep_positions),
            "norm_mean": float(norms.mean().item()),
            "norm_max": float(norms.max().item()),
            "norm_min": float(norms.min().item()),
            "norms_per_cat": [float(x) for x in norms.tolist()],
            "bias_norm": (float(b.detach().norm().item())
                          if b is not None else None),
            "per_cat_avg_kl": per_cat_avg_kl,
            "per_cat_positions": [int(x) for x in ep_cat_cnt.tolist()],
        })

        # -- End-of-epoch no-grad eval on the holdout split.
        # This is the FINAL ("chunk_idx == _ec") sample for the epoch
        # -- the mid-epoch chunk evals above already sampled chunks
        # 1..(_ec - 1).  We reuse the same closure so all chunk evals
        # are recorded identically; the closure also updates the
        # best-holdout snapshot.
        _run_holdout_eval(epoch, _ec, _ec, ep_loss_sum, ep_positions)
    pbar.close()
    if select_best_holdout and best_V_cpu is not None:
        print(f"    [joint] selecting best-holdout snapshot from "
              f"epoch {best_epoch} (holdout_kl={best_holdout_kl:.4f})",
              flush=True)
        _emit({"phase": desc, "event": "best_holdout_selected",
               "epoch": int(best_epoch),
               "avg_kl": float(best_holdout_kl)})
        if metrics_fh is not None:
            metrics_fh.close()
        return best_V_cpu, best_b_cpu, metrics_records
    if metrics_fh is not None:
        metrics_fh.close()
    b_out = (b.detach().float().cpu() if b is not None else None)
    return V.detach().float().cpu(), b_out, metrics_records


def _compute_batch_kl_loss_joint(
    base_model,
    mb: List[int],
    per_example: List[dict],
    by_example: Dict[int, List[Tuple[int, int, torch.Tensor, torch.Tensor]]],
    V: torch.Tensor,
    steer_layer: int,
    pad_token_id: int,
    *,
    kl_mode: str = "topk",
    train_topk: int = 3,
    b: Optional[torch.Tensor] = None,
    bias_frozen: Optional[torch.Tensor] = None,
) -> Tuple[torch.Tensor, int]:
    """Forward one padded minibatch with V injected per-position by
    category, return (per_position_losses, position_category_indices).

    kl_mode controls the loss:
      * ``full_vocab`` (legacy):
            loss_i = - sum_k p_think[i,k] * log p_base_full[i, idx_k]
        where p_base_full = softmax(base_logits, dim=-1) over the full
        vocab. Equivalent (up to a v-indep constant) to
        KL(p_think_topk || p_base_full).
      * ``topk`` (recommended, default):
            p_think_norm = p_think[topk] / sum(p_think[topk])
            p_base_norm  = softmax(base_logits[topk], dim=-1)   # over K tokens
            loss_i       = - sum_k p_think_norm[k] * log p_base_norm[k]
        This is the top-K-restricted cross-entropy where both
        distributions are renormalised within the K candidate tokens.
        Reduces dilution from the 100K+ irrelevant tokens in full-vocab
        KL and gives a cleaner steering gradient.
      * ``ce`` (sharpest signal):
            target = rollout_token[i+1]        # hard label from the
                                               # thinking-model rollout
            loss_i = - log softmax(base_logits)[target]
        Full-vocab cross-entropy against the actual next token from
        the thinking rollout (= the token the base model gets "wrong"
        at a disagreement position).  Uses no thinking-model
        distribution info, just the single correct token.
        ``train_topk`` is ignored in this mode.

    ``train_topk`` truncates the cached top-K' to the first K'
    thinking-model indices during loss computation (the cached K is
    typically 50; default K' = 3 focuses the loss on the tokens that
    matter most).  ``train_topk`` must be <= the cached K.
    """
    device = next(base_model.parameters()).device
    B = len(mb)
    Lmax = max(per_example[e]["ids"].shape[0] for e in mb)
    ids_batch = torch.full((B, Lmax), pad_token_id,
                           device=device, dtype=torch.long)
    attn = torch.zeros((B, Lmax), device=device, dtype=torch.long)
    pos_bids: List[int] = []
    pos_tids: List[int] = []
    pos_cats: List[int] = []
    pos_targets: List[int] = []
    topk_lps: List[torch.Tensor] = []
    topk_ixs: List[torch.Tensor] = []
    for bi, ex_idx in enumerate(mb):
        ex_ids = per_example[ex_idx]["ids"]
        L = ex_ids.shape[0]
        ids_batch[bi, :L] = ex_ids.to(device, non_blocking=True)
        attn[bi, :L] = 1
        ex_ids_cpu = per_example[ex_idx]["ids"]
        for p, c, tlp, tix in by_example[ex_idx]:
            pos_bids.append(bi)
            pos_tids.append(p)
            pos_cats.append(c)
            # Target token = rollout token at base position p+1.  Used
            # by ``kl_mode='ce'``; ignored otherwise.
            pos_targets.append(int(ex_ids_cpu[p + 1].item()))
            topk_lps.append(tlp)
            topk_ixs.append(tix)
    if not pos_bids:
        empty = torch.zeros((0,), device=device)
        empty_cats = torch.zeros((0,), device=device, dtype=torch.long)
        return empty, empty_cats

    pos_bids_t = torch.tensor(pos_bids, device=device, dtype=torch.long)
    pos_tids_t = torch.tensor(pos_tids, device=device, dtype=torch.long)
    pos_cats_t = torch.tensor(pos_cats, device=device, dtype=torch.long)
    pos_tgt_t = torch.tensor(pos_targets, device=device, dtype=torch.long)
    topk_lp_t = torch.stack(topk_lps).to(device, torch.float32)
    topk_idx_t = torch.stack(topk_ixs).to(device, torch.long)
    # Truncate to train_topk (<= cached K) ------------------------------
    if train_topk is not None and 0 < train_topk < topk_lp_t.shape[-1]:
        topk_lp_t = topk_lp_t[:, :train_topk]
        topk_idx_t = topk_idx_t[:, :train_topk]
    topk_probs = topk_lp_t.exp()

    hook = _InjectMultiHook(V, pos_bids_t, pos_tids_t, pos_cats_t, b=b,
                            bias_frozen=bias_frozen)
    with _inject_at_layer(base_model, steer_layer, hook):
        body_out = base_model.model(
            input_ids=ids_batch, attention_mask=attn, use_cache=False)
    hidden = body_out.last_hidden_state
    h_dev = hidden.device
    selected = hidden[pos_bids_t.to(h_dev), pos_tids_t.to(h_dev), :]
    logits = base_model.lm_head(selected).float()
    log_dev = logits.device
    topk_idx_t = topk_idx_t.to(log_dev)
    topk_probs = topk_probs.to(log_dev)
    pos_tgt_t_dev = pos_tgt_t.to(log_dev)

    if kl_mode == "full_vocab":
        base_lp = torch.log_softmax(logits, dim=-1)
        base_lp_at_topk = base_lp.gather(-1, topk_idx_t)
        per_pos = -(topk_probs * base_lp_at_topk).sum(dim=-1)
    elif kl_mode == "topk":
        # Renormalise thinking-model probs within the top-K' subset (the
        # cached probs came from a full-vocab softmax, so they sum to
        # < 1 over just the top-K'; divide through).
        tprob_sum = topk_probs.sum(dim=-1, keepdim=True).clamp_min(1e-12)
        p_think_norm = topk_probs / tprob_sum
        # Renormalise base probs within the top-K' subset.
        base_logits_topk = logits.gather(-1, topk_idx_t)        # (N, K')
        base_lp_topk = torch.log_softmax(base_logits_topk, dim=-1)
        per_pos = -(p_think_norm * base_lp_topk).sum(dim=-1)
    elif kl_mode == "ce":
        # Hard-label CE against the rollout target token over the full
        # vocab.  Sharpest possible training signal.
        base_lp = torch.log_softmax(logits, dim=-1)
        per_pos = -base_lp.gather(-1, pos_tgt_t_dev.unsqueeze(-1)).squeeze(-1)
    else:
        raise ValueError(f"Unknown kl_mode={kl_mode!r}; "
                         f"expected 'topk', 'full_vocab', or 'ce'")
    # Move per_pos back to the trainer's primary device so downstream
    # scatter_add_ ops with pos_cats_t / pos_bids_t see matching devices.
    return per_pos.to(device), pos_cats_t


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    # Initialise DDP early so per-rank GPU + log prefix is set before
    # any other CUDA/torch ops happen.  Outside DDP this is a no-op
    # and (rank, world_size, local_rank) defaults to (0, 1, 0).
    rank, world_size, local_rank = _ddp_setup()
    if _is_ddp():
        # Tag stdout so per-rank tqdm bars are distinguishable.
        print(f"[ddp rank={rank}/{world_size} local_rank={local_rank}] "
              f"initialised", flush=True)

    p = argparse.ArgumentParser()
    p.add_argument("--base_model", type=str, required=True)
    p.add_argument("--thinking_model", type=str, required=True,
                   help="HF id of the thinking model (e.g. "
                        "Open-Reasoner-Zero/Open-Reasoner-Zero-0.5B); "
                        "used ONLY to compute soft KL targets at "
                        "disagreement positions.")
    p.add_argument("--thinking_model_short", type=str, required=True,
                   help="Short name used in annotated_responses_<name>.json")
    p.add_argument("--steer_layer", type=int, default=None,
                   help="Single steering layer (legacy; use --steer_layers "
                        "for a sweep).  Ignored if --steer_layers is given.")
    p.add_argument("--steer_layers", type=str, default=None,
                   help="Comma-separated list of candidate steering layers "
                        "to sweep, e.g. '4,9,14,19'.  For each (layer, "
                        "category) we train a vector, evaluate on a "
                        "holdout split, and keep the layer with the lowest "
                        "holdout KL.  Writes a layer_map.json in save_dir.")
    p.add_argument("--holdout_frac", type=float, default=0.1,
                   help="Fraction of *responses* held out for "
                        "generalisation tracking.  Split is made once "
                        "at the response level so positions from the "
                        "same response never cross train/holdout. "
                        "When >0, the trainer runs a no-grad eval pass "
                        "on both splits at the end of every epoch and "
                        "logs per-category KL to training_metrics_*.jsonl.")
    p.add_argument("--train_global_bias", action="store_true",
                   help="Additionally train a single global-bias vector on "
                        "the UNION of all disagreement positions (ignoring "
                        "category labels).  Also layer-swept.  Saved as "
                        "<model>_bias_global.pt + bias_layer.json.  This "
                        "is intended as an ablation baseline -- 'one "
                        "direction for everything'.")
    p.add_argument("--joint_cats_and_bias", action="store_true",
                   help="JOINT cats+bias training: at every disagreement "
                        "position apply  V[cat[p]] + b  in a single "
                        "forward/backward pass, and update both the "
                        "category matrix V and the shared bias b with "
                        "ONE Adam step.  The optimiser routes "
                        "category-agnostic corrections into b and "
                        "category-specific corrections into V[k], which "
                        "matches the inference-time `cat + bias` "
                        "composition exactly (no double-counting).  When "
                        "set, the separate Phase 3b bias-only training "
                        "is skipped and the co-trained b is saved as "
                        "<model>_bias_global.pt.")
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--topk", type=int, default=50,
                   help="Number of top-K thinking tokens cached per "
                        "disagreement position (soft-KL target).")
    p.add_argument("--train_topk", type=int, default=3,
                   help="Truncate the cached top-K to this many tokens "
                        "at TRAINING time. The loss is computed over "
                        "just these K' tokens (default K'=3). Must be "
                        "<= --topk. Smaller values focus gradient on "
                        "the thinking model's most-likely alternatives "
                        "and avoid dilution from the long tail.")
    p.add_argument("--kl_mode", type=str, default="topk",
                   choices=["topk", "full_vocab", "ce"],
                   help="Loss formulation. 'topk' (default) = KL over "
                        "the thinking top-K' indices with BOTH "
                        "distributions renormalised within that subset "
                        "(p_think / sum and softmax(base_logits[topk])). "
                        "'full_vocab' (legacy) = the base log-softmax "
                        "is over the full vocab and we gather at the "
                        "topk indices -- this implicitly pushes mass "
                        "away from the 100K+ irrelevant tokens too, "
                        "diluting the gradient.  'ce' = hard-label "
                        "full-vocab cross-entropy against the rollout "
                        "target token at each disagreement position "
                        "(sharpest signal; --train_topk is ignored).")
    p.add_argument("--n_responses", type=int, default=1000,
                   help="Cap on number of annotated responses to "
                        "process. With --adaptive_n_responses, this "
                        "acts as a CEILING only.")
    p.add_argument("--adaptive_n_responses", action="store_true",
                   help="Two-pass collection: process "
                        "--adaptive_probe_size responses to estimate "
                        "per-response disagreement rate, then "
                        "continue collecting until a data-adaptive "
                        "target is reached.  Target total positions "
                        "= --positions_per_dim * hidden_size * "
                        "n_categories_observed.  Capped at "
                        "--n_responses and available annotated data.")
    p.add_argument("--adaptive_probe_size", type=int, default=150,
                   help="Number of responses used to estimate "
                        "per-response disagreement rate before "
                        "extrapolating.")
    p.add_argument("--positions_per_dim", type=float, default=1.0,
                   help="Data-adaptive collection target: aim for "
                        "average-share >= C * hidden_size positions "
                        "per category (C = --positions_per_dim).  "
                        "With the share-gate of 0.1, the worst "
                        "surviving category then has >= 0.1 * C * d "
                        "positions.  C=1 is a sensible default "
                        "(well-posed linear regression on equal share).")
    p.add_argument("--max_seq_len", type=int, default=2048,
                   help="Skip responses whose (prompt + thinking) tokenises "
                        "longer than this")
    p.add_argument("--max_positions_per_example", type=int, default=64,
                   help="Cap disagreement positions per example per "
                        "category, to avoid one response dominating")
    p.add_argument("--n_epochs", type=int, default=5)
    p.add_argument("--example_batch_size", type=int, default=4)
    p.add_argument("--lr", type=float, default=1e-2)
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--max_norm", type=float, default=-1.0,
                   help="Per-step row-wise norm clip on V (and b if "
                        "trained).  Negative (default) = AUTO: cap "
                        "every vector at "
                        "norm_cap_frac * mean_residual_norm at the "
                        "steer layer (probed at startup).  0 disables "
                        "the cap.  Positive = explicit cap value.")
    p.add_argument("--norm_cap_frac", type=float, default=0.5,
                   help="When --max_norm < 0, cap each vector (bias "
                        "AND every category vector) at "
                        "norm_cap_frac * mean_residual_norm.  Default "
                        "0.5 keeps every learned vector at most half "
                        "the typical activation magnitude.")
    p.add_argument("--norm_cap_probe_examples", type=int, default=32,
                   help="Examples used to probe the mean residual-"
                        "stream norm at the steer layer for the auto "
                        "norm cap (--max_norm < 0).")
    p.add_argument("--no_select_best_holdout", action="store_true",
                   help="Disable best-holdout-KL snapshot selection. "
                        "By default we save the V from the epoch that "
                        "achieved the lowest holdout KL.")
    p.add_argument("--eval_chunks_per_epoch", type=int, default=4,
                   help="How many times PER EPOCH we run a no-grad "
                        "holdout eval (and emit train_kl + holdout_kl "
                        "in the same record).  4 means at "
                        "25/50/75/100%% of every epoch -- denser "
                        "monitoring than the historical 1/epoch.")
    p.add_argument("--per_example_loss", action="store_true",
                   help="Use per-example mean -> mean over examples "
                        "loss aggregation (balances per-example "
                        "contribution). Default is per-position mean "
                        "(matches historical pipeline).")
    p.add_argument("--cap_resample_each_epoch", action="store_true",
                   help="Resample the per-(example,category) position "
                        "cap each epoch using a fresh RNG. Default OFF "
                        "(matches historical pipeline: deterministic "
                        "subsample held fixed across epochs).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--n_seeds", type=int, default=1,
                   help="Number of independent seeds to train (sequentially "
                        "in this process; each call shares the loaded base "
                        "model).  Seeds used are seed, seed+1, ..., "
                        "seed+N-1.  After all runs, the V with the lowest "
                        "best-holdout KL is kept; the others are discarded "
                        "(per-seed metrics still written to "
                        "training_metrics_cats_seed{i}.jsonl).  Combine with "
                        "--init_v_from_activations to give different seeds "
                        "genuinely different starting directions.")
    p.add_argument("--init_v_from_activations", action="store_true",
                   help="Compute the mean L2 norm of the residual stream at "
                        "--steer_layer over a small probe set BEFORE "
                        "training, and initialize each correction vector as "
                        "a random unit-norm direction scaled to that "
                        "magnitude.  Default OFF (legacy: zero init).  "
                        "Combined with --n_seeds>1, each seed gets a "
                        "different random direction.")
    p.add_argument("--init_probe_examples", type=int, default=16,
                   help="Number of examples used by --init_v_from_activations.")
    p.add_argument("--frozen_bias_path", type=str, default=None,
                   help="Path to a previously-trained global bias vector "
                        "(e.g. <prev_save_dir>/<model>_bias_global.pt).  "
                        "When set: (1) the bias is hooked into the BASE "
                        "model's residual stream at --frozen_bias_layer "
                        "(all positions) during disagreement collection, "
                        "so the collected positions are the ones where "
                        "BASE+BIAS still disagrees with thinking; (2) "
                        "during category-vector training, the hook adds "
                        "``bias_frozen + V[cat[p]]`` at each disagreement "
                        "position, so V learns the category-specific "
                        "RESIDUAL on top of the static bias.  No grad "
                        "flows into the bias.  Pair with --skip_cats_phase=False "
                        "to actually train the cats.  At inference, point "
                        "hybrid_eval at this same bias path with "
                        "--bias_vector_path so the steering composition "
                        "matches training.")
    p.add_argument("--frozen_bias_layer", type=int, default=None,
                   help="Layer at which the frozen bias is applied.  "
                        "Defaults to --steer_layer.")
    p.add_argument("--sae_classify_layer", type=int, default=None,
                   help="If set, classify EACH disagreement position by "
                        "running the thinking-model activation at this "
                        "layer through the SAE -- exactly mirroring "
                        "hybrid_eval.py's last-token classification "
                        "(but applied per-position during collection).  "
                        "When set, the (sentence-level) annotated "
                        "categories are bypassed for category "
                        "assignment, eliminating the train/eval "
                        "category-distribution mismatch.")
    p.add_argument("--sae_n_clusters", type=int, default=10,
                   help="n_clusters of the SAE used for "
                        "--sae_classify_layer.  Default 10 (matches the "
                        "ORZ-7B / QwQ-32B SAEs).")
    p.add_argument("--sae_disable_mean", action="store_true",
                   help="If set, skip activation-mean centering during "
                        "SAE classification (matches hybrid_eval.py "
                        "--disable_sae_mean).  Auto-on when the SAE "
                        "checkpoint has no activation_mean.")
    p.add_argument("--skip_cats_phase", action="store_true",
                   help="Skip Phase 3a (per-category vector training) "
                        "entirely.  Use together with --train_global_bias "
                        "to train ONLY the global bias vector "
                        "(stage 1 of the bias-first pipeline).")
    p.add_argument("--collection_mode", type=str, default="disagreement",
                   choices=["disagreement", "entropy", "union"],
                   help="Position-selection rule during phase 1. "
                        "'disagreement' (default): include only positions "
                        "where the base model's argmax differs from the "
                        "rollout token (legacy behaviour). "
                        "'entropy': include all positions whose category "
                        "is labelled and the THINKING model's next-token "
                        "entropy is >= --entropy_threshold (in nats), "
                        "regardless of agreement. "
                        "'union': include positions where base disagrees "
                        "OR thinking entropy >= threshold -- combines "
                        "explicit hard-error signal with extra soft-target "
                        "signal from same-top-1 high-entropy positions.")
    p.add_argument("--entropy_threshold", type=float, default=1.0,
                   help="Used only when --collection_mode=entropy: "
                        "minimum thinking-model next-token entropy (in "
                        "nats) for a position to be retained.  A value "
                        "of ~ln(k) corresponds to roughly k effective "
                        "candidates: 0.7 ~ 2 candidates, 1.1 ~ 3, "
                        "1.6 ~ 5, 2.3 ~ 10.  Default 1.0.")
    p.add_argument("--responses_dir", type=str,
                   default="../generate-responses/results/vars")
    p.add_argument("--min_disagreements", type=int, default=1,
                   help="Absolute lower bound on positions a category "
                        "needs before we train its vector.  Default 1 "
                        "means we train every category with at least "
                        "one disagreement position; categories with "
                        "literally zero data are dropped.  Final gate "
                        "is max(--min_disagreements, "
                        "ceil(--min_disagreements_ratio * hidden_size), "
                        "share*equal).")
    p.add_argument("--min_disagreements_ratio", type=float, default=0.0,
                   help="LEGACY dim-aware gate: drop any category with "
                        "fewer than ratio * hidden_size positions. "
                        "Off by default.  Gate is the MAX with the "
                        "other floors.")
    p.add_argument("--min_category_share", type=float, default=0.0,
                   help="Data-adaptive gate: drop any category with "
                        "fewer positions than SHARE * (total_positions "
                        "/ n_categories).  Default 0.0 == off; we "
                        "train every category with any data.  Set to "
                        "e.g. 0.1 to require >= 10%% of the equal-split "
                        "share, scaling with both the observed "
                        "disagreement rate (few "
                        "disagreements -> small gate) and the "
                        "category granularity (many categories -> "
                        "small gate). Set 0 to disable.")
    p.add_argument("--collect_only", action="store_true",
                   help="Run ONLY phase 1 (collect disagreements + top-K "
                        "thinking targets), write to <save_dir>/"
                        "disagreements.pt, and exit.  No base model is "
                        "needed for training afterwards -- run the same "
                        "script again with --load_collected to fit the "
                        "vectors.  Splitting the job in two processes "
                        "guarantees the thinking model's VRAM is "
                        "released by the OS on exit.")
    p.add_argument("--load_collected", action="store_true",
                   help="Skip phase 1; instead load <save_dir>/"
                        "disagreements.pt produced by a prior "
                        "--collect_only invocation.  In this mode the "
                        "thinking model is NEVER loaded.")
    args = p.parse_args()

    os.makedirs(args.save_dir, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    if args.collect_only and args.load_collected:
        raise ValueError("--collect_only and --load_collected are mutually "
                         "exclusive")
    dump_path = os.path.join(args.save_dir, "disagreements.pt")

    # ---- Load optional frozen-bias vector (cpu).  Will be moved to the
    # base model's device after that model is loaded.
    frozen_bias_cpu: Optional[torch.Tensor] = None
    if args.frozen_bias_path:
        print(f"Loading frozen bias from {args.frozen_bias_path} ...",
              flush=True)
        _bobj = torch.load(args.frozen_bias_path, map_location="cpu",
                           weights_only=False)
        if isinstance(_bobj, dict):
            frozen_bias_cpu = _bobj.get(
                "bias", next(iter(_bobj.values())))
        else:
            frozen_bias_cpu = _bobj
        if frozen_bias_cpu.dim() == 2 and frozen_bias_cpu.shape[0] == 1:
            frozen_bias_cpu = frozen_bias_cpu.squeeze(0)
        frozen_bias_cpu = frozen_bias_cpu.float().detach()
        print(f"  frozen_bias: shape={tuple(frozen_bias_cpu.shape)}, "
              f"norm={float(frozen_bias_cpu.norm().item()):.3f}",
              flush=True)

    # ========================================================================
    # PHASE 1: collect disagreements -- runs both base and thinking models.
    # If --load_collected we jump straight to phase 2 without loading either
    # model here; only the base model is loaded for training.
    # ========================================================================
    if args.load_collected:
        print(f"[load] reading disagreement dump from {dump_path}",
              flush=True)
        blob = torch.load(dump_path, map_location="cpu", weights_only=False)
        per_example = blob["per_example"]
        per_category = blob["per_category"]
        base_tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        if base_tokenizer.pad_token_id is None:
            base_tokenizer.pad_token = base_tokenizer.eos_token

        # Apply load-time max_seq_len filter: the dump may have been
        # collected with a higher cap (eg 2048).  Keeping that data when
        # the user wants 1536 to fit memory means we'd OOM mid-training
        # at the longest batch.  Drop overlength examples here AND every
        # disagreement record that points at them so per_category stays
        # consistent.
        if args.max_seq_len and args.max_seq_len > 0:
            keep_ex_ids = {
                i for i, ex in enumerate(per_example)
                if ex.get("ids") is not None
                and ex["ids"].shape[0] <= args.max_seq_len}
            n_dropped = len(per_example) - len(keep_ex_ids)
            if n_dropped > 0:
                old_n_pos = sum(len(v) for v in per_category.values())
                # Filter records by ex_idx (records are 4-tuples
                # (ex_idx, pos, topk_lp, topk_idx)).
                for k in list(per_category.keys()):
                    per_category[k] = [
                        r for r in per_category[k] if r[0] in keep_ex_ids]
                new_n_pos = sum(len(v) for v in per_category.values())
                _ddp_print(
                    f"  [seq_filter] dropped {n_dropped}/{len(per_example)} "
                    f"examples > {args.max_seq_len} tokens "
                    f"({old_n_pos} -> {new_n_pos} positions)")

        print(f"  loaded {len(per_example)} examples, "
              f"{sum(len(v) for v in per_category.values())} positions "
              f"across {len(per_category)} categories", flush=True)
    else:
        # ---- Load base model ----
        print(f"Loading base model {args.base_model}...", flush=True)
        base_tokenizer = AutoTokenizer.from_pretrained(args.base_model)
        if base_tokenizer.pad_token_id is None:
            base_tokenizer.pad_token = base_tokenizer.eos_token
        # In DDP each rank loads the full (frozen) base model on its own
        # GPU; outside DDP we let HF auto-shard across visible GPUs.
        _device_map = ({"": local_rank} if _is_ddp()
                       else "auto")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model, device_map=_device_map, dtype=torch.bfloat16)
        for p_ in base_model.parameters():
            p_.requires_grad = False

        if frozen_bias_cpu is not None:
            _bdev = next(base_model.parameters()).device
            frozen_bias_cpu = frozen_bias_cpu.to(_bdev)
            assert frozen_bias_cpu.shape[-1] \
                == base_model.config.hidden_size, (
                    f"frozen_bias shape {tuple(frozen_bias_cpu.shape)} "
                    f"!= hidden_size={base_model.config.hidden_size}")

        # ---- Load thinking model (for soft KL targets only) ----
        print(f"Loading thinking model {args.thinking_model}...", flush=True)
        thinking_tokenizer = AutoTokenizer.from_pretrained(args.thinking_model)
        if thinking_tokenizer.pad_token_id is None:
            thinking_tokenizer.pad_token = thinking_tokenizer.eos_token
        thinking_model = AutoModelForCausalLM.from_pretrained(
            args.thinking_model, device_map=_device_map, dtype=torch.bfloat16)
        for p_ in thinking_model.parameters():
            p_.requires_grad = False
        thinking_model.eval()
        # Vocab size must match; in addition verify a few canonical text
        # spans tokenise identically, which is what our cross-model index
        # alignment and top-K target gathering rely on.
        assert thinking_model.config.vocab_size \
            == base_model.config.vocab_size, (
                f"vocab mismatch: base={base_model.config.vocab_size} "
                f"thinking={thinking_model.config.vocab_size}")
        _probe_strs = [
            " the quick brown fox jumps over the lazy dog.",
            "\n\nLet me think step by step.\n\n",
            "Wait, let's backtrack and reconsider.",
            "Therefore, the answer is \\boxed{42}.",
        ]
        for _s in _probe_strs:
            b_ids = base_tokenizer(_s, add_special_tokens=False)["input_ids"]
            t_ids = thinking_tokenizer(_s, add_special_tokens=False)[
                "input_ids"]
            assert b_ids == t_ids, (
                f"Tokenizer mismatch on probe string {_s!r}: "
                f"base={b_ids[:10]}... thinking={t_ids[:10]}...  "
                "Base-position top-K targets cannot be aligned cross-model; "
                "use a matching tokenizer family.")

        # ---- Optional: load SAE for per-position classification --------
        sae_classifier_fn = None
        if args.sae_classify_layer is not None:
            from utils.sae import load_sae
            # Match hybrid_eval.py's SAE filename convention exactly:
            # sae_<short-lower>_layer<L>_clusters<N>.pt
            think_id = args.thinking_model.split("/")[-1].lower()
            sae_obj, _ = load_sae(think_id, args.sae_classify_layer,
                                  args.sae_n_clusters,
                                  require_activation_mean=False)
            sae_obj = sae_obj.to(next(thinking_model.parameters()).device)
            sae_obj.eval()
            for _p in sae_obj.parameters():
                _p.requires_grad = False
            sae_disable_mean = args.sae_disable_mean
            if (not hasattr(sae_obj, "activation_mean")) and \
                    not sae_disable_mean:
                sae_disable_mean = True
                print("  [sae-cat] no activation_mean on SAE; auto-set "
                      "--sae_disable_mean (matches hybrid_eval).",
                      flush=True)
            sae_act_mean = (sae_obj.activation_mean
                            if hasattr(sae_obj, "activation_mean")
                            and not sae_disable_mean else None)
            sae_dev = next(sae_obj.parameters()).device

            @torch.no_grad()
            def sae_classifier_fn(acts):  # noqa: E306
                # acts: (Lt, hidden) on the thinking-model device.
                x = acts.float().to(sae_dev)
                if sae_act_mean is not None:
                    x = x - sae_act_mean.to(sae_dev)
                    x = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
                la = sae_obj.encoder(x - sae_obj.b_dec)
                ids = la.argmax(dim=-1)
                return ids.cpu()

            print(f"  [sae-cat] loaded SAE for {think_id} layer "
                  f"{args.sae_classify_layer} clusters "
                  f"{args.sae_n_clusters} (disable_mean="
                  f"{sae_disable_mean})", flush=True)

        # ---- Load annotated responses ----
        responses_path = os.path.join(
            args.responses_dir, f"responses_{args.thinking_model_short}.json")
        annotated_path = os.path.join(
            args.responses_dir,
            f"annotated_responses_{args.thinking_model_short}.json")
        print(f"Loading responses from {responses_path}")
        with open(responses_path) as f:
            raw = json.load(f)
        print(f"Loading annotated responses from {annotated_path}")
        with open(annotated_path) as f:
            ann = json.load(f)
        merged = []
        for i, r in enumerate(raw):
            if i >= len(ann):
                break
            a = ann[i]
            if (r.get("question_id") == a.get("question_id")
                    and r.get("dataset_name") == a.get("dataset_name")
                    and a.get("annotated_thinking")):
                m = dict(r)
                m["annotated_thinking"] = a["annotated_thinking"]
                merged.append(m)
        print(f"  {len(merged)} responses with annotations")

        random.shuffle(merged)

        if args.adaptive_n_responses:
            # -------- Pass 1: probe --------------------------------------
            probe_n = min(args.adaptive_probe_size, len(merged),
                          args.n_responses)
            print(f"\n[Phase 1.probe] processing {probe_n} responses to "
                  "estimate disagreement rate...", flush=True)
            per_example, per_category = collect_disagreements(
                base_model, thinking_model, base_tokenizer,
                merged[:probe_n],
                max_seq_len=args.max_seq_len,
                max_examples=probe_n,
                topk=args.topk,
                thinking_tokenizer=thinking_tokenizer,
                collection_mode=args.collection_mode,
                entropy_threshold=args.entropy_threshold,
                frozen_bias=frozen_bias_cpu,
                frozen_bias_layer=(args.frozen_bias_layer
                                   if args.frozen_bias_layer is not None
                                   else args.steer_layer),
                sae_classifier=sae_classifier_fn,
                sae_classify_layer=args.sae_classify_layer,
                sae_n_clusters=args.sae_n_clusters)
            probe_total = sum(len(v) for v in per_category.values())
            probe_idx_cats = [k for k in per_category.keys()
                              if k.startswith("idx") and k[3:].isdigit()]
            n_cats_probe = len(probe_idx_cats)
            pos_per_resp = probe_total / max(probe_n, 1)
            hidden_probe = base_model.config.hidden_size
            target_total_pos = (args.positions_per_dim * hidden_probe
                                * max(n_cats_probe, 1))
            if pos_per_resp > 0:
                target_n = math.ceil(target_total_pos / pos_per_resp)
            else:
                target_n = args.n_responses
            target_n = max(target_n, probe_n)
            target_n = min(target_n, args.n_responses, len(merged))
            print(f"[Phase 1.probe] probe: {len(per_example)} retained / "
                  f"{probe_n}, {probe_total} positions "
                  f"({pos_per_resp:.3f}/resp), {n_cats_probe} idx cats",
                  flush=True)
            print(f"[Phase 1.probe] target_total_pos = "
                  f"{args.positions_per_dim:g} * hidden={hidden_probe} * "
                  f"n_cats={n_cats_probe} = {target_total_pos:.0f}",
                  flush=True)
            print(f"[Phase 1.probe] -> collect total {target_n} responses "
                  f"(ceiling={args.n_responses}, available={len(merged)})",
                  flush=True)
            # -------- Pass 2: remainder ----------------------------------
            if target_n > probe_n:
                print(f"\n[Phase 1.extra] collecting {target_n - probe_n} "
                      "more responses...", flush=True)
                pe_more, pc_more = collect_disagreements(
                    base_model, thinking_model, base_tokenizer,
                    merged[probe_n:target_n],
                    max_seq_len=args.max_seq_len,
                    max_examples=target_n - probe_n,
                    topk=args.topk,
                    thinking_tokenizer=thinking_tokenizer,
                    collection_mode=args.collection_mode,
                    entropy_threshold=args.entropy_threshold,
                    frozen_bias=frozen_bias_cpu,
                    frozen_bias_layer=(args.frozen_bias_layer
                                       if args.frozen_bias_layer is not None
                                       else args.steer_layer),
                    sae_classifier=sae_classifier_fn,
                    sae_classify_layer=args.sae_classify_layer,
                    sae_n_clusters=args.sae_n_clusters)
                # Merge pe_more/pc_more into per_example/per_category,
                # offsetting example indices.
                offset = len(per_example)
                per_example.extend(pe_more)
                for cat, recs in pc_more.items():
                    per_category[cat].extend(
                        (ex_idx + offset, pos, lp, ix)
                        for (ex_idx, pos, lp, ix) in recs)
            final_total = sum(len(v) for v in per_category.values())
            print(f"\n[Phase 1] collected {len(per_example)} examples, "
                  f"{final_total} total positions "
                  f"across {len(per_category)} categories", flush=True)
        else:
            per_example, per_category = collect_disagreements(
                base_model, thinking_model, base_tokenizer, merged,
                max_seq_len=args.max_seq_len,
                max_examples=args.n_responses,
                topk=args.topk,
                thinking_tokenizer=thinking_tokenizer,
                collection_mode=args.collection_mode,
                entropy_threshold=args.entropy_threshold,
                frozen_bias=frozen_bias_cpu,
                frozen_bias_layer=(args.frozen_bias_layer
                                   if args.frozen_bias_layer is not None
                                   else args.steer_layer),
                sae_classifier=sae_classifier_fn,
                sae_classify_layer=args.sae_classify_layer,
                sae_n_clusters=args.sae_n_clusters)

        # Persist and exit if this is the collect-only phase -- the caller
        # will re-run the script with --load_collected on a fresh process.
        # defaultdict is not torch.save friendly; convert to plain dict.
        blob = {"per_example": per_example,
                "per_category": dict(per_category),
                "base_model": args.base_model,
                "thinking_model": args.thinking_model,
                "thinking_model_short": args.thinking_model_short,
                "n_responses": args.n_responses,
                "max_seq_len": args.max_seq_len,
                "topk": args.topk,
                "seed": args.seed}
        print(f"\n[save] writing disagreement dump to {dump_path}",
              flush=True)
        torch.save(blob, dump_path)
        sz_mb = os.path.getsize(dump_path) / (1024 ** 2)
        print(f"  dump size: {sz_mb:.1f} MB", flush=True)

        if args.collect_only:
            print("\n[collect_only] exiting; re-run with --load_collected "
                  "to train.", flush=True)
            return

        # Otherwise (legacy single-process mode) free the thinking model.
        try:
            thinking_model.to("meta")
        except Exception:
            pass
        del thinking_model, thinking_tokenizer
        import gc
        gc.collect()
        gc.collect()
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        mem_alloc = torch.cuda.memory_allocated() / (1024 ** 3)
        mem_reserv = torch.cuda.memory_reserved() / (1024 ** 3)
        print(f"\n[post-cleanup] GPU allocated={mem_alloc:.1f} GB  "
              f"reserved={mem_reserv:.1f} GB", flush=True)

    # ========================================================================
    # From here on: training-only path.  Only the base model is needed.
    # ========================================================================
    if args.load_collected:
        print(f"Loading base model {args.base_model}...", flush=True)
        _device_map = ({"": local_rank} if _is_ddp() else "auto")
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model, device_map=_device_map, dtype=torch.bfloat16)
        for p_ in base_model.parameters():
            p_.requires_grad = False
        if frozen_bias_cpu is not None:
            _bdev = next(base_model.parameters()).device
            frozen_bias_cpu = frozen_bias_cpu.to(_bdev)
            assert frozen_bias_cpu.shape[-1] \
                == base_model.config.hidden_size, (
                    f"frozen_bias shape {tuple(frozen_bias_cpu.shape)} "
                    f"!= hidden_size={base_model.config.hidden_size}")
    hidden = base_model.config.hidden_size
    n_layers = base_model.config.num_hidden_layers

    # NOTE: we do NOT enable gradient_checkpointing here.  Our forward hook
    # at `steer_layer` injects a learnable vector into the layer's output,
    # and under `use_reentrant=False` checkpointing the hook runs a
    # different number of saved tensors during the original forward vs
    # the recomputation, raising CheckpointError.  Memory is controlled
    # instead by EX_BATCH + length-bucketed batching.  At EX_BATCH=4,
    # L=2048, H=5120, bf16, 26-layer (38..63) backward needs ~40 GB of
    # saved activations, which fits comfortably alongside the 62 GB of
    # base weights on a 141 GB H200.
    base_model.eval()

    # ---- Phase 2: pick the (hardcoded) training layer ----
    if args.steer_layer is None:
        raise ValueError("Must pass --steer_layer (layer sweep is no "
                         "longer supported in joint training mode; "
                         "--steer_layers is ignored).")
    steer_layer = args.steer_layer
    assert 0 <= steer_layer < n_layers, (
        f"steer_layer={steer_layer} out of range [0,{n_layers})")
    print(f"\nSteer layer: {steer_layer} (model has {n_layers} layers)",
          flush=True)

    # Ignore any pre-existing "_global" entry from a previous partial run.
    per_category.pop("_global", None)

    # ---- Phase 3: JOINT training ----
    # Training cost: n_epochs * (n_train_examples / batch) forward+backward
    # passes for the n_cats category vectors simultaneously, plus the same
    # for the bias.  For 2000 examples, bs=4, 5 epochs -> 2 * 2500 = 5000
    # steps total (vs the old per-category code's 11 * 2500 = 27500).
    category_keys = sorted(
        [k for k in per_category.keys()
         if k.startswith("idx") and k[3:].isdigit()],
        key=lambda k: int(k[3:]))
    print(f"\nCategories found: {category_keys}", flush=True)
    # Data-adaptive gate:  a category must have at least
    #     SHARE * (total_positions / n_categories)
    # positions to qualify.  This self-scales with the observed
    # disagreement rate (sparse disagreements -> small gate) and the
    # category granularity (more categories -> smaller equal share).
    # We also combine with the legacy dim-aware floor
    #     ratio * hidden_size
    # (off by default) and an absolute floor --min_disagreements.
    # Effective gate: max(abs_floor, ratio*hidden, share*avg).
    total_pos = sum(len(per_category[k]) for k in category_keys)
    n_cats_raw = max(len(category_keys), 1)
    equal_share = total_pos / n_cats_raw
    share_gate = math.floor(max(args.min_category_share, 0.0) * equal_share)
    ratio_gate = math.ceil(max(args.min_disagreements_ratio, 0.0) * hidden)
    gate = max(int(args.min_disagreements), ratio_gate, share_gate)
    print(f"  gate: n_c >= {gate}", flush=True)
    print(f"    abs_floor={args.min_disagreements}   "
          f"ratio*hidden={ratio_gate} "
          f"(ratio={args.min_disagreements_ratio:g}, hidden={hidden})   "
          f"share*equal={share_gate} "
          f"(share={args.min_category_share:g}, "
          f"equal={total_pos}/{n_cats_raw}={equal_share:.1f})",
          flush=True)
    cat_sizes = [(k, len(per_category[k])) for k in category_keys]
    cat_sizes.sort(key=lambda kv: -kv[1])
    for k, n in cat_sizes:
        mark = "keep" if n >= gate else "DROP"
        print(f"    [{mark}] {k}: {n} positions", flush=True)
    active_keys = [k for k in category_keys
                   if len(per_category[k]) >= gate]
    skipped = [k for k in category_keys if k not in active_keys]
    if skipped:
        print(f"  SKIPPED (< {gate} positions): {skipped}", flush=True)
    if not active_keys:
        raise RuntimeError(
            f"All categories fell below the gate of {gate} positions; "
            "lower --min_disagreements_ratio or collect more responses.")
    key_to_cat = {k: i for i, k in enumerate(active_keys)}
    n_cats = len(active_keys)

    # ---- Response-level train / holdout split --------------------------
    # We split *responses*, not positions, so a held-out response's
    # positions never leak into training.  Seed ensures the same split
    # across cats + bias phases.
    all_ex_idx = sorted({ex for k in active_keys
                         for ex, _p, _lp, _ix in per_category[k]})
    holdout_frac = max(0.0, min(0.5, float(args.holdout_frac)))
    n_holdout = int(round(holdout_frac * len(all_ex_idx)))
    split_rng = random.Random(f"holdout-split-{args.seed}")
    shuffled = list(all_ex_idx)
    split_rng.shuffle(shuffled)
    holdout_ex = set(shuffled[:n_holdout])
    train_ex = set(shuffled[n_holdout:])
    print(f"  holdout split: {len(train_ex)} train / "
          f"{len(holdout_ex)} holdout responses "
          f"(holdout_frac={holdout_frac:g})", flush=True)

    # Flatten category records into (ex_idx, pos, cat_idx, topk_lp, topk_idx)
    joint_records: List[
        Tuple[int, int, int, torch.Tensor, torch.Tensor]] = []
    holdout_records: List[
        Tuple[int, int, int, torch.Tensor, torch.Tensor]] = []
    for k in active_keys:
        c = key_to_cat[k]
        for ex_idx, pos, tlp, tix in per_category[k]:
            rec = (ex_idx, pos, c, tlp, tix)
            if ex_idx in holdout_ex:
                holdout_records.append(rec)
            else:
                joint_records.append(rec)
    print(f"  {len(joint_records)} train positions, "
          f"{len(holdout_records)} holdout positions across "
          f"{n_cats} active category vectors", flush=True)

    # ---- Optional: compute target init magnitude from base activations.
    # We measure the mean residual-stream L2 norm at the steering layer
    # over a small probe set so that random-direction inits start at the
    # same scale as the activations they will be added to.
    init_v_norm_value: float = 0.0
    if args.init_v_from_activations:
        print("\n[Phase 2.5] Probing residual-stream norm at "
              f"layer {steer_layer} over "
              f"{args.init_probe_examples} examples...", flush=True)
        init_v_norm_value = compute_mean_activation_magnitude(
            base_model, per_example, steer_layer,
            base_tokenizer.pad_token_id,
            n_examples=args.init_probe_examples, seed=args.seed)
        print(f"  mean residual-stream norm at layer {steer_layer}: "
              f"{init_v_norm_value:.4f} -> using as init magnitude",
              flush=True)

    # ---- Auto norm cap (run unless explicitly disabled).
    # When --max_norm < 0 we probe the mean residual-stream L2 norm at
    # the steer layer and cap EVERY trained vector (bias AND each
    # category vector) at norm_cap_frac * mean.  This prevents any
    # single vector from blowing past the typical activation magnitude
    # while still letting the optimiser route signal across categories.
    # Re-use the init probe result if we already computed it.
    if args.max_norm < 0:
        if args.norm_cap_frac <= 0:
            print("\n[Phase 2.6] norm_cap_frac <= 0 -> norm cap "
                  "DISABLED (max_norm=0)", flush=True)
            args.max_norm = 0.0
        else:
            if init_v_norm_value > 0:
                mean_norm_for_cap = init_v_norm_value
                print("\n[Phase 2.6] Reusing init-probe mean norm = "
                      f"{mean_norm_for_cap:.4f} for auto norm cap",
                      flush=True)
            else:
                print("\n[Phase 2.6] Probing mean residual-stream norm "
                      f"at layer {steer_layer} over "
                      f"{args.norm_cap_probe_examples} examples for "
                      "auto norm cap...", flush=True)
                mean_norm_for_cap = compute_mean_activation_magnitude(
                    base_model, per_example, steer_layer,
                    base_tokenizer.pad_token_id,
                    n_examples=args.norm_cap_probe_examples,
                    seed=args.seed)
            args.max_norm = float(mean_norm_for_cap * args.norm_cap_frac)
            print(f"  mean_norm at layer {steer_layer} = "
                  f"{mean_norm_for_cap:.4f}; auto-cap each vector at "
                  f"{args.max_norm:.4f}  "
                  f"(= {args.norm_cap_frac:g} * mean)", flush=True)
    elif args.max_norm == 0:
        print("\n[Phase 2.6] max_norm = 0 -> norm cap DISABLED",
              flush=True)
    else:
        print(f"\n[Phase 2.6] using explicit max_norm = {args.max_norm:g}",
              flush=True)

    if args.skip_cats_phase:
        print("\n[Phase 3a] SKIPPED (--skip_cats_phase): no per-category "
              "vectors will be trained or saved this run.  Phase 3b "
              "(global bias) still runs if --train_global_bias is set.",
              flush=True)
    elif args.joint_cats_and_bias:
        print("\n[Phase 3] Joint training of category vectors + bias "
              "(single sweep)...", flush=True)
    else:
        print("\n[Phase 3a] Joint training of all category vectors...",
              flush=True)

    n_seeds = max(1, int(args.n_seeds))
    if n_seeds > 1:
        print(f"  multi-seed training: running {n_seeds} seeds "
              f"(seed = {args.seed} .. {args.seed + n_seeds - 1}); "
              f"keeping the V with lowest best-holdout KL.", flush=True)

    def _suffix(i: int) -> str:
        return f"_seed{args.seed + i}" if n_seeds > 1 else ""

    def _best_holdout_from_metrics(metrics: List[dict]) -> float:
        for rec in metrics:
            if rec.get("event") == "best_holdout_selected":
                return float(rec.get("avg_kl", float("inf")))
        last_holdout = float("inf")
        for rec in metrics:
            if (rec.get("event") == "epoch_eval"
                    and rec.get("split") == "holdout"):
                last_holdout = float(rec.get("avg_kl", float("inf")))
        return last_holdout

    seed_results: List[dict] = []
    best_seed: int = args.seed
    best_holdout_kl: float = float("inf")
    V_cat: Optional[torch.Tensor] = None
    b_joint: Optional[torch.Tensor] = None
    cats_metrics: List[dict] = []
    layer_map_partial: Dict[str, int] = {}
    model_short = args.base_model.split("/")[-1].lower()
    if args.skip_cats_phase:
        # We still need the layer_map for downstream metadata writing,
        # even though no cats are saved this run.  Leave it empty -- the
        # bias-only artefact (bias_global.pt + bias_layer.json) is what
        # the next pipeline stage will consume.
        pass
    for i in (range(n_seeds) if not args.skip_cats_phase else range(0)):
        seed_i = args.seed + i
        if n_seeds > 1:
            print(f"\n  --- seed {i+1}/{n_seeds} (seed={seed_i}) ---",
                  flush=True)
        cats_metrics_path = os.path.join(
            args.save_dir, f"training_metrics_cats{_suffix(i)}.jsonl")
        V_cat_i, b_joint_i, cats_metrics_i = train_vectors_joint(
            base_model, per_example, joint_records, n_cats,
            steer_layer=steer_layer,
            hidden_size=hidden,
            lr=args.lr,
            n_epochs=args.n_epochs,
            example_batch_size=args.example_batch_size,
            max_positions_per_example=args.max_positions_per_example,
            seed=seed_i,
            weight_decay=args.weight_decay,
            pad_token_id=base_tokenizer.pad_token_id,
            desc=("cats+bias" if args.joint_cats_and_bias else "cats"),
            metrics_path=cats_metrics_path,
            cat_key_lookup=active_keys,
            kl_mode=args.kl_mode,
            train_topk=min(args.train_topk, args.topk),
            holdout_positions_with_cat=(holdout_records if holdout_records
                                        else None),
            train_bias=bool(args.joint_cats_and_bias),
            max_norm=float(args.max_norm),
            select_best_holdout=(not args.no_select_best_holdout),
            per_example_loss=bool(args.per_example_loss),
            cap_resample_each_epoch=bool(args.cap_resample_each_epoch),
            init_v_norm=init_v_norm_value,
            bias_frozen=frozen_bias_cpu,
            checkpoint_dir=args.save_dir,
            checkpoint_prefix=f"{model_short}_cats_seed{seed_i}",
            eval_chunks_per_epoch=int(args.eval_chunks_per_epoch))
        kl_i = _best_holdout_from_metrics(cats_metrics_i)
        seed_results.append({"seed": seed_i, "best_holdout_kl": kl_i})
        print(f"  [seed {seed_i}] best_holdout_kl = {kl_i:.4f}", flush=True)
        if kl_i < best_holdout_kl:
            best_holdout_kl = kl_i
            best_seed = seed_i
            V_cat = V_cat_i
            b_joint = b_joint_i
            cats_metrics = cats_metrics_i

    if n_seeds > 1:
        print("\n  multi-seed summary:", flush=True)
        for r in seed_results:
            marker = " <-- best" if r["seed"] == best_seed else ""
            print(f"    seed={r['seed']}  "
                  f"best_holdout_kl={r['best_holdout_kl']:.4f}{marker}",
                  flush=True)
        print(f"  selected seed={best_seed}  "
              f"holdout_kl={best_holdout_kl:.4f}", flush=True)
        with open(os.path.join(args.save_dir, "seed_selection.json"), "w") as f:
            json.dump({
                "n_seeds": n_seeds,
                "selected_seed": int(best_seed),
                "selected_holdout_kl": float(best_holdout_kl),
                "init_v_norm": float(init_v_norm_value),
                "all_seeds": seed_results,
            }, f, indent=2)

    # ---- Phase 3a.1: persist cats vectors immediately so a crash or
    # interrupt during the bias phase doesn't wipe out the ~hours of
    # category training we just finished.  Files written here are the
    # final cats artefacts; the bias phase below only adds the bias
    # files, never re-touches these.
    if not args.skip_cats_phase and _is_rank_zero():
        for k in active_keys:
            c = key_to_cat[k]
            v = V_cat[c]
            out_path = os.path.join(
                args.save_dir, f"{model_short}_{k}_linear.pt")
            torch.save({k: v}, out_path)
            layer_map_partial[k] = steer_layer
            print(f"  [interim] saved {k} norm={v.norm().item():.3f}  -> "
                  f"{out_path}", flush=True)
        with open(os.path.join(args.save_dir, "layer_map.json"), "w") as f:
            json.dump(layer_map_partial, f, indent=2)
        print(f"  [interim] cats vectors safely persisted "
              f"({len(active_keys)} keys + layer_map.json)", flush=True)
    if dist.is_initialized():
        dist.barrier()

    # ---- Phase 3b: global bias (single vector on union of all pos) ----
    # Skipped entirely when --joint_cats_and_bias is used; in that mode
    # the bias was co-trained with V_cat in Phase 3 above and we just
    # promote ``b_joint`` into the bias save path below.
    bias_V = None
    bias_metrics: List[dict] = []
    if args.joint_cats_and_bias and b_joint is not None:
        bias_V = b_joint.unsqueeze(0)   # match downstream (1, hidden) shape
        print(f"  [joint-cats+bias] bias vector folded out of co-training "
              f"(norm={float(b_joint.norm().item()):.3f})", flush=True)
    elif args.train_global_bias:
        print("\n[Phase 3b] Joint training of single global bias vector...",
              flush=True)
        bias_records: List[
            Tuple[int, int, int, torch.Tensor, torch.Tensor]] = []
        bias_holdout_records: List[
            Tuple[int, int, int, torch.Tensor, torch.Tensor]] = []
        for k in active_keys:
            for ex_idx, pos, tlp, tix in per_category[k]:
                rec = (ex_idx, pos, 0, tlp, tix)
                if ex_idx in holdout_ex:
                    bias_holdout_records.append(rec)
                else:
                    bias_records.append(rec)
        print(f"  {len(bias_records)} train / "
              f"{len(bias_holdout_records)} holdout positions for bias",
              flush=True)
        bias_metrics_path = os.path.join(
            args.save_dir, "training_metrics_bias.jsonl")
        bias_V, _b_unused, bias_metrics = train_vectors_joint(
            base_model, per_example, bias_records, n_cats=1,
            steer_layer=steer_layer,
            hidden_size=hidden,
            lr=args.lr,
            n_epochs=args.n_epochs,
            example_batch_size=args.example_batch_size,
            max_positions_per_example=args.max_positions_per_example,
            seed=args.seed,
            weight_decay=args.weight_decay,
            pad_token_id=base_tokenizer.pad_token_id,
            desc="bias",
            metrics_path=bias_metrics_path,
            cat_key_lookup=["_bias"],
            kl_mode=args.kl_mode,
            train_topk=min(args.train_topk, args.topk),
            holdout_positions_with_cat=(bias_holdout_records
                                        if bias_holdout_records else None),
            max_norm=float(args.max_norm),
            checkpoint_dir=args.save_dir,
            checkpoint_prefix=f"{model_short}_bias",
            eval_chunks_per_epoch=int(args.eval_chunks_per_epoch))

    # ---- Phase 4: finalise metadata.  Cats vectors + layer_map.json are
    # already on disk from the Phase 3a.1 interim save above; we just
    # re-read them into the summary structure here so the meta JSON is
    # accurate.
    layer_map: Dict[str, int] = dict(layer_map_partial)
    saved_summary: List[dict] = []
    if not args.skip_cats_phase:
        for k in active_keys:
            c = key_to_cat[k]
            v = V_cat[c]
            out_path = os.path.join(
                args.save_dir, f"{model_short}_{k}_linear.pt")
            norm = float(v.norm().item())
            saved_summary.append({"key": k, "layer": steer_layer,
                                  "path": out_path, "norm": norm,
                                  "n_positions": len(per_category[k])})
            print(f"[{k}] norm={norm:.3f}  -> {out_path}", flush=True)

    if bias_V is not None and _is_rank_zero():
        bv = bias_V[0]
        bias_path = os.path.join(args.save_dir,
                                 f"{model_short}_bias_global.pt")
        torch.save({"bias": bv}, bias_path)
        with open(os.path.join(args.save_dir, "bias_layer.json"), "w") as f:
            json.dump({"layer": steer_layer,
                       "norm": float(bv.norm().item())}, f, indent=2)
        print(f"[_global] norm={bv.norm().item():.3f}  -> {bias_path}",
              flush=True)
    if dist.is_initialized():
        dist.barrier()

    if not _is_rank_zero():
        # Non-rank-0 ranks have done their share of the training/eval
        # work; rank 0 owns saving the meta file.
        if dist.is_initialized():
            dist.barrier()
            _ddp_cleanup()
        return
    meta_path = os.path.join(
        args.save_dir, f"{model_short}_correction_meta.json")
    with open(meta_path, "w") as f:
        json.dump({"base_model": args.base_model,
                   "thinking_model": args.thinking_model,
                   "thinking_model_short": args.thinking_model_short,
                   "steer_layer": steer_layer,
                   "training_mode": ("joint-cats-and-bias"
                                     if args.joint_cats_and_bias
                                     else "joint-multi-vector"),
                   "training_objective":
                       f"hard-gate-{args.kl_mode}-KL"
                       f"(train_topk={min(args.train_topk, args.topk)}"
                       f"{', cats+bias-cotrained' if args.joint_cats_and_bias else ''})",
                   "kl_mode": args.kl_mode,
                   "train_topk": int(min(args.train_topk, args.topk)),
                   "topk": args.topk,
                   "hook_type": "forward_hook (post), multi-vector",
                   "n_responses_used": len(per_example),
                   "n_train_responses": int(len(train_ex)),
                   "n_holdout_responses": int(len(holdout_ex)),
                   "holdout_frac": float(holdout_frac),
                   "layer_map": layer_map,
                   "skipped_keys": skipped,
                   "active_keys": active_keys,
                   "saved": saved_summary,
                   "has_bias": bias_V is not None,
                   "args": vars(args)}, f, indent=2)
    print(f"\nSaved {len(saved_summary)} category vectors "
          f"+ {'1 bias' if bias_V is not None else 'no bias'}, "
          f"skipped {len(skipped)}: {skipped}")
    print(f"Metadata at {meta_path}")
    if dist.is_initialized():
        dist.barrier()
        _ddp_cleanup()


if __name__ == "__main__":
    main()
