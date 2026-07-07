"""Staged CE-only correction-vector training pipeline.

Pipeline (single invocation per (base, thinking) model pair):

  Phase A.  Collect disagreements over --n_mmlu_examples annotated
            MMLU-Pro responses.  Each retained position satisfies
                argmax(base[i])  !=  rollout_token[i+1]
            and is labelled with the SAE category obtained by running
            the thinking-model activation at --sae_layer through the
            SAE for the (model, sae_layer, sae_n_clusters) triple.
            (This is the exact same labelling used at inference time
            by hybrid_eval.py.)

  Phase B.  Train BIAS vector b with full-vocab top-1 cross-entropy on
            ALL collected disagreement positions:
                loss(p)  =  - log softmax(base + b)[rollout_token(p)]
            One vector, optimised over the full disagreement set.

  Phase C.  Compute the residual disagreement subset by applying
            (base + b) at every disagreement position in one forward
            pass and dropping positions where the steered argmax now
            equals the rollout token.  Strict subset of Phase A.

  Phase D.  Train per-category CAT vectors V[c] with full-vocab top-1
            cross-entropy on the residual subset, with the (frozen) bias
            b hooked in as a static offset at the same layer:
                loss(p) = - log softmax(base + b + V[cat(p)])[rollout_token(p)]
            Per-cat balanced loss: each minibatch computes the mean loss
            within each present category, then averages across present
            categories.  Cats with more positions do NOT dominate the
            global gradient direction.

Saved artefacts (compatible with hybrid_eval.py's loaders out of the
box):

  {save_dir}/{base_short}_bias_linear.pt    {"bias": (hidden,) fp32 tensor}
  {save_dir}/{base_short}_idx{C}_linear.pt  {"idx{C}": (hidden,) fp32 tensor}
  {save_dir}/layer_map.json                 {"idx{C}": steer_layer, ...}
  {save_dir}/bias_layer.json                {"layer": steer_layer, "norm": ...}
  {save_dir}/training_meta.json             full args + summary

Inference (hybrid_eval.py):
  --old_vectors_dir {save_dir} --old_vectors_layer {steer_layer}
  --bias_vector_path {save_dir}/{base_short}_bias_linear.pt
  --pg_bias_cat_sweep
  --coef_select think_top1
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

import numpy as np
import dotenv

dotenv.load_dotenv("../.env")

# FSDP helpers (lazy import for non-distributed runs)
try:
    from fsdp_utils import (
        init_distributed, get_rank, get_world_size, is_main,
        sync_gradients,
    )
except ImportError:
    pass

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import utils  # noqa: E402,F401  (forces utils package to load for side-effects)
from utils.responses import extract_thinking_process  # noqa: E402

# Patch transformers to tolerate safetensors files with metadata=None
# (e.g. Open-Reasoner-Zero-32B whose shards lack a "format" key).
import transformers.modeling_utils as _tmu
_orig_load_state_dict = _tmu.load_state_dict

def _patched_load_state_dict(checkpoint_file, is_quantized=False, map_location="cpu", weights_only=True):
    from safetensors import safe_open as _safe_open
    if str(checkpoint_file).endswith(".safetensors"):
        with _safe_open(checkpoint_file, framework="pt") as _f:
            meta = _f.metadata()
        if meta is None:
            with _safe_open(checkpoint_file, framework="pt") as _f:
                return {k: _f.get_tensor(k) for k in _f.keys()}
    return _orig_load_state_dict(checkpoint_file, is_quantized=is_quantized,
                                  map_location=map_location, weights_only=weights_only)

_tmu.load_state_dict = _patched_load_state_dict


# ---------------------------------------------------------------------------
# Tokenizer alignment between base and thinking-model rollouts
# ---------------------------------------------------------------------------

def _aligned_tokenize_pair(
    base_tokenizer, base_prompt: str,
    think_tokenizer, think_prompt: str,
    thinking: str,
):
    """Tokenize ``{base,think}_prompt + thinking`` on both sides and find a
    shared character anchor inside ``thinking``.

    The thinking text past the anchor produces identical token sequences
    on both sides (same tokenizer family, deterministic BPE), so the base
    position ``i`` lines up with thinking position ``i + (t_anchor -
    b_anchor)``.  Returns ``b_anchor = t_anchor = -1`` if no shared
    boundary is found.
    """
    base_full = base_prompt + thinking
    think_full = think_prompt + thinking
    enc_b = base_tokenizer(base_full, return_offsets_mapping=True,
                           truncation=False)
    enc_t = think_tokenizer(think_full, return_offsets_mapping=True,
                            truncation=False)
    bp = len(base_prompt)
    tp = len(think_prompt)
    b_ends: Dict[int, int] = {}
    for i, (s, e) in enumerate(enc_b["offset_mapping"]):
        if s >= bp and e > bp:
            b_ends.setdefault(e - bp, i)
    t_ends: Dict[int, int] = {}
    for i, (s, e) in enumerate(enc_t["offset_mapping"]):
        if s >= tp and e > tp:
            t_ends.setdefault(e - tp, i)
    common = set(b_ends.keys()) & set(t_ends.keys())
    if not common:
        return {"b_ids": enc_b["input_ids"], "t_ids": enc_t["input_ids"],
                "b_anchor": -1, "t_anchor": -1}
    anchor_c = min(common)
    return {"b_ids": enc_b["input_ids"],
            "t_ids": enc_t["input_ids"],
            "b_anchor": b_ends[anchor_c] + 1,
            "t_anchor": t_ends[anchor_c] + 1}


def _build_base_prompt(
    question: str,
    style: str = "default",
    *,
    thinking_tokenizer=None,
    think_family: str = "other",
    is_math_question: bool = False,
    math_directive_enabled: bool = False,
) -> str:
    """Construct the base-model prompt.

    - ``default``: bare ``"User: {q}\\nAssistant:"`` -- no directive.
    - ``stepwise`` (final_final v1): prepends "Answer the following
      question. Respond step by step." inside the user turn.
    - ``boxed`` (final_final v2): places QwQ/R1+ORZ \\boxed{} directive
      AFTER the question.
    - ``legacy_task`` (final_final v3): the structured Task/Question/
      Answer scaffolding from origin/main's hybrid_*.py +
      optimize_steering_vectors.py -- ``"Task: Answer the question
      below. Explain your reasoning step by step.\\n\\n\\n\\nQuestion:
      \\n{q}\\n\\nStep by step answer:\\n"``.
    - ``think_template``: apply the thinking model's chat template to
      the family-shaped user content (ORZ Table-5 / R1+QwQ math
      directive).  The resulting string IS the base prompt -- base
      receives exactly what the think model receives.  Requires
      ``thinking_tokenizer``; the family/is_math/directive flags
      mirror the think-side shaping.
    - ``simple``: minimal "User: {q}\\nAssistant: <think>\\n" --
      no chat-template, no family shaping, no math directive.  The
      opening ``<think>`` tag invites the base model into
      thinking-style completion.

    MUST match ``--base_prompt_style`` in ``hybrid_eval.py`` and
    ``vllm-serve/generate_rollouts.py``.
    """
    if style == "stepwise":
        return ("User: Answer the following question. Respond step "
                f"by step.\n\n{question}\nAssistant:")
    if style == "boxed":
        return (f"User: {question}\n\nPlease reason step by step, "
                "and put your final answer within "
                "\\boxed{}.\nAssistant:")
    if style == "legacy_task":
        return ("Task: Answer the question below. Explain your "
                "reasoning step by step.\n\n\n\nQuestion:\n"
                f"{question}\n\nStep by step answer:\n")
    if style == "simple":
        return f"User: {question}\nAssistant: <think>\n"
    if style == "qa_response":
        return f"Question: {question}\nResponse: "
    if style == "qa_instr":
        return f"Answer the following question:\nQ: {question}\nA:"
    if style == "think_qa":
        return (
            "Your task is to answer the following question. First, "
            "carefully think through the question and then provide "
            "your final answer.\n"
            f"Q: {question}\nA:"
        )
    if style == "think_qa_marker":
        return (
            "Your task is to answer the following question. First, "
            "carefully think through the question and then provide "
            "your final answer.\n"
            f"Q: {question}\nThink:"
        )
    if style == "think_qa_think":
        return (
            "Your task is to answer the following question. First, "
            "carefully think through the question and then provide "
            "your final answer.\n"
            f"Q: {question}\nA: Think:"
        )
    if style == "think_word":
        return f"User: {question}\nAssistant: think:\n"
    if style == "convo_marker":
        return (
            'A conversation between User and Assistant. The User asks a '
            'question, and the Assistant solves it. The Assistant '
            'reasons step by step following the "think" marker, and when '
            'done provides their final answer after the "answer" marker.'
            f'\n\nUser: {question}\n\nA:\nthink:\n'
        )
    if style == "convo_marker_v2":
        return (
            'A conversation between User and Assistant. The User asks a '
            'question, and the Assistant solves it. The Assistant '
            'reasons step by step following the "think" marker, and when '
            'done provides their final answer after the "answer" marker.'
            f'\nUser: {question}\nAssistant:\nthink:\n'
        )
    if style == "convo_continue":
        return (
            'A conversation between user and assistant. User asks a '
            'question, assistant responds.'
            f'\n\nUser:\n{question}\n\nAssistant:\n'
        )
    if style == "convo_reason":
        return (
            'A conversation between User and Assistant. User asks a '
            'question and Assistant reasons through it step by step '
            'until figuring out the correct answer.'
            f'\n\nUser question: {question}\n\nAssistant reasoning: '
        )
    if style == "orz_think_template":
        return (
            'A conversation between User and Assistant. The User asks a '
            'question, and the Assistant solves it. The Assistant first '
            'thinks about the reasoning process in the mind and then '
            'provides the User with the answer. The reasoning process is '
            'enclosed within <think> </think> and answer is enclosed '
            'within <answer> </answer> tags, respectively, i.e., <think> '
            'reasoning process here </think> <answer> answer here '
            '</answer>. User: You must put your answer inside <answer> '
            '</answer> tags, i.e., <answer> answer here </answer>. And '
            'your final answer will be extracted automatically by the '
            '\\boxed{} tag.\n' + question +
            '\n\nPlease reason step by step, and put your final answer '
            'within \\boxed{}.\nAssistant: <think>'
        )
    if style == "r1_think_template":
        return (
            '<｜User｜>' + question +
            '\n\nPlease reason step by step, and put your final answer '
            'within \\boxed{}.<｜Assistant｜><think>\n'
        )
    if style == "qwq_think_template":
        return (
            '<|im_start|>user\n' + question +
            '\n\nPlease reason step by step, and put your final answer '
            'within \\boxed{}.<|im_end|>\n<|im_start|>assistant\n<think>\n'
        )
    if style == "plain_chat_math":
        return (
            f'user\n{question}\nPlease reason step by step, and put '
            f'your final answer within \\boxed{{}}.\n\nassistant\n'
            f'<think>\n'
        )
    if style == "convo_think":
        return (
            'A conversation between a User and Assistant. The User asks '
            'a question, and the Assistant solves it. The Assistant '
            'first thinks about the reasoning process in the mind and '
            'then provides the User with the answer. The reasoning '
            'process is enclosed within <think> </think> followed by '
            'the answer.'
            f'\nUser: \n{question}\nAssistant: <think>\n'
        )
    if style == "mini_preamble":
        return (
            'User asks a question. Assistant solves it by thinking '
            'through the question.'
            f'\n\nUser: {question}\nAssistant: <think>\n'
        )
    if style == "orz_full":
        return (
            'A conversation between User and Assistant. The User asks '
            'a question, and the Assistant solves it. The Assistant '
            'first thinks about the reasoning process in the mind and '
            'then provides the User with the answer. The reasoning '
            'process is enclosed within <think> </think> and answer '
            'is enclosed within <answer> </answer> tags, respectively, '
            'i.e., <think> reasoning process here </think> <answer> '
            'answer here </answer>. User: You must put your answer '
            'inside <answer> </answer> tags, i.e., <answer> answer '
            'here </answer>. And your final answer will be extracted '
            'automatically by the \\boxed{} tag.'
            f'\n{question}\nAssistant: <think>\n'
        )
    if style == "r1_plain":
        return (
            f'User:\n{question}\nPlease reason step by step, and put '
            f'your final answer within \\boxed{{}}.\n\nAssistant:\n'
            f'<think>\n'
        )
    if style == "qwq_plain":
        return (
            f'User: {question}\n\nPlease reason step by step, and put '
            f'your final answer within \\boxed{{}}. \n\nAssistant: '
            f'<think>\n'
        )
    if style == "step_preamble":
        return (
            f'Please reason step by step\n\n'
            f'User: {question}\nAssistant:<think>'
        )
    if style == "think_template":
        if thinking_tokenizer is None:
            raise ValueError(
                "base_prompt_style=think_template requires "
                "thinking_tokenizer.")
        shaped = _shape_user_content(
            question, think_family,
            is_math_question=is_math_question,
            math_directive_enabled=math_directive_enabled,
        )
        return thinking_tokenizer.apply_chat_template(
            [{"role": "user", "content": shaped}],
            tokenize=False, add_generation_prompt=True)
    return f"User: {question}\nAssistant:"


# ---------------------------------------------------------------------------
# Model-family-aware user-content shaping for the thinking model.
# Kept in sync with vllm-serve/generate_rollouts.py and hybrid/hybrid_eval.py.
# Disagreement extraction teacher-forces (prompt + response) tokens; the
# prompt MUST match what was sent at generation time, otherwise activations
# are collected from a prompt the model never saw.
# ---------------------------------------------------------------------------
ORZ_USER_PREFIX = (
    "You must put your answer inside <answer> </answer> tags, i.e., "
    "<answer> answer here </answer>. And your final answer will be "
    "extracted automatically by the \\boxed{} tag."
)
MATH_DIRECTIVE = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)
TRAINMIX_MATH_SOURCES = {"hendrycks_math", "natural_reasoning"}
_OOS_MATH_DATASETS = {"math500", "gsm8k", "aime24", "aime25"}


def _detect_think_family(model_id: str) -> str:
    low = (model_id or "").lower()
    if "open-reasoner-zero" in low or "orz" in low:
        return "orz"
    if "r1-distill" in low or "deepseek-r1" in low:
        return "r1"
    if "qwq" in low:
        return "qwq"
    return "other"


def _shape_user_content(question: str, family: str, *,
                        is_math_question: bool,
                        math_directive_enabled: bool) -> str:
    """Pre-wrap the user content sent to the thinking model.

    MUST match the prompt shaping used at rollout-generation time in
    ``vllm-serve/generate_rollouts.py`` so that the SAE classification
    and disagreement-target tokens reflect the same prompt distribution
    that the cached think rollouts were sampled from.

    - ``orz`` : prepend the Table-5 user instruction; ALSO append the
      math directive when ``is_math_question`` (and
      ``math_directive_enabled``).  Matches generate_rollouts.py's ORZ
      math branch.
    - ``r1`` / ``qwq`` : append the math directive iff
      ``is_math_question`` and ``math_directive_enabled``.
    - other : unchanged.
    """
    if family == "orz":
        content = f"{ORZ_USER_PREFIX}\n{question}"
        if is_math_question and math_directive_enabled:
            content = f"{content}\n\n{MATH_DIRECTIVE}"
        return content
    if family in ("r1", "qwq") and is_math_question and math_directive_enabled:
        return f"{question}\n\n{MATH_DIRECTIVE}"
    return question


def _row_is_math(entry: dict, math_directive_mode: str) -> bool:
    """Decide whether an in-memory disagreement-collection entry should
    receive the R1/QwQ math directive.

    Modes:
      'none'   : never.
      'always' : every row.
      'auto'   : if dataset_name is a known math benchmark
                  OR if dataset_name == 'trainmix' and the row's source
                  is in TRAINMIX_MATH_SOURCES.
    """
    if math_directive_mode == "none":
        return False
    if math_directive_mode == "always":
        return True
    if math_directive_mode != "auto":
        raise ValueError(f"Unknown math directive mode: {math_directive_mode!r}")
    ds = entry.get("dataset_name", "")
    if ds in _OOS_MATH_DATASETS:
        return True
    if ds == "trainmix":
        src = entry.get("source")
        if src and src in TRAINMIX_MATH_SOURCES:
            return True
    return False


# ---------------------------------------------------------------------------
# Out-of-distribution response loaders (math500, gsm8k).  We pair the public
# dataset's question text with the already-cached thinking-model rollout in
# hybrid/results/response_cache/thinking_<short>_<ds>_temp0_max2000.jsonl
# so we don't have to regenerate text.  Returned dicts mimic the MMLU-Pro
# response format consumed by collect_disagreements().
# ---------------------------------------------------------------------------

def _rollout_filename(thinking_model_short: str, dataset_name: str,
                      temp_label: str = "0",
                      max_tokens: int = 2000,
                      sample_idx: int = -1) -> str:
    """Build the canonical thinking-rollout cache filename.

    Examples:
        thinking_<short>_trainmix_temp0_max2000.jsonl              (legacy greedy)
        thinking_<short>_math500_temp0.6_max2048_s0.jsonl          (final, sample 0)
    """
    s = f"_s{sample_idx}" if sample_idx >= 0 else ""
    return (f"thinking_{thinking_model_short}_{dataset_name}"
            f"_temp{temp_label}_max{max_tokens}{s}.jsonl")


def _truncate_to_last_boxed(text: str) -> Optional[str]:
    """Truncate ``text`` to end immediately after the LAST ``\\boxed{...}``
    expression in the string.

    Uses brace matching so nested braces (e.g. ``\\boxed{\\frac{1}{2}}``)
    are handled correctly.  Returns ``None`` when no well-formed
    ``\\boxed{}`` is found (caller should drop such examples when running
    in --truncate_answer_box mode, so we don't train on garbage).
    """
    last_start = text.rfind("\\boxed{")
    if last_start < 0:
        return None
    i = last_start + len("\\boxed{")
    depth = 1
    n = len(text)
    while i < n and depth > 0:
        c = text[i]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
        i += 1
    if depth != 0:
        # Unbalanced -- treat as no boxed answer.
        return None
    return text[:i]


def _apply_truncate_answer_box(out: List[dict], *, dataset_label: str) -> List[dict]:
    """Apply boxed-answer truncation to every record's ``full_response``
    in-place; drop records that have no well-formed ``\\boxed{...}``.

    Logs counts so the caller can see how many examples survived.
    """
    kept: List[dict] = []
    n_no_box = 0
    n_truncated = 0
    n_total_chars_before = 0
    n_total_chars_after = 0
    for rec in out:
        original = rec.get("full_response", "") or ""
        n_total_chars_before += len(original)
        truncated = _truncate_to_last_boxed(original)
        if truncated is None:
            n_no_box += 1
            continue
        if len(truncated) < len(original):
            n_truncated += 1
        rec["full_response"] = truncated
        n_total_chars_after += len(truncated)
        kept.append(rec)
    n_in = len(out)
    n_out = len(kept)
    avg_before = (n_total_chars_before / max(n_in, 1))
    avg_after = (n_total_chars_after / max(n_out, 1))
    print(
        f"  [truncate_answer_box:{dataset_label}] kept {n_out}/{n_in} "
        f"(dropped {n_no_box} with no \\boxed{{}}); "
        f"truncated {n_truncated} responses; "
        f"avg chars {avg_before:.0f} -> {avg_after:.0f}",
        flush=True,
    )
    return kept


def _load_oos_responses(dataset_name: str, n_take: int,
                        thinking_model_short: str,
                        cache_root: str,
                        *,
                        thinking_model=None,
                        thinking_tokenizer=None,
                        max_new_tokens: int = 1024,
                        rollouts_temp_label: str = "0",
                        rollouts_max_tokens: int = 2000,
                        rollouts_sample_idx: int = -1,
                        think_family: str = "other",
                        math_directive_mode: str = "none",
                        truncate_answer_box: bool = False) -> List[dict]:
    """Return up to ``n_take`` ``{original_message, full_response,
    annotated_thinking}`` dicts for an out-of-distribution dataset.

    Pulls thinking rollouts from
    hybrid/results/response_cache(_final)/thinking_<short>_<ds>_temp<L>_max<T>[_s<N>].jsonl
    when available.  If the cache is missing or short, falls back to
    on-the-fly greedy generation using ``thinking_model`` /
    ``thinking_tokenizer`` (so the OOS holdout can be evaluated even
    when no precomputed rollouts exist).  Freshly generated rollouts
    are persisted back to the cache for reuse on subsequent runs.
    """
    from datasets import load_dataset

    cache_path = os.path.join(
        cache_root,
        _rollout_filename(thinking_model_short, dataset_name,
                          temp_label=rollouts_temp_label,
                          max_tokens=rollouts_max_tokens,
                          sample_idx=rollouts_sample_idx))
    cache_by_idx: Dict[int, str] = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            for line in f:
                try:
                    r = json.loads(line)
                except json.JSONDecodeError:
                    continue
                cache_by_idx[int(r["dataset_idx"])] = r["response"]
        print(f"  [oos:{dataset_name}] cache hits: "
              f"{len(cache_by_idx)} idx-keyed rollouts", flush=True)

    if dataset_name == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main")["test"]
        q_key = "question"
    elif dataset_name == "math500":
        ds = load_dataset("HuggingFaceH4/MATH-500")["test"]
        q_key = "problem"
    elif dataset_name == "hendrycks_math":
        # Hendrycks MATH benchmark train split (12000 problems).
        # NOTE: this dataset's `test` split equals MATH-500 and must NOT
        # be used as training data.  We hardcode `train` here so callers
        # can't accidentally point at the leaking split.
        ds = load_dataset("nlile/hendrycks-MATH-benchmark")["train"]
        q_key = "problem"
    elif dataset_name == "mmlu_auxiliary_train":
        # MMLU's auxiliary_train split: 99,842 MCQA questions drawn from
        # ARC, MC_TEST, OBQA, RACE. Disjoint from the MMLU test set (and
        # so disjoint from MMLU-Pro questions), and disjoint from
        # MATH500 / GSM8K -- safe to use as a TRAINING source whose
        # distribution mirrors MMLU but is ~8x larger.
        ds = load_dataset("cais/mmlu", "all", split="auxiliary_train")
        q_key = "question"
    else:
        raise ValueError(f"Unsupported OOS dataset: {dataset_name}")

    out: List[dict] = []
    need_gen: List[int] = []
    for i in range(len(ds)):
        if len(out) + len(need_gen) >= n_take:
            break
        if i in cache_by_idx and cache_by_idx[i] and cache_by_idx[i].strip():
            out.append({
                "original_message": {"role": "user",
                                     "content": ds[i][q_key]},
                "full_response": cache_by_idx[i],
                "annotated_thinking": {"_oos": dataset_name},
                "dataset_name": dataset_name,
                "question_id": i,
            })
        else:
            need_gen.append(i)

    if need_gen and thinking_model is not None and thinking_tokenizer is not None:
        os.makedirs(cache_root, exist_ok=True)
        print(f"  [oos:{dataset_name}] generating {len(need_gen)} "
              f"rollouts (cache miss)...", flush=True)
        # Shape the OOS-fallback prompt the same way generate_rollouts.py
        # would shape it at gen-time.  For OOS datasets the row's
        # ``source`` defaults to dataset_name, so _row_is_math() picks up
        # math500 / gsm8k / aime via the _OOS_MATH_DATASETS set.
        _fallback_is_math = dataset_name in _OOS_MATH_DATASETS and \
            math_directive_mode != "none"
        device = next(thinking_model.parameters()).device
        with open(cache_path, "a") as cache_f:
            for idx_pos, didx in enumerate(need_gen):
                q = ds[didx][q_key]
                q_shaped = _shape_user_content(
                    q, think_family,
                    is_math_question=_fallback_is_math,
                    math_directive_enabled=(math_directive_mode != "none"))
                try:
                    prompt_text = thinking_tokenizer.apply_chat_template(
                        [{"role": "user", "content": q_shaped}],
                        tokenize=False, add_generation_prompt=True)
                except Exception:
                    prompt_text = f"User: {q}\nAssistant:"
                enc = thinking_tokenizer(prompt_text, return_tensors="pt",
                                         truncation=True, max_length=1024)
                input_ids = enc["input_ids"].to(device)
                attn = enc["attention_mask"].to(device)
                with torch.no_grad():
                    gen = thinking_model.generate(
                        input_ids=input_ids, attention_mask=attn,
                        max_new_tokens=max_new_tokens,
                        do_sample=False, temperature=1.0,
                        pad_token_id=(thinking_tokenizer.pad_token_id
                                       or thinking_tokenizer.eos_token_id))
                new_ids = gen[0, input_ids.shape[1]:]
                resp = thinking_tokenizer.decode(
                    new_ids, skip_special_tokens=True)
                cache_f.write(json.dumps({
                    "dataset_idx": int(didx),
                    "response": resp,
                    "n_tokens": int(new_ids.shape[0]),
                    "eos": True,
                }) + "\n")
                cache_f.flush()
                out.append({
                    "original_message": {"role": "user", "content": q},
                    "full_response": resp,
                    "annotated_thinking": {"_oos": dataset_name},
                    "dataset_name": dataset_name,
                    "question_id": int(didx),
                })
                if (idx_pos + 1) % 10 == 0:
                    print(f"    [oos:{dataset_name}] generated "
                          f"{idx_pos+1}/{len(need_gen)}", flush=True)

    print(f"  [oos:{dataset_name}] loaded {len(out)}/{n_take} responses "
          f"(cache={cache_path})", flush=True)
    if truncate_answer_box:
        out = _apply_truncate_answer_box(out, dataset_label=f"oos:{dataset_name}")
    return out


# ---------------------------------------------------------------------------
# Residual-stream norm calibration for the norm-cap (alpha * ||h_resid||).
# Used to constrain ||bias|| and each ||cat_k|| at training time so that the
# (bias + max cat) perturbation stays in the regime where steering helps
# rather than overpowers the residual stream at smaller model sizes.
# ---------------------------------------------------------------------------

@torch.no_grad()
def calibrate_h_norm(base_model, per_example: List[dict],
                     records: List[Tuple[int, int, int]],
                     *,
                     steer_layer: int,
                     pad_token_id: int,
                     n_samples: int = 256,
                     seed: int = 0) -> float:
    """Compute median ||h_t||_2 at ``steer_layer`` over a sample of the
    disagreement positions that the bias / cat training will perturb.

    Reuses the loaded base_model and per_example token caches -- no
    additional model loading required.  Returns a scalar float used as
    ``h_bar`` reference for the norm-cap projection.
    """
    if not records:
        return float("nan")
    device = next(base_model.parameters()).device
    rng = random.Random(seed)
    sample = rng.sample(records, min(n_samples, len(records)))

    # Group by example so we forward each example at most once.
    by_ex: Dict[int, List[int]] = defaultdict(list)
    for r in sample:
        ex_idx = int(r[0])
        pos = int(r[1])
        by_ex[ex_idx].append(pos)

    captured: Dict[str, Optional[torch.Tensor]] = {"h": None}

    def _hook(_mod, _inp, output):
        h = output[0] if isinstance(output, tuple) else output
        captured["h"] = h.detach().float()

    handle = base_model.model.layers[steer_layer].register_forward_hook(_hook)
    norms: List[float] = []
    try:
        for ex_idx, positions in by_ex.items():
            ids = per_example[ex_idx]["ids"].to(device).unsqueeze(0)
            _ = base_model(ids)
            h = captured["h"][0]  # (T, D)
            for p in positions:
                if 0 <= p < h.shape[0]:
                    norms.append(h[p].norm().item())
    finally:
        handle.remove()
        captured["h"] = None

    if not norms:
        return float("nan")
    return float(torch.tensor(norms).median().item())


# ---------------------------------------------------------------------------
# Forward-hook utilities: inject (a) a per-position cat vector, optionally
# with frozen bias added, or (b) a uniform bias at every position.
# ---------------------------------------------------------------------------

class _BiasHook:
    """Add a single fixed vector at ALL positions of the residual stream.
    Used at training time as a frozen offset under cat training, and as a
    learnable parameter under bias training (just pass the .grad-tracked
    tensor)."""

    def __init__(self, bias: torch.Tensor):
        self.bias = bias  # (hidden,)

    def __call__(self, _module, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        shifted = h + self.bias.to(h.device, h.dtype).view(1, 1, -1)
        return (shifted,) + out[1:] if isinstance(out, tuple) else shifted


class _CatHook:
    """Add ``V[cat(p)] + bias_frozen`` at each of N selected (b, t)
    positions.  ``V`` is a (n_cats, hidden) trainable matrix; gradients at
    each selected position only flow into V[cat(p)] (other rows receive
    no gradient), so all cat vectors can be trained jointly in one
    forward/backward pass.
    """

    def __init__(self, V: torch.Tensor,
                 pos_bids: torch.Tensor,
                 pos_tids: torch.Tensor,
                 pos_cats: torch.Tensor,
                 bias_frozen: Optional[torch.Tensor] = None):
        self.V = V                      # (n_cats, hidden) trainable
        self.pos_bids = pos_bids        # (N,) long
        self.pos_tids = pos_tids        # (N,) long
        self.pos_cats = pos_cats        # (N,) long in [0, n_cats)
        self.bias_frozen = bias_frozen  # (hidden,) detached, or None

    def __call__(self, _module, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h_dev = h.device
        update = torch.zeros_like(h)
        cats = self.pos_cats.to(h_dev)
        per_pos = self.V.to(h_dev)[cats]                       # (N, hidden)
        if self.bias_frozen is not None:
            per_pos = per_pos + self.bias_frozen.to(h_dev).unsqueeze(0)
        bids = self.pos_bids.to(h_dev)
        tids = self.pos_tids.to(h_dev)
        update[bids, tids, :] = per_pos.to(h.dtype)
        shifted = h + update
        return (shifted,) + out[1:] if isinstance(out, tuple) else shifted


class _CatMLPHook:
    """Add ``alpha(h, cat) * V[cat]`` at each of N selected (b, t)
    positions, where alpha is predicted by a CatCoefMLP from the
    un-shifted residual stream.  Both V and the MLP are trainable.
    """

    def __init__(self, V: torch.Tensor, mlp,
                 pos_bids: torch.Tensor, pos_tids: torch.Tensor,
                 pos_cats: torch.Tensor):
        self.V = V                      # (n_cats, hidden) trainable
        self.mlp = mlp                  # CatCoefMLP, trainable
        self.pos_bids = pos_bids        # (N,) long
        self.pos_tids = pos_tids        # (N,) long
        self.pos_cats = pos_cats        # (N,) long in [0, n_cats)
        self.last_alphas = None         # cached for logging

    def __call__(self, _module, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        h_dev = h.device
        bids = self.pos_bids.to(h_dev)
        tids = self.pos_tids.to(h_dev)
        cats = self.pos_cats.to(h_dev)
        h_at_pos = h[bids, tids, :].float()
        mlp_dev = next(self.mlp.parameters()).device
        alpha = self.mlp(h_at_pos.to(mlp_dev), cats.to(mlp_dev))  # (N,)
        self.last_alphas = alpha.detach()
        v_per_pos = self.V.to(mlp_dev)[cats.to(mlp_dev)]  # (N, hidden) on mlp_dev
        shift = alpha.unsqueeze(-1) * v_per_pos            # (N, hidden) on mlp_dev
        update = torch.zeros_like(h)
        update[bids, tids, :] = shift.to(h_dev).to(h.dtype)
        shifted = h + update
        return (shifted,) + out[1:] if isinstance(out, tuple) else shifted


@contextmanager
def _hook_at(model, layer_idx: int, hook):
    try:
        target = model.model.layers[layer_idx]
    except AttributeError:
        target = model.module.model.layers[layer_idx]
    h = target.register_forward_hook(hook)
    try:
        yield
    finally:
        h.remove()


# ---------------------------------------------------------------------------
# Phase A: collect disagreement positions with per-position SAE labelling
# ---------------------------------------------------------------------------

@torch.no_grad()
def collect_disagreements(
    base_model,
    thinking_model,
    base_tokenizer,
    thinking_tokenizer,
    responses: List[dict],
    *,
    max_seq_len: int,
    max_examples: int,
    sae_classifier,
    sae_classify_layer: int,
    collect_batch_size: int = 8,
    think_family: str = "other",
    math_directive_mode: str = "none",
    base_prompt_style: str = "default",
) -> Tuple[List[dict], Dict[str, List[Tuple[int, int, int]]]]:
    """Collect ``(ex_idx, base_pos, target_token)`` records for every
    position where ``argmax(base[i]) != rollout_token[i+1]``, labelling
    each with the SAE category at the corresponding thinking position.

    Returns:
      per_example[i]   = {"ids": LongTensor(L,), "prompt_len": int}
      per_category[c]  = list of (ex_idx, base_pos, target_token_id)
    """
    per_example: List[dict] = []
    per_category: Dict[str, List[Tuple[int, int, int]]] = defaultdict(list)
    per_category_acts: Dict[str, List[float]] = defaultdict(list)

    base_device = next(base_model.parameters()).device
    think_device = next(thinking_model.parameters()).device

    # Capture thinking-model activations at sae_classify_layer.
    _sae_state: Dict[str, Optional[torch.Tensor]] = {"acts": None}

    def _capture(_mod, _inp, out):
        h = out[0] if isinstance(out, tuple) else out
        _sae_state["acts"] = h.detach()

    try:
        _target_layer = thinking_model.model.layers[sae_classify_layer]
    except Exception:
        _target_layer = thinking_model.module.model.layers[sae_classify_layer]
    _sae_handle = _target_layer.register_forward_hook(_capture)

    it = responses[:max_examples] if max_examples else responses
    pad_id = (base_tokenizer.pad_token_id
              if base_tokenizer.pad_token_id is not None else 0)
    think_pad_id = (thinking_tokenizer.pad_token_id
                    if thinking_tokenizer.pad_token_id is not None else pad_id)

    pending: List[dict] = []

    def _flush():
        if not pending:
            return
        B = len(pending)
        Lb_max = max(it_p["base_ids"].shape[0] for it_p in pending)
        Lt_max = max(it_p["think_ids"].shape[0] for it_p in pending)

        base_ids_pad = torch.full((B, Lb_max), pad_id, dtype=torch.long)
        base_attn = torch.zeros((B, Lb_max), dtype=torch.long)
        think_ids_pad = torch.full((B, Lt_max), think_pad_id, dtype=torch.long)
        think_attn = torch.zeros((B, Lt_max), dtype=torch.long)
        for bi, p_it in enumerate(pending):
            Lb_p = p_it["base_ids"].shape[0]
            Lt_p = p_it["think_ids"].shape[0]
            base_ids_pad[bi, :Lb_p] = p_it["base_ids"]
            base_attn[bi, :Lb_p] = 1
            think_ids_pad[bi, :Lt_p] = p_it["think_ids"]
            think_attn[bi, :Lt_p] = 1

        out_b = base_model(
            input_ids=base_ids_pad.to(base_device),
            attention_mask=base_attn.to(base_device),
            use_cache=False)
        pred_b_batch = out_b.logits.argmax(dim=-1).cpu()  # (B, Lb_max)
        del out_b

        thinking_model(
            input_ids=think_ids_pad.to(think_device),
            attention_mask=think_attn.to(think_device),
            use_cache=False)
        sae_acts_batch = _sae_state["acts"]
        _sae_state["acts"] = None
        if sae_acts_batch is None:
            raise RuntimeError("SAE classify hook did not fire on thinking.")

        for bi, item in enumerate(pending):
            base_ids = item["base_ids"]
            think_ids = item["think_ids"]
            Lb = base_ids.shape[0]
            Lt = think_ids.shape[0]
            b_anchor = item["b_anchor"]
            t_anchor = item["t_anchor"]
            pred_b = pred_b_batch[bi, :Lb]

            # Per-position SAE category labels for the thinking sequence.
            cat_ids, cat_vals = sae_classifier(sae_acts_batch[bi, :Lt])
            cat_per_t = [f"idx{int(c)}" for c in cat_ids.tolist()]
            cat_val_per_t = cat_vals.tolist()

            offset = t_anchor - b_anchor
            ex_idx = len(per_example)
            found_any = False
            for i in range(max(b_anchor - 1, 0), Lb - 1):
                target = int(base_ids[i + 1].item())
                if int(pred_b[i].item()) == target:
                    continue
                i_t = i + offset
                if i_t < 0 or i_t + 1 >= Lt:
                    continue
                if int(think_ids[i_t + 1].item()) != target:
                    continue
                if i_t < 0 or i_t >= len(cat_per_t):
                    continue
                cat = cat_per_t[i_t]
                per_category[cat].append((ex_idx, i, target))
                per_category_acts[cat].append(float(cat_val_per_t[i_t]))
                found_any = True

            if found_any:
                per_example.append({"ids": base_ids.cpu(),
                                    "prompt_len": int(b_anchor)})
        pending.clear()

    n_too_long = n_no_anchor = 0
    for resp in tqdm(it, desc="Collecting disagreements"):
        ann = resp.get("annotated_thinking")
        if not ann:
            continue
        question = resp["original_message"]["content"]
        thinking = extract_thinking_process(resp["full_response"])
        if not thinking or not thinking.strip():
            continue
        # Shape the user content per think family so the prompt teacher-forced
        # here matches what was sent at generation time.  For
        # base_prompt_style=think_template, the base prompt IS the think
        # model's chat-templated string (no separate scaffold).
        _is_math = _row_is_math(resp, math_directive_mode)
        _md_on = (math_directive_mode != "none")
        base_prompt = _build_base_prompt(
            question, base_prompt_style,
            thinking_tokenizer=thinking_tokenizer,
            think_family=think_family,
            is_math_question=_is_math,
            math_directive_enabled=_md_on,
        )
        shaped_question = _shape_user_content(
            question, think_family,
            is_math_question=_is_math,
            math_directive_enabled=_md_on,
        )
        try:
            think_prompt_text = thinking_tokenizer.apply_chat_template(
                [{"role": "user", "content": shaped_question}],
                tokenize=False, add_generation_prompt=True)
        except Exception:
            think_prompt_text = base_prompt
        align = _aligned_tokenize_pair(
            base_tokenizer, base_prompt,
            thinking_tokenizer, think_prompt_text,
            thinking)
        base_ids = torch.tensor(align["b_ids"], dtype=torch.long)
        think_ids = torch.tensor(align["t_ids"], dtype=torch.long)
        if base_ids.shape[0] < 8:
            continue
        if (base_ids.shape[0] > max_seq_len
                or think_ids.shape[0] > max_seq_len):
            n_too_long += 1
            continue
        if align["b_anchor"] < 0:
            n_no_anchor += 1
            continue
        pending.append({
            "base_ids": base_ids,
            "think_ids": think_ids,
            "b_anchor": int(align["b_anchor"]),
            "t_anchor": int(align["t_anchor"])})
        if len(pending) >= max(1, int(collect_batch_size)):
            _flush()
    _flush()
    try:
        _sae_handle.remove()
    except Exception:
        pass

    print(f"  collection: {len(per_example)} examples retained, "
          f"{n_too_long} too long, {n_no_anchor} no anchor",
          flush=True)
    for cat in sorted(per_category.keys(),
                      key=lambda k: int(k[3:]) if k.startswith("idx")
                      and k[3:].isdigit() else -1):
        print(f"    {cat}: {len(per_category[cat])} disagreements",
              flush=True)
    return per_example, per_category, per_category_acts


# ---------------------------------------------------------------------------
# Phase B: train bias with full-vocab top-1 CE on all disagreements
# ---------------------------------------------------------------------------

def _group_records_by_example(
    records: List[Tuple[int, int, int]],
) -> Dict[int, List[Tuple[int, int]]]:
    """records = [(ex_idx, pos, target)]; returns {ex_idx: [(pos, target)]}"""
    by_ex: Dict[int, List[Tuple[int, int]]] = defaultdict(list)
    for ex_idx, pos, target in records:
        by_ex[ex_idx].append((pos, target))
    return by_ex


@torch.no_grad()
def _holdout_eval_bias(
    base_model,
    per_example: List[dict],
    records: List[Tuple[int, int, int]],
    bias: torch.Tensor,
    *,
    steer_layer: int,
    pad_token_id: int,
    batch_size: int = 8,
) -> Tuple[float, int]:
    """Return (mean_CE, n_positions) over a static holdout set under the
    current (frozen) bias.  No grad; cheap."""
    if not records:
        return 0.0, 0
    device = next(base_model.parameters()).device
    bias_dev = bias.to(device, dtype=torch.float32).detach()
    by_ex = _group_records_by_example(records)
    ex_ids = sorted(by_ex.keys())
    total_sum = 0.0
    total_n = 0
    for s in range(0, len(ex_ids), batch_size):
        mb = ex_ids[s:s + batch_size]
        per_pos, n_pts = _ce_loss_batch_bias(
            base_model, mb, per_example, by_ex, bias_dev,
            steer_layer, pad_token_id)
        if n_pts == 0:
            continue
        total_sum += float(per_pos.detach().sum().item())
        total_n += int(n_pts)
    return (total_sum / max(total_n, 1)), total_n


def _ce_loss_batch_bias(
    base_model,
    mb: List[int],
    per_example: List[dict],
    by_example: Dict[int, List[Tuple[int, int]]],
    b: torch.Tensor,
    steer_layer: int,
    pad_token_id: int,
) -> Tuple[torch.Tensor, int]:
    """Forward one minibatch under a learnable bias hooked at steer_layer,
    return ``(per_position_ce_losses, n_pts)``."""
    device = next(base_model.parameters()).device
    B = len(mb)
    Lmax = max(per_example[e]["ids"].shape[0] for e in mb)
    ids = torch.full((B, Lmax), pad_token_id,
                     device=device, dtype=torch.long)
    attn = torch.zeros((B, Lmax), device=device, dtype=torch.long)
    pos_bids: List[int] = []
    pos_tids: List[int] = []
    pos_tgt: List[int] = []
    for bi, ex_idx in enumerate(mb):
        ex_ids = per_example[ex_idx]["ids"]
        L = ex_ids.shape[0]
        ids[bi, :L] = ex_ids.to(device, non_blocking=True)
        attn[bi, :L] = 1
        for p, tgt in by_example[ex_idx]:
            pos_bids.append(bi)
            pos_tids.append(p)
            pos_tgt.append(int(tgt))
    if not pos_bids:
        empty = torch.zeros((0,), device=device)
        return empty, 0
    pb = torch.tensor(pos_bids, device=device, dtype=torch.long)
    pt = torch.tensor(pos_tids, device=device, dtype=torch.long)
    tg = torch.tensor(pos_tgt, device=device, dtype=torch.long)
    with _hook_at(base_model, steer_layer, _BiasHook(b)):
        body = base_model.model(input_ids=ids, attention_mask=attn,
                                use_cache=False)
    hidden = body.last_hidden_state
    h_dev = hidden.device
    selected = hidden[pb.to(h_dev), pt.to(h_dev), :]
    logits = base_model.lm_head(selected).float()
    log_dev = logits.device
    base_lp = torch.log_softmax(logits, dim=-1)
    per_pos = -base_lp.gather(-1, tg.to(log_dev).unsqueeze(-1)).squeeze(-1)
    return per_pos.to(device), len(pos_bids)


def train_bias_ce(
    base_model,
    per_example: List[dict],
    records: List[Tuple[int, int, int]],
    *,
    steer_layer: int,
    hidden_size: int,
    n_epochs: int,
    batch_size: int,
    lr: float,
    pad_token_id: int,
    seed: int = 42,
    max_positions_per_example: int = 64,
    holdout_sets: Optional[Dict[str, List[Tuple[int, int, int]]]] = None,
    holdout_per_example: Optional[Dict[str, List[dict]]] = None,
    patience: int = 0,
    early_stop_metric: str = "trainmix_holdout",
    weight_decay: float = 0.0,
    norm_cap_R: Optional[float] = None,
) -> Tuple[torch.Tensor, List[dict]]:
    """Train a single bias vector with full-vocab top-1 CE on all
    disagreement positions.

    If ``holdout_sets`` is provided, after each epoch we evaluate CE on
    each named holdout set and log it.  When ``patience > 0`` and an
    early_stop_metric key is present in holdout_sets, we save the
    best-by-that-metric snapshot and stop training after ``patience``
    epochs without improvement.

    Returns ``(bias_cpu_fp32, metrics)``.
    """
    device = next(base_model.parameters()).device
    by_ex_full = _group_records_by_example(records)

    rng = random.Random(seed)
    by_ex: Dict[int, List[Tuple[int, int]]] = {}
    for ex_idx, recs in by_ex_full.items():
        if (max_positions_per_example
                and len(recs) > max_positions_per_example):
            local = random.Random(f"bias-{seed}-{ex_idx}")
            recs = local.sample(recs, max_positions_per_example)
        by_ex[ex_idx] = recs

    ex_ids = sorted(by_ex.keys(),
                    key=lambda e: per_example[e]["ids"].shape[0])
    n_pos_total = sum(len(v) for v in by_ex.values())

    b = torch.zeros(hidden_size, device=device, dtype=torch.float32,
                    requires_grad=True)
    if weight_decay > 0:
        opt = torch.optim.AdamW([b], lr=lr, weight_decay=weight_decay)
    else:
        opt = torch.optim.Adam([b], lr=lr)

    print(f"  [bias] {len(ex_ids)} examples, {n_pos_total} positions, "
          f"layer={steer_layer}, bs={batch_size}, lr={lr}, "
          f"weight_decay={weight_decay}, "
          f"epochs={n_epochs}, patience={patience}, "
          f"stop_metric={early_stop_metric}, "
          f"norm_cap_R={'OFF' if norm_cap_R is None else f'{norm_cap_R:.3f}'}",
          flush=True)
    if holdout_sets:
        for k, recs in holdout_sets.items():
            print(f"    [bias] holdout '{k}': {len(recs)} positions",
                  flush=True)

    metrics: List[dict] = []
    steps_per_epoch = math.ceil(len(ex_ids) / batch_size)
    pbar = tqdm(total=n_epochs * steps_per_epoch, desc="bias",
                mininterval=1.0)

    best_ce: Optional[float] = None
    best_b: Optional[torch.Tensor] = None
    best_epoch: int = 0
    no_improve = 0

    # Epoch-0 baseline: bias=0, so all holdout CEs are the natural
    # unsteered base-model CE on those disagreement positions.
    if holdout_sets:
        ep0_h: Dict[str, Tuple[float, int]] = {}
        for hkey, hrecs in holdout_sets.items():
            assert holdout_per_example is not None
            ho_ex = holdout_per_example[hkey]
            ce_h, n_h = _holdout_eval_bias(
                base_model, ho_ex, hrecs, b.detach(),
                steer_layer=steer_layer,
                pad_token_id=pad_token_id, batch_size=batch_size)
            ep0_h[hkey] = (ce_h, n_h)
            print(f"    [bias] epoch 0 (no-bias) holdout '{hkey}': "
                  f"ce={ce_h:.4f}  n={n_h}", flush=True)
        metrics.append({"phase": "bias", "epoch": 0,
                        "avg_ce": None, "n_positions": 0,
                        "bias_norm": 0.0,
                        "holdout_ce": {k: v[0] for k, v in ep0_h.items()},
                        "holdout_n":  {k: v[1] for k, v in ep0_h.items()}})
        if early_stop_metric in ep0_h:
            best_ce = ep0_h[early_stop_metric][0]
            best_b = b.detach().clone().cpu()
            best_epoch = 0

    for epoch in range(n_epochs):
        starts = list(range(0, len(ex_ids), batch_size))
        rng.shuffle(starts)
        ep_sum = 0.0
        ep_n = 0
        for s in starts:
            mb = ex_ids[s:s + batch_size]
            opt.zero_grad(set_to_none=True)
            per_pos, n_pts = _ce_loss_batch_bias(
                base_model, mb, per_example, by_ex, b,
                steer_layer, pad_token_id)
            if n_pts == 0:
                pbar.update(1)
                continue
            loss = per_pos.mean()
            loss.backward()
            opt.step()
            # Norm-cap projection: clamp ||b|| to the L2 ball of radius
            # norm_cap_R after each Adam step.  No-op when norm_cap_R is
            # None or ||b|| already inside the ball.
            if norm_cap_R is not None and norm_cap_R > 0.0:
                with torch.no_grad():
                    bn = b.norm()
                    if bn > norm_cap_R:
                        b.mul_(norm_cap_R / bn)
            ep_sum += float(per_pos.detach().sum().item())
            ep_n += int(n_pts)
            pbar.set_postfix(ep=f"{epoch+1}/{n_epochs}",
                             ce=f"{ep_sum/max(ep_n,1):.4f}",
                             bnrm=f"{b.detach().norm().item():.2f}")
            pbar.update(1)
        ce_epoch = ep_sum / max(ep_n, 1)
        bnorm = float(b.detach().norm().item())

        # ---- holdout eval after this epoch ----
        h_ces: Dict[str, float] = {}
        h_ns: Dict[str, int] = {}
        if holdout_sets:
            for hkey, hrecs in holdout_sets.items():
                ho_ex = holdout_per_example[hkey]
                ce_h, n_h = _holdout_eval_bias(
                    base_model, ho_ex, hrecs, b.detach(),
                    steer_layer=steer_layer,
                    pad_token_id=pad_token_id, batch_size=batch_size)
                h_ces[hkey] = ce_h
                h_ns[hkey] = n_h

        h_str = ("  " + "  ".join(f"{k}_ce={v:.4f}"
                                   for k, v in h_ces.items())
                 if h_ces else "")
        print(f"    [bias] epoch {epoch+1}/{n_epochs}: "
              f"train_ce={ce_epoch:.4f}  positions={ep_n}  "
              f"||b||={bnorm:.3f}{h_str}", flush=True)
        metrics.append({"phase": "bias", "epoch": epoch + 1,
                        "avg_ce": ce_epoch, "n_positions": ep_n,
                        "bias_norm": bnorm,
                        "holdout_ce": h_ces, "holdout_n": h_ns})

        # ---- early stopping / best snapshot ----
        if early_stop_metric in h_ces:
            cur = h_ces[early_stop_metric]
            if best_ce is None or cur < best_ce - 1e-6:
                best_ce = cur
                best_b = b.detach().clone().cpu()
                best_epoch = epoch + 1
                no_improve = 0
            else:
                no_improve += 1
                if patience > 0 and no_improve >= patience:
                    print(f"    [bias] early stop after epoch {epoch+1} "
                          f"(no improvement on '{early_stop_metric}' for "
                          f"{patience} epochs; best epoch={best_epoch} "
                          f"ce={best_ce:.4f})", flush=True)
                    break
    pbar.close()

    if best_b is not None:
        print(f"  [bias] picking best epoch={best_epoch} ce={best_ce:.4f}",
              flush=True)
        return best_b.float().cpu(), metrics
    return b.detach().float().cpu(), metrics


# ---------------------------------------------------------------------------
# Phase C: filter disagreements that the trained bias already resolves
# ---------------------------------------------------------------------------

@torch.no_grad()
def filter_residual_disagreements(
    base_model,
    per_example: List[dict],
    per_category: Dict[str, List[Tuple[int, int, int]]],
    bias: torch.Tensor,
    *,
    steer_layer: int,
    pad_token_id: int,
    batch_size: int = 8,
) -> Dict[str, List[Tuple[int, int, int]]]:
    """Return a per-category dict containing only the disagreement
    positions where (base + bias) argmax STILL disagrees with the
    rollout token.  Strict subset of the input.
    """
    device = next(base_model.parameters()).device
    bias_dev = bias.to(device, torch.float32)

    ex_to_records: Dict[int, List[Tuple[str, int, int, int]]] = defaultdict(list)
    for cat_key, records in per_category.items():
        for ri, (ex_idx, pos, target) in enumerate(records):
            ex_to_records[ex_idx].append((cat_key, ri, pos, target))

    keep = {k: [False] * len(v) for k, v in per_category.items()}

    ex_indices = sorted(ex_to_records.keys())
    for s in range(0, len(ex_indices), batch_size):
        batch_exs = ex_indices[s:s + batch_size]
        B = len(batch_exs)
        Lmax = max(per_example[e]["ids"].shape[0] for e in batch_exs)
        ids = torch.full((B, Lmax), pad_token_id,
                         device=device, dtype=torch.long)
        attn = torch.zeros((B, Lmax), device=device, dtype=torch.long)
        for bi, ex_idx in enumerate(batch_exs):
            ex_ids = per_example[ex_idx]["ids"]
            L = ex_ids.shape[0]
            ids[bi, :L] = ex_ids.to(device)
            attn[bi, :L] = 1
        with _hook_at(base_model, steer_layer, _BiasHook(bias_dev)):
            body = base_model.model(input_ids=ids, attention_mask=attn,
                                    use_cache=False)
        hidden = body.last_hidden_state
        for bi, ex_idx in enumerate(batch_exs):
            for cat_key, ri, pos, target in ex_to_records[ex_idx]:
                h = hidden[bi, pos, :].unsqueeze(0)
                logit = base_model.lm_head(
                    h.to(base_model.lm_head.weight.dtype))
                steered_argmax = int(logit.argmax(dim=-1).item())
                if steered_argmax != target:
                    keep[cat_key][ri] = True

    n_before = sum(len(v) for v in per_category.values())
    filtered: Dict[str, List[Tuple[int, int, int]]] = {}
    for cat_key, records in per_category.items():
        filtered[cat_key] = [r for r, k in zip(records, keep[cat_key]) if k]
    n_after = sum(len(v) for v in filtered.values())
    print(f"  [filter] {n_before} -> {n_after} positions remain after "
          f"bias resolves {n_before - n_after} "
          f"({100 * n_after / max(n_before, 1):.1f}% residual)",
          flush=True)
    return filtered, keep


# ---------------------------------------------------------------------------
# Phase D: train per-category cat vectors with per-cat balanced top-1 CE
# ---------------------------------------------------------------------------

def _group_cat_records_by_example(
    records: List[Tuple[int, int, int, int]],
) -> Dict[int, List[Tuple[int, int, int]]]:
    """records = [(ex_idx, pos, cat_idx, target)]; returns
    {ex_idx: [(pos, cat_idx, target)]}"""
    by_ex: Dict[int, List[Tuple[int, int, int]]] = defaultdict(list)
    for ex_idx, pos, cat_idx, target in records:
        by_ex[ex_idx].append((pos, cat_idx, target))
    return by_ex


@torch.no_grad()
def _holdout_eval_cats(
    base_model,
    per_example: List[dict],
    records: List[Tuple[int, int, int, int]],
    V: torch.Tensor,
    bias_frozen: torch.Tensor,
    *,
    n_cats: int,
    steer_layer: int,
    pad_token_id: int,
    batch_size: int = 8,
) -> Tuple[float, float, List[Optional[float]], List[int]]:
    """Return ``(sample_weighted_ce, cat_balanced_ce, per_cat_ce,
    per_cat_n)`` on a static cats holdout under (frozen V, frozen bias).

    Sample-weighted CE = ``sum_positions(CE) / total_positions`` -- this
    is the metric we LOG (it matches the train_loss reporter at line
    1414 and reflects how the model performs on the average position).

    Cat-balanced CE = ``mean_c(mean_in_cat_c(CE))`` -- the legacy metric
    we kept for diagnostics so we can still see whether tiny categories
    dominate or are dominated by big ones.

    The training loss is unchanged (still per-cat balanced) -- only the
    *holdout* logging is sample-weighted by default."""
    if not records:
        return 0.0, 0.0, [None] * n_cats, [0] * n_cats
    device = next(base_model.parameters()).device
    V_dev = V.to(device, dtype=torch.float32).detach()
    bias_dev = bias_frozen.to(device, dtype=torch.float32).detach()
    by_ex = _group_cat_records_by_example(records)
    ex_ids = sorted(by_ex.keys())

    ep_cat_sum = torch.zeros(n_cats, device=device, dtype=torch.float64)
    ep_cat_cnt = torch.zeros(n_cats, device=device, dtype=torch.long)
    for s in range(0, len(ex_ids), batch_size):
        mb = ex_ids[s:s + batch_size]
        per_pos, pos_cats, n_pts = _ce_loss_batch_cats(
            base_model, mb, per_example, by_ex, V_dev, bias_dev,
            steer_layer, pad_token_id)
        if n_pts == 0:
            continue
        pp = per_pos.detach().double()
        ep_cat_sum.scatter_add_(0, pos_cats, pp)
        ep_cat_cnt.scatter_add_(
            0, pos_cats,
            torch.ones_like(pos_cats, dtype=torch.long))
    per_cat_ce: List[Optional[float]] = []
    present: List[float] = []
    for i in range(n_cats):
        n_i = int(ep_cat_cnt[i].item())
        if n_i > 0:
            ce_i = float((ep_cat_sum[i] / ep_cat_cnt[i]).item())
            per_cat_ce.append(ce_i)
            present.append(ce_i)
        else:
            per_cat_ce.append(None)
    per_cat_n = [int(x) for x in ep_cat_cnt.tolist()]
    cat_balanced = (sum(present) / len(present)) if present else 0.0
    total_n = int(ep_cat_cnt.sum().item())
    sample_weighted = (float(ep_cat_sum.sum().item()) / total_n
                       if total_n > 0 else 0.0)
    return sample_weighted, cat_balanced, per_cat_ce, per_cat_n


def _ce_loss_batch_cats(
    base_model,
    mb: List[int],
    per_example: List[dict],
    by_example: Dict[int, List[Tuple[int, int, int]]],
    V: torch.Tensor,
    bias_frozen: torch.Tensor,
    steer_layer: int,
    pad_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Forward one minibatch with V[cat(p)] + bias_frozen injected at
    each disagreement position.  Returns
    ``(per_pos_ce, pos_cats, n_pts)``."""
    device = next(base_model.parameters()).device
    B = len(mb)
    Lmax = max(per_example[e]["ids"].shape[0] for e in mb)
    ids = torch.full((B, Lmax), pad_token_id,
                     device=device, dtype=torch.long)
    attn = torch.zeros((B, Lmax), device=device, dtype=torch.long)
    pb_l: List[int] = []
    pt_l: List[int] = []
    pc_l: List[int] = []
    tg_l: List[int] = []
    for bi, ex_idx in enumerate(mb):
        ex_ids = per_example[ex_idx]["ids"]
        L = ex_ids.shape[0]
        ids[bi, :L] = ex_ids.to(device, non_blocking=True)
        attn[bi, :L] = 1
        for p, c, tgt in by_example[ex_idx]:
            pb_l.append(bi)
            pt_l.append(p)
            pc_l.append(c)
            tg_l.append(int(tgt))
    if not pb_l:
        empty = torch.zeros((0,), device=device)
        empty_cats = torch.zeros((0,), device=device, dtype=torch.long)
        return empty, empty_cats, 0
    pb = torch.tensor(pb_l, device=device, dtype=torch.long)
    pt = torch.tensor(pt_l, device=device, dtype=torch.long)
    pc = torch.tensor(pc_l, device=device, dtype=torch.long)
    tg = torch.tensor(tg_l, device=device, dtype=torch.long)
    hook = _CatHook(V, pb, pt, pc, bias_frozen=bias_frozen)
    with _hook_at(base_model, steer_layer, hook):
        body = base_model.model(input_ids=ids, attention_mask=attn,
                                use_cache=False)
    hidden = body.last_hidden_state
    h_dev = hidden.device
    selected = hidden[pb.to(h_dev), pt.to(h_dev), :]
    logits = base_model.lm_head(selected).float()
    log_dev = logits.device
    base_lp = torch.log_softmax(logits, dim=-1)
    per_pos = -base_lp.gather(-1, tg.to(log_dev).unsqueeze(-1)).squeeze(-1)
    return per_pos.to(device), pc, len(pb_l)


def train_cats_ce_balanced(
    base_model,
    per_example: List[dict],
    records: List[Tuple[int, int, int, int]],
    n_cats: int,
    bias_frozen: torch.Tensor,
    *,
    steer_layer: int,
    hidden_size: int,
    n_epochs: int,
    batch_size: int,
    lr: float,
    pad_token_id: int,
    seed: int = 42,
    max_positions_per_example: int = 64,
    cat_key_lookup: Optional[List[str]] = None,
    holdout_sets: Optional[Dict[str, List[Tuple[int, int, int, int]]]] = None,
    holdout_per_example: Optional[Dict[str, List[dict]]] = None,
    patience: int = 0,
    early_stop_metric: str = "trainmix_holdout",
    weight_decay: float = 0.0,
    per_cat_best: bool = True,
    norm_cap_R: Optional[float] = None,
) -> Tuple[torch.Tensor, List[dict]]:
    """Train ``n_cats`` per-category vectors jointly with full-vocab
    top-1 CE.  Bias is hooked in as a frozen offset.

    Per-cat balanced loss: for each minibatch, compute the mean per-cat
    CE within the present cats, then average across present cats.  Cats
    with more positions in the batch do not dominate the global
    gradient direction.

    Per-cat best snapshot: when ``per_cat_best`` is True (default) and
    ``early_stop_metric`` is present in ``holdout_sets``, we track each
    cat's best epoch independently on the per-cat holdout CE and save
    its V row at that epoch.  Tiny cats stop improving early and freeze
    at a small-norm V; data-rich cats can run longer.  The global
    training loop stops only when NO cat has improved for ``patience``
    consecutive epochs.

    Returns ``(V_cpu_fp32 (n_cats, hidden), metrics)``.
    """
    device = next(base_model.parameters()).device
    by_ex_full = _group_cat_records_by_example(records)

    by_ex: Dict[int, List[Tuple[int, int, int]]] = {}
    for ex_idx, recs in by_ex_full.items():
        # Per-(example, cat) cap.
        if max_positions_per_example and max_positions_per_example > 0:
            by_cat_local: Dict[int, List[Tuple[int, int, int]]] = defaultdict(list)
            for p, c, t in recs:
                by_cat_local[c].append((p, c, t))
            kept: List[Tuple[int, int, int]] = []
            for c, lr_ in by_cat_local.items():
                if len(lr_) > max_positions_per_example:
                    local = random.Random(f"cats-{seed}-{ex_idx}-{c}")
                    lr_ = local.sample(lr_, max_positions_per_example)
                kept.extend(lr_)
            recs = kept
        by_ex[ex_idx] = recs

    ex_ids = sorted(by_ex.keys(),
                    key=lambda e: per_example[e]["ids"].shape[0])

    V = torch.zeros((n_cats, hidden_size), device=device,
                    dtype=torch.float32, requires_grad=True)
    bias_dev = bias_frozen.to(device, dtype=torch.float32).detach()
    if weight_decay > 0:
        opt = torch.optim.AdamW([V], lr=lr, weight_decay=weight_decay)
    else:
        opt = torch.optim.Adam([V], lr=lr)

    n_pos_total = sum(len(v) for v in by_ex.values())
    print(f"  [cats] {n_cats} vectors, {len(ex_ids)} examples, "
          f"{n_pos_total} positions, layer={steer_layer}, "
          f"bs={batch_size}, lr={lr}, weight_decay={weight_decay}, "
          f"epochs={n_epochs}, patience={patience}, "
          f"stop_metric={early_stop_metric} "
          f"per_cat_best={per_cat_best} "
          f"norm_cap_R={'OFF' if norm_cap_R is None else f'{norm_cap_R:.3f}'} "
          f"(loss = per-cat-balanced CE)", flush=True)
    if holdout_sets:
        for k, recs in holdout_sets.items():
            print(f"    [cats] holdout '{k}': {len(recs)} positions",
                  flush=True)

    metrics: List[dict] = []
    steps_per_epoch = math.ceil(len(ex_ids) / batch_size)
    pbar = tqdm(total=n_epochs * steps_per_epoch, desc="cats",
                mininterval=1.0)
    rng = random.Random(seed)

    # Aggregate snapshot (for backward compatibility / logging).
    best_ce: Optional[float] = None
    best_V: Optional[torch.Tensor] = None
    best_epoch: int = 0
    no_improve = 0

    # Per-cat snapshot: track each cat's best epoch on the per-cat
    # holdout CE under early_stop_metric independently.
    best_per_cat_ce: List[Optional[float]] = [None] * n_cats
    best_per_cat_V: torch.Tensor = torch.zeros(
        (n_cats, hidden_size), dtype=torch.float32)
    best_per_cat_epoch: List[int] = [0] * n_cats
    no_improve_per_cat: List[int] = [0] * n_cats

    # Epoch-0 baseline: V=0, so all holdout CEs reflect (base + bias)
    # CE (i.e., what's left after Phase B).
    if holdout_sets:
        # Each entry is (sample_weighted, cat_balanced, per_cat_ce, per_cat_n).
        ep0_h: Dict[str, Tuple[float, float, List[Optional[float]], List[int]]] = {}
        for hkey, hrecs in holdout_sets.items():
            assert holdout_per_example is not None
            ho_ex = holdout_per_example[hkey]
            sw_ce, cb_ce, per_cat_ce, per_cat_n = _holdout_eval_cats(
                base_model, ho_ex, hrecs, V.detach(), bias_dev,
                n_cats=n_cats, steer_layer=steer_layer,
                pad_token_id=pad_token_id, batch_size=batch_size)
            ep0_h[hkey] = (sw_ce, cb_ce, per_cat_ce, per_cat_n)
            n_tot = sum(per_cat_n)
            print(f"    [cats] epoch 0 (V=0) holdout '{hkey}': "
                  f"sample_ce={sw_ce:.4f}  cat_bal_ce={cb_ce:.4f}  "
                  f"n={n_tot}", flush=True)
        metrics.append({
            "phase": "cats", "epoch": 0,
            "mean_ce": None,
            "per_cat_ce": None, "per_cat_count": None,
            "norms_per_cat": [0.0] * n_cats,
            # holdout_ce = sample-weighted (new default for logging /
            # plotting / early-stopping); cat-balanced kept as
            # diagnostic only.  Training loss is unchanged (still
            # per-cat balanced minibatch loss).
            "holdout_ce": {k: v[0] for k, v in ep0_h.items()},
            "holdout_ce_cat_balanced": {k: v[1] for k, v in ep0_h.items()},
            "holdout_per_cat_ce": {k: v[2] for k, v in ep0_h.items()},
            "holdout_per_cat_n":  {k: v[3] for k, v in ep0_h.items()}})
        if early_stop_metric in ep0_h:
            best_ce = ep0_h[early_stop_metric][0]
            best_V = V.detach().clone().cpu()
            best_epoch = 0
            # Per-cat: V=0 is the natural baseline. We treat epoch-0 CE
            # as the initial best for each cat that has holdout
            # coverage; later epochs must IMPROVE on this to displace.
            per_cat_ce_0 = ep0_h[early_stop_metric][2]
            per_cat_n_0 = ep0_h[early_stop_metric][3]
            for c in range(n_cats):
                if per_cat_n_0[c] > 0 and per_cat_ce_0[c] is not None:
                    best_per_cat_ce[c] = float(per_cat_ce_0[c])
                    best_per_cat_V[c] = V.detach()[c].clone().cpu()
                    best_per_cat_epoch[c] = 0

    for epoch in range(n_epochs):
        starts = list(range(0, len(ex_ids), batch_size))
        rng.shuffle(starts)
        ep_cat_sum = torch.zeros(n_cats, device=device, dtype=torch.float64)
        ep_cat_cnt = torch.zeros(n_cats, device=device, dtype=torch.long)
        for s in starts:
            mb = ex_ids[s:s + batch_size]
            opt.zero_grad(set_to_none=True)
            per_pos, pos_cats, n_pts = _ce_loss_batch_cats(
                base_model, mb, per_example, by_ex, V, bias_dev,
                steer_layer, pad_token_id)
            if n_pts == 0:
                pbar.update(1)
                continue

            # Per-cat balanced loss: compute mean within each present
            # cat, then average across present cats.  Differentiable
            # because we use scatter_add into a (n_cats,) accumulator.
            cat_sum = torch.zeros(n_cats, device=device,
                                  dtype=per_pos.dtype)
            cat_cnt = torch.zeros(n_cats, device=device,
                                  dtype=per_pos.dtype)
            cat_sum.scatter_add_(0, pos_cats, per_pos)
            cat_cnt.scatter_add_(0, pos_cats, torch.ones_like(per_pos))
            mask = cat_cnt > 0
            cat_mean = cat_sum[mask] / cat_cnt[mask]
            loss = cat_mean.mean()
            loss.backward()
            opt.step()
            # Norm-cap projection: per-row clamp to L2 ball of radius
            # norm_cap_R (each cat vector independently).
            if norm_cap_R is not None and norm_cap_R > 0.0:
                with torch.no_grad():
                    row_n = V.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                    scale = torch.clamp_max(norm_cap_R / row_n, 1.0)
                    V.mul_(scale)

            with torch.no_grad():
                pp = per_pos.detach().double()
                ep_cat_sum.scatter_add_(0, pos_cats, pp)
                ep_cat_cnt.scatter_add_(
                    0, pos_cats,
                    torch.ones_like(pos_cats, dtype=torch.long))
            postfix = {"ep": f"{epoch+1}/{n_epochs}",
                       "ce": f"{float(loss.detach().item()):.4f}",
                       "nrm": f"[{V.detach().norm(dim=-1).mean().item():.1f}]"}
            pbar.set_postfix(**postfix)
            pbar.update(1)

        per_cat_kl = [
            (float((ep_cat_sum[i] / ep_cat_cnt[i]).item())
             if int(ep_cat_cnt[i].item()) > 0 else None)
            for i in range(n_cats)]
        norms = V.detach().norm(dim=-1)
        mean_ce = (float(ep_cat_sum.sum().item())
                   / max(int(ep_cat_cnt.sum().item()), 1))

        # ---- holdout eval after this epoch ----
        h_ces: Dict[str, float] = {}                  # sample-weighted (logged)
        h_ces_cb: Dict[str, float] = {}               # cat-balanced (diagnostic)
        h_per_cat_ce: Dict[str, List[Optional[float]]] = {}
        h_per_cat_n: Dict[str, List[int]] = {}
        if holdout_sets:
            for hkey, hrecs in holdout_sets.items():
                ho_ex = holdout_per_example[hkey]
                sw_ce, cb_ce, per_cat_ce_h, per_cat_n_h = _holdout_eval_cats(
                    base_model, ho_ex, hrecs, V.detach(), bias_dev,
                    n_cats=n_cats, steer_layer=steer_layer,
                    pad_token_id=pad_token_id, batch_size=batch_size)
                h_ces[hkey] = sw_ce
                h_ces_cb[hkey] = cb_ce
                h_per_cat_ce[hkey] = per_cat_ce_h
                h_per_cat_n[hkey] = per_cat_n_h

        h_str = ("  " + "  ".join(f"{k}_ce={v:.4f}"
                                   for k, v in h_ces.items())
                 if h_ces else "")
        print(f"    [cats] epoch {epoch+1}/{n_epochs}: "
              f"mean_ce={mean_ce:.4f}  norms=[{float(norms.min()):.2f}-"
              f"{float(norms.max()):.2f}]{h_str}", flush=True)
        if cat_key_lookup is not None:
            parts = []
            for i in range(n_cats):
                k = cat_key_lookup[i]
                ce = per_cat_kl[i]
                ces = f"{ce:.3f}" if ce is not None else "---"
                parts.append(f"{k}={ces} (n={int(ep_cat_cnt[i].item())}, "
                             f"||V||={float(norms[i].item()):.2f})")
            print("      [per-cat ce] " + "  ".join(parts), flush=True)
        metrics.append({"phase": "cats", "epoch": epoch + 1,
                        "mean_ce": mean_ce,
                        "per_cat_ce": per_cat_kl,
                        "per_cat_count": [int(x) for x
                                          in ep_cat_cnt.tolist()],
                        "norms_per_cat": [float(x) for x
                                          in norms.tolist()],
                        "holdout_ce": h_ces,
                        "holdout_ce_cat_balanced": h_ces_cb,
                        "holdout_per_cat_ce": h_per_cat_ce,
                        "holdout_per_cat_n":  h_per_cat_n})

        # ---- early stopping / best snapshot ----
        global_improved_this_epoch = False
        if early_stop_metric in h_ces:
            cur = h_ces[early_stop_metric]
            if best_ce is None or cur < best_ce - 1e-6:
                best_ce = cur
                best_V = V.detach().clone().cpu()
                best_epoch = epoch + 1

            # ---- per-cat best snapshot ----
            per_cat_ce_h = h_per_cat_ce.get(early_stop_metric, [])
            per_cat_n_h = h_per_cat_n.get(early_stop_metric, [])
            V_cpu_now = V.detach().cpu()
            for c in range(n_cats):
                if (c >= len(per_cat_n_h) or per_cat_n_h[c] <= 0
                        or per_cat_ce_h[c] is None):
                    no_improve_per_cat[c] += 1
                    continue
                cur_c = float(per_cat_ce_h[c])
                if (best_per_cat_ce[c] is None
                        or cur_c < best_per_cat_ce[c] - 1e-6):
                    best_per_cat_ce[c] = cur_c
                    best_per_cat_V[c] = V_cpu_now[c].clone()
                    best_per_cat_epoch[c] = epoch + 1
                    no_improve_per_cat[c] = 0
                    global_improved_this_epoch = True
                else:
                    no_improve_per_cat[c] += 1

            if global_improved_this_epoch:
                no_improve = 0
            else:
                no_improve += 1
                if patience > 0 and no_improve >= patience:
                    n_done = sum(1 for c in range(n_cats)
                                 if no_improve_per_cat[c] >= patience)
                    print(f"    [cats] early stop after epoch {epoch+1} "
                          f"(no per-cat improvement on '{early_stop_metric}' "
                          f"for {patience} epochs; "
                          f"{n_done}/{n_cats} cats hit patience)",
                          flush=True)
                    break
    pbar.close()

    # Decide which V to return.
    if per_cat_best:
        # Use per-cat best rows; fallback to current V for cats that
        # never saw a valid holdout signal.
        V_out = V.detach().float().cpu().clone()
        for c in range(n_cats):
            if best_per_cat_ce[c] is not None:
                V_out[c] = best_per_cat_V[c]
        print(f"  [cats] per-cat best snapshot: epochs="
              f"{best_per_cat_epoch}  per_cat_best_ce="
              f"{[f'{x:.3f}' if x is not None else 'None' for x in best_per_cat_ce]}",
              flush=True)
        return V_out, metrics
    if best_V is not None:
        print(f"  [cats] picking best epoch={best_epoch} ce={best_ce:.4f}",
              flush=True)
        return best_V.float().cpu(), metrics
    return V.detach().float().cpu(), metrics


# ---------------------------------------------------------------------------
# Phase D (MLP variant): jointly train V + CatCoefMLP
# ---------------------------------------------------------------------------

def _ce_loss_batch_cats_mlp(
    base_model,
    mb: List[int],
    per_example: List[dict],
    by_example: Dict[int, List[Tuple[int, int, int]]],
    V: torch.Tensor,
    mlp,
    steer_layer: int,
    pad_token_id: int,
) -> Tuple[torch.Tensor, torch.Tensor, int, Optional[torch.Tensor]]:
    """Forward one minibatch with MLP-predicted alpha * V[cat] at each
    disagreement position.  Returns (per_pos_ce, pos_cats, n_pts, alphas)."""
    device = next(base_model.parameters()).device
    B = len(mb)
    Lmax = max(per_example[e]["ids"].shape[0] for e in mb)
    ids = torch.full((B, Lmax), pad_token_id,
                     device=device, dtype=torch.long)
    attn = torch.zeros((B, Lmax), device=device, dtype=torch.long)
    pb_l, pt_l, pc_l, tg_l = [], [], [], []
    for bi, ex_idx in enumerate(mb):
        ex_ids = per_example[ex_idx]["ids"]
        L = ex_ids.shape[0]
        ids[bi, :L] = ex_ids.to(device, non_blocking=True)
        attn[bi, :L] = 1
        for p, c, tgt in by_example[ex_idx]:
            pb_l.append(bi)
            pt_l.append(p)
            pc_l.append(c)
            tg_l.append(int(tgt))
    has_positions = len(pb_l) > 0
    if has_positions:
        pb = torch.tensor(pb_l, device=device, dtype=torch.long)
        pt = torch.tensor(pt_l, device=device, dtype=torch.long)
        pc = torch.tensor(pc_l, device=device, dtype=torch.long)
        tg = torch.tensor(tg_l, device=device, dtype=torch.long)
    else:
        pb = torch.zeros((1,), device=device, dtype=torch.long)
        pt = torch.zeros((1,), device=device, dtype=torch.long)
        pc = torch.zeros((1,), device=device, dtype=torch.long)
        tg = torch.zeros((1,), device=device, dtype=torch.long)

    hook = _CatMLPHook(V, mlp, pb, pt, pc)
    with _hook_at(base_model, steer_layer, hook):
        body = base_model.model(input_ids=ids, attention_mask=attn,
                                use_cache=False)
    hidden = body.last_hidden_state
    del body
    h_dev = hidden.device
    selected = hidden[pb.to(h_dev), pt.to(h_dev), :]
    logits = base_model.lm_head(selected).float()
    del hidden

    if not has_positions:
        empty = torch.zeros((0,), device=device)
        empty_cats = torch.zeros((0,), device=device, dtype=torch.long)
        return empty, empty_cats, 0, None

    base_lp = torch.log_softmax(logits, dim=-1)
    per_pos = -base_lp.gather(-1, tg.to(logits.device).unsqueeze(-1)).squeeze(-1)
    return per_pos.to(device), pc, len(pb_l), hook.last_alphas


@torch.no_grad()
def _holdout_eval_cats_mlp(
    base_model, per_example, records, V, mlp,
    *, n_cats, steer_layer, pad_token_id, batch_size=8,
    distributed=False,
):
    """Holdout eval with MLP hook (no grad).  In distributed mode,
    examples are sharded across ranks and per-cat sums are all-reduced.

    Returns ``(sample_weighted_ce, cat_balanced_ce, per_cat_ce,
    per_cat_n)``.  Sample-weighted is the *new logging default*; the
    cat-balanced value is kept as a secondary diagnostic.  Training
    loss is unchanged."""
    if not records:
        return float("nan"), float("nan"), [None] * n_cats, [0] * n_cats
    device = next(base_model.parameters()).device
    by_ex = _group_cat_records_by_example(records)
    ex_ids = sorted(by_ex.keys(),
                    key=lambda e: per_example[e]["ids"].shape[0])
    if distributed:
        _rank = get_rank()
        _ws = get_world_size()
        ex_ids = ex_ids[_rank::_ws]
    cat_sum = torch.zeros(n_cats, device=device, dtype=torch.float64)
    cat_cnt = torch.zeros(n_cats, device=device, dtype=torch.long)
    for s in range(0, len(ex_ids), batch_size):
        mb = ex_ids[s:s + batch_size]
        per_pos, pos_cats, n_pts, _ = _ce_loss_batch_cats_mlp(
            base_model, mb, per_example, by_ex, V, mlp,
            steer_layer, pad_token_id)
        if n_pts == 0:
            continue
        cat_sum.scatter_add_(0, pos_cats, per_pos.detach().double())
        cat_cnt.scatter_add_(0, pos_cats,
                             torch.ones_like(pos_cats, dtype=torch.long))
    if distributed:
        import torch.distributed as _dist
        _dist.all_reduce(cat_sum, op=_dist.ReduceOp.SUM)
        _dist.all_reduce(cat_cnt, op=_dist.ReduceOp.SUM)
    per_cat_ce = []
    per_cat_n = []
    for c in range(n_cats):
        n = int(cat_cnt[c].item())
        per_cat_n.append(n)
        per_cat_ce.append(
            float((cat_sum[c] / cat_cnt[c]).item()) if n > 0 else None)
    mask = cat_cnt > 0
    if mask.any():
        cat_balanced = float((cat_sum[mask] / cat_cnt[mask]).mean().item())
    else:
        cat_balanced = float("nan")
    total_n = int(cat_cnt.sum().item())
    sample_weighted = (float(cat_sum.sum().item()) / total_n
                       if total_n > 0 else float("nan"))
    return sample_weighted, cat_balanced, per_cat_ce, per_cat_n


@torch.no_grad()
def _steer_metrics_batch_mlp(
    base_model, mb, per_example, by_example, V, mlp,
    steer_layer, pad_token_id,
):
    """One forward with MLP-predicted alpha * V[cat] at each disagreement
    position. Returns (per_pos_ce, correct_bool, pos_cats, n_pts) where
    ``correct_bool[i]`` is True iff the STEERED argmax == think target
    token at that position.  Unsteered argmax != target by construction of
    the disagreement set, so this is a token-level 'gap recovered'."""
    device = next(base_model.parameters()).device
    B = len(mb)
    Lmax = max(per_example[e]["ids"].shape[0] for e in mb)
    ids = torch.full((B, Lmax), pad_token_id, device=device, dtype=torch.long)
    attn = torch.zeros((B, Lmax), device=device, dtype=torch.long)
    pb_l, pt_l, pc_l, tg_l = [], [], [], []
    for bi, ex_idx in enumerate(mb):
        ex_ids = per_example[ex_idx]["ids"]
        L = ex_ids.shape[0]
        ids[bi, :L] = ex_ids.to(device, non_blocking=True)
        attn[bi, :L] = 1
        for p, c, tgt in by_example[ex_idx]:
            pb_l.append(bi); pt_l.append(p); pc_l.append(c); tg_l.append(int(tgt))
    if not pb_l:
        empty = torch.zeros((0,), device=device)
        emptyb = torch.zeros((0,), device=device, dtype=torch.bool)
        emptyc = torch.zeros((0,), device=device, dtype=torch.long)
        return empty, emptyb, emptyc, 0
    pb = torch.tensor(pb_l, device=device, dtype=torch.long)
    pt = torch.tensor(pt_l, device=device, dtype=torch.long)
    pc = torch.tensor(pc_l, device=device, dtype=torch.long)
    tg = torch.tensor(tg_l, device=device, dtype=torch.long)
    hook = _CatMLPHook(V, mlp, pb, pt, pc)
    with _hook_at(base_model, steer_layer, hook):
        body = base_model.model(input_ids=ids, attention_mask=attn,
                                use_cache=False)
    hidden = body.last_hidden_state
    del body
    selected = hidden[pb.to(hidden.device), pt.to(hidden.device), :]
    logits = base_model.lm_head(selected).float()
    del hidden
    log_dev = logits.device
    tg_ld = tg.to(log_dev)
    base_lp = torch.log_softmax(logits, dim=-1)
    per_pos = -base_lp.gather(-1, tg_ld.unsqueeze(-1)).squeeze(-1)
    correct = (logits.argmax(dim=-1) == tg_ld)
    return per_pos.to(device), correct.to(device), pc, len(pb_l)


@torch.no_grad()
def _holdout_eval_cats_mlp_full(
    base_model, per_example, records, V, mlp,
    *, n_cats, steer_layer, pad_token_id, batch_size=8, distributed=False,
):
    """Like ``_holdout_eval_cats_mlp`` but also returns top-1 steering
    accuracy (fraction of disagreement positions whose STEERED argmax hits
    the think target), overall and per-category, computed in the SAME
    forward pass as CE.

    Returns ``(sample_weighted_ce, cat_balanced_ce, per_cat_ce, per_cat_n,
    overall_acc, per_cat_acc)``."""
    if not records:
        nan = float("nan")
        return nan, nan, [None]*n_cats, [0]*n_cats, nan, [None]*n_cats
    device = next(base_model.parameters()).device
    by_ex = _group_cat_records_by_example(records)
    ex_ids = sorted(by_ex.keys(),
                    key=lambda e: per_example[e]["ids"].shape[0])
    if distributed:
        ex_ids = ex_ids[get_rank()::get_world_size()]
    cat_sum = torch.zeros(n_cats, device=device, dtype=torch.float64)
    cat_cnt = torch.zeros(n_cats, device=device, dtype=torch.long)
    cat_cor = torch.zeros(n_cats, device=device, dtype=torch.long)
    for s in range(0, len(ex_ids), batch_size):
        mb = ex_ids[s:s + batch_size]
        per_pos, correct, pos_cats, n_pts = _steer_metrics_batch_mlp(
            base_model, mb, per_example, by_ex, V, mlp,
            steer_layer, pad_token_id)
        if n_pts == 0:
            continue
        cat_sum.scatter_add_(0, pos_cats, per_pos.detach().double())
        cat_cnt.scatter_add_(0, pos_cats,
                             torch.ones_like(pos_cats, dtype=torch.long))
        cat_cor.scatter_add_(0, pos_cats, correct.long())
    if distributed:
        import torch.distributed as _dist
        _dist.all_reduce(cat_sum, op=_dist.ReduceOp.SUM)
        _dist.all_reduce(cat_cnt, op=_dist.ReduceOp.SUM)
        _dist.all_reduce(cat_cor, op=_dist.ReduceOp.SUM)
    per_cat_ce, per_cat_n, per_cat_acc = [], [], []
    for c in range(n_cats):
        n = int(cat_cnt[c].item())
        per_cat_n.append(n)
        per_cat_ce.append(float((cat_sum[c] / cat_cnt[c]).item()) if n > 0 else None)
        per_cat_acc.append(float(cat_cor[c].item()) / n if n > 0 else None)
    mask = cat_cnt > 0
    cat_balanced = (float((cat_sum[mask] / cat_cnt[mask]).mean().item())
                    if mask.any() else float("nan"))
    total_n = int(cat_cnt.sum().item())
    sample_weighted = (float(cat_sum.sum().item()) / total_n
                       if total_n > 0 else float("nan"))
    overall_acc = (float(cat_cor.sum().item()) / total_n
                   if total_n > 0 else float("nan"))
    return (sample_weighted, cat_balanced, per_cat_ce, per_cat_n,
            overall_acc, per_cat_acc)


def train_cats_mlp_coef(
    base_model,
    per_example: List[dict],
    records: List[Tuple[int, int, int, int]],
    n_cats: int,
    *,
    steer_layer: int,
    hidden_size: int,
    n_epochs: int,
    batch_size: int,
    cats_lr: float,
    mlp_lr: float,
    mlp_hidden_dim: int,
    mlp_grad_clip: float,
    pad_token_id: int,
    mlp_per_cat: bool = False,
    seed: int = 42,
    max_positions_per_example: int = 64,
    cat_key_lookup: Optional[List[str]] = None,
    holdout_sets: Optional[Dict[str, List[Tuple[int, int, int, int]]]] = None,
    holdout_per_example: Optional[Dict[str, List[dict]]] = None,
    patience: int = 0,
    early_stop_metric: str = "trainmix_holdout",
    weight_decay: float = 0.01,
    per_cat_best: bool = True,
    norm_cap_R: Optional[float] = None,
    distributed: bool = False,
    save_per_epoch_dir: Optional[str] = None,
    freeze_cats: bool = False,
    init_cat_norms: Optional[List[float]] = None,
    rand_cats_seed: int = 1337,
) -> Tuple[torch.Tensor, object, List[dict]]:
    """Train n_cats category vectors V jointly with a CatCoefMLP.

    Returns (V_cpu, mlp_cpu, metrics).

    distributed=True: data-parallel across torchrun ranks.  Each rank
    loads its own copy of the (frozen) base model on its local GPU,
    processes its shard of training examples, and gradients for V+MLP
    (the tiny trainables) are all-reduced after each backward pass.
    Only rank 0 prints/saves; all ranks participate in holdout eval
    via sharded all-reduce.

    ABLATION KNOBS:
      ``freeze_cats``: if True, V is excluded from the optimiser and
          left at its initial value for the full run.
      ``init_cat_norms``: optional list of per-row L2 norms.  V is
          drawn from N(0, I) (seeded by ``rand_cats_seed``) and each
          row is rescaled so ||V[k]|| == init_cat_norms[k].  Pass with
          ``freeze_cats=True`` to run the "frozen random norm-matched
          cat vectors, only MLP trains" ablation.
    """
    from coef_mlp import CatCoefMLP

    if distributed:
        rank = get_rank()
        world_size = get_world_size()
    else:
        rank = 0
        world_size = 1
    main_rank = (rank == 0)

    device = next(base_model.parameters()).device
    by_ex_full = _group_cat_records_by_example(records)
    by_ex: Dict[int, List[Tuple[int, int, int]]] = {}
    for ex_idx, recs in by_ex_full.items():
        if max_positions_per_example and max_positions_per_example > 0:
            by_cat_local: Dict[int, List[Tuple[int, int, int]]] = defaultdict(list)
            for p, c, t in recs:
                by_cat_local[c].append((p, c, t))
            kept: List[Tuple[int, int, int]] = []
            for c, lr_ in by_cat_local.items():
                if len(lr_) > max_positions_per_example:
                    local = random.Random(f"cats-{seed}-{ex_idx}-{c}")
                    lr_ = local.sample(lr_, max_positions_per_example)
                kept.extend(lr_)
            recs = kept
        by_ex[ex_idx] = recs

    ex_ids = sorted(by_ex.keys(),
                    key=lambda e: per_example[e]["ids"].shape[0])

    # Shard examples across ranks (pad so all ranks have equal counts).
    if distributed and world_size > 1:
        n_pad = (-len(ex_ids)) % world_size
        if n_pad > 0:
            ex_ids = ex_ids + ex_ids[-n_pad:]
        ex_ids_local = ex_ids[rank::world_size]
    else:
        ex_ids_local = ex_ids

    # Seed both V and MLP identically across ranks (call only once).
    if distributed:
        torch.manual_seed(seed)
    if init_cat_norms is not None:
        # ABLATION: random gaussian directions, per-row rescaled to the
        # supplied reference norms.  Identical across DDP ranks via a
        # CPU generator + .to(device) (CUDA generators are device-local
        # and can drift across ranks even with the same seed).
        g_cpu = torch.Generator(device="cpu").manual_seed(int(rand_cats_seed))
        V_init = torch.randn(
            (n_cats, hidden_size), generator=g_cpu, dtype=torch.float32)
        cur_norms = V_init.norm(dim=-1, keepdim=True).clamp_min(1e-12)
        if len(init_cat_norms) != n_cats:
            raise ValueError(
                f"init_cat_norms has {len(init_cat_norms)} entries but "
                f"n_cats={n_cats}")
        target = torch.tensor(
            init_cat_norms, dtype=torch.float32).view(-1, 1)
        V_init = V_init * (target / cur_norms)
        V = V_init.to(device)
        V.requires_grad_(not freeze_cats)
        if main_rank:
            print(f"  [cats+mlp] V initialised RANDOM (seed="
                  f"{rand_cats_seed}), per-row norm-matched to "
                  f"{[f'{n:.3f}' for n in init_cat_norms]}; "
                  f"freeze_cats={freeze_cats}", flush=True)
    else:
        V = torch.zeros((n_cats, hidden_size), device=device,
                        dtype=torch.float32, requires_grad=not freeze_cats)
        if main_rank and freeze_cats:
            print(f"  [cats+mlp] V initialised to ZEROS and FROZEN "
                  f"(unusual: typically pair --freeze_cats with "
                  f"--init_cats_from_dir).", flush=True)
    mlp = CatCoefMLP(d_in=hidden_size, n_cats=n_cats,
                     d_hidden=mlp_hidden_dim,
                     per_cat=mlp_per_cat).to(device).float()

    opt_param_groups = []
    if not freeze_cats:
        opt_param_groups.append({"params": [V], "lr": cats_lr})
    opt_param_groups.append({"params": mlp.parameters(), "lr": mlp_lr})
    opt = torch.optim.AdamW(opt_param_groups, weight_decay=weight_decay)

    n_pos_total = sum(len(v) for v in by_ex.values())

    if main_rank:
        print(f"  [cats+mlp] {n_cats} vectors + MLP(h={mlp_hidden_dim}, "
              f"per_cat={mlp_per_cat}), "
              f"{len(ex_ids)} examples ({len(ex_ids_local)} local), "
              f"{n_pos_total} positions, "
              f"layer={steer_layer}, bs={batch_size}, "
              f"cats_lr={cats_lr}, mlp_lr={mlp_lr}, "
              f"mlp_grad_clip={mlp_grad_clip}, wd={weight_decay}, "
              f"epochs={n_epochs}, patience={patience}"
              f"{', DDP=' + str(world_size) + 'xGPU' if distributed else ''}",
              flush=True)
        if holdout_sets:
            for k, recs in holdout_sets.items():
                print(f"    [cats+mlp] holdout '{k}': {len(recs)} positions",
                      flush=True)

    metrics: List[dict] = []
    steps_per_epoch = math.ceil(len(ex_ids_local) / batch_size)
    pbar = tqdm(total=n_epochs * steps_per_epoch, desc="cats+mlp",
                mininterval=1.0, disable=not main_rank)
    rng = random.Random(seed)

    best_ce: Optional[float] = None
    best_V: Optional[torch.Tensor] = None
    best_mlp_state: Optional[dict] = None
    best_epoch: int = 0
    no_improve = 0

    best_per_cat_ce: List[Optional[float]] = [None] * n_cats
    best_per_cat_V: torch.Tensor = torch.zeros(
        (n_cats, hidden_size), dtype=torch.float32)
    best_per_cat_epoch: List[int] = [0] * n_cats
    no_improve_per_cat: List[int] = [0] * n_cats

    if holdout_sets:
        # Each entry: (sample_weighted, cat_balanced, per_cat_ce, per_cat_n)
        ep0_h = {}
        for hkey, hrecs in holdout_sets.items():
            ho_ex = holdout_per_example[hkey]
            sw_ce, cb_ce, per_cat_ce, per_cat_n = _holdout_eval_cats_mlp(
                base_model, ho_ex, hrecs, V.detach(), mlp,
                n_cats=n_cats, steer_layer=steer_layer,
                pad_token_id=pad_token_id, batch_size=batch_size,
                distributed=distributed)
            ep0_h[hkey] = (sw_ce, cb_ce, per_cat_ce, per_cat_n)
            if main_rank:
                print(f"    [cats+mlp] epoch 0 holdout '{hkey}': "
                      f"sample_ce={sw_ce:.4f}  cat_bal_ce={cb_ce:.4f}",
                      flush=True)
        metrics.append({
            "phase": "cats_mlp", "epoch": 0,
            "mean_ce": None, "alpha_stats": None,
            # logged metric is sample-weighted (training loss unchanged).
            "holdout_ce": {k: v[0] for k, v in ep0_h.items()},
            "holdout_ce_cat_balanced": {k: v[1] for k, v in ep0_h.items()},
            "holdout_per_cat_ce": {k: v[2] for k, v in ep0_h.items()},
        })
        if early_stop_metric in ep0_h:
            best_ce = ep0_h[early_stop_metric][0]
            best_V = V.detach().clone().cpu()
            best_mlp_state = {k: v.cpu().clone()
                              for k, v in mlp.state_dict().items()}
            best_epoch = 0
            per_cat_ce_0 = ep0_h[early_stop_metric][2]
            per_cat_n_0 = ep0_h[early_stop_metric][3]
            for c in range(n_cats):
                if per_cat_n_0[c] > 0 and per_cat_ce_0[c] is not None:
                    best_per_cat_ce[c] = float(per_cat_ce_0[c])
                    best_per_cat_V[c] = V.detach()[c].clone().cpu()
                    best_per_cat_epoch[c] = 0

    for epoch in range(n_epochs):
        starts = list(range(0, len(ex_ids_local), batch_size))
        rng.shuffle(starts)
        ep_cat_sum = torch.zeros(n_cats, device=device, dtype=torch.float64)
        ep_cat_cnt = torch.zeros(n_cats, device=device, dtype=torch.long)
        ep_alpha_sum = torch.zeros(n_cats, device=device, dtype=torch.float64)
        ep_alpha_cnt = torch.zeros(n_cats, device=device, dtype=torch.long)

        import time as _time_mod
        _step_t0 = _time_mod.time()
        for si, s in enumerate(starts):
            mb = ex_ids_local[s:s + batch_size]
            # In DDP: materialise grad tensors for every trainable param
            # *before* backward, so sync_gradients sees the same set of
            # grads on every rank (otherwise per-category MLP heads that
            # are unused on this rank will have grad=None while on other
            # ranks they have grads, leading to a mismatched number of
            # all-reduce calls and a NCCL deadlock).
            if distributed:
                opt.zero_grad(set_to_none=False)
                if (not freeze_cats) and V.grad is None:
                    V.grad = torch.zeros_like(V)
                for _p in mlp.parameters():
                    if _p.grad is None:
                        _p.grad = torch.zeros_like(_p)
            else:
                opt.zero_grad(set_to_none=True)
            per_pos, pos_cats, n_pts, alphas = \
                _ce_loss_batch_cats_mlp(
                    base_model, mb, per_example, by_ex, V, mlp,
                    steer_layer, pad_token_id)

            if main_rank and (si + 1) % 50 == 0:
                _elapsed = _time_mod.time() - _step_t0
                _rate = (si + 1) / _elapsed
                _remaining = (len(starts) - si - 1) / _rate if _rate > 0 else 0
                print(f"    [step {si+1}/{len(starts)}] ep={epoch+1} "
                      f"{_elapsed:.0f}s elapsed, {_rate:.2f} step/s, "
                      f"~{_remaining/60:.1f}m left in epoch", flush=True)

            if n_pts == 0:
                if distributed:
                    # Dummy backward so all ranks stay in sync (rare edge
                    # case: this batch had no positions on this rank but
                    # other ranks may have positions and call collectives).
                    # When V is frozen we only need a path through the
                    # mlp parameters.
                    dummy = V.sum() * 0.0 if not freeze_cats else \
                        torch.zeros((), device=device, dtype=torch.float32)
                    for _p in mlp.parameters():
                        dummy = dummy + _p.sum() * 0.0
                    dummy.backward()
                    sync_gradients(V, mlp)
                    if mlp_grad_clip > 0:
                        torch.nn.utils.clip_grad_norm_(
                            mlp.parameters(), mlp_grad_clip)
                    opt.step()
                pbar.update(1)
                continue

            cat_sum = torch.zeros(n_cats, device=device,
                                  dtype=per_pos.dtype)
            cat_cnt = torch.zeros(n_cats, device=device,
                                  dtype=per_pos.dtype)
            cat_sum.scatter_add_(0, pos_cats, per_pos)
            cat_cnt.scatter_add_(0, pos_cats, torch.ones_like(per_pos))
            mask = cat_cnt > 0
            cat_mean = cat_sum[mask] / cat_cnt[mask]
            loss = cat_mean.mean()
            loss.backward()

            if distributed:
                sync_gradients(V, mlp)

            if mlp_grad_clip > 0:
                torch.nn.utils.clip_grad_norm_(
                    mlp.parameters(), mlp_grad_clip)
            opt.step()

            if (norm_cap_R is not None and norm_cap_R > 0.0
                    and not freeze_cats):
                with torch.no_grad():
                    row_n = V.norm(dim=-1, keepdim=True).clamp_min(1e-12)
                    scale = torch.clamp_max(norm_cap_R / row_n, 1.0)
                    V.mul_(scale)

            with torch.no_grad():
                ep_cat_sum.scatter_add_(0, pos_cats,
                                        per_pos.detach().double())
                ep_cat_cnt.scatter_add_(
                    0, pos_cats,
                    torch.ones_like(pos_cats, dtype=torch.long))
                if alphas is not None:
                    ep_alpha_sum.scatter_add_(
                        0, pos_cats, alphas.double().to(device))
                    ep_alpha_cnt.scatter_add_(
                        0, pos_cats,
                        torch.ones_like(pos_cats, dtype=torch.long))
            pbar.set_postfix(
                ep=f"{epoch+1}/{n_epochs}",
                ce=f"{loss.item():.4f}",
                nrm=f"[{V.detach().norm(dim=-1).mean().item():.1f}]",
                alpha=f"{alphas.mean().item():.2f}" if alphas is not None else "?")
            pbar.update(1)

        norms = V.detach().norm(dim=-1)
        mean_ce = (float(ep_cat_sum.sum().item())
                   / max(int(ep_cat_cnt.sum().item()), 1))
        alpha_mean_per_cat = [
            float((ep_alpha_sum[i] / ep_alpha_cnt[i]).item())
            if int(ep_alpha_cnt[i].item()) > 0 else None
            for i in range(n_cats)]

        # Gradient norms for diagnostics
        v_grad_norm = V.grad.norm().item() if V.grad is not None else 0.0
        if getattr(mlp, "per_cat", False):
            # Per-category MLPs: report combined grad norm under "heads".
            mlp_trunk_gnorm = 0.0
            mlp_heads_gnorm = sum(
                p.grad.norm().item() ** 2
                for sub in mlp.mlps for p in sub.parameters()
                if p.grad is not None) ** 0.5
        else:
            mlp_trunk_gnorm = sum(
                p.grad.norm().item() ** 2
                for p in mlp.trunk.parameters() if p.grad is not None) ** 0.5
            mlp_heads_gnorm = sum(
                p.grad.norm().item() ** 2
                for head in mlp.heads for p in head.parameters()
                if p.grad is not None) ** 0.5

        # Holdout eval (all ranks participate; results all-reduced inside)
        h_ces: Dict[str, float] = {}       # sample-weighted (logged primary)
        h_ces_cb: Dict[str, float] = {}    # cat-balanced (secondary diagnostic)
        h_per_cat_ce: Dict[str, List[Optional[float]]] = {}
        h_per_cat_n: Dict[str, List[int]] = {}
        if holdout_sets:
            for hkey, hrecs in holdout_sets.items():
                ho_ex = holdout_per_example[hkey]
                sw_ce, cb_ce, per_cat_ce_h, per_cat_n_h = _holdout_eval_cats_mlp(
                    base_model, ho_ex, hrecs, V.detach(), mlp,
                    n_cats=n_cats, steer_layer=steer_layer,
                    pad_token_id=pad_token_id, batch_size=batch_size,
                    distributed=distributed)
                h_ces[hkey] = sw_ce
                h_ces_cb[hkey] = cb_ce
                h_per_cat_ce[hkey] = per_cat_ce_h
                h_per_cat_n[hkey] = per_cat_n_h

        h_str = ("  " + "  ".join(f"{k}_ce={v:.4f}"
                                   for k, v in h_ces.items())
                 if h_ces else "")
        if main_rank:
            print(f"    [cats+mlp] epoch {epoch+1}/{n_epochs}: "
                  f"mean_ce={mean_ce:.4f}  norms=[{float(norms.min()):.2f}-"
                  f"{float(norms.max()):.2f}]  alpha_mean="
                  f"{[f'{x:.2f}' if x is not None else '-' for x in alpha_mean_per_cat]}"
                  f"  grad_V={v_grad_norm:.3f}  grad_trunk={mlp_trunk_gnorm:.3f}"
                  f"  grad_heads={mlp_heads_gnorm:.3f}"
                  f"{h_str}", flush=True)

        metrics.append({
            "phase": "cats_mlp", "epoch": epoch + 1,
            "mean_ce": mean_ce,
            "norms_per_cat": [float(x) for x in norms.tolist()],
            "alpha_mean_per_cat": alpha_mean_per_cat,
            "grad_norms": {"V": v_grad_norm,
                           "mlp_trunk": mlp_trunk_gnorm,
                           "mlp_heads": mlp_heads_gnorm},
            "holdout_ce": h_ces,
            "holdout_ce_cat_balanced": h_ces_cb,
            "holdout_per_cat_ce": h_per_cat_ce,
            "holdout_per_cat_n": h_per_cat_n,
        })

        # Per-epoch checkpoint of V + MLP (rank 0 only in DDP mode)
        if save_per_epoch_dir is not None and main_rank:
            os.makedirs(save_per_epoch_dir, exist_ok=True)
            _ck = {
                "epoch": epoch + 1,
                "V": V.detach().float().cpu().clone(),
                "mlp_state": {k: v.detach().cpu().clone()
                              for k, v in mlp.state_dict().items()},
                "cat_keys": list(cat_key_lookup)
                            if cat_key_lookup is not None else None,
                "mean_ce": mean_ce,
                "holdout_ce": h_ces,
                "norms_per_cat": [float(x) for x in norms.tolist()],
                "alpha_mean_per_cat": alpha_mean_per_cat,
            }
            _ck_path = os.path.join(
                save_per_epoch_dir, f"epoch_{epoch+1:02d}.pt")
            torch.save(_ck, _ck_path)

        # Early stopping
        global_improved = False
        if early_stop_metric in h_ces:
            cur = h_ces[early_stop_metric]
            if best_ce is None or cur < best_ce - 1e-6:
                best_ce = cur
                best_V = V.detach().clone().cpu()
                best_mlp_state = {k: v.cpu().clone()
                                  for k, v in mlp.state_dict().items()}
                best_epoch = epoch + 1

            per_cat_ce_h = h_per_cat_ce.get(early_stop_metric, [])
            per_cat_n_h = h_per_cat_n.get(early_stop_metric, [])
            V_cpu_now = V.detach().cpu()
            for c in range(n_cats):
                if (c >= len(per_cat_n_h) or per_cat_n_h[c] <= 0
                        or per_cat_ce_h[c] is None):
                    no_improve_per_cat[c] += 1
                    continue
                cur_c = float(per_cat_ce_h[c])
                if (best_per_cat_ce[c] is None
                        or cur_c < best_per_cat_ce[c] - 1e-6):
                    best_per_cat_ce[c] = cur_c
                    best_per_cat_V[c] = V_cpu_now[c].clone()
                    best_per_cat_epoch[c] = epoch + 1
                    no_improve_per_cat[c] = 0
                    global_improved = True
                else:
                    no_improve_per_cat[c] += 1

            if global_improved:
                no_improve = 0
            else:
                no_improve += 1
                if patience > 0 and no_improve >= patience:
                    if main_rank:
                        print(f"    [cats+mlp] early stop at epoch {epoch+1} "
                              f"(no improvement for {patience} epochs)",
                              flush=True)
                    break
    pbar.close()

    # Assemble final V
    if per_cat_best:
        V_out = V.detach().float().cpu().clone()
        for c in range(n_cats):
            if best_per_cat_ce[c] is not None:
                V_out[c] = best_per_cat_V[c]
        if main_rank:
            print(f"  [cats+mlp] per-cat best: epochs={best_per_cat_epoch}",
                  flush=True)
    elif best_V is not None:
        V_out = best_V.float().cpu()
        if main_rank:
            print(f"  [cats+mlp] global best epoch={best_epoch} ce={best_ce:.4f}",
                  flush=True)
    else:
        V_out = V.detach().float().cpu()

    # Restore best MLP state
    mlp_out = CatCoefMLP(d_in=hidden_size, n_cats=n_cats,
                         d_hidden=mlp_hidden_dim,
                         per_cat=mlp_per_cat)
    if best_mlp_state is not None:
        mlp_out.load_state_dict(best_mlp_state)
    else:
        mlp_out.load_state_dict(mlp.state_dict())

    return V_out, mlp_out, metrics


# ---------------------------------------------------------------------------
# Custom training-mix data loader (for --train_data_file mode)
# ---------------------------------------------------------------------------

def _load_trainmix_responses(data_file: str, thinking_model_short: str,
                              cache_root: str, *, n_take: int = 999999,
                              thinking_model=None,
                              thinking_tokenizer=None,
                              max_new_tokens: int = 1024,
                              rollouts_temp_label: str = "0",
                              rollouts_max_tokens: int = 2000,
                              rollouts_sample_idx: int = -1,
                              truncate_answer_box: bool = False) -> List[dict]:
    """Load questions from a prepared JSONL training-mix file and pair
    them with cached thinking-model rollouts.

    The rollout cache filename follows ``_rollout_filename(...)``; see
    docstring there.  Each cached line: {"dataset_idx": int, "response": str, ...}
    where dataset_idx matches the "idx" field in the data_file.
    """
    import json as _json

    questions: Dict[int, str] = {}
    sources: Dict[int, str] = {}
    with open(data_file) as f:
        for line in f:
            rec = _json.loads(line)
            questions[rec["idx"]] = rec["question"]
            if "source" in rec:
                sources[rec["idx"]] = rec["source"]
    print(f"  [trainmix] loaded {len(questions)} questions from "
          f"{data_file}", flush=True)

    cache_path = os.path.join(
        cache_root,
        _rollout_filename(thinking_model_short, "trainmix",
                          temp_label=rollouts_temp_label,
                          max_tokens=rollouts_max_tokens,
                          sample_idx=rollouts_sample_idx))
    cache_by_idx: Dict[int, str] = {}
    if os.path.exists(cache_path):
        with open(cache_path) as f:
            for line in f:
                try:
                    r = _json.loads(line)
                except _json.JSONDecodeError:
                    continue
                cache_by_idx[int(r["dataset_idx"])] = r["response"]
        print(f"  [trainmix] cache hits: {len(cache_by_idx)} in "
              f"{cache_path}", flush=True)
    else:
        print(f"  [trainmix] WARNING: no cache at {cache_path}", flush=True)

    out: List[dict] = []
    n_miss = 0
    for idx in sorted(questions.keys()):
        if len(out) >= n_take:
            break
        if idx in cache_by_idx and cache_by_idx[idx].strip():
            out.append({
                "original_message": {"role": "user",
                                     "content": questions[idx]},
                "full_response": cache_by_idx[idx],
                "annotated_thinking": {"_trainmix": True},
                "dataset_name": "trainmix",
                "source": sources.get(idx),
                "question_id": idx,
            })
        else:
            n_miss += 1
    print(f"  [trainmix] loaded {len(out)} responses, {n_miss} cache misses",
          flush=True)
    if truncate_answer_box:
        out = _apply_truncate_answer_box(out, dataset_label="trainmix")
    return out


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--base_model", type=str, required=True)
    p.add_argument("--thinking_model", type=str, required=True)
    p.add_argument("--thinking_model_short", type=str, required=True,
                   help="Short name used in annotated_responses_<name>.json")
    p.add_argument("--steer_layer", type=int, required=True,
                   help="Layer at which bias and cat vectors are added "
                        "to the base model's residual stream.  Must "
                        "match hybrid_eval's --old_vectors_layer at "
                        "inference.")
    p.add_argument("--sae_layer", type=int, required=True,
                   help="Layer at which the SAE reads the thinking-"
                        "model activation to label each position.  "
                        "Must match hybrid_eval's --sae_layer.")
    p.add_argument("--sae_n_clusters", type=int, required=True,
                   help="Number of SAE clusters / categories (matches "
                        "hybrid_eval's --n_clusters).")
    p.add_argument("--n_train_examples", "--n_mmlu_examples",
                   type=int, default=10000, dest="n_mmlu_examples",
                   help="Total MMLU-Pro pool size (BEFORE holdout "
                        "split).  Caps at the number of available "
                        "annotated responses.  --mmlu_holdout_frac of "
                        "this is set aside as a held-out selection "
                        "set; the remainder is the cats train pool. "
                        "Bias trains on a (possibly smaller) subset "
                        "of the cats train pool, controlled by "
                        "--n_bias_examples.")
    p.add_argument("--n_bias_examples", type=int, default=4096,
                   help="Number of MMLU-Pro train-pool examples used "
                        "for BIAS training.  Must be <= the cats "
                        "train pool size; bias_train is a random "
                        "subset of cats_train so the bias holdout "
                        "and cats holdout are the same MMLU subset.")
    p.add_argument("--mmlu_categories", type=str, default="",
                   help="Comma-separated MMLU-Pro categories to keep "
                        "for training (e.g. 'math,physics,chemistry,"
                        "engineering').  Empty = use all 14 categories. "
                        "Filtering happens AFTER merging responses with "
                        "annotations and BEFORE the train/holdout split.")
    p.add_argument("--train_dataset", type=str, default="mmlu_pro",
                   choices=["mmlu_pro", "math500", "hendrycks_math",
                            "mmlu_auxiliary_train"],
                   help="Training data source.  'mmlu_pro' (default) "
                        "uses the annotated MMLU-Pro responses; "
                        "'math500' uses the cached MATH-500 thinking "
                        "rollouts as the train pool (also reserves a "
                        "tail slice as in-distribution test); "
                        "'hendrycks_math' uses the 12000-problem "
                        "Hendrycks MATH `train` split (disjoint from "
                        "MATH-500) as the train pool; "
                        "'mmlu_auxiliary_train' uses MMLU's "
                        "auxiliary_train split (~99,842 MCQA from "
                        "ARC/MC_TEST/OBQA/RACE, disjoint from MMLU/"
                        "MMLU-Pro/MATH/GSM8K test sets) -- MMLU-shape "
                        "at ~8x the data of MMLU-Pro train.  Under "
                        "'hendrycks_math' / 'mmlu_auxiliary_train' "
                        "the math500_oos and gsm8k_oos diagnostics "
                        "remain TRUE OOS (loaded from their own caches) "
                        "so per-epoch curves track generalisation, not "
                        "memorisation.  Under 'math500' the script "
                        "loads exactly --n_mmlu_examples MATH-500 "
                        "items from index 0 and uses the next "
                        "--oos_math500_n items (in dataset order) as "
                        "the in-distribution held-out test set.")
    p.add_argument("--max_seq_len", type=int, default=2048,
                   help="Skip responses whose tokenised "
                        "prompt+thinking exceeds this length.")
    p.add_argument("--max_positions_per_example", type=int, default=64,
                   help="Per-(example, cat) position cap during "
                        "training, prevents single responses from "
                        "dominating.  Bias training uses the same cap "
                        "across the unioned set.")
    p.add_argument("--collect_batch_size", type=int, default=8,
                   help="Batch size for the Phase A collection forward "
                        "(base + thinking forwards padded together).")
    p.add_argument("--train_batch_size", type=int, default=4,
                   help="Batch size for bias + cats backward passes.")
    p.add_argument("--filter_batch_size", type=int, default=8,
                   help="Batch size for Phase C residual filter "
                        "forwards (no backward).")
    p.add_argument("--bias_lr", type=float, default=1e-2)
    p.add_argument("--cats_lr", type=float, default=1e-2)
    p.add_argument("--bias_epochs", type=int, default=25,
                   help="Max bias epochs.  With --patience>0 we stop "
                        "early on no MMLU-holdout improvement.")
    p.add_argument("--cats_epochs", type=int, default=10,
                   help="Max cats epochs.  With --patience>0 we stop "
                        "early on no MMLU-holdout improvement.")
    p.add_argument("--patience", type=int, default=5,
                   help="Early-stopping patience on the MMLU holdout "
                        "(0 = disabled, always run full epochs).")
    p.add_argument("--train_holdout_frac", "--mmlu_holdout_frac",
                   type=float, default=0.10, dest="train_holdout_frac",
                   help="Fraction of MMLU-Pro responses held out from "
                        "training and used to track holdout CE every "
                        "epoch.  Drawn after shuffling.")
    p.add_argument("--oos_math500_n", type=int, default=500,
                   help="Number of MATH500 examples to pull as an OOS "
                        "DIAGNOSTIC holdout set (no training, no "
                        "selection, holdout CE only -- used for "
                        "reporting OOD transfer).")
    p.add_argument("--oos_gsm8k_n", type=int, default=500,
                   help="Number of GSM8K examples to pull as an OOS "
                        "DIAGNOSTIC holdout set (no training, no "
                        "selection, holdout CE only -- used for "
                        "reporting OOD transfer).")
    p.add_argument("--weight_decay", type=float, default=0.01,
                   help="L2 weight decay applied to bias and cat "
                        "vectors via AdamW.  Acts as a Gaussian prior "
                        "centred at zero; small-data cats stay near "
                        "the prior, data-rich cats pull free.  Set 0 "
                        "to disable (uses plain Adam).")
    p.add_argument("--norm_cap_alpha", type=float, default=0.0,
                   help="If > 0, enforce ||bias|| <= alpha*h_bar and "
                        "||cat_k|| <= alpha*h_bar after every optimizer "
                        "step via hard L2 projection.  h_bar = median "
                        "||h_resid||_2 at the steer layer measured on a "
                        "sample of disagreement positions (calibrated "
                        "once at training start).  Recommended: 0.5. "
                        "0 disables the cap (legacy behaviour).")
    p.add_argument("--norm_cap_calibration_n", type=int, default=256,
                   help="Number of disagreement positions sampled to "
                        "compute h_bar.  Only used if "
                        "--norm_cap_alpha > 0.")
    p.add_argument("--cat_topk_pct", type=float, default=1.0,
                   help="If < 1.0, for each cat keep only the top-x%% of "
                        "disagreement records (residual after bias filter) "
                        "ranked by SAE activation magnitude at collection "
                        "time.  E.g. 0.25 keeps the top-25%% strongest-"
                        "firing positions per cat for Phase C training.  "
                        "Holdout sets are NOT filtered.  1.0 (default) "
                        "disables the filter.  Ignored when "
                        "--cat_filter=gmm_valley is set.")
    p.add_argument("--cat_filter", type=str, default="none",
                   choices=["none", "topk", "gmm_valley"],
                   help="Phase C activation filter mode.  "
                        "'none': use all residual disagreements.  "
                        "'topk': keep top-x%% by activation magnitude "
                        "(set --cat_topk_pct).  "
                        "'gmm_valley': fit a 2-component GMM per category "
                        "and keep only positions above the valley between "
                        "the two components.  For unimodal categories "
                        "(separation ratio < 1.5) falls back to p75.")
    p.add_argument("--cats_per_cat_best", action="store_true",
                   default=True,
                   help="If set (default), save each cat's V row at "
                        "its individually best epoch on per-cat MMLU "
                        "holdout CE.  Required for robust handling of "
                        "cat-size imbalance (tiny cats hit best fast).")
    p.add_argument("--cats_global_best", dest="cats_per_cat_best",
                   action="store_false",
                   help="Save all V rows at the same global-best "
                        "epoch (legacy v2/v3 behaviour).")
    p.add_argument("--oos_cache_dir", type=str,
                   default="../hybrid/results/response_cache",
                   help="Directory containing cached thinking rollouts "
                        "for the trainmix + OOS datasets, in the format "
                        "thinking_<short>_<dataset>_temp<L>_max<T>[_s<N>].jsonl.")
    p.add_argument("--rollouts_temp_label", type=str, default="0",
                   help="Temperature label substring in rollout "
                        "filenames (e.g. '0' or '0.6'). Must match the "
                        "label produced by vllm-serve/generate_rollouts.py.")
    p.add_argument("--rollouts_max_tokens", type=int, default=2000,
                   help="Max-tokens substring in rollout filenames "
                        "(e.g. 2000 for legacy, 2048 for final).")
    p.add_argument("--rollouts_sample_idx", type=int, default=-1,
                   help="Sample-index substring '_s<N>' in rollout "
                        "filenames. -1 (default) omits the suffix.")
    p.add_argument("--think_prompt_family", type=str, default="auto",
                   choices=["auto", "orz", "r1", "qwq", "other"],
                   help="Family used to shape the thinking-model user "
                        "content during disagreement collection.  Must "
                        "match the shaping used at generation time so "
                        "teacher-forced (prompt+response) tokens line up "
                        "with what the model actually saw.  'auto' (the "
                        "default) detects from --thinking_model.")
    p.add_argument("--math_directive_mode", type=str, default="none",
                   choices=["none", "always", "auto"],
                   help="Whether to append the R1/QwQ 'Please reason step "
                        "by step ...' directive when shaping the user "
                        "content.  'none' (default, legacy) never; "
                        "'always' every row; 'auto' on known math "
                        "benchmarks (math500, gsm8k, aime*) and on "
                        "trainmix rows whose source ∈ {hendrycks_math, "
                        "natural_reasoning}.  Set to 'auto' for the final "
                        "run if generate_rollouts.py used "
                        "--math_directive_mode auto/always.  ORZ shaping "
                        "is unconditional and supersedes this flag.")
    p.add_argument("--base_prompt_style", type=str, default="default",
                   choices=["default", "stepwise", "boxed",
                            "legacy_task", "think_template", "simple",
                            "qa_response", "qa_instr", "think_qa",
                            "think_qa_marker",
                            "think_word", "convo_marker", "convo_marker_v2",
                            "convo_continue", "convo_reason",
                            "orz_think_template", "r1_think_template",
                            "qwq_think_template",
                            "plain_chat_math",
                            "convo_think", "mini_preamble",
                            "orz_full", "r1_plain", "qwq_plain",
                            "step_preamble"],
                   help="Base-model prompt format used when teacher-forcing "
                        "base activations during disagreement collection.  "
                        "'default' bare 'User: {q}\\nAssistant:'. "
                        "'stepwise' (ff v1) prepends step-by-step "
                        "directive. 'boxed' (ff v2) places QwQ/R1+ORZ "
                        "\\boxed{} directive after question. "
                        "'legacy_task' (ff v3) uses the origin/main "
                        "Task/Question/Answer scaffolding. MUST match "
                        "the prompt style of the cached base rollouts.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--save_dir", type=str, required=True)
    p.add_argument("--responses_dir", type=str,
                   default="../generate-responses/results/vars")
    p.add_argument("--sae_disable_mean", action="store_true",
                   help="Skip activation-mean centering during SAE "
                        "classification (auto-enabled if the SAE "
                        "checkpoint has no activation_mean).")
    # --- MLP coefficient learning mode ---
    p.add_argument("--no_bias", action="store_true",
                   help="Skip Phase B (bias training) and Phase C "
                        "(residual filter).  Bias is a zero vector.")
    p.add_argument("--mlp_coef", action="store_true",
                   help="Enable MLP-based coefficient prediction.  "
                        "Jointly trains V (cat vectors) and a CatCoefMLP "
                        "that predicts per-position coefficients from "
                        "the residual stream.  Requires --no_bias.")
    p.add_argument("--mlp_hidden_dim", type=int, default=128,
                   help="Hidden dimension of the CatCoefMLP trunk. "
                        "Doubled from 64 to 128 for the final run.")
    p.add_argument("--mlp_per_cat", action="store_true",
                   help="Train one independent MLP per category "
                        "(Linear(d_in,d_hidden)->GELU->Linear(d_hidden,1) "
                        "each) instead of a shared trunk + per-cat linear "
                        "heads. Each category's coefficient predictor has "
                        "its own weights.")
    p.add_argument("--mlp_lr", type=float, default=1e-3,
                   help="Learning rate for MLP parameters (V uses "
                        "--cats_lr).")
    p.add_argument("--mlp_grad_clip", type=float, default=1.0,
                   help="Gradient clipping for MLP parameters.")
    p.add_argument("--freeze_cats", action="store_true",
                   help=("ABLATION: freeze the category-vector matrix V "
                         "during cats+MLP training; only the MLP's "
                         "parameters are updated.  Typically used with "
                         "--init_cats_from_dir to initialise V from "
                         "frozen random gaussian vectors whose per-row "
                         "norms match a previously-trained reference."))
    p.add_argument("--init_cats_from_dir", type=str, default=None,
                   help=("Path to a previous run's save_dir containing "
                         "<model_short>_idx<k>_linear.pt files.  When "
                         "set, V is initialised from N(0, I) and each "
                         "row is rescaled so its L2 norm matches the "
                         "corresponding trained vector.  Direction is "
                         "random and unrelated to the reference."))
    p.add_argument("--rand_cats_seed", type=int, default=1337,
                   help=("Seed for the random gaussian used to populate "
                         "V when --init_cats_from_dir is set."))
    p.add_argument("--train_data_file", type=str, default=None,
                   help="Path to custom training JSONL (from "
                        "prepare_training_mix.py).  Each line: "
                        "{\"idx\": int, \"question\": str, ...}.  "
                        "When set, ignores --train_dataset.")
    p.add_argument("--val_data_file", type=str, default=None,
                   help="Path to custom validation JSONL (from "
                        "prepare_training_mix.py).")
    p.add_argument("--max_memory_per_gpu", type=str, default=None,
                   help="Per-GPU memory limit for the base model load, "
                        "e.g. '35GiB'.  Forces the base model to spread "
                        "across all GPUs (useful for 32B models on 4+ "
                        "GPUs so both models share all GPUs instead of "
                        "the base greedily filling the first GPUs).")
    p.add_argument("--use_fsdp", action="store_true",
                   help="(deprecated, ignored)")
    p.add_argument("--distributed", action="store_true",
                   help="Data-parallel training via torchrun.  Each rank "
                        "loads a full copy of the (frozen) base model on "
                        "its local GPU, processes its shard of examples, "
                        "and gradients for V+MLP are all-reduced.")
    p.add_argument("--save_per_epoch_ckpts", action="store_true",
                   help="If set, save V + MLP state at the end of every "
                        "epoch under {save_dir}/epoch_checkpoints/ so we "
                        "can experiment with intermediate vectors.")
    p.add_argument("--eval_percat_only", action="store_true",
                   help="Skip training: load the saved V (per-cat linear "
                        "files) + CatCoefMLP from --save_dir and recompute "
                        "per-category holdout CE on trainmix_holdout, "
                        "math500_oos, gsm8k_oos (same sets used during "
                        "training logging). Writes --eval_percat_out.")
    p.add_argument("--eval_percat_out", type=str, default="per_cat_ce_eval.json",
                   help="Filename (under --save_dir) for the eval_percat_only "
                        "JSON dump.")
    p.add_argument("--eval_percat_holdouts", type=str,
                   default="trainmix_holdout,math500_oos,gsm8k_oos",
                   help="Comma-separated holdout sets to evaluate in "
                        "--eval_percat_only mode. For cheap vector SELECTION "
                        "pass 'trainmix_holdout' only (avoids touching the "
                        "math500/gsm8k eval sets).")
    p.add_argument("--disagree_cache", type=str, default=None,
                   help="Path to cached disagreement data (.pt).  If "
                        "provided and exists, skip Phase A and load from "
                        "cache.  If --collect_only, save to this path.")
    p.add_argument("--collect_only", action="store_true",
                   help="Only run Phase A (disagreement collection), "
                        "save to --disagree_cache, and exit.  Used to "
                        "separate collection from FSDP training.")
    p.add_argument("--single_vector", action="store_true",
                   help="Merge all SAE categories into a single category "
                        "(cat 0) and train one global vector + 1-head MLP. "
                        "Tests whether gains come from a global shift vs "
                        "category-specific steering.")
    p.add_argument("--truncate_answer_box", action="store_true",
                   help="Truncate every loaded thinking-rollout response "
                        "to end immediately after the LAST \\boxed{...} "
                        "expression in the string (brace-matched).  "
                        "Records with no well-formed \\boxed{} are dropped. "
                        "Useful for models whose rollouts degenerate into "
                        "post-answer junk (e.g. orz-0.5b looping </answer> "
                        "tags) so the disagreement cache and MLP only "
                        "learn from clean reasoning + final answer.")
    args = p.parse_args()

    # Resolve 'auto' for --think_prompt_family from the HF model id.
    if args.think_prompt_family == "auto":
        args.think_prompt_family = _detect_think_family(args.thinking_model)
    print(f"[prompt-shaping] think_prompt_family="
          f"{args.think_prompt_family}  "
          f"math_directive_mode={args.math_directive_mode}", flush=True)

    # ---- DDP init (before any CUDA ops, before model load) ----
    _ddp_local_rank = 0
    if args.distributed:
        _ddp_local_rank = init_distributed()
        if is_main():
            os.makedirs(args.save_dir, exist_ok=True)
        import torch.distributed as _dist
        _dist.barrier()
    else:
        os.makedirs(args.save_dir, exist_ok=True)

    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # ---- Load base tokenizer (always needed) -------------------------
    base_tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    if base_tokenizer.pad_token_id is None:
        base_tokenizer.pad_token = base_tokenizer.eos_token

    # ---- Phase A: disagreement collection ----------------------------
    # When using FSDP, Phase A should have been done beforehand and
    # cached in --disagree_cache.  Otherwise run it now.
    _skip_phase_a = (args.disagree_cache
                     and os.path.exists(args.disagree_cache)
                     and not args.collect_only)

    thinking_model = None
    base_model = None

    if not _skip_phase_a:
        # Load both models with device_map="auto" for collection
        base_max_mem = None
        if args.max_memory_per_gpu:
            n_gpus = torch.cuda.device_count()
            base_max_mem = {i: args.max_memory_per_gpu for i in range(n_gpus)}
            print(f"  [multi-gpu] {n_gpus} GPUs, base model max_memory="
                  f"{args.max_memory_per_gpu}/GPU", flush=True)

        print(f"Loading base model {args.base_model}...", flush=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model, device_map="auto", torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            max_memory=base_max_mem)
        for p_ in base_model.parameters():
            p_.requires_grad = False
        base_model.eval()

        print(f"Loading thinking model {args.thinking_model}...", flush=True)
        thinking_tokenizer = AutoTokenizer.from_pretrained(
            args.thinking_model)
        if thinking_tokenizer.pad_token_id is None:
            thinking_tokenizer.pad_token = thinking_tokenizer.eos_token
        thinking_model = AutoModelForCausalLM.from_pretrained(
            args.thinking_model, device_map="auto",
            torch_dtype=torch.bfloat16)
        for p_ in thinking_model.parameters():
            p_.requires_grad = False
        thinking_model.eval()
    else:
        print(f"Loading thinking model tokenizer for alignment...",
              flush=True)
        thinking_tokenizer = AutoTokenizer.from_pretrained(
            args.thinking_model)
        if thinking_tokenizer.pad_token_id is None:
            thinking_tokenizer.pad_token = thinking_tokenizer.eos_token

    # Read hidden_size / n_layers from config (no model load needed)
    from transformers import AutoConfig
    _cfg = AutoConfig.from_pretrained(args.base_model)
    hidden = _cfg.hidden_size
    n_layers = _cfg.num_hidden_layers
    assert 0 <= args.steer_layer < n_layers, (
        f"steer_layer={args.steer_layer} out of range [0,{n_layers})")

    # Vocab + a few canonical strings must tokenise identically so base
    # positions and thinking positions align after the anchor.
    if thinking_model is not None and base_model is not None:
        assert thinking_model.config.vocab_size == base_model.config.vocab_size, (
            "base/thinking vocab mismatch -- the cross-model alignment is "
            "only valid for paired models from the same tokenizer family.")
    for _s in (" the quick brown fox jumps over the lazy dog.",
               "\n\nLet me think step by step.\n\n",
               "Therefore, the answer is \\boxed{42}."):
        b_ids = base_tokenizer(_s, add_special_tokens=False)["input_ids"]
        t_ids = thinking_tokenizer(_s, add_special_tokens=False)["input_ids"]
        assert b_ids == t_ids, (
            f"Tokenizer mismatch on probe {_s!r}: "
            f"base={b_ids[:10]} thinking={t_ids[:10]}")

    # ---- Load SAE classifier (only needed for Phase A collection) -----
    sae_obj = None
    sae_act_mean = None
    sae_dev = None
    sae_disable_mean = False
    if not _skip_phase_a:
        from utils.sae import load_sae
        think_id = args.thinking_model.split("/")[-1].lower()
        sae_obj, _ = load_sae(think_id, args.sae_layer, args.sae_n_clusters,
                              require_activation_mean=False)
        sae_obj = sae_obj.to(next(thinking_model.parameters()).device)
        sae_obj.eval()
        for _p in sae_obj.parameters():
            _p.requires_grad = False
        sae_disable_mean = (args.sae_disable_mean
                            or not hasattr(sae_obj, "activation_mean"))
        sae_act_mean = (sae_obj.activation_mean
                        if hasattr(sae_obj, "activation_mean")
                        and not sae_disable_mean else None)
        sae_dev = next(sae_obj.parameters()).device

    @torch.no_grad()
    def sae_classifier(acts):  # (Lt, hidden) on the thinking device
        x = acts.float().to(sae_dev)
        if sae_act_mean is not None:
            x = x - sae_act_mean.to(sae_dev)
            x = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
        la = sae_obj.encoder(x - sae_obj.b_dec)
        ids = la.argmax(dim=-1)
        vals = la[torch.arange(la.shape[0], device=sae_dev), ids]
        return ids.cpu(), vals.float().cpu()

    if sae_obj is not None:
        print(f"  [sae] {args.thinking_model.split('/')[-1].lower()} "
              f"layer {args.sae_layer} "
              f"clusters {args.sae_n_clusters} "
              f"disable_mean={sae_disable_mean}",
              flush=True)

    # ---- Load training responses ------------------------------------
    if args.train_data_file:
        # Custom training-mix mode (from prepare_training_mix.py)
        print(f"Loading training data from {args.train_data_file}",
              flush=True)
        merged = _load_trainmix_responses(
            args.train_data_file, args.thinking_model_short,
            args.oos_cache_dir,
            thinking_model=thinking_model,
            thinking_tokenizer=thinking_tokenizer,
            rollouts_temp_label=args.rollouts_temp_label,
            rollouts_max_tokens=args.rollouts_max_tokens,
            rollouts_sample_idx=args.rollouts_sample_idx,
            truncate_answer_box=args.truncate_answer_box)
        random.shuffle(merged)
        oos_math_preloaded = None

        # Also load val data if provided
        val_data_preloaded = None
        if args.val_data_file:
            print(f"Loading val data from {args.val_data_file}", flush=True)
            val_data_preloaded = _load_trainmix_responses(
                args.val_data_file, args.thinking_model_short,
                args.oos_cache_dir,
                thinking_model=thinking_model,
                thinking_tokenizer=thinking_tokenizer,
                rollouts_temp_label=args.rollouts_temp_label,
                rollouts_max_tokens=args.rollouts_max_tokens,
                rollouts_sample_idx=args.rollouts_sample_idx,
                truncate_answer_box=args.truncate_answer_box)
        else:
            val_data_preloaded = None

    elif args.train_dataset == "math500":
        # In-distribution-test mode.  Load all MATH-500 thinking
        # rollouts (cap = n_mmlu_examples + oos_math500_n), use the
        # first n_mmlu_examples as the train pool (then split into
        # cats_train + mmlu_holdout for selection), and the next
        # oos_math500_n examples as the in-distribution test set
        # (named math500_oos to flow through the same per-epoch
        # holdout-CE machinery, but it is really the
        # held-out *test* set, not OOS).
        n_pool = int(args.n_mmlu_examples) + int(args.oos_math500_n)
        print(f"Loading MATH500 train pool (train_dataset=math500, "
              f"want {n_pool} = {args.n_mmlu_examples} train + "
              f"{args.oos_math500_n} held-out test)", flush=True)
        all_math = _load_oos_responses(
            "math500", n_pool,
            args.thinking_model_short, args.oos_cache_dir,
            thinking_model=thinking_model,
            thinking_tokenizer=thinking_tokenizer,
            truncate_answer_box=args.truncate_answer_box)
        if len(all_math) < n_pool:
            print(f"  WARNING: only {len(all_math)} MATH500 rollouts "
                  f"available; using all of them.  Train pool will be "
                  f"shorter than requested.", flush=True)
        merged = all_math[: int(args.n_mmlu_examples)]
        oos_math_preloaded = all_math[
            int(args.n_mmlu_examples) :
            int(args.n_mmlu_examples) + int(args.oos_math500_n)
        ]
        # The MMLU-Pro category filter is meaningless for math500 --
        # MATH-500 has no MMLU category strings.
        if args.mmlu_categories.strip():
            print("  --mmlu_categories ignored under "
                  "--train_dataset math500", flush=True)
    elif args.train_dataset == "hendrycks_math":
        # Math-only training on the 12000-problem Hendrycks MATH train
        # split.  Disjoint from MATH-500 and GSM8K -- those remain TRUE
        # OOS holdouts (loaded normally further down).  We deliberately
        # do NOT slice math500 out of the train pool here.
        print(f"Loading Hendrycks MATH train pool "
              f"({args.n_mmlu_examples} requested)", flush=True)
        merged = _load_oos_responses(
            "hendrycks_math", int(args.n_mmlu_examples),
            args.thinking_model_short, args.oos_cache_dir,
            thinking_model=thinking_model,
            thinking_tokenizer=thinking_tokenizer,
            truncate_answer_box=args.truncate_answer_box)
        print(f"  {len(merged)} hendrycks_math responses", flush=True)
        if args.mmlu_categories.strip():
            print("  --mmlu_categories ignored under "
                  "--train_dataset hendrycks_math", flush=True)
        # Shuffle so the 10%/4096 train/bias/holdout split sees a
        # random subject mix.  Deterministic given args.seed (set above).
        random.shuffle(merged)
        oos_math_preloaded = None  # math500 loaded as TRUE OOS below
    elif args.train_dataset == "mmlu_auxiliary_train":
        # MMLU auxiliary_train: ~99,842 MCQA questions sourced from
        # ARC/MC_TEST/OBQA/RACE.  Disjoint from MMLU/MMLU-Pro test sets
        # and from MATH/GSM8K, so math500_oos and gsm8k_oos diagnostics
        # remain TRUE OOS.
        print(f"Loading MMLU auxiliary_train pool "
              f"({args.n_mmlu_examples} requested)", flush=True)
        merged = _load_oos_responses(
            "mmlu_auxiliary_train", int(args.n_mmlu_examples),
            args.thinking_model_short, args.oos_cache_dir,
            thinking_model=thinking_model,
            thinking_tokenizer=thinking_tokenizer)
        print(f"  {len(merged)} mmlu_auxiliary_train responses",
              flush=True)
        if args.mmlu_categories.strip():
            print("  --mmlu_categories ignored under "
                  "--train_dataset mmlu_auxiliary_train", flush=True)
        # Shuffle so the train/bias/holdout split sees a random mix
        # across the (mixed) source datasets.
        random.shuffle(merged)
        oos_math_preloaded = None  # math500 loaded as TRUE OOS below
    else:
        # ---- Load annotated MMLU-Pro responses ----------------------
        responses_path = os.path.join(
            args.responses_dir,
            f"responses_{args.thinking_model_short}.json")
        annotated_path = os.path.join(
            args.responses_dir,
            f"annotated_responses_{args.thinking_model_short}.json")
        print(f"Loading responses from {responses_path}", flush=True)
        with open(responses_path) as f:
            raw = json.load(f)
        print(f"Loading annotations from {annotated_path}", flush=True)
        with open(annotated_path) as f:
            ann = json.load(f)
        merged: List[dict] = []
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
        print(f"  {len(merged)} responses with annotations", flush=True)

        # ---- Optional MMLU-Pro category filter ----------------------
        cat_filter_raw = (args.mmlu_categories or "").strip()
        if cat_filter_raw:
            wanted = {c.strip().lower()
                      for c in cat_filter_raw.split(",") if c.strip()}
            before = len(merged)
            merged = [m for m in merged
                      if str(m.get("category", "")).strip().lower() in wanted]
            from collections import Counter as _Counter
            kept_dist = _Counter(m.get("category") for m in merged)
            print(f"  category filter: kept {len(merged)}/{before} "
                  f"responses (filter={sorted(wanted)})", flush=True)
            for k, v in sorted(kept_dist.items(), key=lambda x: -x[1]):
                print(f"    {k:<24}  n={v}", flush=True)
            if not merged:
                raise RuntimeError(
                    "Category filter left zero responses; check "
                    "--mmlu_categories against the actual category "
                    "strings in annotated_responses_<short>.json.")

        random.shuffle(merged)
        oos_math_preloaded = None  # OOS math will be loaded below

    # ---- Split train pool into train + holdout ----------------------
    if args.train_data_file and val_data_preloaded is not None:
        # Custom mode: train and val are separate files
        cats_train = merged
        trainmix_holdout_data = val_data_preloaded
        n_bias = min(args.n_bias_examples, len(cats_train))
        bias_train = cats_train[:n_bias]
        print(f"  train_data_file mode: "
              f"cats_train={len(cats_train)}  "
              f"trainmix_holdout={len(trainmix_holdout_data)}  "
              f"bias_train={len(bias_train)} (subset)", flush=True)
    else:
        # Legacy mode: single pool split by fraction
        n_total = min(args.n_mmlu_examples, len(merged))
        n_holdout = int(round(args.train_holdout_frac * n_total))
        n_train = n_total - n_holdout
        cats_train = merged[:n_train]
        trainmix_holdout_data = merged[n_train:n_total]
        n_bias = min(args.n_bias_examples, n_train)
        bias_train = cats_train[:n_bias]
        print(f"  train_dataset={args.train_dataset} split: "
              f"cats_train={len(cats_train)}  "
              f"bias_train={len(bias_train)} (subset)  "
              f"trainmix_holdout={len(trainmix_holdout_data)}", flush=True)

    # ---- Load OOS / held-out test holdouts --------------------------
    oos_math = oos_gsm = []
    if oos_math_preloaded is not None:
        # math500 train mode: oos_math was already sliced off above.
        oos_math = oos_math_preloaded
        print(f"  [oos:math500] reusing held-out test slice "
              f"({len(oos_math)} responses, idx "
              f"{args.n_mmlu_examples}..{args.n_mmlu_examples + len(oos_math) - 1})",
              flush=True)
    elif args.oos_math500_n > 0:
        oos_math = _load_oos_responses(
            "math500", args.oos_math500_n,
            args.thinking_model_short, args.oos_cache_dir,
            thinking_model=thinking_model,
            thinking_tokenizer=thinking_tokenizer,
            rollouts_temp_label=args.rollouts_temp_label,
            rollouts_max_tokens=args.rollouts_max_tokens,
            rollouts_sample_idx=args.rollouts_sample_idx,
            think_family=args.think_prompt_family,
            math_directive_mode=args.math_directive_mode,
            truncate_answer_box=args.truncate_answer_box)
    if args.oos_gsm8k_n > 0:
        oos_gsm = _load_oos_responses(
            "gsm8k", args.oos_gsm8k_n,
            args.thinking_model_short, args.oos_cache_dir,
            thinking_model=thinking_model,
            thinking_tokenizer=thinking_tokenizer,
            rollouts_temp_label=args.rollouts_temp_label,
            rollouts_max_tokens=args.rollouts_max_tokens,
            rollouts_sample_idx=args.rollouts_sample_idx,
            truncate_answer_box=args.truncate_answer_box,
            think_family=args.think_prompt_family,
            math_directive_mode=args.math_directive_mode)

    # ================================================================
    # Phase A. Collect disagreements (or load from cache).
    # ================================================================
    _disagree_loaded = False
    if args.disagree_cache and os.path.exists(args.disagree_cache) \
            and not args.collect_only:
        print(f"[Phase A] loading cached disagreements from "
              f"{args.disagree_cache}", flush=True)
        _cached = torch.load(args.disagree_cache, map_location="cpu",
                             weights_only=False)
        per_example        = _cached["per_example"]
        per_category       = _cached["per_category"]
        per_category_acts  = _cached["per_category_acts"]
        bias_per_example   = _cached.get("bias_per_example", [])
        bias_per_category  = _cached.get("bias_per_category", {})
        holdout_per_example  = _cached.get("holdout_per_example", {})
        holdout_per_category = _cached.get("holdout_per_category", {})
        full_cat_keys      = _cached.get("full_cat_keys",
                                         [f"idx{c}" for c in range(args.sae_n_clusters)])
        _disagree_loaded = True
        cats_union_records: List[Tuple[int, int, int]] = []
        for k in full_cat_keys:
            per_category.setdefault(k, [])
            cats_union_records.extend(per_category[k])
        print(f"  loaded {len(per_example)} examples, "
              f"{len(cats_union_records)} positions, "
              f"{len(full_cat_keys)} categories", flush=True)
    else:
        print(f"\n[Phase A] collecting disagreements ("
              f"cats_train: {len(cats_train)}, "
              f"bias_train (subset): {len(bias_train)}, "
              f"trainmix_holdout: {len(trainmix_holdout_data)}, "
              f"math500_oos: {len(oos_math)}, gsm8k_oos: {len(oos_gsm)})",
              flush=True)
        print(f"  [phase A: cats_train MMLU]", flush=True)
        per_example, per_category, per_category_acts = collect_disagreements(
            base_model, thinking_model, base_tokenizer, thinking_tokenizer,
            cats_train,
            max_seq_len=args.max_seq_len,
            max_examples=len(cats_train),
            sae_classifier=sae_classifier,
            sae_classify_layer=args.sae_layer,
            collect_batch_size=args.collect_batch_size,
            think_family=args.think_prompt_family,
            math_directive_mode=args.math_directive_mode,
            base_prompt_style=args.base_prompt_style)

        if not args.no_bias:
            bias_train_ids = {id(r) for r in bias_train}
            print(f"  [phase A: bias_train subset]", flush=True)
            bias_per_example, bias_per_category, _bias_per_cat_acts = collect_disagreements(
                base_model, thinking_model, base_tokenizer, thinking_tokenizer,
                bias_train,
                max_seq_len=args.max_seq_len,
                max_examples=len(bias_train),
                sae_classifier=sae_classifier,
                sae_classify_layer=args.sae_layer,
                collect_batch_size=args.collect_batch_size,
                think_family=args.think_prompt_family,
                math_directive_mode=args.math_directive_mode,
                base_prompt_style=args.base_prompt_style)
        else:
            bias_per_example = []
            bias_per_category = {}

        holdout_per_example: Dict[str, List[dict]] = {}
        holdout_per_category: Dict[str, Dict[str, List[Tuple[int, int, int]]]] = {}
        for hkey, hsrc in (("trainmix_holdout", trainmix_holdout_data),
                           ("math500_oos", oos_math),
                           ("gsm8k_oos", oos_gsm)):
            if not hsrc:
                holdout_per_example[hkey] = []
                holdout_per_category[hkey] = {}
                continue
            print(f"  [phase A: {hkey}]", flush=True)
            h_ex, h_cat, _ = collect_disagreements(
                base_model, thinking_model, base_tokenizer, thinking_tokenizer,
                hsrc,
                max_seq_len=args.max_seq_len,
                max_examples=len(hsrc),
                sae_classifier=sae_classifier,
                sae_classify_layer=args.sae_layer,
                collect_batch_size=args.collect_batch_size,
                think_family=args.think_prompt_family,
                math_directive_mode=args.math_directive_mode,
                base_prompt_style=args.base_prompt_style)
            holdout_per_example[hkey] = h_ex
            holdout_per_category[hkey] = h_cat

        full_cat_keys = [f"idx{c}" for c in range(args.sae_n_clusters)]
        for k in full_cat_keys:
            per_category.setdefault(k, [])
        print(f"  cats_train category keys (with counts): "
              f"{[(k, len(per_category[k])) for k in full_cat_keys]}",
              flush=True)

        cats_union_records = []
        for k in full_cat_keys:
            cats_union_records.extend(per_category[k])
        print(f"  cats_train disagreements (union): "
              f"{len(cats_union_records)}", flush=True)
        if not cats_union_records:
            raise RuntimeError("No disagreement positions collected on "
                               "cats_train; check data/cache paths.")

        # Save cache if requested
        if args.disagree_cache:
            print(f"  Saving disagreement cache to {args.disagree_cache}",
                  flush=True)
            os.makedirs(os.path.dirname(args.disagree_cache) or ".",
                        exist_ok=True)
            torch.save({
                "per_example": per_example,
                "per_category": per_category,
                "per_category_acts": per_category_acts,
                "bias_per_example": bias_per_example,
                "bias_per_category": bias_per_category,
                "holdout_per_example": holdout_per_example,
                "holdout_per_category": holdout_per_category,
                "full_cat_keys": full_cat_keys,
            }, args.disagree_cache)

        if args.collect_only:
            print("  --collect_only: done.", flush=True)
            return

    # Free thinking model + SAE: rest of pipeline only needs base.
    if thinking_model is not None:
        try:
            thinking_model.to("meta")
        except Exception:
            pass
        del thinking_model
    import gc
    gc.collect()
    torch.cuda.empty_cache()
    if torch.cuda.is_available():
        torch.cuda.synchronize()

    if args.distributed:
        # Each rank loads a full copy on its local GPU.
        if base_model is not None:
            del base_model
            gc.collect()
            torch.cuda.empty_cache()
        if is_main():
            print(f"[DDP] Loading base model {args.base_model} on "
                  f"{get_world_size()} GPUs...", flush=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model, torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            device_map={"": _ddp_local_rank})
        for p_ in base_model.parameters():
            p_.requires_grad = False
        base_model.eval()
        import torch.distributed as _dist
        _dist.barrier()
        if is_main():
            print(f"[DDP] Base model loaded (rank 0 of {get_world_size()})",
                  flush=True)
    elif base_model is None:
        print(f"Loading base model {args.base_model}...", flush=True)
        base_max_mem = None
        if args.max_memory_per_gpu:
            n_gpus = torch.cuda.device_count()
            base_max_mem = {i: args.max_memory_per_gpu
                           for i in range(n_gpus)}
        base_model = AutoModelForCausalLM.from_pretrained(
            args.base_model, device_map="auto",
            torch_dtype=torch.bfloat16,
            attn_implementation="flash_attention_2",
            max_memory=base_max_mem)
        for p_ in base_model.parameters():
            p_.requires_grad = False
        base_model.eval()

    model_short = args.base_model.split("/")[-1].lower()
    n_cats = len(full_cat_keys)
    key_to_cat = {k: i for i, k in enumerate(full_cat_keys)}

    # Norm calibration
    h_bar: Optional[float] = None
    norm_cap_R: Optional[float] = None
    if args.norm_cap_alpha and args.norm_cap_alpha > 0.0:
        ref_ex = per_example
        ref_recs = cats_union_records
        h_bar = calibrate_h_norm(
            base_model, ref_ex, ref_recs,
            steer_layer=args.steer_layer,
            pad_token_id=base_tokenizer.pad_token_id,
            n_samples=int(args.norm_cap_calibration_n),
            seed=args.seed)
        norm_cap_R = float(args.norm_cap_alpha) * float(h_bar)
        print(f"[norm-cap] h_bar={h_bar:.3f}  R={norm_cap_R:.3f}", flush=True)
    else:
        print("[norm-cap] DISABLED", flush=True)

    # ================================================================
    # Branch: MLP coefficient mode vs legacy bias+cat mode
    # ================================================================
    if args.no_bias and args.mlp_coef:
        # ============================================================
        # MLP MODE: no bias, train V + CatCoefMLP jointly
        # ============================================================

        # --- Single-vector mode: merge all categories into cat 0 ---
        if args.single_vector:
            n_cats = 1
            full_cat_keys = ["global"]
            key_to_cat = {"global": 0}
            merged_positions: List[Tuple[int, int, int]] = []
            merged_acts: List[float] = []
            for k in list(per_category.keys()):
                merged_positions.extend(per_category[k])
                merged_acts.extend(per_category_acts.get(k, [0.0] * len(per_category[k])))
            per_category = {"global": merged_positions}
            per_category_acts = {"global": merged_acts}
            for hkey in list(holdout_per_category.keys()):
                h_merged: List[Tuple[int, int, int]] = []
                for krecs in holdout_per_category[hkey].values():
                    h_merged.extend(krecs)
                holdout_per_category[hkey] = {"global": h_merged}
            print(f"\n[SINGLE VECTOR MODE] Merged all categories into 1 "
                  f"global vector ({len(merged_positions)} positions)",
                  flush=True)
        else:
            print(f"\n[MLP MODE] No bias vector. Training {n_cats} cat vectors "
                  f"+ CatCoefMLP jointly.", flush=True)

        cat_records: List[Tuple[int, int, int, int]] = []
        for k in full_cat_keys:
            ci = key_to_cat[k]
            for ex_idx, pos, target in per_category.get(k, []):
                cat_records.append((ex_idx, pos, ci, target))
        print(f"  total training records: {len(cat_records)}", flush=True)

        holdout_cat_records: Dict[str, List[Tuple[int, int, int, int]]] = {}
        for hkey in ("trainmix_holdout", "math500_oos", "gsm8k_oos"):
            recs_h: List[Tuple[int, int, int, int]] = []
            for k, krecs in holdout_per_category.get(hkey, {}).items():
                if k not in key_to_cat:
                    continue
                ci = key_to_cat[k]
                for ex_idx, pos, target in krecs:
                    recs_h.append((ex_idx, pos, ci, target))
            if recs_h:
                holdout_cat_records[hkey] = recs_h
        print("  holdout sizes: "
              + ", ".join(f"{k}={len(v)}" for k, v in
                          holdout_cat_records.items()),
              flush=True)

        # ABLATION: load per-cat norms from a previous run if requested.
        init_cat_norms: Optional[List[float]] = None
        if args.init_cats_from_dir:
            init_cat_norms = []
            missing = []
            for k in full_cat_keys:
                cand = os.path.join(
                    args.init_cats_from_dir,
                    f"{model_short}_{k}_linear.pt")
                if not os.path.exists(cand):
                    missing.append(cand)
                    continue
                d = torch.load(cand, map_location="cpu", weights_only=False)
                # File stores {k: tensor}; tolerate either shape.
                if isinstance(d, dict):
                    vec = d.get(k, next(iter(d.values())))
                else:
                    vec = d
                init_cat_norms.append(float(vec.float().norm().item()))
            if missing:
                raise FileNotFoundError(
                    "init_cats_from_dir missing per-cat files: "
                    + ", ".join(missing))
            print(f"  [ablation] loaded reference norms from "
                  f"{args.init_cats_from_dir}: "
                  f"{[f'{n:.3f}' for n in init_cat_norms]}", flush=True)

        if args.eval_percat_only:
            from coef_mlp import CatCoefMLP
            device = next(base_model.parameters()).device
            # Rebuild V from the saved per-cat linear files (cat order via
            # key_to_cat so it matches the saved MLP / training).
            V_eval = torch.zeros((n_cats, hidden), dtype=torch.float32)
            for k in full_cat_keys:
                ci = key_to_cat[k]
                vp = os.path.join(args.save_dir, f"{model_short}_{k}_linear.pt")
                dd = torch.load(vp, map_location="cpu", weights_only=False)
                if isinstance(dd, dict):
                    vec = dd.get(k, next(iter(dd.values())))
                else:
                    vec = dd
                V_eval[ci] = vec.float().view(-1)
            V_eval = V_eval.to(device)
            mlp_eval = CatCoefMLP(d_in=hidden, n_cats=n_cats,
                                  d_hidden=args.mlp_hidden_dim,
                                  per_cat=args.mlp_per_cat).to(device).float()
            sd = torch.load(os.path.join(args.save_dir, "cat_coef_mlp.pt"),
                            map_location="cpu", weights_only=False)
            mlp_eval.load_state_dict(sd)
            mlp_eval.eval()
            out = {"per_cat_keys": full_cat_keys, "n_cats": n_cats,
                   "steer_layer": args.steer_layer,
                   "sae_layer": args.sae_layer,
                   "sae_n_clusters": args.sae_n_clusters,
                   "holdouts": {}}
            _req_holdouts = [h.strip() for h in
                             args.eval_percat_holdouts.split(",") if h.strip()]
            for hkey in _req_holdouts:
                recs_h = holdout_cat_records.get(hkey)
                if not recs_h:
                    out["holdouts"][hkey] = None
                    continue
                with torch.no_grad():
                    sw, cb, pcce, pcn, sacc, pcacc = _holdout_eval_cats_mlp_full(
                        base_model, holdout_per_example[hkey], recs_h,
                        V_eval, mlp_eval, n_cats=n_cats,
                        steer_layer=args.steer_layer,
                        pad_token_id=base_tokenizer.pad_token_id,
                        batch_size=args.train_batch_size, distributed=False)
                out["holdouts"][hkey] = {
                    "sample_weighted_ce": sw, "cat_balanced_ce": cb,
                    "per_cat_ce": pcce, "per_cat_n": pcn,
                    "steer_top1_acc": sacc, "steer_top1_acc_per_cat": pcacc}
                print(f"  [eval_percat] {hkey}: sw={sw:.4f} cb={cb:.4f} "
                      f"steer_top1_acc={sacc*100:.2f}%", flush=True)
            outp = os.path.join(args.save_dir, args.eval_percat_out)
            with open(outp, "w") as f:
                json.dump(out, f, indent=2)
            print(f"[eval_percat_only] wrote {outp}", flush=True)
            return

        V_cpu, mlp_out, cats_metrics = train_cats_mlp_coef(
            base_model, per_example, cat_records, n_cats,
            steer_layer=args.steer_layer,
            hidden_size=hidden,
            n_epochs=args.cats_epochs,
            batch_size=args.train_batch_size,
            cats_lr=args.cats_lr,
            mlp_lr=args.mlp_lr,
            mlp_hidden_dim=args.mlp_hidden_dim,
            mlp_grad_clip=args.mlp_grad_clip,
            pad_token_id=base_tokenizer.pad_token_id,
            mlp_per_cat=args.mlp_per_cat,
            seed=args.seed,
            max_positions_per_example=args.max_positions_per_example,
            cat_key_lookup=full_cat_keys,
            holdout_sets=holdout_cat_records,
            holdout_per_example=holdout_per_example,
            patience=args.patience,
            early_stop_metric="trainmix_holdout",
            weight_decay=args.weight_decay,
            per_cat_best=args.cats_per_cat_best,
            norm_cap_R=norm_cap_R,
            distributed=args.distributed,
            save_per_epoch_dir=(
                os.path.join(args.save_dir, "epoch_checkpoints")
                if args.save_per_epoch_ckpts else None),
            freeze_cats=args.freeze_cats,
            init_cat_norms=init_cat_norms,
            rand_cats_seed=args.rand_cats_seed)

        _do_save = (not args.distributed) or is_main()
        if _do_save:
            layer_map: Dict[str, int] = {}
            for k in full_cat_keys:
                ci = key_to_cat[k]
                v = V_cpu[ci].clone()
                out_path = os.path.join(
                    args.save_dir, f"{model_short}_{k}_linear.pt")
                torch.save({k: v}, out_path)
                layer_map[k] = args.steer_layer
                print(f"  saved {k}  norm={float(v.norm().item()):.3f}  -> "
                      f"{out_path}", flush=True)
            with open(os.path.join(args.save_dir, "layer_map.json"), "w") as f:
                json.dump(layer_map, f, indent=2)

            mlp_path = os.path.join(args.save_dir, "cat_coef_mlp.pt")
            torch.save(mlp_out.state_dict(), mlp_path)
            mlp_config = {
                "d_in": hidden,
                "n_cats": n_cats,
                "d_hidden": args.mlp_hidden_dim,
                "per_cat": bool(args.mlp_per_cat),
            }
            mlp_config_path = os.path.join(args.save_dir, "mlp_config.json")
            with open(mlp_config_path, "w") as f:
                json.dump(mlp_config, f, indent=2)
            print(f"  saved MLP -> {mlp_path}", flush=True)
            print(f"  saved MLP config -> {mlp_config_path}", flush=True)

        if not _do_save:
            import torch.distributed as _dist
            _dist.barrier()
            _dist.destroy_process_group()
            return
        # Summary metadata (only rank 0 reaches here in DDP mode)
        meta = {
            "base_model": args.base_model,
            "thinking_model": args.thinking_model,
            "thinking_model_short": args.thinking_model_short,
            "steer_layer": args.steer_layer,
            "sae_layer": args.sae_layer,
            "sae_n_clusters": args.sae_n_clusters,
            "mode": "mlp_coef_single" if args.single_vector else "mlp_coef",
            "single_vector": bool(args.single_vector),
            "train_data_file": args.train_data_file,
            "val_data_file": args.val_data_file,
            "n_train_examples_used": len(per_example),
            "n_disagreements": len(cat_records),
            "training_objective": "top1-CE (MLP alpha * V[cat])",
            "cat_loss_aggregation": "per-cat-balanced",
            "weight_decay": args.weight_decay,
            "mlp_hidden_dim": args.mlp_hidden_dim,
            "mlp_lr": args.mlp_lr,
            "mlp_grad_clip": args.mlp_grad_clip,
            "cats_lr": args.cats_lr,
            "cats_per_cat_best": bool(args.cats_per_cat_best),
            "cat_norms": [float(V_cpu[i].norm().item()) for i in range(n_cats)],
            "active_cat_keys": full_cat_keys,
            "cats_metrics": cats_metrics,
            "holdout_sizes": {
                k: len(v) for k, v in holdout_cat_records.items()},
            "norm_cap": {
                "alpha": float(args.norm_cap_alpha),
                "h_bar": float(h_bar) if h_bar is not None else None,
                "R": float(norm_cap_R) if norm_cap_R is not None else None,
            },
            "early_stop": {
                "patience": args.patience,
                "metric": "trainmix_holdout",
            },
            "args": vars(args),
        }
        meta_path = os.path.join(
            args.save_dir, f"{model_short}_correction_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"\n[done] {n_cats} cats + MLP saved to "
              f"{args.save_dir}\n  meta -> {meta_path}", flush=True)

        # ---- Final-run-friendly artifacts ----------------------------
        # train_log.jsonl: one JSON line per epoch, easy to grep/parse
        try:
            tl_path = os.path.join(args.save_dir, "train_log.jsonl")
            with open(tl_path, "w") as _tlf:
                for m in cats_metrics:
                    _tlf.write(json.dumps({
                        "phase": m.get("phase", "cats_mlp"),
                        "epoch": m.get("epoch"),
                        # mean_ce here is already sample-weighted (sum
                        # over positions / total positions) -- see the
                        # per-epoch loop that produces it.
                        "mean_ce": m.get("mean_ce"),
                        # holdout_ce is sample-weighted (new default).
                        "holdout_ce": m.get("holdout_ce", {}),
                        # cat-balanced version kept for diagnostics.
                        "holdout_ce_cat_balanced": m.get("holdout_ce_cat_balanced", {}),
                        "holdout_per_cat_n": m.get("holdout_per_cat_n", {}),
                        "norms_per_cat": m.get("norms_per_cat"),
                        "alpha_mean_per_cat": m.get("alpha_mean_per_cat"),
                    }) + "\n")
            print(f"  saved train_log -> {tl_path}", flush=True)
        except Exception as _e:
            print(f"  warn: train_log.jsonl write failed: {_e}", flush=True)

        # best_meta.json: explicit best-epoch pointer based on trainmix_holdout CE
        try:
            best_epoch, best_val = None, None
            for m in cats_metrics:
                _hces = m.get("holdout_ce") or {}
                if "trainmix_holdout" not in _hces:
                    continue
                _v = float(_hces["trainmix_holdout"])
                if best_val is None or _v < best_val:
                    best_val = _v
                    best_epoch = m.get("epoch")
            bm_path = os.path.join(args.save_dir, "best_meta.json")
            with open(bm_path, "w") as _bmf:
                json.dump({
                    "best_epoch": best_epoch,
                    "metric": "trainmix_holdout",
                    "best_value": best_val,
                    "patience": args.patience,
                    "n_epochs_run": len(cats_metrics),
                    "vectors": "cat_coef_mlp.pt",
                    "mlp_config": "mlp_config.json",
                    "epoch_checkpoints": (
                        "epoch_checkpoints/" if args.save_per_epoch_ckpts
                        else None),
                }, _bmf, indent=2)
            print(f"  saved best_meta -> {bm_path}  "
                  f"(epoch={best_epoch} ce={best_val})", flush=True)
        except Exception as _e:
            print(f"  warn: best_meta.json write failed: {_e}", flush=True)

        # run_meta.json: provenance (git sha, full argv, key params)
        try:
            import subprocess as _sp
            try:
                _git = _sp.check_output(
                    ["git", "rev-parse", "HEAD"],
                    cwd=os.path.dirname(os.path.abspath(__file__)),
                    stderr=_sp.DEVNULL).decode().strip()
            except Exception:
                _git = None
            rm_path = os.path.join(args.save_dir, "run_meta.json")
            with open(rm_path, "w") as _rmf:
                json.dump({
                    "git_sha": _git,
                    "argv": sys.argv,
                    "started_at_utc": time.strftime(
                        "%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                    "rollouts_temp_label": args.rollouts_temp_label,
                    "rollouts_max_tokens": args.rollouts_max_tokens,
                    "rollouts_sample_idx": args.rollouts_sample_idx,
                    "cats_epochs": args.cats_epochs,
                    "patience": args.patience,
                    "early_stop_metric": "trainmix_holdout",
                    "save_dir": args.save_dir,
                    "save_per_epoch_ckpts": bool(args.save_per_epoch_ckpts),
                }, _rmf, indent=2)
            print(f"  saved run_meta -> {rm_path}", flush=True)
        except Exception as _e:
            print(f"  warn: run_meta.json write failed: {_e}", flush=True)

        if args.distributed:
            import torch.distributed as _dist
            _dist.barrier()
            _dist.destroy_process_group()
        return

    else:
        # ============================================================
        # LEGACY MODE: bias + cats (original pipeline)
        # ============================================================
        for k in full_cat_keys:
            bias_per_category.setdefault(k, [])

        bias_union_records: List[Tuple[int, int, int]] = []
        for k in full_cat_keys:
            bias_union_records.extend(bias_per_category[k])
        print(f"  bias_train disagreements (union): "
              f"{len(bias_union_records)}", flush=True)

        holdout_bias_records: Dict[str, List[Tuple[int, int, int]]] = {}
        for hkey in ("trainmix_holdout", "math500_oos", "gsm8k_oos"):
            unioned: List[Tuple[int, int, int]] = []
            for k, recs in holdout_per_category.get(hkey, {}).items():
                unioned.extend(recs)
            if unioned:
                holdout_bias_records[hkey] = unioned

        # Phase B: train bias
        print(f"\n[Phase B] training bias ({len(bias_union_records)} positions)",
              flush=True)
        bias_cpu, bias_metrics = train_bias_ce(
            base_model, bias_per_example, bias_union_records,
            steer_layer=args.steer_layer,
            hidden_size=hidden,
            n_epochs=args.bias_epochs,
            batch_size=args.train_batch_size,
            lr=args.bias_lr,
            pad_token_id=base_tokenizer.pad_token_id,
            seed=args.seed,
            max_positions_per_example=args.max_positions_per_example,
            holdout_sets=holdout_bias_records,
            holdout_per_example=holdout_per_example,
            patience=args.patience,
            early_stop_metric="trainmix_holdout",
            weight_decay=args.weight_decay,
            norm_cap_R=norm_cap_R)

        bias_path = os.path.join(args.save_dir,
                                 f"{model_short}_bias_linear.pt")
        torch.save({"bias": bias_cpu}, bias_path)
        with open(os.path.join(args.save_dir, "bias_layer.json"), "w") as f:
            json.dump({"layer": args.steer_layer,
                       "norm": float(bias_cpu.norm().item())}, f, indent=2)
        print(f"  saved bias -> {bias_path}", flush=True)

        # Phase C: filter residual disagreements
        print(f"\n[Phase C] filtering residual disagreements", flush=True)
        filtered, _filter_keep = filter_residual_disagreements(
            base_model, per_example, per_category, bias_cpu,
            steer_layer=args.steer_layer,
            pad_token_id=base_tokenizer.pad_token_id,
            batch_size=args.filter_batch_size)

        filtered_acts: Dict[str, List[float]] = {}
        for cat_key, vals in per_category_acts.items():
            km = _filter_keep.get(cat_key, [True] * len(vals))
            filtered_acts[cat_key] = [v for v, k in zip(vals, km) if k]

        _cat_filter = getattr(args, "cat_filter", "none")
        if _cat_filter == "none" and float(getattr(args, "cat_topk_pct", 1.0)) < 1.0:
            _cat_filter = "topk"

        if _cat_filter == "topk":
            cat_topk_pct = float(getattr(args, "cat_topk_pct", 0.25))
            before_total = sum(len(v) for v in filtered.values())
            for cat_key in list(filtered.keys()):
                recs_f = filtered.get(cat_key, [])
                acts_f = filtered_acts.get(cat_key, [])
                n = len(recs_f)
                if n == 0 or len(acts_f) != n:
                    continue
                n_keep = max(1, int(round(n * cat_topk_pct)))
                if n_keep >= n:
                    continue
                order = sorted(range(n), key=lambda i: -acts_f[i])
                keep_idx = sorted(order[:n_keep])
                filtered[cat_key] = [recs_f[i] for i in keep_idx]
                filtered_acts[cat_key] = [acts_f[i] for i in keep_idx]
            after_total = sum(len(v) for v in filtered.values())
            print(f"  [topk] {before_total} -> {after_total}", flush=True)

        elif _cat_filter == "gmm_valley":
            import numpy as np
            from sklearn.mixture import GaussianMixture
            before_total = sum(len(v) for v in filtered.values())
            for cat_key in sorted(filtered.keys()):
                recs_f = filtered.get(cat_key, [])
                acts_f = filtered_acts.get(cat_key, [])
                n = len(recs_f)
                if n < 10 or len(acts_f) != n:
                    continue
                vals = np.array(acts_f, dtype=np.float64)
                if n >= 30:
                    gm = GaussianMixture(n_components=2, random_state=0)
                    gm.fit(vals.reshape(-1, 1))
                    mu = sorted(gm.means_.flatten())
                    sds = [s**0.5 for _, s in sorted(
                        zip(gm.means_.flatten(), gm.covariances_.flatten()),
                        key=lambda x: x[0])]
                    sep_ratio = abs(mu[1] - mu[0]) / (max(sds[0], sds[1]) + 1e-8)
                    if sep_ratio > 1.5:
                        xs = np.linspace(mu[0], mu[1], 300)
                        valley = float(xs[np.argmin(
                            gm.score_samples(xs.reshape(-1, 1)))])
                    else:
                        valley = float(np.percentile(vals, 75))
                else:
                    valley = float(np.percentile(vals, 75))
                keep_mask = vals >= valley
                if keep_mask.sum() == 0:
                    keep_mask[np.argmax(vals)] = True
                filtered[cat_key] = [r for r, km in zip(recs_f, keep_mask) if km]
                filtered_acts[cat_key] = [a for a, km in zip(acts_f, keep_mask) if km]
            after_total = sum(len(v) for v in filtered.values())
            print(f"  [gmm_valley] {before_total} -> {after_total}", flush=True)

        # Phase D: train cats
        cat_records: List[Tuple[int, int, int, int]] = []
        for k in full_cat_keys:
            ci = key_to_cat[k]
            for ex_idx, pos, target in filtered.get(k, []):
                cat_records.append((ex_idx, pos, ci, target))
        print(f"  residual records: {len(cat_records)}", flush=True)

        holdout_cat_records: Dict[str, List[Tuple[int, int, int, int]]] = {}
        for hkey in ("trainmix_holdout", "math500_oos", "gsm8k_oos"):
            recs_h: List[Tuple[int, int, int, int]] = []
            for k, krecs in holdout_per_category.get(hkey, {}).items():
                if k not in key_to_cat:
                    continue
                ci = key_to_cat[k]
                for ex_idx, pos, target in krecs:
                    recs_h.append((ex_idx, pos, ci, target))
            if recs_h:
                holdout_cat_records[hkey] = recs_h

        if cat_records:
            V_cpu, cats_metrics = train_cats_ce_balanced(
                base_model, per_example, cat_records, n_cats, bias_cpu,
                steer_layer=args.steer_layer,
                hidden_size=hidden,
                n_epochs=args.cats_epochs,
                batch_size=args.train_batch_size,
                lr=args.cats_lr,
                pad_token_id=base_tokenizer.pad_token_id,
                seed=args.seed,
                max_positions_per_example=args.max_positions_per_example,
                cat_key_lookup=full_cat_keys,
                holdout_sets=holdout_cat_records,
                holdout_per_example=holdout_per_example,
                patience=args.patience,
                early_stop_metric="trainmix_holdout",
                weight_decay=args.weight_decay,
                per_cat_best=args.cats_per_cat_best,
                norm_cap_R=norm_cap_R)
        else:
            V_cpu = torch.zeros((n_cats, hidden), dtype=torch.float32)
            cats_metrics = []

        # Save cat vectors
        layer_map: Dict[str, int] = {}
        for k in full_cat_keys:
            ci = key_to_cat[k]
            v = V_cpu[ci].clone()
            out_path = os.path.join(
                args.save_dir, f"{model_short}_{k}_linear.pt")
            torch.save({k: v}, out_path)
            layer_map[k] = args.steer_layer
        with open(os.path.join(args.save_dir, "layer_map.json"), "w") as f:
            json.dump(layer_map, f, indent=2)

        meta = {
            "base_model": args.base_model,
            "thinking_model": args.thinking_model,
            "steer_layer": args.steer_layer,
            "sae_layer": args.sae_layer,
            "sae_n_clusters": args.sae_n_clusters,
            "train_dataset": args.train_dataset,
            "n_train_examples": len(per_example),
            "n_disagreements": len(cat_records),
            "bias_norm": float(bias_cpu.norm().item()),
            "cat_norms": [float(V_cpu[i].norm().item()) for i in range(n_cats)],
            "bias_metrics": bias_metrics,
            "cats_metrics": cats_metrics,
            "args": vars(args),
        }
        meta_path = os.path.join(
            args.save_dir, f"{model_short}_correction_meta.json")
        with open(meta_path, "w") as f:
            json.dump(meta, f, indent=2)
        print(f"\n[done] {n_cats} cats + 1 bias saved to "
              f"{args.save_dir}\n  meta -> {meta_path}", flush=True)


if __name__ == "__main__":
    main()
