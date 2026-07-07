"""Hybrid model evaluation: thinking model + base model + SAE-guided steering.

Three-phase pipeline:
  1. Standalone generation for thinking and base models (cached).
  2. KV-cached batched hybrid generation with SAE-category steering and
     token-disagreement-gated coefficient sweep (paper recipe).
  3. Concurrent LLM judging with majority vote.
"""

import dotenv
dotenv.load_dotenv("../.env")

import sys
import os
import torch
import json
import re
import gc
import time
import random
import argparse
import math
from typing import Optional, Dict, Tuple, List
from collections import Counter

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

try:
    from tqdm.auto import tqdm
except Exception:
    tqdm = None

from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer

# Patch transformers.modeling_utils.load_state_dict to handle .safetensors
# files whose metadata is None (some community model exports, e.g. ORZ-32B,
# omit the "format" key).  Default to {"format": "pt"} so the loader accepts
# the file instead of raising AttributeError on metadata.get("format").
try:
    import transformers.modeling_utils as _tm_modutils
    from safetensors import safe_open as _safe_open
    from safetensors.torch import load_file as _safe_load_file
    _orig_load_state_dict = _tm_modutils.load_state_dict
    def _patched_load_state_dict(checkpoint_file, *args, **kwargs):
        if str(checkpoint_file).endswith(".safetensors"):
            with _safe_open(checkpoint_file, framework="pt") as f:
                md = f.metadata()
            if md is None or md.get("format") not in ["pt", "tf", "flax", "mlx"]:
                return _safe_load_file(checkpoint_file)
        return _orig_load_state_dict(checkpoint_file, *args, **kwargs)
    _tm_modutils.load_state_dict = _patched_load_state_dict
except Exception as _e:
    print(f"[warn] could not patch transformers.load_state_dict: {_e}")

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from utils.sae import load_sae
from utils.utils import center_and_l2_normalize_torch, chat_batch
from utils.clustering import get_latent_descriptions


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

CODING_DATASETS = {"mbpp", "livecodebench"}
MCQA_DATASETS = {"medqa", "gpqa"}
TEXT_CLASSIFICATION_DATASETS = {"legalbench"}
GPQA_LETTERS = ["A", "B", "C", "D"]

def parse_args():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--dataset", type=str, default="math500",
                   choices=["gsm8k", "math500", "aime24", "aime25", "mbpp",
                            "livecodebench", "medqa", "gpqa", "legalbench",
                            "natreason", "holdoutmix", "hendrycks_holdout"])
    p.add_argument("--natreason_file", type=str, default=None,
                   help="Path to the natural_reasoning eval jsonl (fields: "
                        "question, reference_answer). Required when "
                        "--dataset natreason.")
    p.add_argument("--holdoutmix_file", type=str, default=None,
                   help="Path to the trainmix VAL-holdout eval jsonl (fields: "
                        "question, reference_answer). Required when "
                        "--dataset holdoutmix. Used as a cheap in-distribution "
                        "vector-SELECTION signal (real gap recovery).")
    p.add_argument("--hendrycks_holdout_file", type=str, default=None,
                   help="Path to the hendrycks-MATH holdout eval jsonl (fields: "
                        "question, reference_answer), disjoint from train/val "
                        "and math500. Required when --dataset hendrycks_holdout.")
    p.add_argument("--thinking_model", type=str, default="Qwen/QwQ-32B")
    p.add_argument("--base_model", type=str, default="Qwen/Qwen2.5-32B")
    p.add_argument("--sae_layer", type=int, default=27)
    p.add_argument("--n_clusters", type=int, default=10)
    p.add_argument("--n_tasks", type=int, default=0, help="0 = all")
    p.add_argument("--batch_gen_size", type=int, default=32)
    p.add_argument("--hybrid_gen_batch_size", type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=5000)
    p.add_argument("--max_thinking_tokens", type=int, default=5000)
    p.add_argument("--eval_start_idx", type=int, default=0)
    p.add_argument("--eval_indices", type=str, default="",
                   help="Optional comma/space-separated dataset indices to "
                        "evaluate exactly, in the provided order. Overrides "
                        "--eval_start_idx/--n_tasks for task selection.")
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--decode_temperature", type=float, default=0.0,
                   help="Per-step sampling temperature for the hybrid "
                        "decode loop. T=0 preserves historical pure-argmax "
                        "behaviour. For T>0, base/think candidate tokens "
                        "used by the disagreement gate are sampled from "
                        "softmax(logits/T), and emitted base/steered tokens "
                        "are sampled from their corresponding distributions.")
    p.add_argument("--decode_seed", type=int, default=0,
                   help="RNG seed for --decode_temperature sampling. "
                        "Different values produce different samples "
                        "from the same prompt.")
    p.add_argument("--skip_hybrid", action="store_true",
                   help="Skip Phase 3 (hybrid generation + judging) and "
                        "only generate + judge the standalone base and "
                        "thinking rollouts.  Writes a rolling file with "
                        "answers/judges/eos/n_tokens for {thinking, base} "
                        "only.  Useful for quick base-vs-think accuracy "
                        "comparisons across prompt styles without paying "
                        "for the dual-model hybrid decode.")
    p.add_argument("--cold_start_n_tokens", type=int, default=0,
                   help="If >0, inject the first N tokens of the THINKING "
                        "model's rollout (decoded to text) as a fixed "
                        "prefix into BOTH the standalone base response and "
                        "the hybrid (base+think) prefill, then continue "
                        "with normal base / hybrid rollout from that "
                        "shared starting point.  Bypasses base response "
                        "cache (--no_response_cache for base only) because "
                        "the prompt now varies per task.  Hybrid's "
                        "thinking-side prefill is also extended with the "
                        "same text via thinking_continuation_text so the "
                        "think model sees the prefix as 'already produced' "
                        "and parallel-decode resumes cleanly.")
    p.add_argument("--dom_vectors_dir", type=str, required=True)
    p.add_argument("--dom_vectors_model_short", type=str, default=None)
    p.add_argument("--old_vectors_dir", type=str, default=None,
                   help="Load old-format per-index optimized vectors from this dir "
                        "(files: <model_short>_idx<N>_linear.pt, all at --old_vectors_layer)")
    p.add_argument("--old_vectors_layer", type=int, default=10,
                   help="Layer at which old-format vectors are applied")
    p.add_argument("--randomize_vectors", action="store_true",
                   help="After loading steering vectors, replace each category "
                        "vector with a Gaussian-random direction of the same "
                        "norm (control/ablation). Bias vector (if any) is kept.")
    p.add_argument("--randomize_vectors_unit_norm", action="store_true",
                   help="Ablation (reproduces bug in collaborator's hybrid_token.py "
                        "'random_vectors' path): replace each per-category vector "
                        "with a UNIT-NORM Gaussian-random direction (norm=1) "
                        "instead of preserving the learned-vector norm. Bias (if "
                        "any) is kept untouched.  Since learned-vector norms are "
                        "~40-60 and bias has comparable norm, this effectively "
                        "collapses the random-vectors ablation toward bias-only.")
    p.add_argument("--random_seed", type=int, default=0,
                   help="Seed for --randomize_vectors / --random_firing / "
                        "--random_guardrail.")
    p.add_argument("--bias_vector_path", type=str, default=None,
                   help="Path to a .pt bias vector (a dict {'bias': tensor} or raw "
                        "tensor) to add on top of the per-category steering vector, "
                        "matching the OLD/paper pipeline's global bias term.")
    p.add_argument("--bias_always_on", action="store_true",
                       help="Apply the bias vector at EVERY token position "
                            "(not just disagreements) via a separate always-on "
                            "hook.  Cat vectors still fire only on disagreement. "
                            "Requires --bias_vector_path.  When set, the bias is "
                            "NOT folded into the cat vectors.")
    p.add_argument("--bias_only", action="store_true",
                   help="Ablation: zero out every per-category steering vector "
                        "AFTER loading, so only the --bias_vector_path remains. "
                        "Requires --bias_vector_path.")
    p.add_argument("--bias_layer", type=int, default=None,
                   help="When --bias_only: register the bias at this layer for "
                        "every category key (overrides per-category layer_map). "
                        "If omitted, tries to read <bias_vector_path>'s sibling "
                        "bias_layer.json, else falls back to --old_vectors_layer.")
    p.add_argument("--random_firing", action="store_true",
                   help="Ablation: at each decode step, ignore the SAE argmax "
                        "and pick a uniform-random category key per batch row "
                        "(simulates random latent firing).")
    p.add_argument("--random_firing_exclude_top_k_keys", type=int, default=0,
                   help="When --random_firing is set: exclude the top-K SAE "
                        "activated category keys (per row, ordered by latent "
                        "activation) from the random-pick pool. 0 disables "
                        "(uniform across all keys, the default). 3 means "
                        "pick uniformly from keys that are NOT among the top-3 "
                        "SAE-activated keys at the current disagreement "
                        "position.")
    p.add_argument("--pure_steer_base_eos", action="store_true",
                   help="Cleanest hybrid mode: DISABLES </think> close detection "
                        "AND base-EOS suppression. The hybrid just keeps the "
                        "disagreement->SAE->MLP+cat steering protocol running "
                        "every token, and terminates only when the BASE model "
                        "naturally emits EOS. Avoids premature transitions "
                        "triggered by thinking-model quirks (e.g. tool-use cues "
                        "like ```python in ORZ-32B). NOTE: 'Final answer:' is "
                        "never injected in this mode -- the row is purely "
                        "base-driven, steered at every disagreement.")
    p.add_argument("--continuation_mode", action="store_true",
                   help="Phase-2 4096-extension: load existing pure-mode rolling "
                        "rows from --continuation_source_rolling (a glob or "
                        "comma-list of JSONL files), restrict tasks to rows "
                        "where eos.hybrid==False, and continue the hybrid loop "
                        "for --max_new_tokens additional tokens with both KV "
                        "caches prefilled from the cached think/hybrid text.")
    p.add_argument("--continuation_source_rolling", type=str, default="",
                   help="Comma-separated paths or glob to source rolling JSONL "
                        "files used as continuation seeds (e.g. the two pure_h* "
                        "rolling files for this pair).  REQUIRED with "
                        "--continuation_mode.")
    p.add_argument("--firing_replace_with_min_cosine", action="store_true",
                   help="Ablation: at each disagreement position, replace the "
                        "SAE-selected category with the category whose steering "
                        "vector has MINIMUM cosine similarity to the SAE-top-1 "
                        "vector. The MLP coefficient + V for that anti-cat is "
                        "then used. Deterministic (no randomness). Independent "
                        "of --random_firing.")
    p.add_argument("--random_steer_prob", type=float, default=0.0,
                   help="Ablation: REPLACE the disagreement-position gate with "
                        "a per-step Bernoulli(p) draw. At every decoding step, "
                        "each batch row is treated as 'eligible for steering' "
                        "with probability p REGARDLESS of whether think and "
                        "base actually disagree. All other gates (finished / "
                        "forced-queue / answer-phase / warmup) still suppress "
                        "steering. p should match the empirical fraction of "
                        "positions steered by the full pipeline for an "
                        "apples-to-apples 'random position' comparison "
                        "(e.g. p=0.0332 for ORZ-1.5B / math500, p=0.0393 for "
                        "ORZ-1.5B / gsm8k). Seeded via --random_seed.")
    p.add_argument("--random_guardrail", action="store_true",
                   help="Ablation: replace the thinking-model perplexity "
                        "guardrail with a uniform random choice among the "
                        "coefficient sweep candidates.")
    p.add_argument("--fixed_coef", type=float, default=None,
                   help="Apply the steering vector at this fixed coefficient "
                        "at every disagreement position, with no sweep or "
                        "selection heuristic.  Overrides --coef_sweep and "
                        "--coef_select when set.")
    p.add_argument("--coef_sweep", type=str, default="0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0",
                   help="Comma-separated coefficient sweep used by the "
                        "guardrail. Default is the paper's 10-point grid "
                        "[0.1..1.0]. Ignored when --fixed_coef is set.")
    p.add_argument("--coef_select", type=str, default="pg",
                   choices=["pg", "kl_top3", "kl_topk",
                            "think_top1", "think_top1_match",
                            "think_top1_match_maxconf", "fixed", "mlp",
                            "mlp_pg"],
                   help="Coefficient-selection rule. 'fixed': use --fixed_coef "
                        "directly, no sweep. 'pg' (default): pick coef whose "
                        "steered-base argmax has highest log-prob under thinking "
                        "model. 'kl_top3'/'kl_topk': minimise CE vs thinking "
                        "top-K. 'think_top1': oracle ceiling. 'mlp': use the "
                        "MLP-predicted alpha directly, no sweep. 'mlp_pg': "
                        "sweep effective coef = mlp_alpha * --coef_sweep and "
                        "pick the multiplier whose steered argmax has highest "
                        "thinking-model log-prob (perplexity guardrail on top "
                        "of the MLP).")
    p.add_argument("--kl_topk", type=int, default=3,
                   help="K for --coef_select=kl_topk (and kl_top3 alias).")
    p.add_argument("--warmup_until_sentence_end", action="store_true", default=False,
                   help="If set, the hybrid protocol (steering + EOS suppression "
                        "+ </think> detection) is *gated off* until the base model "
                        "emits its first sentence-ending token (a token whose "
                        "decoded form contains '[.?!] '/'[.?!]\\n').  During that "
                        "warmup window each row generates with base alone and the "
                        "thinking model is fed the base's emitted tokens to keep "
                        "KVs aligned.  Diagnostic test for the 0.5B auto-completion "
                        "failure: lets the base commit to its own response style "
                        "and stop naturally before steering kicks in.")
    p.add_argument("--warmup_max_tokens", type=int, default=60,
                   help="Hard cap on the per-row warmup window when "
                        "--warmup_until_sentence_end is set.  If no sentence end "
                        "is detected within this many tokens, the hybrid protocol "
                        "engages anyway so generation makes progress.")
    p.add_argument("--think_prompt_family", default="auto",
                   choices=["auto", "orz", "r1", "qwq", "other"],
                   help="Family used to shape the thinking-model user "
                        "content. 'auto' (default) detects from "
                        "--thinking_model: 'open-reasoner-zero'/'orz' -> "
                        "'orz' (prepends the Table-5 user instruction); "
                        "'r1-distill'/'deepseek-r1' -> 'r1'; 'qwq' -> "
                        "'qwq'; otherwise 'other' (no shaping). The base "
                        "model's prompt is shaped independently via "
                        "--base_prompt_style.")
    p.add_argument("--base_prompt_style", default="default",
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
                   help="Base-model prompt format. 'default' is the bare "
                        "'User: {q}\\nAssistant:' completion prompt.  "
                        "'think_template' applies the thinking model's "
                        "chat template (with family-specific user-content "
                        "shaping + math directive) to the question, then "
                        "feeds that string to the base model as a raw "
                        "completion prompt -- i.e. base sees exactly the "
                        "same prompt as the think model.  "
                        "'stepwise' (final_final v1) builds 'User: "
                        "Answer the following question. Respond step by "
                        "step.\\n\\n{q}\\nAssistant:'.  'boxed' "
                        "(final_final v2) places the QwQ/R1 post-hoc "
                        "math directive + ORZ \\boxed{} anchor AFTER the "
                        "question: 'User: {q}\\n\\nPlease reason step "
                        "by step, and put your final answer within "
                        "\\boxed{}.\\nAssistant:'.  'legacy_task' (v3) "
                        "is the structured Task/Question/Answer prompt "
                        "used on origin/main's hybrid/hybrid_*.py and "
                        "train-vectors/optimize_steering_vectors.py: "
                        "'Task: Answer the question below. Explain your "
                        "reasoning step by step.\\n\\n\\n\\nQuestion:"
                        "\\n{q}\\n\\nStep by step answer:\\n'.  MUST "
                        "match the prompt used by the cached base "
                        "rollouts and during steering-vector training "
                        "(same flag in generate_rollouts.py and "
                        "optimize_correction_vectors.py).")
    p.add_argument("--math_directive", action="store_true", default=False,
                   help="When set AND the thinking family is r1/qwq AND "
                        "the dataset is a math benchmark (math500, gsm8k, "
                        "aime24, aime25), append the DeepSeek-R1/QwQ "
                        "'Please reason step by step, and put your final "
                        "answer within \\boxed{}.' directive to the user "
                        "content sent to the thinking model.  ORZ "
                        "shaping is independent of this flag (Table-5 "
                        "always applied).  IMPORTANT: must match the "
                        "directive used at vLLM generation time for the "
                        "cached think rollouts -- mismatch silently "
                        "produces a different prompt than what was "
                        "generated, so set this for math500/gsm8k whenever "
                        "the cached rollouts were generated with "
                        "--math_directive_mode always.")
    p.add_argument("--free_fly_until_think_eos", action="store_true", default=False,
                   help="Diagnostic 'free-fly' mode (e.g. for ORZ-0.5B where "
                        "the thinking model never emits a clean </think>).  "
                        "Skips </think>/</answer> close-detection and the "
                        "ANSWER-phase transition entirely: the hybrid stays "
                        "in REASONING with steering active at every "
                        "disagreement, and a row terminates ONLY when the "
                        "thinking model itself predicts EOS at that step "
                        "(base EOS is always suppressed regardless of "
                        "--disable_eos_suppression).  Default OFF -- "
                        "behavior for all other model pairs is unchanged.")
    p.add_argument("--disable_eos_suppression", action="store_true", default=False,
                   help="If set, the hybrid state machine does NOT substitute "
                        "the thinking model's argmax when the base model emits "
                        "EOS during the reasoning phase.  Effectively: when "
                        "base says EOS, hybrid terminates.  Diagnostic for "
                        "small bases that emit \\boxed{} cleanly and want to "
                        "stop, but were being forced to auto-complete because "
                        "the thinking model hadn't yet emitted </think>.")
    p.add_argument("--no_termination", action="store_true", default=False,
                   help="Disable ALL early termination conditions.  Like "
                        "--free_fly_until_think_eos (close-detection skipped, "
                        "base EOS always suppressed) but additionally never "
                        "terminates on think EOS either.  Row only stops when "
                        "max_new_tokens is reached.  Diagnostic mode used by "
                        "the orz-1.5b termination-mode comparison experiment.")
    p.add_argument("--eos_prob_warmup", action="store_true", default=False,
                   help="Linearly scale base P(EOS) from 0 at hybrid step 0 "
                        "to the unmodified base P(EOS) at "
                        "--eos_prob_warmup_steps.  Implemented exactly in "
                        "probability space: at step n out of T, base "
                        "softmax probability of EOS is multiplied by "
                        "alpha = n / T and the displaced mass is "
                        "redistributed proportionally to non-EOS tokens.  "
                        "This converts the binary 'allow / suppress base EOS' "
                        "behaviour of --pure_steer_base_eos into a smooth "
                        "ramp, preventing the small base from terminating "
                        "long before the budget is used while preserving "
                        "the base's natural EOS prior near the budget.")
    p.add_argument("--eos_prob_warmup_steps", type=int, default=0,
                   help="Total ramp length (in hybrid generation tokens) for "
                        "--eos_prob_warmup.  0 (default) uses --max_new_tokens. "
                        "alpha = min(1, n_gen / T) is multiplied onto base "
                        "P(EOS) at every step.")
    p.add_argument("--accept_answer_close", action="store_true", default=False,
                   help="Also accept '</answer>' as a reasoning-phase close "
                        "trigger in the hybrid state machine (in addition to "
                        "'</think>').  ORZ-0.5B's thinking model uses a "
                        "non-standard template '\\boxed{X} <answer>...</answer> "
                        "</think>' where </answer> consistently appears ~108 "
                        "chars after \\boxed{} while </think> may only appear "
                        "much later or not at all.  Catching </answer> lets the "
                        "hybrid exit the reasoning phase soon after the answer "
                        "is emitted, preventing post-boxed auto-completion drift.")
    p.add_argument("--suppress_boxed_first_n_tokens", type=int, default=0,
                   help="If > 0, suppress the 'boxed' tokens (79075 'boxed' and "
                        "73664 ' boxed', which exclusively appear inside the "
                        "'\\\\boxed{' construction in Qwen2.5 tokenizer) from "
                        "being emitted during the first N hybrid-generation "
                        "tokens.  Diagnostic for the 0.5B quirk where steering "
                        "drives the base into emitting the final answer too "
                        "early.  Implemented as a post-emit override: any "
                        "selected 'boxed' token is replaced with the next-best "
                        "unsteered base argmax.")
    p.add_argument("--pg_bias_cat_sweep", action="store_true", default=False,
                   help="Sweep the cartesian product of (bias_coef, cat_coef) "
                        "per disagreement step and pick the (b,c) pair with "
                        "highest thinking-model log-prob at the base steered "
                        "argmax. Disables the legacy load-time bias-into-cat "
                        "folding so bias and cat have independent coefficients. "
                        "Overrides --coef_select / --fixed_coef / --coef_sweep.")
    p.add_argument("--pg_bias_coefs", type=str, default="0.0,0.5,1.0",
                   help="Bias coefficient candidates for --pg_bias_cat_sweep.")
    p.add_argument("--pg_cat_coefs", type=str, default="0.0,0.5,1.0",
                   help="Cat coefficient candidates for --pg_bias_cat_sweep.")
    p.add_argument("--token_window", type=int, default=0,
                   help="When > 0, force full-sequence forward (no KV cache) "
                        "during the coefficient sweep and apply the shift "
                        "c*(bias+cat) only to the LAST `token_window` positions "
                        "of layer `steering_layer`. Matches paper's "
                        "`--token_windows -N` semantics (positive integer here). "
                        "0 means 'use the legacy --steer_all_positions* flags'.")
    p.add_argument("--steer_all_positions_full", action="store_true",
                   help="Faithful reproduction of hybrid_token.py semantics: "
                        "during the coefficient sweep, drop the base-model KV "
                        "cache and run a FULL forward on the entire sequence "
                        "(input + generated so far) with the hook applying "
                        "c*(bias+vec) to ALL positions of layer `steering_layer`. "
                        "Each coef gets its own full forward. Expensive "
                        "(O(N^2) per task) but matches `--token_windows [1]` "
                        "exactly. After the sweep, the winning token is "
                        "committed and the base model's KV cache advances "
                        "normally (unsteered). Mutually exclusive with "
                        "--steer_all_positions.")
    p.add_argument("--steer_all_positions", action="store_true",
                   help="Reproduce collaborator's hybrid_token.py default "
                        "'--token_windows [1]' (internally window_size=0 -> "
                        "'all tokens') semantics. Instead of reverting the "
                        "steering shift from the KV cache after each "
                        "disagreement step, the winning-coefficient shift is "
                        "persisted: we re-run the current position's forward "
                        "with the best per-row coef so layers > sae_layer "
                        "have steered K/V for that position. Across many "
                        "disagreement steps this accumulates so attention at "
                        "subsequent steps sees shifted K/V for all past "
                        "steered positions, approximating their 'apply "
                        "c*(bias+vec) to every position's layer-24 output on "
                        "every forward pass' behaviour.")
    p.add_argument("--calibrate_coef", action="store_true", default=False,
                   help="Before the full hybrid eval, run a 10%% calibration "
                        "sweep on tasks where base=wrong, think=correct.  "
                        "Sweeps --calibrate_coef_grid coefs and picks the "
                        "one with highest gap recovery.  Then runs the full "
                        "eval with the winning coef.")
    p.add_argument("--calibrate_coef_grid", type=str,
                   default="0.25,0.5,1.0,1.5,2.0",
                   help="Comma-separated coefficient candidates for "
                        "the calibration sweep (used with --calibrate_coef).")
    p.add_argument("--calibrate_pct", type=float, default=0.10,
                   help="Fraction of base-wrong/think-correct tasks to use "
                        "for calibration (default 0.10 = 10%%).")
    p.add_argument("--stratified_calibrate", action="store_true", default=False,
                   help="Stratified calibration: judge all tasks, sample a "
                        "stratified 10%% subset matching the base/think accuracy "
                        "distribution, sweep --calibrate_coef_grid, pick the "
                        "coef with best hybrid accuracy on the subset.")
    p.add_argument("--save_best_coef", type=str, default=None,
                   help="Path to save the best calibrated coefficient as JSON. "
                        "Used to persist the bias coef for subsequent "
                        "bias+cat runs.")
    p.add_argument("--fixed_bias_coef", type=float, default=None,
                   help="Lock the bias coefficient during stratified calibration "
                        "to this value and only sweep the cat coefficient. "
                        "Typically set to the best_coef from a prior bias-only "
                        "run.  Implies --pg_bias_cat_sweep.")
    p.add_argument("--act_modulate_stats", type=str, default=None,
                   help="Path to sae_act_stats.json produced by "
                        "tools/compute_act_stats_v8.py.  When set, "
                        "during _classify() the per-cat steering vector "
                        "is scaled by a function of the live SAE "
                        "activation magnitude: v_eff = v * f(val, cat).  "
                        "See --act_modulate_fn for the function.")
    p.add_argument("--act_modulate_fn", type=str, default="p10p90",
                   choices=["p10p90", "p25p75", "linear_minmax"],
                   help="Modulation function used when "
                        "--act_modulate_stats is set.  "
                        "'p10p90': clip((val - p10) / (p90 - p10), 0, 1) "
                        "(robust linear, default).  "
                        "'p25p75': same with p25/p75 (more aggressive).  "
                        "'linear_minmax': uses (min, max) instead.")
    p.add_argument("--mlp_coef_path", type=str, default=None,
                   help="Path to cat_coef_mlp.pt state dict")
    p.add_argument("--mlp_config_path", type=str, default=None,
                   help="Path to mlp_config.json")
    p.add_argument("--mlp_coef_scale", type=float, default=1.0,
                   help="Constant multiplier applied to the MLP-predicted "
                        "alpha when --coef_select=mlp or mlp_pg.  e.g. 2.0 "
                        "amplifies the learned steering strength by 2x at "
                        "every disagreement position.")
    p.add_argument("--judge_model", type=str, default="openai/gpt-5.2")
    p.add_argument("--judge_repetitions", type=int, default=1)
    # ---- Explicit rollout-cache slug overrides (for the final run) ----
    p.add_argument("--think_cache_temp_label", type=str, default=None,
                   help="Override the temperature label in the THINK "
                        "cache filename (e.g. '0.6'). Defaults to the "
                        "label derived from --temperature.")
    p.add_argument("--think_cache_max_tokens", type=int, default=None,
                   help="Override the max-tokens substring in the THINK "
                        "cache filename. Defaults to --max_thinking_tokens.")
    p.add_argument("--think_cache_sample_idx", type=int, default=-1,
                   help="Sample-index '_s<N>' suffix for the THINK cache. "
                        "-1 (default) omits the suffix.")
    p.add_argument("--base_cache_temp_label", type=str, default=None,
                   help="Override the temperature label in the BASE "
                        "cache filename (e.g. '0' to reuse legacy "
                        "greedy base rollouts). Defaults to the label "
                        "derived from --temperature.")
    p.add_argument("--base_cache_max_tokens", type=int, default=None,
                   help="Override the max-tokens substring in the BASE "
                        "cache filename. Defaults to --max_new_tokens.")
    p.add_argument("--base_cache_sample_idx", type=int, default=-1,
                   help="Sample-index '_s<N>' suffix for the BASE cache. "
                        "-1 (default) omits the suffix.")
    p.add_argument("--hybrid_cache_sample_idx", type=int, default=-1,
                   help="Sample-index '_s<N>' suffix for the HYBRID "
                        "rollout cache. -1 (default) omits the suffix.")
    p.add_argument("--max_concurrent", type=int, default=40)
    p.add_argument("--results_dir", type=str, default="results")
    p.add_argument("--results_suffix", type=str, default="")
    p.add_argument("--no_response_cache", action="store_true")
    p.add_argument("--response_cache_dir", type=str, default=None,
                   help="Override the rollout-cache directory used by "
                        "_cache_path().  When unset (default), caches "
                        "live in '${results_dir}/response_cache/' "
                        "(legacy behaviour).  When set to an explicit "
                        "path, the cache files are read/written there "
                        "directly with no symlink hop.  Use this when "
                        "launching multiple jobs (e.g. math500 + gsm8k) "
                        "that share a --results_dir, to avoid the "
                        "symlink race that the legacy shell-script "
                        "pattern relied on.")
    p.add_argument("--disable_sae_mean", action="store_true")
    p.add_argument("--max_memory_per_gpu", type=str, default=None,
                   help="Per-GPU memory limit for the first model load, "
                        "e.g. '35GiB'.  Forces interleaved placement of "
                        "both models across all GPUs.")
    p.add_argument("--two_gpu_split", action="store_true", default=False,
                   help="Place base model entirely on cuda:0 and thinking "
                        "model entirely on cuda:1 (requires 2 GPUs).  Used "
                        "for the 14B/32B thinking-model evals where each "
                        "model fits on one H200 but neither pair fits on "
                        "a single GPU.  Default behavior (device_map='auto') "
                        "is unchanged for smaller pairs.")
    return p.parse_args()


def _result_suffix(args):
    s = ""
    if args.results_suffix:
        s += ("_" + args.results_suffix) if not args.results_suffix.startswith("_") else args.results_suffix
    return s


def _dataset_type(args):
    if args.dataset in CODING_DATASETS:
        return "coding"
    if args.dataset in MCQA_DATASETS:
        return "mcqa"
    if args.dataset in TEXT_CLASSIFICATION_DATASETS:
        return "classification"
    return "math"


# ---------------------------------------------------------------------------
# Dataset / prompt helpers
# ---------------------------------------------------------------------------

CODING_TASK_PREFIX = "Task: Write a single Python function for the following problem. Do not include tests or examples in your output."
CODING_BASE_SUFFIX = "Algorithmic steps to solve this problem, followed by the Python function:\n"


# ---------------------------------------------------------------------------
# Model-family-aware user-content shaping
# ---------------------------------------------------------------------------
# Kept in sync with vllm-serve/generate_rollouts.py.  This must match the
# user content used at *generation* time, otherwise the hybrid trajectory's
# thinking-model prompt diverges from the cached think rollout.

# ORZ Table-5 prompt: the chunk prepended to {{prompt}} under the single
# User turn.  ORZ's shipped chat_template emits the preamble and the
# closing ``Assistant: <think>``; we only need the user-instruction body.
ORZ_USER_PREFIX = (
    "You must put your answer inside <answer> </answer> tags, i.e., "
    "<answer> answer here </answer>. And your final answer will be "
    "extracted automatically by the \\boxed{} tag."
)

# DeepSeek-R1 / QwQ recommended directive for mathematical problems.
MATH_DIRECTIVE = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)

# Datasets in this module whose questions we treat as "math problems" for
# the purposes of the R1/QwQ step-by-step directive.  Note: hybrid eval
# only runs on benchmark datasets (not trainmix), so this is a small set.
_MATH_DATASETS = {"math500", "gsm8k", "aime24", "aime25", "natreason",
                  "holdoutmix", "hendrycks_holdout"}


def _detect_think_family(model_id: str) -> str:
    """Return one of {'orz','r1','qwq','other'} based on the HF model id.

    Used to pick the appropriate user-content shaping for the thinking
    model.  The base model never gets shaped (its prompt is the legacy
    completion-style ``User: ... Assistant:`` string)."""
    low = (model_id or "").lower()
    if "open-reasoner-zero" in low or "orz" in low:
        return "orz"
    if "r1-distill" in low or "deepseek-r1" in low:
        return "r1"
    if "qwq" in low:
        return "qwq"
    return "other"


def _shape_think_user_content(question: str, family: str, *,
                              is_math_question: bool,
                              math_directive_enabled: bool) -> str:
    """Pre-wrap a question into the per-family user content for the
    thinking model, before ``apply_chat_template`` is called.

    MUST match the prompt shaping used at rollout-generation time in
    ``vllm-serve/generate_rollouts.py`` so that the parallel think
    model's argmax at hybrid time stays on the same distribution as
    the cached think tokens that were used to train the MLP.

    - ``orz`` : prepend the Table-5 user instruction; ALSO append the
      math directive when ``is_math_question`` (and
      ``math_directive_enabled``).  Mirrors generate_rollouts.py:
      ``_format_orz`` math-question branch.
    - ``r1`` / ``qwq`` : append the math directive iff
      ``is_math_question`` and ``math_directive_enabled``.
    - ``other`` : unchanged.
    """
    if family == "orz":
        content = f"{ORZ_USER_PREFIX}\n{question}"
        if is_math_question and math_directive_enabled:
            content = f"{content}\n\n{MATH_DIRECTIVE}"
        return content
    if family in ("r1", "qwq") and is_math_question and math_directive_enabled:
        return f"{question}\n\n{MATH_DIRECTIVE}"
    return question


def _format_gpqa_item(item, index):
    correct = item["Correct Answer"]
    distractors = [item["Incorrect Answer 1"], item["Incorrect Answer 2"],
                   item["Incorrect Answer 3"]]
    all_ans = [correct] + distractors
    random.Random(42 + index).shuffle(all_ans)
    letter = GPQA_LETTERS[all_ans.index(correct)]
    opts = "\n".join(f"{l}. {a}" for l, a in zip(GPQA_LETTERS, all_ans))
    return f"{item['Question']}\n\nOptions:\n{opts}", letter


def _build_task_prompts(item, i, args):
    test_list, starter_code = None, ""
    if args.dataset == "gsm8k":
        q, a = item["question"], item["answer"]
    elif args.dataset in ("aime24", "aime25"):
        q, a = item["problem"], item["answer"]
    elif args.dataset == "math500":
        q, a = item["problem"], item["answer"]
    elif args.dataset == "mbpp":
        q, a = item["text"], item["code"]
        test_list = item["test_list"]
    elif args.dataset == "livecodebench":
        q, a = item["question_content"], ""
        raw = item.get("public_test_cases", "[]")
        pts = json.loads(raw) if raw else []
        test_list = [f"# Test {ti+1}:\n- Input:\n{t['input']}\n- Output:\n{t['output']}"
                     for ti, t in enumerate(pts)] if pts else []
        starter_code = item.get("starter_code", "")
    elif args.dataset == "medqa":
        opts = "\n".join(f"{k}. {v}" for k, v in item["options"].items())
        q = f"{item['question']}\n\nOptions:\n{opts}"
        a = item["answer_idx"]
    elif args.dataset == "gpqa":
        q, a = _format_gpqa_item(item, i)
    elif args.dataset == "legalbench":
        q, a = item.get("text", str(item)), str(item.get("answer", ""))
    elif args.dataset in ("natreason", "holdoutmix", "hendrycks_holdout"):
        q, a = item["question"], str(item.get("reference_answer", ""))
    else:
        q, a = str(item), ""

    # Base-prompt style switch:
    #   'default'      = bare "User: q\nAssistant:"
    #   'stepwise'     = "User: Answer the following question. Respond
    #                    step by step.\n\n{q}\nAssistant:" (ff v1)
    #   'boxed'        = "User: {q}\n\nPlease reason step by step, and
    #                    put your final answer within \boxed{}.\n
    #                    Assistant:" (ff v2; QwQ/R1+ORZ derivative)
    #   'legacy_task'  = "Task: Answer the question below. Explain your
    #                    reasoning step by step.\n\n\n\nQuestion:\n{q}
    #                    \n\nStep by step answer:\n" (ff v3; the
    #                    structured prompt used on origin/main's
    #                    hybrid_*.py / optimize_steering_vectors.py
    #                    pipeline -- step-by-step instruction PLUS
    #                    explicit Task/Question/Answer scaffolding).
    # Coding datasets use their own scaffolding and are intentionally
    # NOT touched.
    base_style = getattr(args, "base_prompt_style", "default")

    # For 'think_template' style, the base prompt = think tokenizer's chat
    # template applied to the family-shaped user content (matches
    # gen_base_thinkprompt.py).  Lazy-load + cache the tokenizer on args.
    if base_style == "think_template":
        from transformers import AutoTokenizer
        if getattr(args, "_think_tok_for_baseprompt", None) is None:
            args._think_tok_for_baseprompt = AutoTokenizer.from_pretrained(
                args.thinking_model, trust_remote_code=True)
        _tt_tok = args._think_tok_for_baseprompt
        _tt_family = getattr(args, "think_prompt_family", "auto")
        if _tt_family == "auto":
            _tt_family = _detect_think_family(args.thinking_model)
        _tt_math = args.dataset in _MATH_DATASETS
        _tt_md_on = bool(getattr(args, "math_directive", False))

    def _bp_for(question: str) -> str:
        if base_style == "stepwise":
            return ("User: Answer the following question. Respond step "
                    f"by step.\n\n{question}\nAssistant:")
        if base_style == "boxed":
            return (f"User: {question}\n\nPlease reason step by step, "
                    "and put your final answer within "
                    "\\boxed{}.\nAssistant:")
        if base_style == "legacy_task":
            return ("Task: Answer the question below. Explain your "
                    "reasoning step by step.\n\n\n\nQuestion:\n"
                    f"{question}\n\nStep by step answer:\n")
        if base_style == "simple":
            return f"User: {question}\nAssistant: <think>\n"
        if base_style == "qa_response":
            return f"Question: {question}\nResponse: "
        if base_style == "qa_instr":
            return f"Answer the following question:\nQ: {question}\nA:"
        if base_style == "think_qa":
            return (
                "Your task is to answer the following question. First, "
                "carefully think through the question and then provide "
                "your final answer.\n"
                f"Q: {question}\nA:"
            )
        if base_style == "think_qa_marker":
            # Same preamble as think_qa, but the terminal cue is
            # "Think:" instead of "A:".  The hypothesis: the base
            # model is far less likely to emit an answer-first token
            # like " 8" right after "Think:" than right after "A:",
            # which gives the hybrid steering more room to keep the
            # response on a reasoning trajectory.
            return (
                "Your task is to answer the following question. First, "
                "carefully think through the question and then provide "
                "your final answer.\n"
                f"Q: {question}\nThink:"
            )
        if base_style == "think_word":
            return f"User: {question}\nAssistant: think:\n"
        if base_style == "convo_marker":
            return (
                'A conversation between User and Assistant. The User asks a '
                'question, and the Assistant solves it. The Assistant '
                'reasons step by step following the "think" marker, and when '
                'done provides their final answer after the "answer" marker.'
                f'\n\nUser: {question}\n\nA:\nthink:\n'
            )
        if base_style == "convo_marker_v2":
            return (
                'A conversation between User and Assistant. The User asks a '
                'question, and the Assistant solves it. The Assistant '
                'reasons step by step following the "think" marker, and when '
                'done provides their final answer after the "answer" marker.'
                f'\nUser: {question}\nAssistant:\nthink:\n'
            )
        if base_style == "convo_continue":
            return (
                'A conversation between user and assistant. User asks a '
                'question, assistant responds.'
                f'\n\nUser:\n{question}\n\nAssistant:\n'
            )
        if base_style == "convo_reason":
            return (
                'A conversation between User and Assistant. User asks a '
                'question and Assistant reasons through it step by step '
                'until figuring out the correct answer.'
                f'\n\nUser question: {question}\n\nAssistant reasoning: '
            )
        if base_style == "orz_think_template":
            # ORZ chat template (rendered), with the math directive baked in.
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
        if base_style == "r1_think_template":
            # DeepSeek-R1-Distill chat template (rendered), BOS token omitted.
            return (
                '<｜User｜>' + question +
                '\n\nPlease reason step by step, and put your final answer '
                'within \\boxed{}.<｜Assistant｜><think>\n'
            )
        if base_style == "qwq_think_template":
            # QwQ-32B (ChatML) chat template, rendered.
            return (
                '<|im_start|>user\n' + question +
                '\n\nPlease reason step by step, and put your final answer '
                'within \\boxed{}.<|im_end|>\n<|im_start|>assistant\n<think>\n'
            )
        if base_style == "plain_chat_math":
            return (
                f'user\n{question}\nPlease reason step by step, and put '
                f'your final answer within \\boxed{{}}.\n\nassistant\n'
                f'<think>\n'
            )
        if base_style == "convo_think":
            return (
                'A conversation between a User and Assistant. The User asks '
                'a question, and the Assistant solves it. The Assistant '
                'first thinks about the reasoning process in the mind and '
                'then provides the User with the answer. The reasoning '
                'process is enclosed within <think> </think> followed by '
                'the answer.'
                f'\nUser: \n{question}\nAssistant: <think>\n'
            )
        if base_style == "mini_preamble":
            return (
                'User asks a question. Assistant solves it by thinking '
                'through the question.'
                f'\n\nUser: {question}\nAssistant: <think>\n'
            )
        if base_style == "orz_full":
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
        if base_style == "r1_plain":
            return (
                f'User:\n{question}\nPlease reason step by step, and put '
                f'your final answer within \\boxed{{}}.\n\nAssistant:\n'
                f'<think>\n'
            )
        if base_style == "qwq_plain":
            return (
                f'User: {question}\n\nPlease reason step by step, and put '
                f'your final answer within \\boxed{{}}. \n\nAssistant: '
                f'<think>\n'
            )
        if base_style == "step_preamble":
            return (
                f'Please reason step by step\n\n'
                f'User: {question}\nAssistant:<think>'
            )
        if base_style == "think_template":
            shaped = _shape_think_user_content(
                question, _tt_family,
                is_math_question=_tt_math,
                math_directive_enabled=_tt_md_on,
            )
            return _tt_tok.apply_chat_template(
                [{"role": "user", "content": shaped}],
                tokenize=False, add_generation_prompt=True)
        return f"User: {question}\nAssistant:"

    if args.dataset == "mbpp":
        tc = "\n".join(test_list) if test_list else ""
        ts = f"\n\nPublic Tests:\n{tc}" if tc else ""
        tp = f"{CODING_TASK_PREFIX}\n\nProblem: {q}{ts}"
        bp = f"{tp}\n\n{CODING_BASE_SUFFIX}"
    elif args.dataset == "livecodebench":
        tc = "\n\n".join(test_list) if test_list else ""
        ts = f"\n\nPublic Tests:\n\n{tc}" if tc else ""
        sh = f"\n\nStarter code:\n```python\n{starter_code}\n```" if starter_code else ""
        tp = f"{CODING_TASK_PREFIX}\n\nProblem: {q}{sh}{ts}"
        bp = f"{tp}\n\n{CODING_BASE_SUFFIX}"
    elif args.dataset in ("medqa", "gpqa"):
        tp = f"{q}\n\nPlease select the correct answer (A, B, C, or D) and explain your reasoning."
        bp = _bp_for(q)
    else:
        tp = q
        bp = _bp_for(q)

    # ── Per-family user-content shaping for the thinking model ────────────
    # ORZ: prepend Table-5 instruction unconditionally.
    # R1 / QwQ: append math directive on math-style benchmarks if enabled.
    # Coding / MCQA prompts already have task-specific scaffolding; we still
    # apply the family shaping on top so that the model sees its expected
    # template ending (e.g. "Assistant: <think>" for ORZ via chat template).
    think_family = getattr(args, "think_prompt_family", "auto")
    if think_family == "auto":
        think_family = _detect_think_family(args.thinking_model)
    math_directive_enabled = bool(getattr(args, "math_directive", False))
    is_math = args.dataset in _MATH_DATASETS
    tp = _shape_think_user_content(
        tp, think_family,
        is_math_question=is_math,
        math_directive_enabled=math_directive_enabled,
    )

    return {"question": q, "correct_answer": a,
            "thinking_prompt": tp, "base_prompt": bp, "test_list": test_list}


# ---------------------------------------------------------------------------
# Standalone batch generation
# ---------------------------------------------------------------------------

def _batch_generate(model, tokenizer, prompts, max_new_tokens, batch_size,
                    *, use_chat_template=False, temperature=0.0,
                    on_batch_done=None, tag=""):
    """Generate completions in batches.

    If ``on_batch_done`` is provided it is invoked after each batch as
    ``on_batch_done(batch_start_idx, batch_results)`` (results use global
    prompt indices).  Use this for incremental cache flushing so an
    interrupted run does not lose earlier batches.
    """
    results = [None] * len(prompts)
    eos_id = int(tokenizer.eos_token_id)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = eos_id

    n_batches = (len(prompts) + batch_size - 1) // batch_size
    for bi, bs in enumerate(range(0, len(prompts), batch_size)):
        batch = prompts[bs:bs + batch_size]
        print(f"  [{tag}] batch {bi + 1}/{n_batches}  "
              f"({bs + 1}..{bs + len(batch)}/{len(prompts)})  generating...",
              flush=True)
        t0 = time.time()
        if use_chat_template:
            texts = [tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                add_generation_prompt=True, tokenize=False) for p in batch]
            enc = tokenizer(texts, return_tensors="pt", padding=True,
                            truncation=False).to(model.device)
        else:
            enc = tokenizer(batch, return_tensors="pt", padding=True,
                            truncation=False).to(model.device)

        gkw = dict(**enc, max_new_tokens=max_new_tokens,
                   pad_token_id=eos_id, eos_token_id=eos_id,
                   do_sample=(temperature > 0))
        if temperature > 0:
            gkw["temperature"] = temperature
        with torch.inference_mode():
            out = model.generate(**gkw)

        pl = enc["input_ids"].shape[1]
        batch_results = []
        for j in range(len(batch)):
            gen = out[j, pl:]
            nt = int(gen.shape[0])
            eos = bool(gen[-1].item() == eos_id) if nt > 0 else True
            rec = {
                "response": tokenizer.decode(gen, skip_special_tokens=True),
                "n_tokens": nt, "eos": eos}
            results[bs + j] = rec
            batch_results.append(rec)
        dt = time.time() - t0
        total_new = sum(r["n_tokens"] for r in batch_results)
        print(f"  [{tag}] batch {bi + 1}/{n_batches} done in {dt:.1f}s "
              f"({total_new} new tokens, "
              f"{total_new / max(dt, 1e-6):.1f} tok/s)", flush=True)
        if on_batch_done is not None:
            try:
                on_batch_done(bs, batch_results)
            except Exception as e:
                print(f"  [{tag}] WARN on_batch_done failed: {e}",
                      flush=True)
        del out, enc
        torch.cuda.empty_cache()
    return results


# ---------------------------------------------------------------------------
# DOM vector loading
# ---------------------------------------------------------------------------

def load_dom_vectors(dom_dir, model_short, descriptions):
    """Load steering vectors + per-category best layer/coeff. No bias vector."""
    vp = os.path.join(dom_dir, f"dom_vectors_multilayer_{model_short}.pt")
    mp = os.path.join(dom_dir, f"dom_metadata_multilayer_{model_short}.json")
    blp = os.path.join(dom_dir, f"dom_best_layers_{model_short}.json")
    bcp = os.path.join(dom_dir, f"dom_best_coeffs_{model_short}.json")

    vecs = torch.load(vp, map_location="cpu")
    with open(mp) as f:
        meta = json.load(f)
    with open(blp) as f:
        best_layers = json.load(f)
    best_coeffs = {}
    if os.path.exists(bcp):
        with open(bcp) as f:
            raw = json.load(f)
        best_coeffs = {k: float(v["best_coeff"]) for k, v in raw.items()}
        print(f"  Per-category best coefficients: {best_coeffs}")

    steering_vectors, layer_map = {}, {}
    for cat_id, lv in vecs.items():
        m = meta[cat_id]
        bl = int(best_layers[cat_id]["best_layer"])
        raw_vec = lv[str(bl)]
        cc = best_coeffs.get(cat_id, 1.0)
        final = raw_vec * cc
        key = f"idx{cat_id}"
        steering_vectors[key] = final.to(torch.float32)
        layer_map[key] = bl
        print(f"  Cat {cat_id} ({m['title']}): key={key}, layer={bl}, "
              f"coeff={cc}, norm={final.norm().item():.2f}")
    return steering_vectors, layer_map


# ---------------------------------------------------------------------------
# KV-cached batched hybrid generation
# ---------------------------------------------------------------------------

def _truncate_kv(kv, n=1):
    """Remove the last *n* positions from a DynamicCache in-place.

    Some transformers versions return the legacy `tuple-of-tuples` cache
    format from the first forward.  Detect that and slice manually so the
    caller doesn't have to special-case it.
    """
    if n <= 0:
        return kv
    if hasattr(kv, "crop"):
        kv.crop(-n)
        return kv
    # Legacy tuple format: tuple(layer)( (K, V) ) with K,V shaped
    # [B, n_heads, seq, head_dim].  Slice off the last n positions.
    return tuple(
        (k[..., :-n, :], v[..., :-n, :]) for (k, v) in kv
    )


def _snapshot_last_n(kv, n):
    """Clone the K and V tensors at positions [-n:] for every layer.

    Returns (snap_keys, snap_vals) where each is a list of [B, H, n, D]
    tensors (one per layer).  Used to make the multi-token KV-window
    sweep round-trip-safe: we mutate the cache during the sweep, then
    write the originals back so the cache is byte-identical to its
    pre-sweep (incrementally-built) state.
    """
    if n <= 0:
        return [], []
    if hasattr(kv, "key_cache"):
        ks = [k[..., -n:, :].clone() for k in kv.key_cache]
        vs = [v[..., -n:, :].clone() for v in kv.value_cache]
    else:
        ks = [k[..., -n:, :].clone() for (k, _) in kv]
        vs = [v[..., -n:, :].clone() for (_, v) in kv]
    return ks, vs


def _restore_last_n(kv, snap_ks, snap_vs):
    """Write `snap_ks` / `snap_vs` back into the cache at positions [-n:].

    Assumes the cache currently has length >= snap_ks[0].shape[-2].  In
    practice the cache will be at its original (pre-sweep) length, since
    each candidate forward truncates by N and re-rolls N tokens, leaving
    the cache at the same length.

    Wrapped in `torch.inference_mode()` because the cache tensors are
    typically themselves inference tensors (created inside the model
    forward's inference_mode context), so in-place mutation must happen
    inside an inference context.
    """
    if not snap_ks:
        return kv
    n = snap_ks[0].shape[-2]
    with torch.inference_mode():
        if hasattr(kv, "key_cache"):
            for li in range(len(kv.key_cache)):
                kv.key_cache[li][..., -n:, :].copy_(snap_ks[li])
                kv.value_cache[li][..., -n:, :].copy_(snap_vs[li])
            return kv
        # Legacy tuple-of-tuples cache (immutable tuple, but inner
        # tensors are mutable).
        for li, (k, v) in enumerate(kv):
            k[..., -n:, :].copy_(snap_ks[li])
            v[..., -n:, :].copy_(snap_vs[li])
    return kv


def hybrid_generate_batched(
    thinking_model, base_model, base_tokenizer,
    thinking_prompts, base_prompts, max_new_tokens,
    sae_layer, sae, steering_vectors, latent_descriptions,
    steering_layer_map, *,
    thinking_tokenizer=None,
    disable_sae_mean=False,
    show_progress=False, collect_details=True,
    random_firing=False, random_firing_exclude_top_k_keys=0,
    firing_replace_with_min_cosine=False,
    pure_steer_base_eos=False,
    random_steer_prob=0.0,
    random_guardrail=False, random_seed=0,
    coef_sweep=None, steer_all_positions=False,
    steer_all_positions_full=False,
    coef_select="pg", kl_topk=3,
    always_on_bias_vec=None,
    always_on_bias_layer=None,
    pg_bias_cat_sweep=False,
    pg_bias_vec=None,
    pg_bias_coefs=(0.0, 0.5, 1.0),
    pg_cat_coefs=(0.0, 0.5, 1.0),
    token_window=0,
    act_modulate: Optional[Dict[str, Tuple[float, float]]] = None,
    mlp_model=None,
    warmup_until_sentence_end: bool = False,
    warmup_max_tokens: int = 60,
    suppress_boxed_first_n_tokens: int = 0,
    accept_answer_close: bool = False,
    disable_eos_suppression: bool = False,
    free_fly_until_think_eos: bool = False,
    no_termination: bool = False,
    eos_prob_warmup: bool = False,
    eos_prob_warmup_steps: int = 0,
    thinking_continuation_text: Optional[List[str]] = None,
    base_continuation_text: Optional[List[str]] = None,
    mlp_coef_scale: float = 1.0,
    decode_temperature: float = 0.0,
    decode_seed: int = 0,
):
    """Batched KV-cached hybrid generation (paper recipe).

    For every decode step: both models produce a candidate next token.
    On token disagreement, sweep 10 coefficients 0.1..1.0 on the
    base-model steering vector (last-position, layer `old_vectors_layer`)
    and always pick the steered candidate with the highest thinking-model
    log-prob.  Steering is reverted from the cached K/V after the step
    so it only acts as a per-step logit nudge (matches the old non-KV
    pipeline's semantics).
    """
    B = len(thinking_prompts)
    assert len(base_prompts) == B and B > 0

    device = next(base_model.parameters()).device
    dtype = next(base_model.parameters()).dtype
    # When base and thinking models live on separate GPUs (--two_gpu_split),
    # we need to bounce small think-model inputs onto the think GPU before
    # each forward, and copy the final-step think logits back onto the base
    # GPU for token-comparison / steering.  When both models share a device
    # the .to() calls below are no-ops.
    think_device = next(thinking_model.parameters()).device
    _split_devices = (think_device != device)
    def _to_think(x):
        return x.to(think_device, non_blocking=True) if _split_devices else x
    hidden_size = base_model.config.hidden_size
    act_mean = (sae.activation_mean if hasattr(sae, "activation_mean")
                and not disable_sae_mean else None)
    eos_id = int(base_tokenizer.eos_token_id)

    default_layer = min(steering_layer_map.values()) if steering_layer_map else 0
    all_steer_layers = sorted(set(steering_layer_map.values()))

    # Pre-compute per-key vector norms + "is this a real (non-zero) vector"
    # flag once, used in per-token debug records to track `shift_norm` and
    # `n_no_vector` (e.g. when bias_only zeroed out category vectors, or when
    # the trainer dropped a category for having too few disagreements).
    vec_norms: Dict[str, float] = {}
    has_vec_map: Dict[str, bool] = {}
    for k, v in steering_vectors.items():
        n = float(v.float().norm().item())
        vec_norms[k] = n
        has_vec_map[k] = n > 1e-8

    # Ablation RNGs. Seeded Python RNGs so runs are reproducible.
    _firing_rng = random.Random(random_seed + 1)
    _guard_rng = random.Random(random_seed + 2)
    # Torch generator for the per-step Bernoulli draw used by
    # --random_steer_prob.  CPU-side so we can construct it once before
    # the model placement.
    _rstpos_gen = torch.Generator(device="cpu").manual_seed(int(random_seed) + 3)
    _firing_keys = [k for k in steering_vectors.keys() if k in steering_layer_map]

    # Pre-compute the deterministic "min-cosine-to-self" anti-cat map for the
    # firing_replace_with_min_cosine ablation.  For each key k_top in the
    # firing pool, find the key k_min (k_min != k_top) whose steering vector
    # has the lowest cosine similarity to V[k_top].  Use full L2-normalised
    # dot product; vectors may live on different layers but the *direction*
    # itself is compared (we're asking which trained anti-direction to apply).
    _min_cos_map = {}
    if firing_replace_with_min_cosine and len(_firing_keys) >= 2:
        _norm_vecs = {}
        for _k in _firing_keys:
            _v = steering_vectors.get(_k)
            if _v is None:
                continue
            _vf = _v.detach().float().flatten()
            _n = float(_vf.norm().item())
            if _n < 1e-8:
                continue
            _norm_vecs[_k] = (_vf / _n).cpu()
        for _kt in list(_norm_vecs.keys()):
            best_k = None
            best_cos = float("inf")
            for _ko in _norm_vecs:
                if _ko == _kt:
                    continue
                _c = float(torch.dot(_norm_vecs[_kt], _norm_vecs[_ko]).item())
                if _c < best_cos:
                    best_cos = _c
                    best_k = _ko
            if best_k is not None:
                _min_cos_map[_kt] = (best_k, best_cos)
        try:
            _summary_lines = ["[ablation:min_cosine] anti-cat map (per SAE top-1 -> min-cos cat):"]
            for _kt, (_km, _c) in _min_cos_map.items():
                _summary_lines.append(f"    {_kt} -> {_km}   cos={_c:+.3f}")
            print("\n".join(_summary_lines), flush=True)
        except Exception:
            pass

    generated_ids = [[] for _ in range(B)]
    token_infos = [[] for _ in range(B)] if collect_details else None
    steer_sels = [[] for _ in range(B)]
    coeff_sels = [[] for _ in range(B)]
    # Parallel per-step bias-coef history (only meaningful when
    # pg_bias_cat_sweep=True; otherwise zeros are recorded).
    bcoef_sels = [[] for _ in range(B)]
    finished = torch.zeros(B, dtype=torch.bool, device=device)

    # ---- hooks: thinking model SAE layer ----
    captured = {}

    def _sae_hook(mod, inp, out):
        h = out[0] if isinstance(out, tuple) else out
        captured["sae_act"] = h[:, -1, :].detach().float()

    handle_sae = thinking_model.model.layers[sae_layer].register_forward_hook(
        _sae_hook)

    # ---- hooks: steering on base model ----
    # steer_s["all_positions"]: when True and the hook sees a multi-token
    # forward (prefill / full-seq rerun during --steer_all_positions_full),
    # the shift c*v is applied to EVERY position's layer-`steering_layer`
    # output, matching hybrid_token.py's `token_windows=0` (all-positions)
    # semantics. When False (default), only the last position is shifted.
    steer_s = {"vecs": None, "layer_masks": {}, "coef": 1.0,
               "all_positions": False,
               # When > 0 and all_positions is True, shift is restricted
               # to the LAST `window_size` positions (matches paper's
               # --token_windows -N). 0 means "all" when all_positions=True.
               "window_size": int(token_window) if token_window > 0 else 0,
               # Independent bias term, used by --pg_bias_cat_sweep. When
               # bias_vec is None the hook behaves exactly as before.
               "bias_vec": None, "bias_coef": 0.0}
    # Dynamic window flag: when token_window > 0 AND we're NOT in the
    # legacy full-seq mode, use a KV-cache path that truncates the cache
    # by N=token_window, re-runs the last N tokens with the steering
    # hook applying the shift to all of them, then reads the last logit.
    # This costs O(N) per candidate forward instead of O(seq_len).
    _kv_window_mode = bool(token_window and int(token_window) > 0
                           and not steer_all_positions_full)
    if _kv_window_mode:
        # The hook needs all_positions=True (we'll set the toggle
        # explicitly around each multi-token candidate forward) and
        # window_size is implicitly the input length, so we leave the
        # legacy window_size=0 in steer_s and rely on h.shape[1] alone.
        pass
    if pg_bias_cat_sweep and pg_bias_vec is not None:
        steer_s["bias_vec"] = pg_bias_vec.to(device=device, dtype=dtype)
    # ---- always-on bias hook (fires at every position, every step) ----
    _bias_handles = []
    if always_on_bias_vec is not None and always_on_bias_layer is not None:
        _bv = always_on_bias_vec.to(device=device)
        def _bias_hook(mod, inp, out):
            h = out[0] if isinstance(out, tuple) else out
            # Only fire on single-token decode steps (h.shape[1]==1).
            # Skip multi-token prefill forwards — the bias was trained on
            # reasoning positions only (after the prompt), so applying it
            # to prompt tokens during prefill would be out-of-distribution.
            if h.shape[1] != 1:
                return out
            _bv_dev = _bv.to(h.device, dtype=h.dtype)
            h = h.clone()
            h[:, -1, :] = h[:, -1, :] + _bv_dev
            return (h,) + out[1:] if isinstance(out, tuple) else h
        _bias_handles.append(
            base_model.model.layers[always_on_bias_layer].register_forward_hook(
                _bias_hook))
        print(f"  [bias_always_on] hook installed at layer {always_on_bias_layer} "
              f"(norm={always_on_bias_vec.float().norm().item():.2f})", flush=True)

    steer_handles = []
    for li in all_steer_layers:
        def _mk(layer_i):
            def hook(mod, inp, out):
                mask = steer_s["layer_masks"].get(layer_i)
                if mask is None or not mask.any():
                    return out
                v = steer_s["vecs"]
                if v is None:
                    return out
                h = out[0] if isinstance(out, tuple) else out
                # When the base model is sharded across multiple GPUs
                # (device_map="auto"), the layer's output `h` lives on
                # whichever device that layer landed on, while `mask`,
                # `v`, and `coef` were created on the trainer's primary
                # `device`.  Move them to h.device for indexing.
                h_dev = h.device
                if mask.device != h_dev:
                    mask = mask.to(h_dev)
                if v.device != h_dev:
                    v = v.to(h_dev, dtype=h.dtype)
                h = h.clone()
                coef = steer_s["coef"]
                if isinstance(coef, torch.Tensor):
                    # Per-row coefficient (used when committing the
                    # winning coef back into the KV cache under
                    # --steer_all_positions).
                    if coef.device != h_dev:
                        coef = coef.to(h_dev)
                    delta = (coef[mask].view(-1, 1, 1)
                             * v[mask].unsqueeze(1))
                else:
                    delta = (coef * v[mask]).unsqueeze(1)
                # Optional independent bias term (only used by
                # --pg_bias_cat_sweep; otherwise bias_vec is None or
                # bias_coef==0 and this branch is a no-op).
                _bias_v = steer_s.get("bias_vec")
                _bias_c = steer_s.get("bias_coef", 0.0)
                if _bias_v is not None and (
                        isinstance(_bias_c, torch.Tensor)
                        or float(_bias_c) != 0.0):
                    if _bias_v.device != h_dev:
                        _bias_v = _bias_v.to(h_dev, dtype=h.dtype)
                    bv = _bias_v.view(1, 1, -1)
                    if isinstance(_bias_c, torch.Tensor):
                        if _bias_c.device != h_dev:
                            _bias_c = _bias_c.to(h_dev)
                        delta = delta + (_bias_c[mask].view(-1, 1, 1) * bv)
                    else:
                        delta = delta + (float(_bias_c) * bv)
                if h.shape[1] > 1 and steer_s["all_positions"]:
                    ws = int(steer_s.get("window_size", 0) or 0)
                    if ws > 0:
                        # Static last-N window (paper's --token_windows -N):
                        # clip the shift to the last ws positions of the
                        # full-sequence forward.
                        n_pos = h.shape[1]
                        if ws >= n_pos:
                            h[mask, :, :] = h[mask, :, :] + delta
                        else:
                            h[mask, -ws:, :] = h[mask, -ws:, :] + delta
                    else:
                        # Full-seq forward with all-positions steering
                        # (matches hybrid_token.py's token_windows=0).
                        h[mask, :, :] = h[mask, :, :] + delta
                elif h.shape[1] > 1:
                    # Match OLD `--token_windows -1`: only steer the
                    # last position of a full-seq forward.
                    h[mask, -1:, :] += delta
                else:
                    h[mask] += delta
                return (h,) + out[1:] if isinstance(out, tuple) else h
            return hook
        steer_handles.append(
            base_model.model.layers[li].register_forward_hook(_mk(li)))

    pbar = (tqdm(total=max_new_tokens, desc=f"Hybrid B={B}", leave=False)
            if (show_progress and tqdm) else None)

    # ---- SAE helpers ----
    def _classify(sae_act_batch):
        """Classify + set up steering vectors/masks in steer_s."""
        # The SAE-layer hook on the thinking model captures activations on
        # whatever device that layer lives on (with device_map="auto" the
        # thinking model can be sharded across multiple GPUs).  The SAE
        # itself was placed on the FIRST thinking-model parameter's device
        # in main(), which may differ.  Move the activation onto the SAE's
        # device before encoding to avoid a cuda:N vs cuda:0 mismatch.
        sae_dev = next(sae.parameters()).device
        if sae_act_batch.device != sae_dev:
            sae_act_batch = sae_act_batch.to(sae_dev)
        if act_mean is not None:
            x = sae_act_batch - act_mean.to(sae_dev)
            x = x / (x.norm(dim=-1, keepdim=True) + 1e-8)
        else:
            x = sae_act_batch
        la = sae.encoder(x - sae.b_dec)
        ids = la.argmax(dim=-1)
        vals = la[torch.arange(B, device=sae_dev), ids]
        ids = ids.to(device)
        vals = vals.to(device)

        vecs = torch.zeros(B, hidden_size, device=device, dtype=dtype)
        assigns = [default_layer] * B
        keys, titles = [], []
        mod_factors = []
        # Precompute top-K latents per row for the exclude-top-k ablation.
        # We pull enough top latents so that we can recover at least K distinct
        # category keys even when multiple latents map to the same key.
        _excl_k = int(random_firing_exclude_top_k_keys)
        _top_lids = None
        if random_firing and _excl_k > 0:
            # Pull a generous number of top latents to ensure K distinct keys.
            _topn = min(la.shape[-1], max(_excl_k * 4, 16))
            _top_lids = la.topk(_topn, dim=-1).indices.to(device)
        for b in range(B):
            lid = ids[b].item()
            k = latent_descriptions[lid]["key"]
            # Deterministic anti-cat replacement (min cosine to SAE top-1).
            # Independent of --random_firing.
            if firing_replace_with_min_cosine and _min_cos_map:
                _ent = _min_cos_map.get(k)
                if _ent is not None:
                    k = _ent[0]
            if random_firing and _firing_keys:
                if _excl_k > 0 and _top_lids is not None:
                    # Build the set of top-K *unique* keys by activation rank.
                    excl = []
                    seen = set()
                    for _l in _top_lids[b].tolist():
                        _kk = latent_descriptions[_l]["key"]
                        if _kk not in seen:
                            seen.add(_kk)
                            excl.append(_kk)
                            if len(excl) >= _excl_k:
                                break
                    pool = [kk for kk in _firing_keys if kk not in seen]
                    if not pool:
                        # Edge case: excluded everything (shouldn't happen if
                        # excl_k < len(_firing_keys)). Fall back to uniform.
                        pool = list(_firing_keys)
                    k = _firing_rng.choice(pool)
                else:
                    # Ablation: override SAE-picked key with a uniform random
                    # category key (same pool the SAE oracle selects from).
                    k = _firing_rng.choice(_firing_keys)
            keys.append(k)
            titles.append(latent_descriptions[lid]["title"])
            sv = steering_vectors.get(k)
            # ---- Optional per-position activation modulation ----
            mod_b = 1.0
            if act_modulate is not None and sv is not None:
                rng = act_modulate.get(k)
                if rng is not None:
                    lo, hi = rng
                    val_b = float(vals[b].item())
                    if hi > lo + 1e-8:
                        mod_b = (val_b - lo) / (hi - lo)
                        if mod_b < 0.0:
                            mod_b = 0.0
                        elif mod_b > 1.0:
                            mod_b = 1.0
                    else:
                        mod_b = 1.0
            mod_factors.append(mod_b)
            if sv is not None:
                if mod_b == 1.0:
                    vecs[b] = sv
                else:
                    vecs[b] = sv * mod_b
            if k in steering_layer_map:
                assigns[b] = steering_layer_map[k]
        if act_modulate is not None:
            steer_s["last_mod_factors"] = mod_factors
        else:
            steer_s["last_mod_factors"] = None
        steer_s["vecs"] = vecs
        steer_s["assigns"] = assigns
        for l in all_steer_layers:
            steer_s["layer_masks"][l] = torch.tensor(
                [a == l for a in assigns], dtype=torch.bool, device=device)
        return ids, vals, keys, titles

    def _clear_steering():
        for l in all_steer_layers:
            steer_s["layer_masks"][l] = torch.zeros(
                B, dtype=torch.bool, device=device)

    _clear_steering()  # start with no steering active

    try:
        # ---- Tokenize ----
        think_tok = thinking_tokenizer if thinking_tokenizer is not None else base_tokenizer
        think_tok.padding_side = "left"
        think_texts = [think_tok.apply_chat_template(
            [{"role": "user", "content": p}],
            add_generation_prompt=True, tokenize=False) for p in thinking_prompts]
        # Continuation mode (Phase-2 4096-extension): append the cached truncated
        # think response *after* the assistant-open header so the KV prefill
        # absorbs it.  No new tokens are generated for this segment; the loop
        # picks up from the seam.  Pure-mode disables all tag detection so the
        # seam carries no protocol state.
        if thinking_continuation_text is not None:
            assert len(thinking_continuation_text) == len(think_texts), (
                f"thinking_continuation_text len mismatch: "
                f"{len(thinking_continuation_text)} vs {len(think_texts)}")
            think_texts = [t + (c or "") for t, c in
                           zip(think_texts, thinking_continuation_text)]
        t_enc = think_tok(think_texts, return_tensors="pt",
                          padding=True, truncation=False).to(device)
        t_ids = t_enc["input_ids"]
        t_mask = t_enc["attention_mask"]
        t_pos = t_mask.long().cumsum(-1) - 1
        t_pos.masked_fill_(t_mask == 0, 0)
        t_lens = t_mask.sum(dim=1)

        base_tokenizer.padding_side = "left"
        # Continuation mode: append the existing hybrid output to the base
        # prompt so the base KV prefill carries the prior trace.  Generated
        # tokens (n_generated) count only the NEW continuation segment.
        if base_continuation_text is not None:
            assert len(base_continuation_text) == len(base_prompts), (
                f"base_continuation_text len mismatch: "
                f"{len(base_continuation_text)} vs {len(base_prompts)}")
            base_prompts = [b + (c or "") for b, c in
                            zip(base_prompts, base_continuation_text)]
        b_enc = base_tokenizer(base_prompts, return_tensors="pt",
                               padding=True, truncation=False).to(device)
        b_ids = b_enc["input_ids"]
        b_mask = b_enc["attention_mask"]
        b_pos = b_mask.long().cumsum(-1) - 1
        b_pos.masked_fill_(b_mask == 0, 0)
        b_lens = b_mask.sum(dim=1)

        # ---- Prefill: thinking model ----
        # logits_to_keep=1 restricts the final lm_head projection to the
        # last position only, which matters when the fp32 lm_head wrapper is
        # active (otherwise we'd allocate a (B, L, vocab) fp32 tensor during
        # prefill -> OOM at large batch sizes).
        with torch.inference_mode():
            think_out = thinking_model(input_ids=_to_think(t_ids),
                                       attention_mask=_to_think(t_mask),
                                       position_ids=_to_think(t_pos),
                                       use_cache=True,
                                       logits_to_keep=1)
        think_kv = think_out.past_key_values
        # Move just the last-step logits back to base device for comparison
        # / steering.  KV cache stays on think_device for next forward.
        think_logits = (think_out.logits[:, -1, :].to(device)
                        if _split_devices else think_out.logits[:, -1, :])
        del think_out

        lat_ids, act_vals, lat_keys, lat_titles = _classify(captured["sae_act"])
        _clear_steering()

        # ---- Prefill: base model (unsteered) ----
        with torch.inference_mode():
            base_out = base_model(input_ids=b_ids, attention_mask=b_mask,
                                  position_ids=b_pos, use_cache=True,
                                  logits_to_keep=1)
        base_kv = base_out.past_key_values
        base_logits = base_out.logits[:, -1, :]
        del base_out
        torch.cuda.empty_cache()

        # Token fed to base model to produce current base_logits (for re-run)
        prev_base_input = b_ids[:, -1].clone()

        # For --steer_all_positions_full: maintain the growing full input
        # sequence (prompt + tokens generated so far) and its attention
        # mask, so the coef sweep can do fresh full forwards that match
        # hybrid_token.py's `base_model.trace(base_output_ids)` exactly.
        base_ids_full = b_ids.clone()
        base_mask_full = b_mask.clone()

        think_pos = t_lens.clone()
        base_pos = b_lens.clone()
        n_gen = 0

        # ---- Think-region state ----
        # Base prompts are "User: {q}\nAssistant:" — they DO NOT contain
        # <think>.  The two models run in parallel with DIFFERENT prompts:
        #   - Thinking model: native template (e.g. "<|begin_of_thinking|>..."
        #     or "User: {q}\nAssistant: <think>") and stays inside <think>.
        #   - Base model:  "User: {q}\nAssistant:" — it generates an answer
        #     directly.  During the reasoning phase its tokens are steered
        #     by the thinking model's hidden states.
        # When the thinking model emits </think>, we transition:
        #   - Thinking KV is advanced by its own </think> token(s).
        #   - Base KV is advanced by a base-friendly transition sequence
        #     (e.g. "\n\nFinal answer: ").
        # Each model has its own forced-token queue because the two
        # transition sequences differ in length and content.
        # After transition, the row is in "answer" phase: NO steering,
        # NO EOS suppression, no disagreement protocol — base generates
        # freely until its natural EOS.  The thinking model is fed the
        # base-emitted tokens to keep its KV advancing, but its logits
        # are no longer used.
        _think_tok_for_tags = think_tok  # thinking tokenizer

        def _close_seq_think():
            return _think_tok_for_tags.encode("</think>", add_special_tokens=False)

        _think_close_seq_think = _close_seq_think()       # think vocab ids for </think>
        _think_close_id_single = (                        # -1 if multi-token in think
            int(_think_close_seq_think[0])
            if len(_think_close_seq_think) == 1 else -1)

        # BPE-aware multi-token detection.
        # For ORZ / Qwen-base tokenizers, "</think>" is split by BPE into
        # context-dependent pieces.  Both the leading AND trailing tokens
        # are context-dependent because BPE merges `</` with the preceding
        # character class AND `>` with the following character class.
        # Examples observed in real ORZ outputs (via offset_mapping):
        #   ORZ-7B:   "...\n</think>\n"   -> ['</', 'think', '>\n']        ids = (522, 26865, 397)
        #   ORZ-32B:  "...\n</think>\n"   -> ['</', 'think', '>\n']        (522, 26865, 397)
        #   ORZ-0.5B: "... </think> "     -> [' </', 'think', '>']         (690, 26865,  29)
        #   ORZ-1.5B: "...).</think>\n"   -> [').</', 'think', '>\n']      (66233, 26865, 397)
        #   ORZ-1.5B: "...</think>\n\n"   -> ['</', 'think', '>\n\n']     (522, 26865, 1339)
        # In every variant the MIDDLE token is invariant: 26865 ('think').
        # We use a 3-state machine over the invariant centre:
        #   state 0 -> 1: any token whose decoded form ends with '</'      ("opener")
        #   state 1 -> 2: token id 26865 ('think')
        #   state 2 -> DETECTED: any token whose decoded form starts with '>'   ("closer")
        # False-positive risk is low: the suffix `think>` only follows
        # `</` in `</think>` context; `<think>` opening tokenises to a
        # different mid-token (e.g. 766 = 'ink' for bare, or doesn't
        # transit out of state 0 because the leading token doesn't end in '</').
        _think_close_mid: int = -1
        _opener_ids: set = set()
        _closer_ids: set = set()
        if _think_close_id_single < 0:
            try:
                _think_close_mid = int(
                    _think_tok_for_tags.encode("think", add_special_tokens=False)[0])
            except Exception:
                # Fallback to the trailing-but-one token of bare "</think>".
                if len(_think_close_seq_think) >= 3:
                    _think_close_mid = int(_think_close_seq_think[-2])
            # Enumerate every token id whose decoded text ends with '</' (opener)
            # and every token id whose decoded text starts with '>' (closer).
            # Vocab is ~150k tokens; one-shot decode is cheap (~20s, once per
            # generation call).
            try:
                _vocab = _think_tok_for_tags.get_vocab()
                for tok_str, tok_id in _vocab.items():
                    s = _think_tok_for_tags.decode([int(tok_id)])
                    if s.endswith("</"):
                        _opener_ids.add(int(tok_id))
                    if s.startswith(">"):
                        _closer_ids.add(int(tok_id))
            except Exception:
                # Fallback: at minimum, include the leading and trailing
                # token IDs of the bare encoding.
                if len(_think_close_seq_think) >= 3:
                    _opener_ids.add(int(_think_close_seq_think[0]))
                    _closer_ids.add(int(_think_close_seq_think[-1]))

        # Base-friendly transition sequence.  Same string for all model
        # families; tokenised in the base vocab (length varies per family).
        _BASE_TRANSITION_STR = "\n\nFinal answer: "
        _base_transition_seq = base_tokenizer.encode(
            _BASE_TRANSITION_STR, add_special_tokens=False)

        if len(_think_close_seq_think) == 1:
            print(f"[hybrid] </think> detection: single-token in think vocab "
                  f"(id={_think_close_id_single}); "
                  f"base transition {_BASE_TRANSITION_STR!r} = "
                  f"{_base_transition_seq} ({len(_base_transition_seq)} tok)")
        else:
            print(f"[hybrid] </think> detection: BPE-suffix in think vocab "
                  f"(mid={_think_close_mid} 'think', "
                  f"openers={len(_opener_ids)} tokens ending in '</', "
                  f"closers={len(_closer_ids)} tokens starting with '>'); "
                  f"base transition {_BASE_TRANSITION_STR!r} = "
                  f"{_base_transition_seq} ({len(_base_transition_seq)} tok)")

        # ---- Optional: also detect '</answer>' as a close trigger ----
        # Same 3-state machine as for </think>, but with 'answer' as the
        # invariant middle token.  Re-uses _opener_ids / _closer_ids since the
        # '</' opener and '>' closer token sets are independent of which tag
        # sits between them.  Single-token close for </answer> is also
        # supported via _answer_close_id_single if applicable.
        _answer_close_mid: int = -1
        _answer_close_id_single: int = -1
        if accept_answer_close:
            try:
                _answer_close_seq = _think_tok_for_tags.encode(
                    "</answer>", add_special_tokens=False)
                if len(_answer_close_seq) == 1:
                    _answer_close_id_single = int(_answer_close_seq[0])
                else:
                    _answer_close_mid = int(_think_tok_for_tags.encode(
                        "answer", add_special_tokens=False)[0])
                print(f"[hybrid] </answer> ALSO accepted as close trigger: "
                      f"mid={_answer_close_mid} 'answer', "
                      f"single_id={_answer_close_id_single}")
            except Exception as e:
                print(f"[hybrid] WARN: could not set up </answer> detection: {e}")
                accept_answer_close = False

        # ---- Warmup-until-first-sentence-end gate (diagnostic) ----
        # When enabled, each row's hybrid protocol (steering + EOS suppression
        # + </think> detection) stays OFF until the base has emitted a
        # sentence-ending token.  Used to test the hypothesis that the 0.5B
        # auto-completion failure is driven by EOS suppression engaging from
        # token-0 before the base has had a chance to commit to its own
        # response style.
        warmup_active = torch.zeros(B, dtype=torch.bool, device=device)
        warmup_tok_count = torch.zeros(B, dtype=torch.long, device=device)
        _sentence_end_ids: set = set()
        if warmup_until_sentence_end:
            import re as _re
            # Require punctuation FOLLOWED BY whitespace inside the same token
            # (e.g. ".\n", ".\n\n", "}.\n", "?\n").  This avoids false-positives
            # on bare-"." tokens that appear inside numbers like "3.14".  In the
            # rare case where the tokenizer splits sentence endings as
            # ["...", "."] + [" ", "..."], the warmup_max_tokens cap will fire.
            _send_re = _re.compile(r"[.?!]\s")
            try:
                _vsize = int(getattr(base_tokenizer, "vocab_size",
                                     len(base_tokenizer)))
            except Exception:
                _vsize = len(base_tokenizer)
            for _tid in range(_vsize):
                try:
                    _txt = base_tokenizer.decode([_tid])
                except Exception:
                    continue
                if _send_re.search(_txt):
                    _sentence_end_ids.add(_tid)
            warmup_active = torch.ones(B, dtype=torch.bool, device=device)
            print(f"[hybrid] warmup_until_sentence_end=True: "
                  f"{len(_sentence_end_ids)} sentence-ender tokens; "
                  f"cap={warmup_max_tokens} tokens/row")

        # One bool per row: True while the thinking model is still inside <think>.
        # Flips to False once the thinking model has emitted </think> and the
        # base model has been advanced by the transition sequence.
        # When warmup is active for a row, inside_think starts False (so the
        # state machine routes through the pass-through branch) and flips to
        # True the step a sentence-end is detected.
        inside_think = ~warmup_active.clone()
        if not warmup_until_sentence_end:
            inside_think = torch.ones(B, dtype=torch.bool, device=device)
        # Per-row queues of token IDs to force.  Independent per model
        # because the transition sequences differ.
        _think_forced_queues: list = [[] for _ in range(B)]
        _base_forced_queues: list = [[] for _ in range(B)]
        # Per-row partial-match counter for multi-token </think> in think vocab.
        _think_close_partial: list = [0] * B
        # Parallel partial-match state for '</answer>' detection.  Only used
        # when accept_answer_close=True.
        _answer_close_partial: list = [0] * B

        # Coefficient sweep. Default is paper's 10-point grid [0.1..1.0];
        # overridable via --coef_sweep (e.g. [0.5..1.0] to match the
        # collaborator's ablation-study report).
        _SWEEP = list(coef_sweep) if coef_sweep else [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
        assert len(_SWEEP) > 0 and all(0.0 <= float(c) <= 10.0 for c in _SWEEP)

        # ---- EOS probability warmup (linear in probability space) ----
        # If --eos_prob_warmup is set, on every step we multiply the base
        # softmax probability of EOS by alpha = min(1, n_gen / T), where
        # T = --eos_prob_warmup_steps (or max_new_tokens by default), and
        # redistribute the displaced mass proportionally to the non-EOS
        # tokens.  The result is fed back through log() into base_logits
        # so all downstream argmax / EOS-suppression / steering decisions
        # see the warmed-up distribution.
        _eos_T = (eos_prob_warmup_steps
                  if eos_prob_warmup_steps > 0 else max_new_tokens)
        _eos_T = max(1, int(_eos_T))

        # ---- Per-batch sampling generator (for decode_temperature > 0) ----
        # In sampling mode, the base/think candidate tokens used for the
        # disagreement gate are sampled from each model's distribution, and the
        # emitted token is sampled from either the base or steered distribution.
        # The generator is seeded per-process so repeated runs of the same
        # prompt with the same seed give reproducible samples.
        _decode_T = float(decode_temperature)
        _do_decode_sample = _decode_T > 0.0
        if _do_decode_sample:
            _decode_gen = torch.Generator(device=device)
            _decode_gen.manual_seed(int(decode_seed))

        def _sample_or_argmax(logits: torch.Tensor) -> torch.Tensor:
            """Return shape-(B,) ids; argmax when T==0, else categorical
            sample from softmax(logits / T)."""
            if not _do_decode_sample:
                return torch.argmax(logits, dim=-1)
            probs = torch.softmax(logits.float() / _decode_T, dim=-1)
            return torch.multinomial(probs, num_samples=1,
                                     generator=_decode_gen).squeeze(-1)

        while n_gen < max_new_tokens:
            # ---- 0. EOS probability warmup ----
            if eos_prob_warmup:
                _alpha = min(1.0, n_gen / float(_eos_T))
                if _alpha < 1.0:
                    _probs = torch.softmax(base_logits, dim=-1)
                    _p_eos = _probs[:, eos_id]                    # (B,)
                    _new_p_eos = _alpha * _p_eos                  # (B,)
                    _denom = (1.0 - _p_eos).clamp(min=1e-12)
                    _scale = ((1.0 - _new_p_eos) / _denom).unsqueeze(-1)
                    _probs = _probs * _scale
                    _probs[:, eos_id] = _new_p_eos
                    base_logits = torch.log(_probs.clamp(min=1e-30))
                    del _probs, _p_eos, _new_p_eos, _denom, _scale

            # ---- 1. Candidate tokens from each model ----
            # T=0: historical argmax gate. T>0: sampled gate, so both the
            # disagreement decision and the emitted token share the same
            # stochastic base / think candidates at this step.
            base_next_toks = _sample_or_argmax(base_logits)
            think_next_toks = _sample_or_argmax(think_logits)
            # Skip steering on:
            #  - finished rows
            #  - rows draining a forced-token queue (transition mid-flight)
            #  - rows already past the think region (answer phase)
            #  - rows where think model is signalling </think> this step
            _is_forced = torch.tensor(
                [len(qb) > 0 or len(qt) > 0
                 for qb, qt in zip(_base_forced_queues, _think_forced_queues)],
                dtype=torch.bool, device=device)
            _not_inside = ~inside_think
            _is_tag = torch.zeros(B, dtype=torch.bool, device=device)
            # In pure_steer_base_eos mode we IGNORE any think-side close-tag
            # signals -- the run is purely base-EOS driven and steering must
            # keep firing at every disagreement position, regardless of
            # whether the think model is emitting </think> / </answer> right
            # now. Free-fly mode follows the same logic for the same reason.
            if not (pure_steer_base_eos or free_fly_until_think_eos):
                if _think_close_id_single >= 0:
                    _is_tag |= (think_next_toks == _think_close_id_single)
                if accept_answer_close and _answer_close_id_single >= 0:
                    _is_tag |= (think_next_toks == _answer_close_id_single)
            if random_steer_prob > 0.0:
                # Ablation: REPLACE the natural disagreement signal with a
                # per-step Bernoulli(p) draw.  Every other gate (finished,
                # forced-queue, answer-phase, warmup, tag) still suppresses
                # steering -- only the (think == base) check is swapped out.
                _rs_draw = torch.rand(B, generator=_rstpos_gen).to(device)
                _synthetic_agree = (_rs_draw >= random_steer_prob)
                token_agree = (_synthetic_agree
                               | finished | _is_tag | _is_forced | _not_inside
                               | warmup_active)
            else:
                token_agree = ((think_next_toks == base_next_toks)
                               | finished | _is_tag | _is_forced | _not_inside
                               | warmup_active)

            best_coeff = torch.zeros(B, device=device)
            # Parallel best-bias-coef tracker (only meaningful under
            # pg_bias_cat_sweep; otherwise stays 0 everywhere).
            best_bcoef = torch.zeros(B, device=device)
            did_steer = torch.zeros(B, dtype=torch.bool, device=device)
            output_toks = base_next_toks

            if not token_agree.all():
                # ---- 2. Coefficient sweep on disagreeing rows ----
                disagree_mask = ~token_agree
                think_lp = torch.log_softmax(think_logits, dim=-1)
                arange_B = torch.arange(B, device=device)

                # KL-top-K mode: pre-compute thinking model's top-K
                # logprobs/ids and probs ONCE per step.  Used to score
                # each candidate base distribution by the same form as
                # the training objective:
                #   score_sc = -sum_k p_t(k) * log p_b_steered(k)
                # (lower = better fit to thinking's top-K target).
                if coef_select in ("kl_top3", "kl_topk"):
                    K = int(kl_topk) if coef_select == "kl_topk" else 3
                    t_topk_lp, t_topk_ix = think_lp.topk(K, dim=-1)
                    t_topk_p = t_topk_lp.exp()  # (B, K)

                # Old pipeline: on disagreement always pick best steered.
                # For PG: best_lp tracks max thinking logp of base argmax.
                # For KL-topK: best_score tracks min(-CE) i.e. max(-CE).
                if coef_select == "pg":
                    best_lp = torch.full_like(
                        think_lp[arange_B, base_next_toks], float("-inf"))
                else:
                    # store the LARGER-is-better quantity: -CE (so we can
                    # reuse the same `better = c_lp > best_lp` logic).
                    best_lp = torch.full(
                        (B,), float("-inf"),
                        device=device, dtype=think_lp.dtype)
                best_tok = base_next_toks.clone()
                # think_top1_match*: track whether any coef in the sweep
                # produced an argmax matching thinking's top-1 for each
                # disagreement row.  Rows that never match stay UNSTEERED.
                if coef_select in ("think_top1_match",
                                   "think_top1_match_maxconf"):
                    matched_row = torch.zeros(B, dtype=torch.bool, device=device)
                raw_vecs = steer_s["vecs"]

                # ----- MLP coefficient prediction -----
                # For coef_select == "mlp": single forward with mlp_alpha, no sweep.
                # For coef_select == "mlp_pg": compute mlp_alpha here, then fall
                #   through to the perplexity-guardrail sweep below where the
                #   effective coef per candidate is mlp_alpha * sweep_coef.
                _mlp_handled = False
                _mlp_alpha = None
                if coef_select in ("mlp", "mlp_pg") and mlp_model is not None:
                    _cap = {}
                    steer_layer_for_cap = min(all_steer_layers)
                    def _cap_hook(_mod, _inp, _out):
                        h = _out[0] if isinstance(_out, tuple) else _out
                        _cap["h"] = h[:, -1, :].detach().float()
                    _cap_handle = base_model.model.layers[steer_layer_for_cap].register_forward_hook(_cap_hook)
                    _clear_steering()
                    base_kv = _truncate_kv(base_kv)
                    with torch.inference_mode():
                        _cap_out = base_model(
                            input_ids=prev_base_input.unsqueeze(1),
                            attention_mask=b_mask,
                            position_ids=(base_pos - 1).unsqueeze(1),
                            past_key_values=base_kv, use_cache=True)
                    base_kv = _cap_out.past_key_values
                    del _cap_out
                    _cap_handle.remove()
                    h_raw = _cap["h"]  # (B, D)

                    _mlp_n_cats = mlp_model.n_cats
                    cat_id_list = []
                    for b in range(B):
                        k = lat_keys[b]
                        cid = int(k.replace("idx", "")) if k.startswith("idx") else 0
                        cat_id_list.append(min(cid, _mlp_n_cats - 1))
                    cat_id_tensor = torch.tensor(cat_id_list, dtype=torch.long, device=h_raw.device)

                    _mlp_dev = next(mlp_model.parameters()).device
                    if h_raw.device != _mlp_dev:
                        mlp_model = mlp_model.to(h_raw.device)
                    with torch.inference_mode():
                        alpha = mlp_model(h_raw, cat_id_tensor)  # (B,)
                    alpha = alpha.to(device=device, dtype=torch.float32)
                    alpha = torch.where(disagree_mask, alpha, torch.zeros_like(alpha))
                    if mlp_coef_scale != 1.0:
                        alpha = alpha * float(mlp_coef_scale)
                    _mlp_alpha = alpha

                if coef_select == "mlp" and _mlp_alpha is not None:
                    alpha = _mlp_alpha
                    steer_s["coef"] = alpha
                    for li in all_steer_layers:
                        steer_s["layer_masks"][li] = torch.tensor(
                            [steer_s["assigns"][b] == li for b in range(B)],
                            dtype=torch.bool, device=device) & disagree_mask
                    base_kv = _truncate_kv(base_kv)
                    with torch.inference_mode():
                        _mlp_out = base_model(
                            input_ids=prev_base_input.unsqueeze(1),
                            attention_mask=b_mask,
                            position_ids=(base_pos - 1).unsqueeze(1),
                            past_key_values=base_kv, use_cache=True)
                    base_kv = _mlp_out.past_key_values
                    mlp_logits = _mlp_out.logits[:, -1, :]
                    # Emission token: sampled from steered logits when
                    # decode_temperature>0, else argmax.
                    best_tok = _sample_or_argmax(mlp_logits)
                    del _mlp_out

                    best_coeff = alpha
                    need_steer = disagree_mask
                    # For no-steering rows, emit the same base candidate that
                    # participated in the gate above.  In sampling mode this
                    # avoids drawing a second independent base token.
                    output_toks = torch.where(need_steer, best_tok, base_next_toks)
                    did_steer = need_steer

                    # Revert KV to unsteered state
                    _clear_steering()
                    steer_s["coef"] = 1.0
                    base_kv = _truncate_kv(base_kv)
                    with torch.inference_mode():
                        _mlp_revert = base_model(
                            input_ids=prev_base_input.unsqueeze(1),
                            attention_mask=b_mask,
                            position_ids=(base_pos - 1).unsqueeze(1),
                            past_key_values=base_kv, use_cache=True)
                    base_kv = _mlp_revert.past_key_values
                    del _mlp_revert
                    _mlp_handled = True

                # Random-guardrail ablation: pre-sample a random coefficient
                # per row; we'll commit whatever the sweep produces at that
                # coefficient instead of picking by thinking-model logp.
                if random_guardrail:
                    chosen_coef = torch.tensor(
                        [_guard_rng.choice(_SWEEP) for _ in range(B)],
                        device=device, dtype=torch.float32)

                # Position_ids for full-seq forward (only used under
                # --steer_all_positions_full).  Built from the growing
                # attention mask so left-padded prompts get correct RoPE
                # positions.
                if steer_all_positions_full:
                    full_pos = base_mask_full.long().cumsum(-1) - 1
                    full_pos.masked_fill_(base_mask_full == 0, 0)

                if not _mlp_handled:
                    # Cartesian sweep over (bias_coef, cat_coef) is controlled
                    # by `pg_bias_cat_sweep` (independent of `coef_select`, which
                    # now only controls the SCORING rule used to rank the
                    # candidates).  For think_top1_match we want the SMALLEST
                    # coef that produces an argmax==thinking top-1, so iterate
                    # sorted.
                    if pg_bias_cat_sweep:
                        # Cartesian product (bias_coef, cat_coef). Both are
                        # passed through steer_s; the hook adds
                        # bias_coef * bias_vec  +  cat_coef * cat_vec
                        # at the disagreement-step positions.  Sort so that
                        # (0, 0) comes first; this lets `last_logits` for the
                        # no-shift candidate be the 1-token base logits and
                        # avoids running a useless forward.
                        _sweep_iter = sorted(
                            [(float(b), float(c))
                             for b in pg_bias_coefs
                             for c in pg_cat_coefs],
                            key=lambda bc: (bc[0] + bc[1], bc[0], bc[1]))
                    else:
                        _sweep_iter = (sorted(_SWEEP)
                                       if coef_select == "think_top1_match"
                                       else _SWEEP)
                    # Track whether any candidate forward actually touched
                    # base_kv (so we know whether the post-sweep restore is
                    # needed).  Stays False when only the no-shift candidate
                    # ran via the short-circuit below.
                    _kv_dirty = False
                    # Snapshot the last N positions of base_kv before any
                    # shifted candidate runs.  After the sweep we copy these
                    # back, giving a cache state that's byte-identical to
                    # the pristine incrementally-built cache (no extra drift
                    # from an unsteered re-roll).  Only relevant in
                    # _kv_window_mode.
                    _kv_snap_ks, _kv_snap_vs = [], []
                    if _kv_window_mode:
                        _N_snap = max(1, min(int(token_window), int(n_gen) + 1))
                        _kv_snap_ks, _kv_snap_vs = _snapshot_last_n(
                            base_kv, _N_snap)
                    for _sw in _sweep_iter:
                        if pg_bias_cat_sweep:
                            _bc, sc = _sw  # (bias_coef, cat_coef)
                            steer_s["bias_coef"] = _bc
                            steer_s["coef"] = sc
                        else:
                            _bc = 0.0
                            sc = _sw
                            if coef_select == "mlp_pg" and _mlp_alpha is not None:
                                # Effective per-row coef = mlp_alpha * sweep_coef.
                                # Zero rows where MLP would steer with alpha=0
                                # (i.e. non-disagree positions) stay zero.
                                steer_s["coef"] = (_mlp_alpha
                                                   * float(sc)).to(device)
                            else:
                                steer_s["coef"] = sc
                        for li in all_steer_layers:
                            steer_s["layer_masks"][li] = torch.tensor(
                                [steer_s["assigns"][b] == li for b in range(B)],
                                dtype=torch.bool, device=device) & disagree_mask

                        if steer_all_positions_full:
                            # Faithful reproduction of hybrid_token.py:
                            # fresh full-sequence forward (no KV cache),
                            # hook applies c*v to ALL positions of layer
                            # `steering_layer`. Logits at last position
                            # are the candidate next-token distribution.
                            steer_s["all_positions"] = True
                            with torch.inference_mode():
                                out = base_model(
                                    input_ids=base_ids_full,
                                    attention_mask=base_mask_full,
                                    position_ids=full_pos,
                                    use_cache=False,
                                    logits_to_keep=1)
                            steer_s["all_positions"] = False
                            last_logits = out.logits[:, -1, :]
                            cand = torch.argmax(last_logits, dim=-1)
                            del out
                        elif _kv_window_mode:
                            # Dynamic last-N window with KV-cache reuse.
                            #   - N_eff = min(token_window, current generation
                            #     length).  We never reach back into prompt
                            #     tokens (would be OOD w.r.t. the bias which
                            #     was trained on reasoning positions only).
                            #
                            # IMPORTANT: when this candidate is the "no-shift"
                            # one (bias_coef == cat_coef == 0), we MUST short-
                            # circuit to the 1-token base decode logits
                            # (`base_logits` / `base_next_toks`) rather than
                            # re-rolling the last N tokens.  Otherwise tiny
                            # bf16/matmul nondeterminism between the
                            # incrementally-built KV cache (built by repeated
                            # 1-token forwards) and the multi-token re-roll
                            # over the same positions can flip the argmax,
                            # turning real "no-shift wins" decisions into
                            # spurious "shift needed" picks downstream.  This
                            # drift is mild for SDPA but enormous for eager
                            # attention on Qwen2.5-1.5B (cache K/V can drift
                            # by O(1) per position after ~60 steps), and is
                            # what was producing the eager runs' negative gap
                            # recovery.  Skipping the forward also saves us
                            # one model call per disagreement step.
                            _is_no_shift = (
                                pg_bias_cat_sweep
                                and float(_bc) == 0.0 and float(sc) == 0.0)
                            if _is_no_shift:
                                last_logits = base_logits
                                cand = base_next_toks
                            else:
                                N_eff = max(1, min(int(token_window), int(n_gen) + 1))
                                base_kv = _truncate_kv(base_kv, n=N_eff)
                                last_N_ids = base_ids_full[:, -N_eff:]
                                pos_ids = (
                                    torch.arange(N_eff, device=device).view(1, -1)
                                    + (base_pos - N_eff).view(-1, 1))
                                # Multi-token forward: ask the hook to shift
                                # ALL of these N positions (= the last N of
                                # the full sequence after re-extending the
                                # cache).
                                steer_s["all_positions"] = True
                                with torch.inference_mode():
                                    out = base_model(
                                        input_ids=last_N_ids,
                                        attention_mask=b_mask,
                                        position_ids=pos_ids,
                                        past_key_values=base_kv,
                                        use_cache=True)
                                steer_s["all_positions"] = False
                                base_kv = out.past_key_values
                                last_logits = out.logits[:, -1, :]
                                cand = torch.argmax(last_logits, dim=-1)
                                _kv_dirty = True
                                del out
                        else:
                            base_kv = _truncate_kv(base_kv)
                            with torch.inference_mode():
                                out = base_model(
                                    input_ids=prev_base_input.unsqueeze(1),
                                    attention_mask=b_mask,
                                    position_ids=(base_pos - 1).unsqueeze(1),
                                    past_key_values=base_kv, use_cache=True)
                            base_kv = out.past_key_values
                            last_logits = out.logits[:, -1, :]
                            cand = torch.argmax(last_logits, dim=-1)
                            del out

                        if random_guardrail:
                            # Commit this candidate for rows whose random draw == sc.
                            picked = (torch.isclose(
                                chosen_coef, torch.tensor(sc, device=device))
                                & disagree_mask)
                            if picked.any():
                                best_coeff[picked] = sc
                                best_tok[picked] = cand[picked]
                        elif coef_select == "think_top1_match":
                            # Strict ceiling: lock in the SMALLEST coef whose
                            # steered argmax already equals thinking's top-1.
                            # Rows that already matched are skipped; rows that
                            # never match stay unsteered (best_coeff stays 0,
                            # best_tok stays base_next_toks).
                            new_match = (
                                (cand == think_next_toks)
                                & disagree_mask
                                & ~matched_row)
                            if new_match.any():
                                matched_row[new_match] = True
                                best_coeff[new_match] = sc
                                best_tok[new_match] = cand[new_match]
                        elif coef_select == "think_top1_match_maxconf":
                            # Low-confound oracle: among coefs whose steered
                            # argmax EQUALS thinking's top-1 token T, pick the
                            # one with the HIGHEST log p_steered(T).  Random
                            # vectors fail the strict argmax==T condition (~1/V
                            # per coef) and fall through to base unsteered.
                            is_match = (
                                (cand == think_next_toks) & disagree_mask)
                            if is_match.any():
                                base_lp = torch.log_softmax(
                                    last_logits.float(), dim=-1)
                                c_lp = base_lp[arange_B, think_next_toks]
                                c_lp = c_lp.to(best_lp.dtype)
                                # Only consider matching rows: non-matchers'
                                # logp shouldn't compete.
                                c_lp = torch.where(
                                    is_match, c_lp,
                                    torch.full_like(c_lp, float("-inf")))
                                better = (c_lp > best_lp) & is_match
                                if better.any():
                                    matched_row[better] = True
                                    best_lp[better] = c_lp[better]
                                    best_coeff[better] = sc
                                    best_tok[better] = cand[better]
                        elif coef_select == "fixed":
                            # Fixed coef: unconditionally commit the result of this
                            # (only) sweep step for all disagreeing rows.
                            best_coeff = torch.where(
                                disagree_mask,
                                torch.tensor(sc, device=device,
                                             dtype=best_coeff.dtype).expand(B),
                                best_coeff)
                            best_tok = torch.where(disagree_mask, cand, best_tok)
                        else:
                            if coef_select in ("pg", "mlp_pg"):
                                c_lp = think_lp[arange_B, cand]
                            elif coef_select == "think_top1":
                                # Oracle / ceiling: pick coef that maximises
                                # the steered-base log-prob at the thinking
                                # model's argmax token T.  Tokenizer alignment
                                # (base vocab == think vocab, 1:1) is verified
                                # at startup so think_next_toks indexes base
                                # logits correctly.
                                base_lp = torch.log_softmax(
                                    last_logits.float(), dim=-1)
                                c_lp = base_lp[arange_B, think_next_toks]
                                c_lp = c_lp.to(best_lp.dtype)
                            else:
                                # KL-top-K score: -CE between thinking top-K
                                # (as soft target) and steered base log-probs.
                                # Larger (closer to 0) = better fit.
                                base_lp = torch.log_softmax(
                                    last_logits.float(), dim=-1)
                                base_lp_at_topk = base_lp.gather(
                                    -1, t_topk_ix)  # (B, K)
                                c_lp = (t_topk_p * base_lp_at_topk).sum(dim=-1)
                                c_lp = c_lp.to(best_lp.dtype)
                            better = (c_lp > best_lp) & disagree_mask
                            if better.any():
                                best_lp[better] = c_lp[better]
                                best_coeff[better] = sc
                                if pg_bias_cat_sweep:
                                    best_bcoef[better] = _bc
                                best_tok[better] = cand[better]

                    if coef_select in ("think_top1_match",
                                       "think_top1_match_maxconf"):
                        # Only count rows that actually matched thinking's top-1
                        # at some coef as "steered"; the rest fall through to
                        # the unsteered base argmax.
                        need_steer = matched_row & disagree_mask
                    elif coef_select == "fixed":
                        need_steer = disagree_mask
                    else:
                        need_steer = disagree_mask
                    output_toks = torch.where(need_steer, best_tok, base_next_toks)
                    did_steer = need_steer

                    if steer_all_positions_full:
                        # Faithful mode: the sweep never touched base_kv
                        # (use_cache=False), so nothing to revert. The KV
                        # cache still reflects the unsteered processing up
                        # to prev_base_input.
                        _clear_steering()
                    elif steer_all_positions:
                        # Reproduce collaborator's hybrid_token.py semantics:
                        # persist the winning-coef shift at this position in
                        # the K/V cache for all layers > sae_layer, rather
                        # than reverting to unsteered. Across decode steps
                        # this accumulates so subsequent attention sees
                        # steered K/V for every past disagreement position,
                        # approximating their 'shift every position's layer-
                        # sae_layer output on every forward pass' behaviour.
                        for li in all_steer_layers:
                            steer_s["layer_masks"][li] = torch.tensor(
                                [steer_s["assigns"][b] == li for b in range(B)],
                                dtype=torch.bool, device=device) & disagree_mask
                        steer_s["coef"] = best_coeff  # per-row winner
                        base_kv = _truncate_kv(base_kv)
                        with torch.inference_mode():
                            commit = base_model(
                                input_ids=prev_base_input.unsqueeze(1),
                                attention_mask=b_mask,
                                position_ids=(base_pos - 1).unsqueeze(1),
                                past_key_values=base_kv, use_cache=True)
                        base_kv = commit.past_key_values
                        del commit
                        steer_s["coef"] = 1.0  # reset
                    elif _kv_window_mode:
                        # Restore K/V at positions [-N:] to the snapshot we
                        # took before the sweep.  This gives back the
                        # pristine, incrementally-built cache state — bit-
                        # identical to "no sweep ever happened", so the next
                        # 1-token decode produces the same logits it would
                        # without our steering machinery.  This replaces the
                        # old "re-roll unsteered" path, which was itself
                        # introducing batched-vs-incremental drift into the
                        # cache at every disagreement step.
                        _clear_steering()
                        if _kv_dirty:
                            _restore_last_n(base_kv, _kv_snap_ks, _kv_snap_vs)
                    else:
                        # Revert the K/V at prev_base_input to unsteered —
                        # steering should act as a per-step logit nudge only,
                        # matching the old non-KV pipeline's semantics.
                        _clear_steering()
                        base_kv = _truncate_kv(base_kv)
                        with torch.inference_mode():
                            revert = base_model(
                                input_ids=prev_base_input.unsqueeze(1),
                                attention_mask=b_mask,
                                position_ids=(base_pos - 1).unsqueeze(1),
                                past_key_values=base_kv, use_cache=True)
                        base_kv = revert.past_key_values
                        del revert

            # ---- 2b. Think-region state machine ----
            # Per-row logic (applied in this priority order):
            #
            # Phase: REASONING (inside_think=True, no forced queue active)
            #   - normal: emit steered base token (already in output_toks);
            #     also advance thinking model's KV with the same token.
            #   - detect </think> on thinking stream → start TRANSITION:
            #       seed _think_forced_queue with </think> tokens (think vocab)
            #       seed _base_forced_queue with "\n\nFinal answer: " (base vocab)
            #   - if base argmax == EOS → suppress (substitute think argmax).
            #
            # Phase: TRANSITION (forced queues being drained)
            #   - base emit:  drain _base_forced_queue → output_toks[b]
            #   - think feed: drain _think_forced_queue → think_feed_toks[b]
            #   - Queues are independent (different lengths).  Row flips to
            #     ANSWER phase once BOTH queues are empty.
            #
            # Phase: ANSWER (inside_think=False, both queues empty)
            #   - emit base argmax as-is (no steering, no EOS suppression).
            #   - thinking model fed the same base token to keep KV moving.
            think_feed_toks = think_next_toks.clone()
            for _b in range(B):
                if finished[_b]:
                    continue
                # Drain forced queues first (transition phase)
                if _base_forced_queues[_b] or _think_forced_queues[_b]:
                    if _base_forced_queues[_b]:
                        output_toks[_b] = _base_forced_queues[_b].pop(0)
                    else:
                        # Base queue drained first; emit base argmax
                        # (no steering — we're past think region).
                        output_toks[_b] = base_next_toks[_b]
                    if _think_forced_queues[_b]:
                        think_feed_toks[_b] = _think_forced_queues[_b].pop(0)
                    else:
                        # Think queue drained first; feed the same token
                        # as base (the answer phase has begun).
                        think_feed_toks[_b] = output_toks[_b]
                    # Once BOTH queues drained, flip to answer phase
                    if (not _base_forced_queues[_b]
                            and not _think_forced_queues[_b]):
                        inside_think[_b] = False
                    continue
                # Answer phase: pass through, no steering, no EOS suppression.
                # think KV keeps moving with the same emitted base token.
                if not inside_think[_b]:
                    think_feed_toks[_b] = output_toks[_b]
                    continue
                # Reasoning phase: detect </think> (and optionally </answer>)
                # in thinking stream
                think_tok_id = int(think_next_toks[_b].item())
                detected_close = False
                in_partial_window = False  # true while building multi-token match
                if _think_close_id_single >= 0:
                    # Single-token detection (DeepSeek distill)
                    if think_tok_id == _think_close_id_single:
                        detected_close = True
                    elif (accept_answer_close
                          and _answer_close_id_single >= 0
                          and think_tok_id == _answer_close_id_single):
                        detected_close = True
                else:
                    # BPE-suffix detection (ORZ / Qwen base).  Both leading
                    # and trailing tokens of the </think> tokenisation are
                    # context-dependent.  We anchor on the INVARIANT middle
                    # token (26865 = 'think') and use opener/closer sets:
                    #   state 0 -> 1: token whose decoded form ends with '</'
                    #   state 1 -> 2: token id 26865 ('think')
                    #   state 2 -> DETECTED: token whose decoded form starts with '>'
                    # During the partial window we feed the thinking model
                    # its OWN predicted token so its KV builds the close
                    # sequence correctly (regardless of what the steered
                    # base model emits this step).
                    s = _think_close_partial[_b]
                    if s == 0:
                        if think_tok_id in _opener_ids:
                            _think_close_partial[_b] = 1
                            in_partial_window = True
                    elif s == 1:
                        if think_tok_id == _think_close_mid:
                            _think_close_partial[_b] = 2
                            in_partial_window = True
                        elif think_tok_id in _opener_ids:
                            # Stay in state 1 (back-to-back `</` openers).
                            in_partial_window = True
                        else:
                            _think_close_partial[_b] = 0
                    elif s == 2:
                        if think_tok_id in _closer_ids:
                            _think_close_partial[_b] = 0
                            detected_close = True
                        elif think_tok_id in _opener_ids:
                            # False alarm on the suffix, but the new token is
                            # itself a fresh `</` opener — restart at state 1.
                            _think_close_partial[_b] = 1
                            in_partial_window = True
                        else:
                            _think_close_partial[_b] = 0

                    # Parallel '</answer>' state machine (only if enabled).
                    # Same opener/closer sets, different middle token.
                    # If EITHER detection fires we treat it as a close.
                    if accept_answer_close and _answer_close_mid >= 0 and not detected_close:
                        sa = _answer_close_partial[_b]
                        if sa == 0:
                            if think_tok_id in _opener_ids:
                                _answer_close_partial[_b] = 1
                                in_partial_window = True
                        elif sa == 1:
                            if think_tok_id == _answer_close_mid:
                                _answer_close_partial[_b] = 2
                                in_partial_window = True
                            elif think_tok_id in _opener_ids:
                                in_partial_window = True
                            else:
                                _answer_close_partial[_b] = 0
                        elif sa == 2:
                            if think_tok_id in _closer_ids:
                                _answer_close_partial[_b] = 0
                                detected_close = True
                            elif think_tok_id in _opener_ids:
                                _answer_close_partial[_b] = 1
                                in_partial_window = True
                            else:
                                _answer_close_partial[_b] = 0
                # Free-fly mode: ignore any detected close.  The hybrid
                # stays in REASONING and steering keeps firing.  The row
                # only terminates when think predicts EOS (handled in the
                # record-output block below).
                # Pure-steer-base-eos mode: same close-detection skip, but
                # the row terminates on BASE EOS (no transition / no Final
                # answer injection / no EOS suppression).
                if free_fly_until_think_eos or pure_steer_base_eos or no_termination:
                    detected_close = False
                    in_partial_window = False
                if detected_close:
                    # Full close detected — start transition.
                    # Seed both queues with their respective transition seqs.
                    # For multi-token close: the close_seq has already been
                    # fed to think KV across the partial-match steps, so the
                    # think queue is empty here.  For single-token close: the
                    # close token is fed THIS step (output below).
                    base_seq = list(_base_transition_seq)
                    output_toks[_b] = base_seq[0]
                    _base_forced_queues[_b] = base_seq[1:]
                    if _think_close_id_single >= 0:
                        # Single-token close: feed the token now, queue empty.
                        think_feed_toks[_b] = _think_close_id_single
                        _think_forced_queues[_b] = []
                    else:
                        # Multi-token close: last close-token fed this step,
                        # all prior close tokens already fed during partial.
                        think_feed_toks[_b] = think_tok_id
                        _think_forced_queues[_b] = []
                    # If base queue exhausted in one step, flip phase now
                    if not _base_forced_queues[_b]:
                        inside_think[_b] = False
                    continue
                if in_partial_window:
                    # Currently building multi-token close-match.  Feed think
                    # its OWN argmax to keep the partial alive in its KV.
                    # Base emits steered argmax as normal (no EOS suppression
                    # since think isn't predicting EOS — it's predicting a
                    # close-sequence token).
                    think_feed_toks[_b] = think_tok_id
                    continue
                # Normal reasoning step: optionally suppress base EOS so the
                # row keeps generating until thinking emits </think>.  When
                # --disable_eos_suppression OR --pure_steer_base_eos is set,
                # base's EOS is allowed through, the row is marked finished
                # in the record-output block.
                # In free-fly mode base EOS is ALWAYS suppressed regardless
                # of the other flags (we only stop on think EOS).
                _eff_supp_off = (disable_eos_suppression or pure_steer_base_eos)
                if ((not _eff_supp_off or free_fly_until_think_eos or no_termination)
                        and int(output_toks[_b].item()) == eos_id):
                    output_toks[_b] = think_next_toks[_b]
                # Feed think model the EMITTED base token (keeps KVs roughly
                # aligned during reasoning).
                think_feed_toks[_b] = output_toks[_b]

            # ---- 2c. Boxed-token suppression (diagnostic) ----
            # If --suppress_boxed_first_n_tokens > 0, replace any emitted
            # 'boxed' (79075) or ' boxed' (73664) token in the first N
            # hybrid-generation tokens with the next-best UNSTEERED base
            # argmax.  These two tokens exclusively form the '\\boxed{' final
            # answer marker in the Qwen2.5 vocabulary, so suppressing them
            # tests the 0.5B quirk where steering drives the model into the
            # final-answer template too early.
            if (suppress_boxed_first_n_tokens > 0
                    and n_gen < suppress_boxed_first_n_tokens):
                _box_mask = ((output_toks == 79075)
                             | (output_toks == 73664)) & (~finished)
                if _box_mask.any():
                    _masked_blogits = base_logits.clone()
                    _masked_blogits[:, 79075] = float("-inf")
                    _masked_blogits[:, 73664] = float("-inf")
                    _alt_toks = torch.argmax(_masked_blogits, dim=-1)
                    output_toks = torch.where(_box_mask, _alt_toks, output_toks)

            # ---- 3. Record output ----
            # Pre-compute per-row debug tensors so the Python loop below
            # just reads scalars.
            base_top_ids = base_next_toks.detach().cpu().tolist()
            think_top_ids = think_next_toks.detach().cpu().tolist()
            disagree_row = (~token_agree).detach().cpu().tolist()
            steer_match_think = (
                did_steer
                & (output_toks == think_next_toks)
            ).detach().cpu().tolist()
            did_steer_row = did_steer.detach().cpu().tolist()
            best_coeff_row = best_coeff.detach().cpu().tolist()
            best_bcoef_row = best_bcoef.detach().cpu().tolist()
            assigns_snapshot = list(steer_s.get("assigns", [default_layer] * B))

            for b in range(B):
                if finished[b]:
                    continue
                tid = output_toks[b].item()
                generated_ids[b].append(tid)
                sel = "steered" if did_steer_row[b] else "base"
                steer_sels[b].append(sel)
                cc = round(best_coeff_row[b], 1) if did_steer_row[b] else 0.0
                coeff_sels[b].append(cc)
                bc = round(best_bcoef_row[b], 1) if did_steer_row[b] else 0.0
                bcoef_sels[b].append(bc)
                if token_infos is not None:
                    k = lat_keys[b]
                    v_norm = vec_norms.get(k, 0.0)
                    s_layer = assigns_snapshot[b] if assigns_snapshot else default_layer
                    token_infos[b].append({
                        "token": base_tokenizer.decode(tid),
                        "latent_id": lat_ids[b].item(),
                        "latent_title": lat_titles[b],
                        "latent_key": k,
                        "activation_value": act_vals[b].item(),
                        "coefficient": cc,
                        "selection": sel,
                        "base_tok_id": int(base_top_ids[b]),
                        "think_tok_id": int(think_top_ids[b]),
                        "disagreed": bool(disagree_row[b]),
                        "has_vector": bool(has_vec_map.get(k, False)),
                        "steer_layer": int(s_layer),
                        "shift_norm": round(cc * v_norm, 3),
                        "steered_matches_think": bool(steer_match_think[b]),
                    })
                if tid == eos_id and not no_termination:
                    finished[b] = True
                # Free-fly: terminate this row when the THINK model
                # predicts EOS at this step, regardless of what base
                # predicted (base EOS is suppressed above).
                # --no_termination overrides this: row only stops at
                # max_new_tokens.
                if (free_fly_until_think_eos
                        and int(think_top_ids[b]) == eos_id
                        and not no_termination):
                    finished[b] = True

            # ---- 3b. Warmup-exit detection ----
            # For rows currently in warmup, check if the just-emitted token
            # ends a sentence (or if we've hit the hard cap).  When the
            # condition fires, the row transitions out of warmup and the
            # hybrid protocol (steering + EOS suppression + </think>
            # detection) engages from the next step on.
            if warmup_until_sentence_end and warmup_active.any():
                # Increment per-row warmup counter for still-warming rows.
                warmup_tok_count = torch.where(
                    warmup_active,
                    warmup_tok_count + 1,
                    warmup_tok_count)
                _to_list = output_toks.detach().cpu().tolist()
                _act_list = warmup_active.cpu().tolist()
                _cnt_list = warmup_tok_count.cpu().tolist()
                exit_idx = []
                for _b in range(B):
                    if not _act_list[_b]:
                        continue
                    if (_to_list[_b] in _sentence_end_ids
                            or _cnt_list[_b] >= warmup_max_tokens):
                        exit_idx.append(_b)
                if exit_idx:
                    for _b in exit_idx:
                        warmup_active[_b] = False
                        inside_think[_b] = True

            n_gen += 1
            if pbar:
                pbar.update(1)
            if finished.all():
                break

            # ---- 4. Advance both models with the chosen output token ----
            t_mask = torch.cat([t_mask, torch.ones(B, 1, device=device, dtype=t_mask.dtype)], dim=1)
            b_mask = torch.cat([b_mask, torch.ones(B, 1, device=device, dtype=b_mask.dtype)], dim=1)

            # Grow the full base sequence for --steer_all_positions_full.
            base_ids_full = torch.cat(
                [base_ids_full, output_toks.unsqueeze(1)], dim=1)
            base_mask_full = torch.cat(
                [base_mask_full,
                 torch.ones(B, 1, device=device, dtype=base_mask_full.dtype)],
                dim=1)

            with torch.inference_mode():
                think_out = thinking_model(
                    input_ids=_to_think(think_feed_toks.unsqueeze(1)),
                    attention_mask=_to_think(t_mask),
                    position_ids=_to_think(think_pos.unsqueeze(1)),
                    past_key_values=think_kv, use_cache=True)
            think_kv = think_out.past_key_values
            think_logits = (think_out.logits[:, -1, :].to(device)
                            if _split_devices else
                            think_out.logits[:, -1, :])
            del think_out
            think_pos += 1

            lat_ids, act_vals, lat_keys, lat_titles = _classify(captured["sae_act"])
            _clear_steering()

            with torch.inference_mode():
                base_out = base_model(
                    input_ids=output_toks.unsqueeze(1),
                    attention_mask=b_mask,
                    position_ids=base_pos.unsqueeze(1),
                    past_key_values=base_kv, use_cache=True)
            base_kv = base_out.past_key_values
            base_logits = base_out.logits[:, -1, :]
            del base_out
            base_pos += 1

            prev_base_input = output_toks.clone()

            if n_gen % 32 == 0:
                torch.cuda.empty_cache()

    finally:
        handle_sae.remove()
        for h in steer_handles:
            h.remove()
        for h in _bias_handles:
            h.remove()
        if pbar:
            pbar.close()
        gc.collect()
        torch.cuda.empty_cache()

    out_list = []
    for b in range(B):
        n_tot = len(generated_ids[b])
        n_st = sum(1 for s in steer_sels[b] if s == "steered")
        cc_hist = {}
        for c in coeff_sels[b]:
            if c > 0:
                k = str(round(c, 1))
                cc_hist[k] = cc_hist.get(k, 0) + 1
        # Joint (bias_coef, cat_coef) histogram for --pg_bias_cat_sweep
        # (recorded only at "steered" positions; (0.0, 0.0) is included
        # so the user can see how often PG picked the no-steer option).
        bc_hist: Dict[str, int] = {}
        for bcv, cv, sel in zip(bcoef_sels[b], coeff_sels[b], steer_sels[b]):
            if sel != "steered":
                continue
            key = f"{round(bcv, 1)}|{round(cv, 1)}"
            bc_hist[key] = bc_hist.get(key, 0) + 1
        # Richer per-example debug aggregates (derived from token_infos).
        infos_b = token_infos[b] if token_infos else []
        n_dis = sum(1 for ti in infos_b if ti.get("disagreed"))
        n_no_vec = sum(1 for ti in infos_b
                       if ti.get("disagreed") and not ti.get("has_vector", True))
        n_match_think = sum(1 for ti in infos_b
                            if ti.get("steered_matches_think"))
        per_cat_count: Dict[str, int] = {}
        per_cat_coef_sum: Dict[str, float] = {}
        per_layer_count: Dict[int, int] = {}
        for ti in infos_b:
            if ti.get("selection") != "steered":
                continue
            k = ti.get("latent_key", "?")
            per_cat_count[k] = per_cat_count.get(k, 0) + 1
            per_cat_coef_sum[k] = per_cat_coef_sum.get(k, 0.0) + float(
                ti.get("coefficient", 0.0))
            s_layer = int(ti.get("steer_layer", -1))
            per_layer_count[s_layer] = per_layer_count.get(s_layer, 0) + 1
        per_cat_mean_coef = {k: round(per_cat_coef_sum[k] / per_cat_count[k], 3)
                             for k in per_cat_count}
        top5_cats = sorted(per_cat_count.items(),
                           key=lambda kv: -kv[1])[:5]
        out_list.append({
            "generated_ids": generated_ids[b],
            "n_generated": n_tot,
            "token_latent_info": (token_infos[b] if token_infos else []),
            "steering_selection": steer_sels[b],
            "ended_by_eos": bool(finished[b]),
            "steering_stats": {
                "n_steered": n_st,
                "n_total": n_tot,
                "frac_steered": round(n_st / max(n_tot, 1), 4),
                "coeff_distribution": cc_hist,
                "bias_cat_coeff_distribution": bc_hist,
                # --- new debug aggregates ---
                "n_disagree": n_dis,
                "n_no_vector": n_no_vec,
                "n_steered_matches_think": n_match_think,
                "frac_steered_matches_think": round(
                    n_match_think / max(n_st, 1), 4),
                "per_category_counts": per_cat_count,
                "per_category_mean_coef": per_cat_mean_coef,
                "per_layer_counts": {str(k): v
                                     for k, v in per_layer_count.items()},
                "top5_categories": [[k, v] for k, v in top5_cats],
            },
        })
    return out_list


# ---------------------------------------------------------------------------
# LLM judge
# ---------------------------------------------------------------------------

def _safe_chat_batch(prompts, model_name, max_tokens=2000, **kwargs):
    import asyncio
    import concurrent.futures
    async def _run():
        return await chat_batch(prompts, model=model_name,
                                max_tokens=max_tokens, **kwargs)
    try:
        loop = asyncio.get_running_loop()
        with concurrent.futures.ThreadPoolExecutor() as ex:
            return ex.submit(lambda: asyncio.run(_run())).result()
    except RuntimeError:
        return asyncio.run(_run())


def _build_judge_prompt(answer, gold, question, ds_type="math", test_list=None):
    if ds_type == "coding":
        tc = "\n\n".join(test_list) if test_list else "No test cases"
        ref = f"\n\nReference:\n```python\n{gold}\n```" if gold else ""
        return (f"Evaluate correctness.\n\nProblem: {question}\n\n"
                f"Response:\n{answer}\n{ref}\n\nTests:\n{tc}\n\n"
                f"Answer YES or NO.")
    elif ds_type == "mcqa":
        return (f"Question: {question}\nCorrect: {gold}\n\n"
                f"Response: {answer}\n\nDid they pick {gold}? YES or NO.")
    elif ds_type == "classification":
        return (f"Question: {question}\nCorrect: {gold}\n\n"
                f"Response: {answer}\n\nCorrect classification? YES or NO.")
    return (f"Question: {question}\nCorrect answer: {gold}\n\n"
            f"Response: {answer}\n\n"
            f"Does the response contain the correct answer? YES or NO.")


def judge_batch(items, judge_model, n_reps=1, max_concurrent=40):
    """Judge items concurrently. Returns list of {correct, raw, reps}."""
    prompts = []
    for it in items:
        p = _build_judge_prompt(it["answer"], it["gold"], it["question"],
                                ds_type=it.get("ds_type", "math"),
                                test_list=it.get("test_list"))
        prompts.extend([p] * max(1, n_reps))

    responses = _safe_chat_batch(prompts, judge_model, max_tokens=100,
                                 max_concurrent_requests=max_concurrent)

    out = []
    idx = 0
    for it in items:
        reps = []
        for _ in range(max(1, n_reps)):
            r = responses[idx] if idx < len(responses) else ""
            if not isinstance(r, str):
                r = ""
            reps.append({"correct": "yes" in r.lower(), "raw": r})
            idx += 1
        n_yes = sum(r["correct"] for r in reps)
        correct = n_yes > len(reps) / 2
        label = it.get("label", "")
        verdicts = ["Y" if r["correct"] else "N" for r in reps]
        print(f"  {label} -> {verdicts}" if n_reps > 1
              else f"  {label} -> {reps[0]['raw'][:40]}")
        out.append({"correct": correct, "raw": reps[0]["raw"], "repetitions": reps})
    return out


# ---------------------------------------------------------------------------
# Response cache
# ---------------------------------------------------------------------------

def _cache_path(results_dir, role, model_id, dataset, temp, max_tok,
                sample_idx=-1, temp_label=None, cache_dir=None):
    """Build the rollout-cache filename.

    ``temp`` is a float (used to derive a default label).  ``temp_label``,
    when given, overrides the derived label exactly (used so the
    hybrid-eval cache filename can point at legacy ``temp0`` base files
    while we run with ``--temperature 0.6`` for think rollouts).

    ``cache_dir``, when given, is used verbatim as the cache directory
    (no ``results_dir`` joining, no ``/response_cache`` suffix).  This
    lets callers point at an explicit cache root and avoids the symlink
    hack the launcher scripts used to use (which races when sibling
    jobs share ``results_dir`` and rewrite ``${results_dir}/response_cache``).
    When unset (default), the legacy ``${results_dir}/response_cache/``
    layout is used.
    """
    cdir = cache_dir if cache_dir else f"{results_dir}/response_cache"
    os.makedirs(cdir, exist_ok=True)
    if temp_label is None:
        temp_label = f"{temp:.2f}".rstrip("0").rstrip(".") or "0"
    s = f"_s{sample_idx}" if sample_idx >= 0 else ""
    return (f"{cdir}/{role}_{model_id}_{dataset}"
            f"_temp{temp_label}_max{max_tok}{s}.jsonl")


def _load_cache(path):
    cache = {}
    if not os.path.exists(path):
        return cache
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            e = json.loads(line)
            cache[e["dataset_idx"]] = {
                "response": e["response"],
                "n_tokens": e["n_tokens"], "eos": e["eos"]}
    return cache


def _append_cache(path, entries):
    if not entries:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a") as f:
        for e in entries:
            f.write(json.dumps(e, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------------------
# Rolling results
# ---------------------------------------------------------------------------

ROLLING_MAX = 90 * 1024 * 1024


def _rolling_prefix(args, base_id, think_id):
    os.makedirs(f"{args.results_dir}/rolling", exist_ok=True)
    s = _result_suffix(args)
    return f"{args.results_dir}/rolling/rolling_{base_id}_{args.dataset}{s}"


def _count_completed(args, base_id, think_id):
    prefix = _rolling_prefix(args, base_id, think_id)
    total = 0
    for path in _list_rolling(prefix):
        with open(path) as f:
            for _ in f:
                total += 1
    return total


def _list_rolling(prefix):
    d = os.path.dirname(prefix)
    b = os.path.basename(prefix)
    files = []
    legacy = os.path.join(d, b + ".jsonl")
    if os.path.exists(legacy):
        files.append(legacy)
    try:
        for fn in sorted(os.listdir(d)):
            if fn.startswith(b + "_") and fn.endswith(".jsonl"):
                files.append(os.path.join(d, fn))
    except FileNotFoundError:
        pass
    return files


def _load_prev_counts(args, base_id, think_id):
    """Load previous majority-vote and per-rep accuracy counts."""
    files = _list_rolling(_rolling_prefix(args, base_id, think_id))
    n = 0
    counts = {"thinking": 0, "base": 0, "hybrid": 0}
    per_rep = {"thinking": [], "base": [], "hybrid": []}
    for path in files:
        with open(path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                for k in counts:
                    j = rec.get("judges", {}).get(k, {})
                    if not isinstance(j, dict):
                        continue
                    if j.get("correct"):
                        counts[k] += 1
                    reps = j.get("repetitions") or []
                    if reps:
                        rep_correct = [bool(r.get("correct")) for r in reps]
                    else:
                        rep_correct = [bool(j.get("correct"))]
                    if not per_rep[k]:
                        per_rep[k] = [0] * len(rep_correct)
                    if len(per_rep[k]) != len(rep_correct):
                        target = max(len(per_rep[k]), len(rep_correct))
                        per_rep[k] = per_rep[k] + [0] * (
                            target - len(per_rep[k]))
                        rep_correct = rep_correct + [False] * (
                            target - len(rep_correct))
                    for i, c in enumerate(rep_correct):
                        if c:
                            per_rep[k][i] += 1
                n += 1
    return n, counts, per_rep


def append_rolling(record, args, base_id, think_id):
    prefix = _rolling_prefix(args, base_id, think_id)
    files = _list_rolling(prefix)
    data = json.dumps(record) + "\n"
    if files:
        target = files[-1]
        if os.path.getsize(target) + len(data.encode()) > ROLLING_MAX:
            idx = len([f for f in files if "_" in os.path.basename(f)])
            target = f"{prefix}_{idx}.jsonl"
    else:
        target = prefix + ".jsonl"
    with open(target, "a", encoding="utf-8") as f:
        f.write(data)


# ---------------------------------------------------------------------------
# Main evaluation
# ---------------------------------------------------------------------------

def main():
    args = parse_args()
    if args.steer_all_positions and args.steer_all_positions_full:
        raise SystemExit(
            "--steer_all_positions and --steer_all_positions_full are "
            "mutually exclusive (both re-interpret the sweep semantics).")
    os.makedirs(args.results_dir, exist_ok=True)
    ds_type = _dataset_type(args)

    # ---- Load dataset ----
    print(f"Loading {args.dataset}...")
    if args.dataset == "gsm8k":
        dataset = load_dataset("openai/gsm8k", "main")["test"]
    elif args.dataset == "math500":
        dataset = load_dataset("HuggingFaceH4/MATH-500")["test"]
    elif args.dataset == "aime24":
        dataset = load_dataset("HuggingFaceH4/aime_2024")["train"]
    elif args.dataset == "aime25":
        dataset = load_dataset("yentinglin/aime_2025")["train"]
    elif args.dataset == "mbpp":
        dataset = load_dataset("google-research-datasets/mbpp", "full")["test"]
    elif args.dataset == "livecodebench":
        dataset = load_dataset("bzantium/livecodebench", "release_v5")["test"]
    elif args.dataset == "medqa":
        dataset = load_dataset("GBaker/MedQA-USMLE-4-options")["test"].select(range(500))
    elif args.dataset == "gpqa":
        dataset = load_dataset("Idavidrein/gpqa", "gpqa_main", split="train")
    elif args.dataset == "natreason":
        if not args.natreason_file or not os.path.exists(args.natreason_file):
            raise ValueError("--natreason_file must point to the eval jsonl "
                             f"(got {args.natreason_file})")
        dataset = [json.loads(l) for l in open(args.natreason_file)
                   if l.strip()]
    elif args.dataset == "holdoutmix":
        if not args.holdoutmix_file or not os.path.exists(args.holdoutmix_file):
            raise ValueError("--holdoutmix_file must point to the eval jsonl "
                             f"(got {args.holdoutmix_file})")
        dataset = [json.loads(l) for l in open(args.holdoutmix_file)
                   if l.strip()]
    elif args.dataset == "hendrycks_holdout":
        if (not args.hendrycks_holdout_file
                or not os.path.exists(args.hendrycks_holdout_file)):
            raise ValueError("--hendrycks_holdout_file must point to the eval "
                             f"jsonl (got {args.hendrycks_holdout_file})")
        dataset = [json.loads(l) for l in open(args.hendrycks_holdout_file)
                   if l.strip()]
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # ---- Quick judge test ----
    try:
        r = _safe_chat_batch(["Reply YES."], args.judge_model, max_tokens=5)
        print(f"Judge API test: {'OK' if r and isinstance(r[0], str) else 'FAIL'}")
    except Exception as e:
        print(f"Judge API test: FAIL ({e})")

    # ---- Load models ----
    think_id = args.thinking_model.split("/")[-1].lower()
    base_id = args.base_model.split("/")[-1].lower()
    dom_model_short = args.dom_vectors_model_short or base_id

    _sdp_force = os.environ.get("SDP_FORCE")
    if _sdp_force:
        kernels = {
            "math":    (False, False, True,  False),
            "mem_eff": (False, True,  True,  False),
            "flash":   (True,  False, True,  False),
        }
        if _sdp_force not in kernels:
            raise ValueError(f"SDP_FORCE must be one of {list(kernels)}; got {_sdp_force}")
        f, m, mt, c = kernels[_sdp_force]
        torch.backends.cuda.enable_flash_sdp(f)
        torch.backends.cuda.enable_mem_efficient_sdp(m)
        torch.backends.cuda.enable_math_sdp(mt)
        try:
            torch.backends.cuda.enable_cudnn_sdp(c)
        except AttributeError:
            pass
        print(f"[sdp] SDP_FORCE={_sdp_force} -> flash={torch.backends.cuda.flash_sdp_enabled()} "
              f"mem_eff={torch.backends.cuda.mem_efficient_sdp_enabled()} "
              f"math={torch.backends.cuda.math_sdp_enabled()}")

    first_max_mem = None
    if args.max_memory_per_gpu:
        n_gpus = torch.cuda.device_count()
        first_max_mem = {i: args.max_memory_per_gpu for i in range(n_gpus)}
        print(f"  [multi-gpu] {n_gpus} GPUs, first model max_memory="
              f"{args.max_memory_per_gpu}/GPU")

    if args.two_gpu_split:
        n_gpus_avail = torch.cuda.device_count()
        if n_gpus_avail < 2:
            raise RuntimeError(f"--two_gpu_split requires >=2 visible GPUs, "
                               f"got {n_gpus_avail}.")
        # Thinking model on cuda:1, base on cuda:0.  Each model pinned to
        # a single device (no sharding).
        think_dm = {"": 1}
        base_dm  = {"": 0}
        print(f"  [two-gpu] base=cuda:0  thinking=cuda:1  ({n_gpus_avail} GPUs visible)")
    else:
        think_dm = "auto"
        base_dm  = "auto"

    print(f"\nLoading thinking model {args.thinking_model}...")
    think_tok = AutoTokenizer.from_pretrained(args.thinking_model)
    if think_tok.pad_token is None:
        think_tok.pad_token = think_tok.eos_token
    think_model = AutoModelForCausalLM.from_pretrained(
        args.thinking_model, torch_dtype=torch.bfloat16, device_map=think_dm,
        max_memory=(None if args.two_gpu_split else first_max_mem),
        attn_implementation=os.environ.get("ATTN_IMPL", "sdpa"))
    think_model.eval()

    print(f"Loading base model {args.base_model}...")
    base_tok = AutoTokenizer.from_pretrained(args.base_model)
    if base_tok.pad_token is None:
        base_tok.pad_token = base_tok.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model, torch_dtype=torch.bfloat16, device_map=base_dm,
        attn_implementation=os.environ.get("ATTN_IMPL", "sdpa"))
    base_model.eval()

    # ---- Tokenizer alignment check ----
    # Several --coef_select modes (pg, kl_top*, think_top1) index base-model
    # logits with thinking-model token ids and vice versa.  This only works
    # if the two tokenizers share the same vocab 1:1.  Verify on a probe set
    # (cheaper than a full vocab scan, catches realistic divergence).
    assert think_model.config.vocab_size == base_model.config.vocab_size, (
        f"vocab_size mismatch: think={think_model.config.vocab_size} "
        f"base={base_model.config.vocab_size} -- coef_select needs aligned "
        f"vocabularies.")
    _probes = [
        " the quick brown fox jumps over the lazy dog.",
        "\n\nLet me think step by step.\n\n",
        "Wait, lets backtrack.",
        "Therefore, the answer is \\boxed{42}.",
        "\\frac{1}{2}",
    ]
    for _s in _probes:
        _b = base_tok(_s, add_special_tokens=False)["input_ids"]
        _t = think_tok(_s, add_special_tokens=False)["input_ids"]
        assert _b == _t, (
            f"Tokenizer mismatch on probe {_s!r}: base={_b} think={_t}. "
            f"coef_select modes that mix indices across models would be "
            f"incorrect; refusing to run.")
    print("Tokenizer alignment: OK (base/thinking vocabs match 1:1 on probes).")

    # Up-cast the final lm_head weight to fp32 on both models.  Hidden
    # states stay bf16; we cast them to fp32 just before the matmul via a
    # wrapper.  Kills most bf16 argmax flips on near-tied logits and is
    # essentially free at runtime.
    def _wrap_lm_head(model):
        lm = model.lm_head
        # For models with tied word embeddings (e.g. Qwen2.5-0.5B,
        # Qwen2.5-1.5B, small Llamas), lm.weight is the SAME tensor as
        # model.model.embed_tokens.weight.  Casting lm to fp32 in-place
        # would also cast the embedding weight, causing the transformer
        # residual stream to go fp32 while decoder-layer weights remain
        # bf16 -> mat1/mat2 dtype mismatch.  Break the tie first by
        # cloning the weight, then cast the copy to fp32.
        try:
            tied = getattr(getattr(model, "config", None),
                           "tie_word_embeddings", False)
        except Exception:
            tied = False
        if tied:
            new_w = torch.nn.Parameter(
                lm.weight.detach().clone().to(torch.float32))
            lm.weight = new_w
        else:
            lm.to(torch.float32)
        orig_forward = lm.forward
        def _fwd(x):
            return orig_forward(x.to(torch.float32))
        lm.forward = _fwd
    _wrap_lm_head(think_model)
    _wrap_lm_head(base_model)

    # ---- Load SAE ----
    print(f"Loading SAE (layer={args.sae_layer}, clusters={args.n_clusters})...")
    sae, _ = load_sae(think_id, args.sae_layer, args.n_clusters,
                      require_activation_mean=False)
    ckpt_path = os.path.join(
        os.path.dirname(__file__),
        f"../train-saes/results/vars/saes/sae_{think_id}_layer{args.sae_layer}"
        f"_clusters{args.n_clusters}.pt")
    if os.path.exists(ckpt_path):
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        if "activation_mean" in ckpt:
            sae.activation_mean = ckpt["activation_mean"]
            print("  Loaded activation_mean from checkpoint")
        del ckpt
    if not hasattr(sae, "activation_mean") and not args.disable_sae_mean:
        args.disable_sae_mean = True
        print("  Auto-set disable_sae_mean (no activation_mean)")
    sae = sae.to(next(think_model.parameters()).device)
    descriptions = get_latent_descriptions(think_id, args.sae_layer, args.n_clusters)

    # ---- Load steering vectors ----
    base_dev = next(base_model.parameters()).device
    base_dt = next(base_model.parameters()).dtype

    if args.old_vectors_dir:
        print(f"Loading OLD optimized vectors from {args.old_vectors_dir}...")
        old_layer = args.old_vectors_layer
        steering_vectors, layer_map = {}, {}
        # Optional per-category layer map: written by the layer-sweep
        # trainer in train-vectors/optimize_correction_vectors.py.  If
        # present, it overrides --old_vectors_layer on a per-key basis.
        per_key_layers = {}
        # Prefer model-specific file to avoid conflicts when models share a save dir
        lm_path_specific = os.path.join(args.old_vectors_dir, f"layer_map_{dom_model_short}.json")
        lm_path_generic  = os.path.join(args.old_vectors_dir, "layer_map.json")
        lm_path = lm_path_specific if os.path.exists(lm_path_specific) else lm_path_generic
        if os.path.exists(lm_path):
            with open(lm_path) as f:
                per_key_layers = {k: int(v) for k, v in json.load(f).items()}
            print(f"  Using per-category layer_map.json: {per_key_layers}")
        global_fpath = os.path.join(args.old_vectors_dir,
                                    f"{dom_model_short}_global_linear.pt")
        if os.path.exists(global_fpath):
            ckpt = torch.load(global_fpath, map_location="cpu",
                              weights_only=False)
            vec = ckpt.get("global", next(iter(ckpt.values())))
            steering_vectors["global"] = vec.to(torch.float32)
            layer_map["global"] = per_key_layers.get("global", old_layer)
            print(f"  global: layer={layer_map['global']}, "
                  f"norm={vec.norm().item():.2f} (single-vector mode)")
        else:
            for cat_id in range(args.n_clusters):
                key = f"idx{cat_id}"
                fpath = os.path.join(args.old_vectors_dir,
                                     f"{dom_model_short}_idx{cat_id}_linear.pt")
                if not os.path.exists(fpath):
                    print(f"  WARNING: {fpath} not found, skipping {key}")
                    continue
                ckpt = torch.load(fpath, map_location="cpu",
                                  weights_only=False)
                vec = ckpt[key]
                steering_vectors[key] = vec.to(torch.float32)
                layer_map[key] = per_key_layers.get(key, old_layer)
                print(f"  {key}: layer={layer_map[key]}, "
                      f"norm={vec.norm().item():.2f}")
    else:
        print(f"Loading DOM vectors from {args.dom_vectors_dir}...")
        steering_vectors, layer_map = load_dom_vectors(
            args.dom_vectors_dir, dom_model_short, descriptions)

    if "global" in steering_vectors and not any(
            k.startswith("idx") for k in steering_vectors):
        gv = steering_vectors.pop("global")
        gl = layer_map.pop("global")
        for cat_id in range(args.n_clusters):
            k = f"idx{cat_id}"
            steering_vectors[k] = gv.clone()
            layer_map[k] = gl
        print(f"  [single-vector] Replicated 'global' vector to "
              f"{args.n_clusters} category keys")

    for k, v in steering_vectors.items():
        steering_vectors[k] = v.to(device=base_dev, dtype=base_dt)

    # ---- Optional: replace each category vector with a random direction of
    # the same norm.  Runs BEFORE bias is folded in, so the bias (if any) is
    # left untouched -- this isolates the contribution of the *category-
    # specific direction* vs norm + bias.
    assert not (args.randomize_vectors and args.randomize_vectors_unit_norm), (
        "--randomize_vectors and --randomize_vectors_unit_norm are mutually "
        "exclusive")
    if args.randomize_vectors:
        print(f"Randomizing steering vectors (seed={args.random_seed}, "
              f"norm-preserving)...")
        g = torch.Generator(device="cpu").manual_seed(int(args.random_seed))
        for k in list(steering_vectors.keys()):
            v = steering_vectors[k]
            orig_norm = v.float().norm().item()
            rnd = torch.randn(v.shape, generator=g, dtype=torch.float32)
            rnd = rnd * (orig_norm / (rnd.norm().item() + 1e-12))
            steering_vectors[k] = rnd.to(device=v.device, dtype=v.dtype)
            print(f"  {k}: orig_norm={orig_norm:.2f}, "
                  f"rand_norm={steering_vectors[k].float().norm().item():.2f}")
    if args.randomize_vectors_unit_norm:
        print(f"Randomizing steering vectors (seed={args.random_seed}, "
              f"UNIT-NORM -- reproducing collaborator bug)...")
        g = torch.Generator(device="cpu").manual_seed(int(args.random_seed))
        for k in list(steering_vectors.keys()):
            v = steering_vectors[k]
            orig_norm = v.float().norm().item()
            rnd = torch.randn(v.shape, generator=g, dtype=torch.float32)
            rnd = rnd / (rnd.norm().item() + 1e-12)
            steering_vectors[k] = rnd.to(device=v.device, dtype=v.dtype)
            print(f"  {k}: orig_norm={orig_norm:.2f}, "
                  f"rand_norm={steering_vectors[k].float().norm().item():.4f}")

    # ---- Optional: zero out per-category vectors (bias-only ablation) ----
    if args.bias_only:
        assert args.bias_vector_path, "--bias_only requires --bias_vector_path"
        print("Bias-only ablation: zeroing all per-category steering vectors "
              "(only the global bias will remain).")
        for k in list(steering_vectors.keys()):
            steering_vectors[k] = torch.zeros_like(steering_vectors[k])

    # ---- Optional: add a global bias vector on top of each category vector ----
    # The OLD (paper) pipeline applies  coef * (steer_vec + bias_vec)  at the
    # steering layer for each (coef, window) candidate.  Since we always apply
    # the category vector and the bias at the same positions with the same
    # coefficient, folding bias into each category vector at load time is
    # mathematically equivalent and keeps the hot loop untouched.
    _always_on_bias_vec: "Optional[torch.Tensor]" = None
    _always_on_bias_layer: "Optional[int]" = None

    if args.bias_vector_path:
        print(f"Loading bias vector from {args.bias_vector_path}...")
        bias_obj = torch.load(args.bias_vector_path, map_location="cpu",
                              weights_only=False)
        if isinstance(bias_obj, dict):
            bias_vec = bias_obj.get("bias", next(iter(bias_obj.values())))
        else:
            bias_vec = bias_obj
        bias_vec = bias_vec.to(device=base_dev, dtype=base_dt)
        print(f"  bias: shape={tuple(bias_vec.shape)}, "
              f"norm={bias_vec.float().norm().item():.2f}")
        hidden = getattr(base_model.config, "hidden_size", 0)
        assert bias_vec.shape[-1] == hidden, (
            f"bias_vec shape {tuple(bias_vec.shape)} incompatible with base model")

        # Load per-category bias scales if present (written by
        # --per_cat_bias_scale during training).  For each key k the
        # effective fold is  alpha[k] * bias  instead of bias, so the cat
        # vector direction is kept clean and only the magnitude is adjusted.
        bias_alpha: dict = {}
        alpha_path = os.path.join(args.dom_vectors_dir, "bias_alpha.json")
        if os.path.exists(alpha_path):
            with open(alpha_path) as _f:
                bias_alpha = json.load(_f)
            _vals = list(bias_alpha.values())
            print(f"  bias_alpha.json loaded: {len(bias_alpha)} keys  "
                  f"min={min(_vals):.3f}  max={max(_vals):.3f}  "
                  f"mean={sum(_vals)/len(_vals):.3f}")

        if getattr(args, "bias_always_on", False):
            # Always-on mode: bias is applied via a separate hook at every
            # position. Do NOT fold it into cat vectors — they stay pure.
            # Resolve the bias layer: bias_layer.json sibling > --bias_layer > steer layer.
            _aon_layer = args.bias_layer
            if _aon_layer is None:
                _sib = os.path.join(os.path.dirname(args.bias_vector_path),
                                    "bias_layer.json")
                if os.path.exists(_sib):
                    with open(_sib) as _f:
                        _aon_layer = int(json.load(_f)["layer"])
            if _aon_layer is None:
                _aon_layer = args.old_vectors_layer
            _always_on_bias_vec = bias_vec
            _always_on_bias_layer = _aon_layer
            print(f"  [bias_always_on] will apply bias at layer {_aon_layer} "
                  "at every position; NOT folded into cat vectors.")
        elif getattr(args, "pg_bias_cat_sweep", False):
            # Cartesian-PG mode: bias and cat are applied via the SAME
            # steering hook but with INDEPENDENT coefficients. We pass
            # the raw (unscaled) bias_vec to the decoder and do NOT fold
            # it into the cat vectors. bias_alpha (per-cat bias scaling)
            # is intentionally ignored in this mode — the whole point is
            # to let the PG choose the bias magnitude freely.
            _always_on_bias_vec = None
            _always_on_bias_layer = None
            print(f"  [pg_bias_cat_sweep] bias is NOT folded into cat "
                  f"vectors; (b,c) ∈ pg_bias_coefs × pg_cat_coefs is "
                  f"swept per disagreement step.")
        else:
            _always_on_bias_vec = None
            _always_on_bias_layer = None
            for k in list(steering_vectors.keys()):
                scale = float(bias_alpha[k]) if k in bias_alpha else 1.0
                steering_vectors[k] = steering_vectors[k] + scale * bias_vec

        # When running the BIAS-ONLY ablation, we want every key to steer at
        # the bias's own chosen layer, not whatever layer each category was
        # individually selected for.  Resolve bias_layer priority:
        #   1. explicit --bias_layer CLI
        #   2. bias_layer.json sibling of --bias_vector_path (written by
        #      optimize_correction_vectors.py when --train_global_bias)
        #   3. --old_vectors_layer (legacy fallback)
        if args.bias_only:
            bias_layer = args.bias_layer
            if bias_layer is None:
                sib = os.path.join(os.path.dirname(args.bias_vector_path),
                                   "bias_layer.json")
                if os.path.exists(sib):
                    with open(sib) as f:
                        bias_layer = int(json.load(f)["layer"])
                    print(f"  Using bias_layer={bias_layer} from {sib}")
            if bias_layer is None:
                bias_layer = args.old_vectors_layer
                print(f"  Falling back to --old_vectors_layer={bias_layer} "
                      f"for bias-only steering layer")
            for k in list(layer_map.keys()):
                layer_map[k] = bias_layer
            print(f"  [bias_only] layer_map overridden to "
                  f"{bias_layer} for all keys")

    # ---- Load CatCoefMLP (optional, for --coef_select=mlp) ----
    mlp_model = None
    if args.mlp_coef_path and args.mlp_config_path:
        import sys as _sys
        _sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "train-vectors"))
        from coef_mlp import CatCoefMLP
        with open(args.mlp_config_path) as f:
            mlp_cfg = json.load(f)
        mlp_model = CatCoefMLP(
            d_in=mlp_cfg["d_in"],
            n_cats=mlp_cfg["n_cats"],
            d_hidden=mlp_cfg["d_hidden"],
            per_cat=bool(mlp_cfg.get("per_cat", False)))
        mlp_model.load_state_dict(
            torch.load(args.mlp_coef_path, map_location="cpu", weights_only=True))
        mlp_model.eval()
        for _p in mlp_model.parameters():
            _p.requires_grad = False
        print(f"Loaded CatCoefMLP from {args.mlp_coef_path} "
              f"(d_in={mlp_cfg['d_in']}, n_cats={mlp_cfg['n_cats']}, "
              f"d_hidden={mlp_cfg['d_hidden']}, "
              f"per_cat={bool(mlp_cfg.get('per_cat', False))})")

    # ---- One-time debug metadata dump (config + loaded vectors) ----
    # This lets us reconstruct AFTER THE FACT exactly what was steered,
    # from which files, at which layers, with which norms.  Cheap (few
    # KB) and written once per run, right before task loop starts.
    try:
        dbg_meta = {
            "base_model": args.base_model,
            "thinking_model": args.thinking_model,
            "base_id": base_id,
            "thinking_id": think_id,
            "dataset": args.dataset,
            "sae_layer": args.sae_layer,
            "n_clusters": args.n_clusters,
            "sae_path": getattr(args, "sae_path", None),
            "coef_sweep": [float(x) for x in args.coef_sweep.split(",")],
            "coef_select": args.coef_select,
            "kl_topk": int(args.kl_topk),
            "bias_only": bool(args.bias_only),
            "bias_vector_path": args.bias_vector_path,
            "bias_layer": (
                int(locals()["bias_layer"])
                if args.bias_only and "bias_layer" in locals()
                and locals()["bias_layer"] is not None else None),
            "random_firing": bool(args.random_firing),
            "random_guardrail": bool(args.random_guardrail),
            "random_seed": int(args.random_seed),
            "steer_all_positions": bool(args.steer_all_positions),
            "steer_all_positions_full": bool(args.steer_all_positions_full),
            "disable_sae_mean": bool(args.disable_sae_mean),
            "old_vectors_dir": getattr(args, "old_vectors_dir", None),
            "old_vectors_layer": getattr(args, "old_vectors_layer", None),
            "n_latent_descriptions": len(descriptions),
            "layer_map": {k: int(v) for k, v in layer_map.items()},
            "steering_vectors": {
                k: {"shape": list(v.shape),
                    "norm": float(v.float().norm().item()),
                    "layer": int(layer_map.get(k, -1))}
                for k, v in steering_vectors.items()},
            "suffix": _result_suffix(args),
        }
        dbg_dir = args.results_dir
        os.makedirs(dbg_dir, exist_ok=True)
        dbg_path = os.path.join(
            dbg_dir,
            f"debug_meta_{base_id}_{think_id}_{args.dataset}"
            f"{_result_suffix(args)}.json")
        with open(dbg_path, "w") as f:
            json.dump(dbg_meta, f, indent=2)
        print(f"[debug] wrote run metadata -> {dbg_path}")
    except Exception as e:
        print(f"[debug] WARNING: could not write debug_meta: {e}")

    # ---- Prepare tasks ----
    eval_indices = []
    if getattr(args, "eval_indices", ""):
        raw = args.eval_indices.replace(",", " ").split()
        eval_indices = [int(x) for x in raw]
        if not eval_indices:
            raise ValueError("--eval_indices was provided but parsed empty")
        args.n_tasks = len(eval_indices)
        print(f"[eval_indices] evaluating explicit dataset indices: "
              f"{eval_indices}")

    if args.n_tasks <= 0:
        args.n_tasks = len(dataset) - args.eval_start_idx

    completed = _count_completed(args, base_id, think_id)
    if completed > 0 and not getattr(args, "continuation_mode", False):
        if completed >= args.n_tasks:
            print(f"Already completed {completed} tasks. Nothing to do.")
            return
        if eval_indices:
            print(f"Resuming explicit-index run: {completed} done, "
                  f"{args.n_tasks - completed} remaining")
            eval_indices = eval_indices[completed:]
            args.n_tasks = len(eval_indices)
        else:
            print(f"Resuming: {completed} done, {args.n_tasks - completed} remaining "
                  f"(eval_start_idx {args.eval_start_idx} -> {args.eval_start_idx + completed})")
            # PRESERVE the original eval_start_idx (important for sharded runs
            # where the shard starts at a non-zero offset into the dataset).
            args.eval_start_idx += completed
            args.n_tasks -= completed
    elif completed > 0 and getattr(args, "continuation_mode", False):
        # In continuation mode the resume offset shortcut doesn't apply
        # (tasks are filtered post-hoc by question text).  Skip resume.
        # User should delete the ext rolling file if they want a fresh run.
        print(f"[continuation] WARN: {completed} ext rows already present; "
              f"continuation_mode will re-run all extended rows.")

    tasks = []
    if eval_indices:
        for i in eval_indices:
            item = dataset[i]
            t = _build_task_prompts(item, i, args)
            t["dataset_idx"] = i
            tasks.append(t)
    else:
        for i, item in enumerate(dataset):
            if i < args.eval_start_idx:
                continue
            if len(tasks) >= args.n_tasks:
                break
            t = _build_task_prompts(item, i, args)
            t["dataset_idx"] = i
            tasks.append(t)
    n_tasks = len(tasks)
    print(f"\n=== {n_tasks} tasks ===")

    # ---- Continuation-mode filter: load source rolling rows, restrict
    # tasks to eos.hybrid==False rows, attach continuation text.
    if getattr(args, "continuation_mode", False):
        src_paths_raw = (args.continuation_source_rolling or "").strip()
        assert src_paths_raw, (
            "--continuation_mode requires --continuation_source_rolling")
        import glob as _glob
        src_paths: List[str] = []
        for chunk in src_paths_raw.split(","):
            chunk = chunk.strip()
            if not chunk:
                continue
            src_paths.extend(sorted(_glob.glob(chunk)) or [chunk])
        print(f"[continuation] source rolling files:")
        for p in src_paths:
            print(f"  - {p}  ({'exists' if os.path.exists(p) else 'MISSING'})")
        # Build {question -> source row}. dataset_idx isn't stored in the
        # rolling file (only question text), so we match by question.
        src_rows: Dict[str, dict] = {}
        for p in src_paths:
            if not os.path.exists(p):
                continue
            with open(p) as fh:
                for line in fh:
                    try:
                        r = json.loads(line)
                    except Exception:
                        continue
                    q = r.get("question")
                    if q is not None:
                        src_rows[q] = r
        print(f"[continuation] loaded {len(src_rows)} source rows across "
              f"{len(src_paths)} file(s)")
        # Filter tasks: only keep rows where eos.hybrid is False.
        kept = []
        n_no_match = 0
        n_already_eos = 0
        for t in tasks:
            r = src_rows.get(t["question"])
            if r is None:
                n_no_match += 1
                continue
            eos_map = r.get("eos") or {}
            if eos_map.get("hybrid"):
                n_already_eos += 1
                continue
            ans = r.get("answers") or {}
            t["thinking_continuation_text"] = ans.get("thinking", "") or ""
            t["base_continuation_text"] = ans.get("hybrid", "") or ""
            # Also pull original truncated lengths for diagnostics.
            ntok = r.get("n_tokens") or {}
            t["_cont_orig_think_toks"] = int(ntok.get("thinking", 0))
            t["_cont_orig_hybrid_toks"] = int(ntok.get("hybrid", 0))
            kept.append(t)
        tasks = kept
        n_tasks = len(tasks)
        print(f"[continuation] dropped {n_already_eos} already-EOS rows, "
              f"{n_no_match} no-match rows; kept {n_tasks} to extend")
        if n_tasks == 0:
            print(f"[continuation] no rows need extension - exiting cleanly")
            return

    # ---- Phase 2: Standalone responses ----
    use_cache = not args.no_response_cache
    cache_dir_override = getattr(args, "response_cache_dir", None) or None
    tc_path = _cache_path(
        args.results_dir, "thinking", think_id, args.dataset,
        args.temperature,
        args.think_cache_max_tokens if args.think_cache_max_tokens is not None
        else args.max_thinking_tokens,
        sample_idx=args.think_cache_sample_idx,
        temp_label=args.think_cache_temp_label,
        cache_dir=cache_dir_override)
    bc_path = _cache_path(
        args.results_dir, "base", base_id, args.dataset,
        args.temperature,
        args.base_cache_max_tokens if args.base_cache_max_tokens is not None
        else args.max_new_tokens,
        sample_idx=args.base_cache_sample_idx,
        temp_label=args.base_cache_temp_label,
        cache_dir=cache_dir_override)
    tc = _load_cache(tc_path) if use_cache else {}
    bc = _load_cache(bc_path) if use_cache else {}
    print(f"  Cache: thinking={len(tc)}, base={len(bc)}")

    uncached_t = [(i, t) for i, t in enumerate(tasks) if t["dataset_idx"] not in tc]
    print(f"\n=== Thinking: {n_tasks - len(uncached_t)} cached, {len(uncached_t)} to generate ===")
    if uncached_t:
        def _flush_t(bstart, brecs):
            if not use_cache:
                return
            entries = []
            for k, r in enumerate(brecs):
                oi, t = uncached_t[bstart + k]
                entries.append(dict(dataset_idx=t["dataset_idx"], **r))
            _append_cache(tc_path, entries)
        res = _batch_generate(think_model, think_tok,
                              [t["thinking_prompt"] for _, t in uncached_t],
                              args.max_thinking_tokens, args.batch_gen_size,
                              use_chat_template=True, temperature=args.temperature,
                              on_batch_done=_flush_t, tag="thinking")
        for (oi, t), r in zip(uncached_t, res):
            t.update(thinking_response=r["response"],
                     thinking_n_tokens=r["n_tokens"], thinking_eos=r["eos"])
        del res; torch.cuda.empty_cache()

    for t in tasks:
        if "thinking_response" not in t:
            c = tc[t["dataset_idx"]]
            t.update(thinking_response=c["response"],
                     thinking_n_tokens=c["n_tokens"], thinking_eos=c["eos"])

    # ---- Optional: cold-start prefix from the thinking model ----
    # Inject the first N tokens (re-encoded with the THINKING tokenizer for
    # exact alignment with the cached rollout) into both standalone base
    # generation and hybrid prefill, so all three rollouts share the same
    # opener and any "first-token jump-to-answer" failure mode is removed.
    cold_n = int(getattr(args, "cold_start_n_tokens", 0) or 0)
    cold_start_active = cold_n > 0
    if cold_start_active:
        print(f"\n=== Cold-start: injecting first {cold_n} think-tokens "
              f"of the thinking rollout into base + hybrid ===")
        for t in tasks:
            tr = t.get("thinking_response", "") or ""
            if not tr:
                t["cold_start_text"] = ""
                continue
            # Re-encode the cached rollout text in the THINKING tokenizer,
            # take the first cold_n tokens, decode back to text.  Using
            # the think-tok side keeps the count exact w.r.t. the model
            # that actually generated those tokens; the base side just
            # re-tokenizes that text (count may differ by ±1).
            ids = think_tok(tr, add_special_tokens=False)["input_ids"][:cold_n]
            cs_text = think_tok.decode(ids, skip_special_tokens=True)
            t["cold_start_text"] = cs_text
            # Hybrid: feed the same prefix into both prefills.
            t["thinking_continuation_text"] = cs_text
            t["base_continuation_text"]     = cs_text
        # Disable base response cache (the base prompt now carries a
        # per-task prefix so existing cache entries are invalid).
        bc = {}
        for ti in range(min(3, len(tasks))):
            print(f"  [{ti}] cold_start={tasks[ti]['cold_start_text']!r}")

    uncached_b = [(i, t) for i, t in enumerate(tasks) if t["dataset_idx"] not in bc]
    print(f"\n=== Base: {n_tasks - len(uncached_b)} cached, {len(uncached_b)} to generate ===")
    if uncached_b:
        def _flush_b(bstart, brecs):
            if not use_cache:
                return
            entries = []
            for k, r in enumerate(brecs):
                oi, t = uncached_b[bstart + k]
                entries.append(dict(dataset_idx=t["dataset_idx"], **r))
            _append_cache(bc_path, entries)
        # Cold-start: append the shared prefix to the raw base prompt so
        # base standalone generation starts from the same opener as the
        # hybrid's base side.
        base_prompts_for_gen = [
            (t["base_prompt"] + (t.get("cold_start_text", "") or ""))
            for _, t in uncached_b]
        res = _batch_generate(base_model, base_tok,
                              base_prompts_for_gen,
                              args.max_new_tokens, args.batch_gen_size,
                              temperature=args.temperature,
                              on_batch_done=_flush_b, tag="base")
        for (oi, t), r in zip(uncached_b, res):
            # Glue the prefix back so the saved response is judgeable
            # end-to-end (base completion = prefix + continuation).
            cs = t.get("cold_start_text", "") or ""
            full = cs + r["response"]
            t.update(base_response=full,
                     base_n_tokens=r["n_tokens"] + cold_n if cs else r["n_tokens"],
                     base_eos=r["eos"])
        del res; torch.cuda.empty_cache()

    for t in tasks:
        if "base_response" not in t:
            c = bc[t["dataset_idx"]]
            t.update(base_response=c["response"],
                     base_n_tokens=c["n_tokens"], base_eos=c["eos"])

    # ---- Optional fast-path: skip hybrid, judge base+think only ----
    if bool(getattr(args, "skip_hybrid", False)):
        print(f"\n=== --skip_hybrid: judging base+think only on "
              f"{n_tasks} tasks ===")
        # Build judge items for all tasks at once.
        judge_items = []
        for ti, t in enumerate(tasks):
            q, gold, tl = t["question"], t["correct_answer"], t["test_list"]
            common = dict(gold=gold, question=q, ds_type=ds_type, test_list=tl)
            judge_items.append(dict(
                answer=re.sub(r'\s+', ' ', t["thinking_response"]).strip(),
                label=f"T{ti+1} Think", **common))
            judge_items.append(dict(
                answer=re.sub(r'\s+', ' ', t["base_response"]).strip(),
                label=f"T{ti+1} Base", **common))
        jr = judge_batch(judge_items, args.judge_model,
                         n_reps=args.judge_repetitions,
                         max_concurrent=args.max_concurrent)
        n_think_correct = 0
        n_base_correct = 0
        for ti, t in enumerate(tasks):
            te, be = jr[2 * ti], jr[2 * ti + 1]
            if te["correct"]:
                n_think_correct += 1
            if be["correct"]:
                n_base_correct += 1
            append_rolling({
                "ts": time.time(), "dataset": args.dataset,
                "question": t["question"], "gold_answer": t["correct_answer"],
                "answers": {"thinking": t["thinking_response"],
                            "base": t["base_response"]},
                "judges": {"thinking": te, "base": be},
                "eos": {"thinking": t["thinking_eos"],
                        "base": t["base_eos"]},
                "n_tokens": {"thinking": t["thinking_n_tokens"],
                             "base": t["base_n_tokens"]},
            }, args, base_id, think_id)
        tp = n_think_correct / n_tasks * 100
        bp = n_base_correct / n_tasks * 100
        print()
        print(f"===== Final (base+think only) =====")
        print(f"Thinking: {n_think_correct}/{n_tasks} ({tp:.1f}%)")
        print(f"Base:     {n_base_correct}/{n_tasks} ({bp:.1f}%)")
        print(f"Gap (Think - Base): {tp - bp:+.1f} pts")
        print("Done.")
        return

    # ---- Phase 2.5: Optional coef calibration sweep ----
    _calibrated_cat_coef = None
    if getattr(args, "calibrate_coef", False):
        # Identify tasks where think=correct AND base=wrong using the judge.
        # We already have think_response and base_response; judge them now.
        cal_pct = float(getattr(args, "calibrate_pct", 0.10))
        cal_grid = [float(x) for x in args.calibrate_coef_grid.split(",")]
        print(f"\n=== Calibration: judging think/base on all {n_tasks} tasks "
              f"to find disagreement subset ===")

        # Judge think and base in one batch
        cal_judge_items = []
        for ti, t in enumerate(tasks):
            q, gold, tl = t["question"], t["correct_answer"], t["test_list"]
            common = dict(gold=gold, question=q, ds_type=ds_type, test_list=tl)
            cal_judge_items.append(dict(
                answer=re.sub(r'\s+', ' ', t["thinking_response"]).strip(),
                label=f"CAL_T{ti}", **common))
            cal_judge_items.append(dict(
                answer=re.sub(r'\s+', ' ', t["base_response"]).strip(),
                label=f"CAL_B{ti}", **common))

        cal_jr = judge_batch(cal_judge_items, args.judge_model,
                             n_reps=1, max_concurrent=args.max_concurrent)

        # Pair up: think_correct[i], base_correct[i]
        think_correct = [cal_jr[2*i]["correct"] for i in range(n_tasks)]
        base_correct  = [cal_jr[2*i+1]["correct"] for i in range(n_tasks)]

        # Select: think=YES, base=NO
        disagree_idx = [i for i in range(n_tasks)
                        if think_correct[i] and not base_correct[i]]
        # Take up to cal_pct of TOTAL benchmark size, capped by available
        n_cal = min(max(1, int(round(n_tasks * cal_pct))),
                    len(disagree_idx))
        cal_idx = disagree_idx[:n_cal]

        print(f"\n=== Calibration: {len(disagree_idx)} think-correct/base-wrong "
              f"tasks, using {len(cal_idx)} (of {n_tasks} total × "
              f"{cal_pct:.0%} = {int(round(n_tasks * cal_pct))}) for sweep ===")
        print(f"  Grid: {cal_grid}")

        if len(cal_idx) >= 2:
            cal_tasks = [tasks[i] for i in cal_idx]
            _cal_hbs = min(len(cal_tasks), args.hybrid_gen_batch_size)
            best_coef, best_gap, best_correct = cal_grid[0], -1.0, 0

            for _cc in cal_grid:
                print(f"\n  -- Calibrating cat_coef={_cc} --")
                _cal_bc = 1.0  # bias coef always 1.0

                # Run hybrid on calibration tasks with this coef
                all_hybrid = []
                for cb_start in range(0, len(cal_tasks), _cal_hbs):
                    cb = cal_tasks[cb_start:cb_start + _cal_hbs]
                    torch.cuda.empty_cache()
                    hr = hybrid_generate_batched(
                        think_model, base_model, base_tok,
                        [t["thinking_prompt"] for t in cb],
                        [t["base_prompt"] for t in cb],
                        args.max_new_tokens, args.sae_layer, sae,
                        steering_vectors, descriptions, layer_map,
                        thinking_tokenizer=think_tok,
                        disable_sae_mean=args.disable_sae_mean,
                        show_progress=False, collect_details=False,
                        random_firing=args.random_firing,
                        random_firing_exclude_top_k_keys=args.random_firing_exclude_top_k_keys,
                        firing_replace_with_min_cosine=args.firing_replace_with_min_cosine,
                        pure_steer_base_eos=args.pure_steer_base_eos,
                        random_steer_prob=args.random_steer_prob,
                        random_guardrail=False,
                        random_seed=args.random_seed,
                        coef_sweep=[1.0],
                        coef_select="think_top1",
                        pg_bias_cat_sweep=True,
                        pg_bias_vec=(locals().get("bias_vec")
                                     if getattr(args, "pg_bias_cat_sweep", False)
                                     else None),
                        pg_bias_coefs=(_cal_bc,),
                        pg_cat_coefs=(_cc,),
                        token_window=int(getattr(args, "token_window", 0)),
                        act_modulate=None)
                    all_hybrid.extend(hr)

                # Judge hybrid responses
                cal_h_items = []
                for j, (t, h) in enumerate(zip(cal_tasks, all_hybrid)):
                    resp = base_tok.decode(h["generated_ids"],
                                           skip_special_tokens=True)
                    q, gold, tl = t["question"], t["correct_answer"], t["test_list"]
                    cal_h_items.append(dict(
                        answer=re.sub(r'\s+', ' ', resp).strip(),
                        label=f"CAL_H_c{_cc}_{j}",
                        gold=gold, question=q, ds_type=ds_type, test_list=tl))

                h_jr = judge_batch(cal_h_items, args.judge_model,
                                   n_reps=1, max_concurrent=args.max_concurrent)
                n_correct = sum(1 for r in h_jr if r["correct"])
                gap_rec = n_correct / len(cal_tasks)  # fraction of fixable tasks fixed
                print(f"  cat_coef={_cc}: {n_correct}/{len(cal_tasks)} correct "
                      f"(gap_rec={gap_rec:.1%})")

                if n_correct > best_correct:
                    best_correct = n_correct
                    best_gap = gap_rec
                    best_coef = _cc

            _calibrated_cat_coef = best_coef
            print(f"\n=== Calibration result: best cat_coef={best_coef} "
                  f"({best_correct}/{len(cal_tasks)} = {best_gap:.1%}) ===")
            print(f"  Using cat_coef={best_coef} for full eval")
        else:
            print("  Too few calibration tasks; defaulting to cat_coef=1.0")
            _calibrated_cat_coef = 1.0

    # ---- Phase 2.6: Stratified calibration sweep ----
    _calibrated_bias_coef = None
    _calibrated_fixed_coef = None
    if getattr(args, "stratified_calibrate", False):
        cal_pct = float(getattr(args, "calibrate_pct", 0.10))
        cal_grid = [float(x) for x in args.calibrate_coef_grid.split(",")]
        print(f"\n=== Stratified calibration: judging think/base on all "
              f"{n_tasks} tasks ===")

        cal_judge_items = []
        for ti, t in enumerate(tasks):
            q, gold, tl = t["question"], t["correct_answer"], t["test_list"]
            common = dict(gold=gold, question=q, ds_type=ds_type, test_list=tl)
            cal_judge_items.append(dict(
                answer=re.sub(r'\s+', ' ', t["thinking_response"]).strip(),
                label=f"STCAL_T{ti}", **common))
            cal_judge_items.append(dict(
                answer=re.sub(r'\s+', ' ', t["base_response"]).strip(),
                label=f"STCAL_B{ti}", **common))

        cal_jr = judge_batch(cal_judge_items, args.judge_model,
                             n_reps=1, max_concurrent=args.max_concurrent)

        think_correct = [cal_jr[2*i]["correct"] for i in range(n_tasks)]
        base_correct  = [cal_jr[2*i+1]["correct"] for i in range(n_tasks)]

        base_acc = sum(base_correct) / n_tasks
        think_acc = sum(think_correct) / n_tasks
        gap = think_acc - base_acc
        print(f"  Base acc={base_acc:.1%}  Think acc={think_acc:.1%}  "
              f"Gap={gap:.1%}")

        # Stratified sampling: partition tasks into 4 buckets by
        # (base_correct, think_correct) and sample proportionally.
        import math as _math
        n_cal = max(1, _math.ceil(n_tasks * cal_pct))
        buckets = {(False, False): [], (False, True): [],
                   (True, False): [], (True, True): []}
        for i in range(n_tasks):
            buckets[(base_correct[i], think_correct[i])].append(i)

        cal_idx = []
        _remaining = n_cal
        _bucket_items = sorted(buckets.items(),
                               key=lambda kv: len(kv[1]))
        for bi, (bkey, bidxs) in enumerate(_bucket_items):
            if not bidxs:
                continue
            frac = len(bidxs) / n_tasks
            want = max(1, round(n_cal * frac)) if len(bidxs) > 0 else 0
            if bi == len(_bucket_items) - 1:
                want = _remaining
            want = min(want, len(bidxs), _remaining)
            random.seed(42)
            chosen = random.sample(bidxs, want)
            cal_idx.extend(chosen)
            _remaining -= want
            bc_lbl = ("Bcorrect" if bkey[0] else "Bwrong")
            tc_lbl = ("Tcorrect" if bkey[1] else "Twrong")
            print(f"  Bucket ({bc_lbl},{tc_lbl}): {len(bidxs)} tasks, "
                  f"sampled {want}")

        print(f"\n  Calibration set: {len(cal_idx)} tasks "
              f"(target {n_cal} = ceil({n_tasks} × {cal_pct}))")
        cal_base_acc = sum(base_correct[i] for i in cal_idx) / len(cal_idx)
        cal_think_acc = sum(think_correct[i] for i in cal_idx) / len(cal_idx)
        print(f"  Cal subset: base_acc={cal_base_acc:.1%}  "
              f"think_acc={cal_think_acc:.1%}  "
              f"(full: {base_acc:.1%} / {think_acc:.1%})")

        if len(cal_idx) >= 2:
            cal_tasks_s = [tasks[i] for i in cal_idx]
            _cal_hbs = min(len(cal_tasks_s), args.hybrid_gen_batch_size)
            best_coef_s, best_correct_s = cal_grid[0], -1
            _sweep_results = {}

            for _cc in cal_grid:
                print(f"\n  -- Stratified sweep coef={_cc} --")
                all_hybrid_s = []
                for cb_start in range(0, len(cal_tasks_s), _cal_hbs):
                    cb = cal_tasks_s[cb_start:cb_start + _cal_hbs]
                    torch.cuda.empty_cache()

                    if args.fixed_bias_coef is not None:
                        _sweep_bias_coefs = (args.fixed_bias_coef,)
                        _sweep_cat_coefs = (_cc,)
                    elif args.bias_only:
                        _sweep_bias_coefs = (_cc,)
                        _sweep_cat_coefs = (0.0,)
                    else:
                        _sweep_bias_coefs = (_cc,)
                        _sweep_cat_coefs = (_cc,)

                    hr = hybrid_generate_batched(
                        think_model, base_model, base_tok,
                        [t["thinking_prompt"] for t in cb],
                        [t["base_prompt"] for t in cb],
                        args.max_new_tokens, args.sae_layer, sae,
                        steering_vectors, descriptions, layer_map,
                        thinking_tokenizer=think_tok,
                        disable_sae_mean=args.disable_sae_mean,
                        show_progress=False, collect_details=False,
                        random_firing=args.random_firing,
                        random_firing_exclude_top_k_keys=args.random_firing_exclude_top_k_keys,
                        firing_replace_with_min_cosine=args.firing_replace_with_min_cosine,
                        pure_steer_base_eos=args.pure_steer_base_eos,
                        random_steer_prob=args.random_steer_prob,
                        random_guardrail=False,
                        random_seed=args.random_seed,
                        coef_sweep=[1.0],
                        coef_select="think_top1",
                        pg_bias_cat_sweep=True,
                        pg_bias_vec=(locals().get("bias_vec")
                                     if locals().get("bias_vec") is not None
                                     else None),
                        pg_bias_coefs=_sweep_bias_coefs,
                        pg_cat_coefs=_sweep_cat_coefs,
                        token_window=int(getattr(args, "token_window", 0)),
                        act_modulate=None)
                    all_hybrid_s.extend(hr)

                cal_h_items_s = []
                for j, (t, h) in enumerate(zip(cal_tasks_s, all_hybrid_s)):
                    resp = base_tok.decode(h["generated_ids"],
                                           skip_special_tokens=True)
                    q, gold, tl = (t["question"], t["correct_answer"],
                                   t["test_list"])
                    cal_h_items_s.append(dict(
                        answer=re.sub(r'\s+', ' ', resp).strip(),
                        label=f"STCAL_H_c{_cc}_{j}",
                        gold=gold, question=q, ds_type=ds_type,
                        test_list=tl))

                h_jr_s = judge_batch(cal_h_items_s, args.judge_model,
                                     n_reps=1,
                                     max_concurrent=args.max_concurrent)
                n_correct_s = sum(1 for r in h_jr_s if r["correct"])
                acc = n_correct_s / len(cal_tasks_s)
                print(f"  coef={_cc}: {n_correct_s}/{len(cal_tasks_s)} "
                      f"correct ({acc:.1%})")
                _sweep_results[str(_cc)] = {
                    "n_correct": n_correct_s,
                    "n_total": len(cal_tasks_s),
                    "accuracy": acc,
                }

                if n_correct_s > best_correct_s:
                    best_correct_s = n_correct_s
                    best_coef_s = _cc

            _calibrated_fixed_coef = best_coef_s
            if args.bias_only:
                _calibrated_bias_coef = best_coef_s
            if args.fixed_bias_coef is not None:
                _calibrated_bias_coef = args.fixed_bias_coef
                _calibrated_cat_coef = best_coef_s
                _calibrated_fixed_coef = None
            print(f"\n=== Stratified calibration: best coef={best_coef_s} "
                  f"({best_correct_s}/{len(cal_tasks_s)} = "
                  f"{best_correct_s/len(cal_tasks_s):.1%}) ===")

            if args.save_best_coef:
                _bucket_info = {}
                for bkey, bidxs in buckets.items():
                    bc_lbl = ("Bcorrect" if bkey[0] else "Bwrong")
                    tc_lbl = ("Tcorrect" if bkey[1] else "Twrong")
                    _sampled = sum(1 for i in cal_idx if i in bidxs)
                    _bucket_info[f"{bc_lbl},{tc_lbl}"] = {
                        "total": len(bidxs), "sampled": _sampled,
                    }
                _coef_info = {
                    "best_coef": best_coef_s,
                    "best_correct": best_correct_s,
                    "cal_size": len(cal_tasks_s),
                    "cal_acc": best_correct_s / len(cal_tasks_s),
                    "base_acc_full": base_acc,
                    "think_acc_full": think_acc,
                    "cal_base_acc": cal_base_acc,
                    "cal_think_acc": cal_think_acc,
                    "bias_only": bool(args.bias_only),
                    "dataset": args.dataset,
                    "base_model": args.base_model,
                    "thinking_model": args.thinking_model,
                    "grid": cal_grid,
                    "sweep_results": _sweep_results,
                    "stratified_buckets": _bucket_info,
                }
                if args.fixed_bias_coef is not None:
                    _coef_info["fixed_bias_coef"] = args.fixed_bias_coef
                    _coef_info["best_cat_coef"] = best_coef_s
                    _coef_info["bias_only"] = False
                os.makedirs(os.path.dirname(args.save_best_coef) or ".",
                            exist_ok=True)
                with open(args.save_best_coef, "w") as f:
                    json.dump(_coef_info, f, indent=2)
                print(f"  Saved best coef info -> {args.save_best_coef}")
        else:
            print("  Too few calibration tasks; defaulting to coef=1.0")
            _calibrated_fixed_coef = 1.0
            if args.bias_only:
                _calibrated_bias_coef = 1.0
            if args.fixed_bias_coef is not None:
                _calibrated_bias_coef = args.fixed_bias_coef
                _calibrated_cat_coef = 1.0
                _calibrated_fixed_coef = None

    # ---- Phase 3: Hybrid + judge ----
    hbs = args.hybrid_gen_batch_size
    print(f"\n=== Hybrid (B={hbs}, KV-cached, coeff-sweep) + judge ===")

    _suffix = _result_suffix(args)
    hc_path = _cache_path(
        args.results_dir, f"hybrid_{base_id}", think_id, args.dataset,
        args.temperature, args.max_new_tokens,
        sample_idx=args.hybrid_cache_sample_idx,
        cache_dir=cache_dir_override)
    if _suffix:
        hc_path = hc_path.replace(".jsonl", f"{_suffix}.jsonl")
    _hc_existing = set()
    if use_cache and os.path.exists(hc_path):
        with open(hc_path) as _hcf:
            for _hcl in _hcf:
                _hcl = _hcl.strip()
                if _hcl:
                    try:
                        _hcj = json.loads(_hcl)
                        _hc_existing.add(_hcj.get("dataset_idx"))
                    except json.JSONDecodeError:
                        pass
    print(f"  Hybrid response cache: {hc_path} ({len(_hc_existing)} existing)")

    # ---- Optional: per-cat activation-magnitude modulation ----
    _act_modulate = None
    if getattr(args, "act_modulate_stats", None):
        with open(args.act_modulate_stats, "r") as f:
            _stats = json.load(f)
        _fn = getattr(args, "act_modulate_fn", "p10p90")
        if _fn == "p25p75":
            _lo_key, _hi_key = "p25", "p75"
        elif _fn == "linear_minmax":
            _lo_key, _hi_key = "min", "max"
        else:
            _lo_key, _hi_key = "p10", "p90"
        _act_modulate = {}
        print(f"\n[act_modulate] loading stats: {args.act_modulate_stats}")
        print(f"[act_modulate] using fn='{_fn}' "
              f"(lo={_lo_key}, hi={_hi_key})")
        print(f"[act_modulate] per-cat (lo, hi):")
        for _k, _s in _stats.get("per_cat", {}).items():
            if _s.get("count", 0) == 0:
                continue
            if _lo_key not in _s or _hi_key not in _s:
                continue
            _act_modulate[_k] = (float(_s[_lo_key]), float(_s[_hi_key]))
            print(f"  {_k}: ({_s[_lo_key]:.3f}, {_s[_hi_key]:.3f})  "
                  f"n={_s['count']}")

    prev_n, prev_counts, prev_per_rep = _load_prev_counts(
        args, base_id, think_id)
    n_reps_eff = max(1, int(args.judge_repetitions))
    results = {"thinking_correct": 0, "base_correct": 0, "hybrid_correct": 0,
               "thinking_per_rep": [0] * n_reps_eff,
               "base_per_rep": [0] * n_reps_eff,
               "hybrid_per_rep": [0] * n_reps_eff,
               "thinking_eos": [], "base_eos": [], "hybrid_eos": [],
               "thinking_lengths": [], "base_lengths": [], "hybrid_lengths": []}
    for _k in ("thinking", "base", "hybrid"):
        if prev_per_rep.get(_k) and len(prev_per_rep[_k]) != n_reps_eff:
            target = max(len(prev_per_rep[_k]), n_reps_eff)
            prev_per_rep[_k] = prev_per_rep[_k] + [0] * (
                target - len(prev_per_rep[_k]))
            results[_k + "_per_rep"] = results[_k + "_per_rep"] + [0] * (
                target - len(results[_k + "_per_rep"]))
            n_reps_eff = target

    for batch_start in range(0, n_tasks, hbs):
        batch = tasks[batch_start:batch_start + hbs]
        B_ = len(batch)
        print(f"\n--- Batch {batch_start//hbs+1} "
              f"(tasks {batch_start+1}-{batch_start+B_}/{n_tasks}) ---")

        torch.cuda.empty_cache()
        _coef_sweep = ([args.fixed_coef]
                       if args.fixed_coef is not None
                       else [float(x) for x in args.coef_sweep.split(",")])
        if getattr(args, "pg_bias_cat_sweep", False):
            _coef_select = args.coef_select
            _pg_bias_coefs = tuple(float(x) for x in args.pg_bias_coefs.split(","))
            _pg_cat_coefs = tuple(float(x) for x in args.pg_cat_coefs.split(","))
            # Override with calibrated coef if available
            if _calibrated_cat_coef is not None:
                _pg_cat_coefs = (_calibrated_cat_coef,)
                print(f"  [calibrated] using cat_coef={_calibrated_cat_coef}")
            if _calibrated_bias_coef is not None:
                _pg_bias_coefs = (_calibrated_bias_coef,)
                print(f"  [strat-calibrated] using bias_coef={_calibrated_bias_coef}")
            if _calibrated_fixed_coef is not None and _calibrated_bias_coef is None:
                _pg_bias_coefs = (_calibrated_fixed_coef,)
                _pg_cat_coefs = (_calibrated_fixed_coef,)
                print(f"  [strat-calibrated] using coef={_calibrated_fixed_coef}")
        else:
            _coef_select = ("fixed"
                            if args.fixed_coef is not None
                            else args.coef_select)
            _pg_bias_coefs = (0.0, 0.5, 1.0)
            _pg_cat_coefs = (0.0, 0.5, 1.0)
        hr = hybrid_generate_batched(
            think_model, base_model, base_tok,
            [t["thinking_prompt"] for t in batch],
            [t["base_prompt"] for t in batch],
            args.max_new_tokens, args.sae_layer, sae,
            steering_vectors, descriptions, layer_map,
            thinking_tokenizer=think_tok,
            disable_sae_mean=args.disable_sae_mean,
            show_progress=True, collect_details=True,
            random_firing=args.random_firing,
            random_firing_exclude_top_k_keys=args.random_firing_exclude_top_k_keys,
            firing_replace_with_min_cosine=args.firing_replace_with_min_cosine,
            pure_steer_base_eos=args.pure_steer_base_eos,
            random_steer_prob=args.random_steer_prob,
            random_guardrail=args.random_guardrail,
            random_seed=args.random_seed,
            coef_sweep=_coef_sweep,
            steer_all_positions=args.steer_all_positions,
            steer_all_positions_full=args.steer_all_positions_full,
            coef_select=_coef_select,
            kl_topk=args.kl_topk,
            always_on_bias_vec=(_always_on_bias_vec
                                if getattr(args, "bias_always_on", False)
                                else None),
            always_on_bias_layer=(_always_on_bias_layer
                                  if getattr(args, "bias_always_on", False)
                                  else None),
            pg_bias_cat_sweep=getattr(args, "pg_bias_cat_sweep", False),
            pg_bias_vec=(locals().get("bias_vec")
                         if getattr(args, "pg_bias_cat_sweep", False)
                         else None),
            pg_bias_coefs=_pg_bias_coefs,
            pg_cat_coefs=_pg_cat_coefs,
            token_window=int(getattr(args, "token_window", 0)),
            act_modulate=_act_modulate,
            mlp_model=locals().get("mlp_model"),
            mlp_coef_scale=float(getattr(args, "mlp_coef_scale", 1.0)),
            decode_temperature=float(getattr(
                args, "decode_temperature", 0.0)),
            decode_seed=int(getattr(args, "decode_seed", 0)),
            warmup_until_sentence_end=bool(getattr(
                args, "warmup_until_sentence_end", False)),
            warmup_max_tokens=int(getattr(args, "warmup_max_tokens", 60)),
            suppress_boxed_first_n_tokens=int(getattr(
                args, "suppress_boxed_first_n_tokens", 0)),
            accept_answer_close=bool(getattr(
                args, "accept_answer_close", False)),
            disable_eos_suppression=bool(getattr(
                args, "disable_eos_suppression", False)),
            free_fly_until_think_eos=bool(getattr(
                args, "free_fly_until_think_eos", False)),
            no_termination=bool(getattr(args, "no_termination", False)),
            eos_prob_warmup=bool(getattr(args, "eos_prob_warmup", False)),
            eos_prob_warmup_steps=int(getattr(
                args, "eos_prob_warmup_steps", 0)),
            thinking_continuation_text=(
                [t.get("thinking_continuation_text", "") or "" for t in batch]
                if any(t.get("thinking_continuation_text") for t in batch)
                else None),
            base_continuation_text=(
                [t.get("base_continuation_text", "") or "" for t in batch]
                if any(t.get("base_continuation_text") for t in batch)
                else None))

        judge_items = []
        batch_meta = []
        for j, (t, h) in enumerate(zip(batch, hr)):
            hybrid_new = base_tok.decode(h["generated_ids"], skip_special_tokens=True)
            # In continuation mode, judge / record the FULL response: existing
            # (truncated) hybrid text + the freshly-generated continuation.
            _cont_base = t.get("base_continuation_text") or ""
            hybrid_resp = (_cont_base + hybrid_new) if _cont_base else hybrid_new
            _orig_hyb_tok = int(t.get("_cont_orig_hybrid_toks", 0))
            hybrid_toks_total = _orig_hyb_tok + int(h["n_generated"])
            q, gold, tl = t["question"], t["correct_answer"], t["test_list"]
            common = dict(gold=gold, question=q, ds_type=ds_type, test_list=tl)
            ti = batch_start + j
            judge_items.append(dict(answer=re.sub(r'\s+', ' ', t["thinking_response"]).strip(),
                                    label=f"T{ti+1} Think", **common))
            judge_items.append(dict(answer=re.sub(r'\s+', ' ', t["base_response"]).strip(),
                                    label=f"T{ti+1} Base", **common))
            judge_items.append(dict(answer=re.sub(r'\s+', ' ', hybrid_resp).strip(),
                                    label=f"T{ti+1} Hybrid", **common))
            # If a cold-start prefix was injected into the hybrid prefill,
            # the per-token info from hybrid_generate_batched only covers
            # the freshly-generated continuation.  Prepend synthetic
            # non-steered entries for the cold-start tokens so the
            # inspector renders the whole hybrid response and the prefix
            # shows up as plain (un-highlighted) tokens.
            tli = h.get("token_latent_info") or []
            cs = (t.get("base_continuation_text") or "")
            if cs:
                cs_ids = base_tok(cs, add_special_tokens=False)["input_ids"]
                cs_entries = []
                for tid in cs_ids:
                    cs_entries.append({
                        "token": base_tok.decode([tid], skip_special_tokens=True),
                        "selection": "base",
                        "latent_key": "",
                        "latent_id": None,
                        "latent_title": "",
                        "coefficient": 0.0,
                        "steer_layer": None,
                        "activation_value": 0.0,
                        "steered_matches_think": None,
                        "disagreed": None,
                        "cold_start": True,
                    })
                tli = cs_entries + (tli if isinstance(tli, list) else [])
            batch_meta.append(dict(
                task_idx=ti, question=q, gold=gold, test_list=tl,
                think_resp=t["thinking_response"], base_resp=t["base_response"],
                hybrid_resp=hybrid_resp, hybrid_eos=h["ended_by_eos"],
                hybrid_toks=hybrid_toks_total,
                think_eos=t["thinking_eos"], base_eos=t["base_eos"],
                think_toks=t["thinking_n_tokens"], base_toks=t["base_n_tokens"],
                steering_stats=h.get("steering_stats"),
                token_latent_info=tli))

        if any(m.get("steering_stats") for m in batch_meta):
            ss_all = [m["steering_stats"] for m in batch_meta if m.get("steering_stats")]
            tot_st = sum(s["n_steered"] for s in ss_all)
            tot_tok = sum(s["n_total"] for s in ss_all)
            tot_dis = sum(s.get("n_disagree", 0) for s in ss_all)
            tot_no_vec = sum(s.get("n_no_vector", 0) for s in ss_all)
            tot_match = sum(s.get("n_steered_matches_think", 0) for s in ss_all)
            agg_cc = {}
            agg_cat = {}
            agg_layer = {}
            for s in ss_all:
                for k, v in s.get("coeff_distribution", {}).items():
                    agg_cc[k] = agg_cc.get(k, 0) + v
                for k, v in s.get("per_category_counts", {}).items():
                    agg_cat[k] = agg_cat.get(k, 0) + v
                for k, v in s.get("per_layer_counts", {}).items():
                    agg_layer[k] = agg_layer.get(k, 0) + v
            top5 = sorted(agg_cat.items(), key=lambda kv: -kv[1])[:5]
            match_pct = tot_match / max(tot_st, 1) * 100
            print(f"  Steering: {tot_st}/{tot_tok} tokens "
                  f"({tot_st/max(tot_tok,1)*100:.1f}%)  "
                  f"disagree={tot_dis}  no_vec={tot_no_vec}  "
                  f"match_think={tot_match} ({match_pct:.1f}%)")
            print(f"    coeff dist: {dict(sorted(agg_cc.items()))}")
            print(f"    top cats:   {top5}")
            if len(agg_layer) > 1:
                print(f"    per-layer:  {dict(sorted(agg_layer.items()))}")

        print(f"\n  Judging {len(judge_items)} items...")
        jr = judge_batch(judge_items, args.judge_model,
                         n_reps=args.judge_repetitions,
                         max_concurrent=args.max_concurrent)

        for j, m in enumerate(batch_meta):
            te, be, he = jr[j*3], jr[j*3+1], jr[j*3+2]
            if te["correct"]: results["thinking_correct"] += 1
            if be["correct"]: results["base_correct"] += 1
            if he["correct"]: results["hybrid_correct"] += 1
            for _key, _entry in (("thinking", te), ("base", be),
                                 ("hybrid", he)):
                _reps = _entry.get("repetitions") or []
                if not _reps:
                    _reps = [{"correct": _entry.get("correct", False)}]
                _per = results[_key + "_per_rep"]
                if len(_per) != len(_reps):
                    _target = max(len(_per), len(_reps))
                    while len(_per) < _target:
                        _per.append(0)
                    while len(_reps) < _target:
                        _reps.append({"correct": False})
                for _i, _r in enumerate(_reps):
                    if _r.get("correct"):
                        _per[_i] += 1
            results["thinking_eos"].append(m["think_eos"])
            results["base_eos"].append(m["base_eos"])
            results["hybrid_eos"].append(m["hybrid_eos"])
            results["thinking_lengths"].append(m["think_toks"])
            results["base_lengths"].append(m["base_toks"])
            results["hybrid_lengths"].append(m["hybrid_toks"])

            append_rolling({
                "ts": time.time(), "dataset": args.dataset,
                "question": m["question"], "gold_answer": m["gold"],
                "answers": {"thinking": m["think_resp"], "base": m["base_resp"],
                            "hybrid": m["hybrid_resp"]},
                "judges": {"thinking": te, "base": be, "hybrid": he},
                "eos": {"thinking": m["think_eos"], "base": m["base_eos"],
                        "hybrid": m["hybrid_eos"]},
                "n_tokens": {"thinking": m["think_toks"], "base": m["base_toks"],
                             "hybrid": m["hybrid_toks"]},
                "steering_stats": m.get("steering_stats"),
                "token_latent_info": m.get("token_latent_info"),
            }, args, base_id, think_id)

            done = m["task_idx"] + 1 + prev_n
            ct = prev_counts["thinking"] + results["thinking_correct"]
            cb = prev_counts["base"] + results["base_correct"]
            ch = prev_counts["hybrid"] + results["hybrid_correct"]
            tp, bp, hp = ct/done*100, cb/done*100, ch/done*100
            gap = abs(tp - bp)
            gs = f"  Gap: {max(0,(hp-min(tp,bp))/gap*100):.1f}%" if gap > 0 else ""
            print(f"  [{done}] T={ct}/{done}({tp:.1f}%) B={cb}/{done}({bp:.1f}%) "
                  f"H={ch}/{done}({hp:.1f}%){gs}")

        del hr; gc.collect(); torch.cuda.empty_cache()

    # ---- Final summary ----
    total = prev_n + len(tasks)
    ct = prev_counts["thinking"] + results["thinking_correct"]
    cb = prev_counts["base"] + results["base_correct"]
    ch = prev_counts["hybrid"] + results["hybrid_correct"]
    print(f"\n===== Final =====")
    print(f"Thinking: {ct}/{total} ({ct/total*100:.1f}%)")
    print(f"Base:     {cb}/{total} ({cb/total*100:.1f}%)")
    print(f"Hybrid:   {ch}/{total} ({ch/total*100:.1f}%)")
    gap = abs(ct/total - cb/total) * 100
    if gap > 0:
        rec = (ch/total*100 - min(ct/total, cb/total)*100) / gap
        print(f"Gap recovered: {max(0, rec)*100:.1f}%")

    # Per-rep accuracy mean / std (judge-noise quantification).
    def _combine_per_rep(prev, cur):
        if prev and cur:
            target = max(len(prev), len(cur))
            prev = prev + [0] * (target - len(prev))
            cur = cur + [0] * (target - len(cur))
            return [a + b for a, b in zip(prev, cur)]
        return list(prev or cur or [])

    rep_totals = {
        "thinking": _combine_per_rep(prev_per_rep.get("thinking"),
                                     results["thinking_per_rep"]),
        "base": _combine_per_rep(prev_per_rep.get("base"),
                                 results["base_per_rep"]),
        "hybrid": _combine_per_rep(prev_per_rep.get("hybrid"),
                                   results["hybrid_per_rep"]),
    }
    if rep_totals["thinking"] and len(rep_totals["thinking"]) > 1:
        import statistics as _stats
        print("\n===== Judge sampling (mean +/- std across reps) =====")
        rep_summary = {}
        for _name, _counts_list in rep_totals.items():
            accs = [c / total * 100 for c in _counts_list]
            mu = _stats.mean(accs)
            sd = _stats.stdev(accs) if len(accs) > 1 else 0.0
            rep_summary[_name] = {"per_rep_pct": accs,
                                  "mean_pct": mu, "std_pct": sd,
                                  "per_rep_correct": list(_counts_list)}
            print(f"{_name.capitalize():9s}: {mu:.2f}% +/- {sd:.2f}%   "
                  f"per-rep: {[f'{a:.2f}%' for a in accs]}")
        # Hybrid gap-recovered mean/std
        t_mean = rep_summary["thinking"]["mean_pct"]
        b_mean = rep_summary["base"]["mean_pct"]
        h_per = rep_summary["hybrid"]["per_rep_pct"]
        gap0 = abs(t_mean - b_mean)
        if gap0 > 0:
            recs = [max(0.0, (h - min(t_mean, b_mean)) / gap0 * 100)
                    for h in h_per]
            mu_r = _stats.mean(recs)
            sd_r = (_stats.stdev(recs) if len(recs) > 1 else 0.0)
            print(f"Gap recovered (per-rep, vs majority t/b means): "
                  f"{mu_r:.2f}% +/- {sd_r:.2f}%")
        # Persist alongside the existing summary JSON.
        try:
            _suffix = _result_suffix(args)
            _sj = (f"{args.results_dir}/judge_reps_{base_id}"
                   f"_{args.dataset}{_suffix}.json")
            with open(_sj, "w") as _f:
                json.dump({"total": total, "per_rep": rep_summary,
                           "n_reps": len(rep_totals['thinking'])}, _f,
                          indent=2)
            print(f"Wrote judge-reps summary to {_sj}")
        except Exception as _e:
            print(f"warn: failed to save judge_reps json: {_e}")

    plt.figure(figsize=(10, 6))
    accs = [cb/total*100, ct/total*100, ch/total*100]
    plt.bar(["Base", "Thinking", "Hybrid"], accs,
            color=["#3498db", "#e74c3c", "#2ecc71"])
    plt.title(f"Accuracy on {total} {args.dataset} Tasks")
    plt.ylabel("Accuracy (%)")
    plt.ylim(0, 100)
    for i, a in enumerate(accs):
        plt.text(i, a + 2, f"{a:.1f}%", ha="center")
    plt.tight_layout()
    s = _result_suffix(args)
    plt.savefig(f"{args.results_dir}/accuracy_{base_id}_{args.dataset}{s}.png")
    plt.close()
    print("Done.")


if __name__ == "__main__":
    main()
