#!/usr/bin/env python3
"""Generate rollouts for datasets via a vLLM OpenAI-compatible server.

Produces JSONL files in the exact format consumed by
  train-vectors/optimize_correction_vectors.py:
    {"dataset_idx": <int>, "response": <str>, "n_tokens": <int>, "eos": <bool>}

The `response` field must be compatible with
  utils/responses.py:extract_thinking_process():
    - For models with <think>…</think>: strip the leading <think> tag
      (the training code's generation prompt already includes it) but
      keep </think> so the parser finds the boundary.
    - For models without think tags (e.g. QwQ): store the raw response;
      extract_thinking_process treats the whole text as reasoning.

Usage:
    python generate_rollouts.py \\
        --base_url http://node-X:8000/v1 \\
        --model Qwen/QwQ-32B \\
        --model_short qwq-32b \\
        --dataset mmlu_auxiliary_train \\
        --n_examples 20000 \\
        --max_tokens 2048 \\
        --output_dir ../hybrid/results/response_cache
"""
import argparse, json, os, sys, time, re
from concurrent.futures import ThreadPoolExecutor, as_completed

# ── Model-specific formatters ──────────────────────────────────────────────

THINK_OPEN = "<think>"
THINK_CLOSE = "</think>"


def _format_orz(content: str) -> str:
    """Open-Reasoner-Zero: outputs <think>reasoning</think>\\n\\nanswer.
    Strip leading <think> to match the existing cache format where
    response = 'reasoning</think>\\n\\nanswer'."""
    s = content.lstrip()
    if s.startswith(THINK_OPEN):
        s = s[len(THINK_OPEN):]
    return s


def _format_r1_distill(content: str) -> str:
    """DeepSeek-R1-Distill-Qwen: outputs <think>reasoning</think>\\n\\nanswer.
    Same treatment as ORZ: strip leading <think>, keep </think>."""
    s = content.lstrip()
    if s.startswith(THINK_OPEN):
        s = s[len(THINK_OPEN):]
    return s


def _format_qwq(content: str) -> str:
    """QwQ-32B: inline reasoning, no think tags.
    Store as-is; extract_thinking_process returns the whole text when
    no <think> is present (think_start=0)."""
    return content


FORMATTERS = {
    "orz":       _format_orz,
    "r1":        _format_r1_distill,
    "qwq":       _format_qwq,
    "passthrough": lambda c: c,
}


# ---------------------------------------------------------------------------
# User-content shaping (Table 5 / R1+QwQ math directive)
# ---------------------------------------------------------------------------
# ORZ Table-5 user instruction (the chunk prepended to {{prompt}} under the
# single User turn).  ORZ's shipped chat_template emits the preamble and the
# closing ``Assistant: <think>`` automatically, so we only need to prepend
# this paragraph to the actual question.
ORZ_USER_PREFIX = (
    "You must put your answer inside <answer> </answer> tags, i.e., "
    "<answer> answer here </answer>. And your final answer will be "
    "extracted automatically by the \\boxed{} tag."
)

# DeepSeek-R1 + QwQ recommended directive for mathematical problems.
MATH_DIRECTIVE = (
    "Please reason step by step, and put your final answer within \\boxed{}."
)

# Source names in the prepared trainmix file whose rows we treat as
# "math problems" for the purposes of the R1/QwQ directive.  Other
# sources (e.g. SciBench, TheoremQA) are deliberately NOT extended --
# matches the user's explicit guidance.
TRAINMIX_MATH_SOURCES = {"hendrycks_math", "natural_reasoning"}


def shape_user_content(question: str, family: str, *,
                       is_math_question: bool = False) -> str:
    """Return the per-question user-message content tuned to the model
    family's required prompt format.

    - ``orz`` : prepends the Table-5 user instruction so that
      ``apply_chat_template([{"role":"user","content": this}], ...)`` emits
      the full Table 5 template (preamble + the prepended directive +
      the question + ``Assistant: <think>``).  When ``is_math_question``
      is True, also appends the R1/QwQ math directive after the
      question (same place as R1/QwQ) so ORZ receives the matching
      "Please reason step by step ... \\boxed{}" instruction on math
      benchmarks.  The Table-5 prefix is retained either way because
      ORZ's <answer> contract depends on it.
    - ``r1`` / ``qwq`` : appends the step-by-step \\boxed{} directive
      iff ``is_math_question`` (per DeepSeek-R1 / QwQ usage docs).
    - other / ``passthrough`` : returns the question unchanged.
    """
    if family == "orz":
        content = f"{ORZ_USER_PREFIX}\n{question}"
        if is_math_question:
            content = f"{content}\n\n{MATH_DIRECTIVE}"
        return content
    if family in ("r1", "qwq") and is_math_question:
        return f"{question}\n\n{MATH_DIRECTIVE}"
    return question

MODEL_FORMAT_MAP = {
    "open-reasoner-zero": "orz",
    "deepseek-r1-distill": "r1",
    "qwq": "qwq",
}


def detect_format(model_id: str) -> str:
    low = model_id.lower()
    if "open-reasoner-zero" in low or "orz" in low:
        return "orz"
    if "r1-distill" in low or "deepseek-r1" in low:
        return "r1"
    if "qwq" in low:
        return "qwq"
    return "passthrough"


# ── Dataset loading ────────────────────────────────────────────────────────

def load_dataset_questions(dataset_name: str, n_examples: int,
                           dataset_file: str = None):
    if dataset_file:
        items = []
        with open(dataset_file) as f:
            for line in f:
                rec = json.loads(line)
                items.append({
                    "dataset_idx": rec["idx"],
                    "question": rec["question"],
                    # carry source so callers can decide whether to
                    # inject the math directive per-row (trainmix only)
                    "source": rec.get("source"),
                })
                if len(items) >= n_examples:
                    break
        return items

    from datasets import load_dataset

    if dataset_name == "mmlu_auxiliary_train":
        ds = load_dataset("cais/mmlu", "all", split="auxiliary_train")
        q_key = "question"
        src = "mmlu_auxiliary_train"
    elif dataset_name == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main")["test"]
        q_key = "question"
        src = "gsm8k"
    elif dataset_name == "math500":
        ds = load_dataset("HuggingFaceH4/MATH-500")["test"]
        q_key = "problem"
        src = "math500"
    elif dataset_name == "hendrycks_math":
        ds = load_dataset("nlile/hendrycks-MATH-benchmark")["train"]
        q_key = "problem"
        src = "hendrycks_math"
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    items = []
    for i in range(min(len(ds), n_examples)):
        items.append({"dataset_idx": i, "question": ds[i][q_key],
                      "source": src})
    return items


def _is_math_question(source: str | None, directive_mode: str,
                      dataset_name: str | None) -> bool:
    """Decide whether a single row counts as a 'math problem' for the
    purposes of the R1/QwQ step-by-step directive.

    - ``directive_mode="none"``  : never.
    - ``directive_mode="always"``: every row (used for math500 / gsm8k benches).
    - ``directive_mode="auto"``  : True iff ``source`` ∈ TRAINMIX_MATH_SOURCES
      (used for trainmix custom-file generation; only hendrycks_math and
      natural_reasoning rows receive the directive, per the usage docs).
    """
    if directive_mode == "none":
        return False
    if directive_mode == "always":
        return True
    if directive_mode == "auto":
        if source and source in TRAINMIX_MATH_SOURCES:
            return True
        if dataset_name in ("math500", "gsm8k", "hendrycks_math"):
            return True
        return False
    raise ValueError(f"Unknown math directive mode: {directive_mode!r}")


# ── Generation ─────────────────────────────────────────────────────────────

def generate_one(client, model, question, max_tokens, temperature,
                 top_p=1.0, seed=None):
    """vLLM chat-completion path.  ``question`` is the user-message
    content; vLLM applies the model's chat_template internally."""
    from openai import OpenAI  # noqa: F811 – lazy for testability
    kwargs = dict(
        model=model,
        messages=[{"role": "user", "content": question}],
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stream=False,
    )
    if seed is not None:
        kwargs["seed"] = int(seed)
    resp = client.chat.completions.create(**kwargs)
    choice = resp.choices[0]
    return {
        "content": choice.message.content or "",
        "n_tokens": resp.usage.completion_tokens,
        "eos": choice.finish_reason == "stop",
    }


def generate_one_completion(client, model, raw_prompt, max_tokens, temperature,
                            top_p=1.0, seed=None):
    """vLLM text-completion path.  ``raw_prompt`` is the literal prompt
    string that gets tokenized AS-IS by vLLM (no chat template).
    Use this for base-model rollouts where we need the exact
    ``User: {q}\\nAssistant:`` text-completion prompt to match the
    hybrid-eval tokenization."""
    kwargs = dict(
        model=model,
        prompt=raw_prompt,
        max_tokens=max_tokens,
        temperature=temperature,
        top_p=top_p,
        stream=False,
    )
    if seed is not None:
        kwargs["seed"] = int(seed)
    resp = client.completions.create(**kwargs)
    choice = resp.choices[0]
    return {
        "content": choice.text or "",
        "n_tokens": resp.usage.completion_tokens,
        "eos": choice.finish_reason == "stop",
    }


def _temp_label(temperature: float) -> str:
    """File-safe temperature label: 0.0 -> '0', 0.6 -> '0.6'."""
    return f"{temperature:.2f}".rstrip("0").rstrip(".") or "0"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--base_url", required=True,
                   help="vLLM server URL, e.g. http://node-X:8000/v1")
    p.add_argument("--model", required=True,
                   help="HuggingFace model ID served by vLLM")
    p.add_argument("--model_short", required=True,
                   help="Short name for cache file, e.g. qwq-32b")
    p.add_argument("--format", default=None,
                   choices=list(FORMATTERS.keys()),
                   help="Response format (auto-detected from model ID)")
    p.add_argument("--dataset", required=True,
                   help="Dataset name: mmlu_auxiliary_train, gsm8k, math500, "
                        "hendrycks_math, or 'custom' with --dataset_file")
    p.add_argument("--dataset_file", default=None,
                   help="Path to custom JSONL question file (each line: "
                        "{\"idx\": int, \"question\": str, ...}). "
                        "When set, --dataset is used only for the output "
                        "filename.")
    p.add_argument("--n_examples", type=int, default=20000)
    p.add_argument("--max_tokens", type=int, default=2048)
    p.add_argument("--temperature", type=float, default=0.0)
    p.add_argument("--top_p", type=float, default=1.0,
                   help="Nucleus-sampling top_p (vLLM honors this).")
    p.add_argument("--seed", type=int, default=None,
                   help="Sampling seed forwarded to vLLM. Use seed = "
                        "base_seed + sample_idx so distinct sample_idx "
                        "values yield distinct, deterministic rollouts.")
    p.add_argument("--sample_idx", type=int, default=-1,
                   help="Sample index appended to the output filename as "
                        "'_s<N>'. -1 (default) omits the suffix, matching "
                        "legacy single-rollout naming.")
    p.add_argument("--output_dir", default=None,
                   help="Output directory (default: ../hybrid/results/"
                        "response_cache)")
    p.add_argument("--max_concurrent", type=int, default=32,
                   help="Max parallel requests to vLLM server")
    p.add_argument("--resume", action="store_true", default=True,
                   help="Skip already-generated dataset_idx entries")
    p.add_argument("--role", type=str, default="thinking",
                   choices=["thinking", "base"],
                   help="Role prefix for output file: 'thinking' or 'base'")
    p.add_argument("--math_directive_mode", default="none",
                   choices=["none", "always", "auto"],
                   help=(
                       "Per-row injection of the R1/QwQ math directive "
                       "'Please reason step by step, and put your final "
                       "answer within \\boxed{}.': 'none' never adds it "
                       "(legacy behaviour); 'always' adds it to every row "
                       "(use for math500/gsm8k benchmarks); 'auto' adds it "
                       "only to rows whose source is a math source "
                       "(hendrycks_math, natural_reasoning) -- intended "
                       "for trainmix custom-file generation. The directive "
                       "is only applied to r1/qwq families; for orz the "
                       "Table-5 user prefix is applied unconditionally and "
                       "supersedes this flag."))
    p.add_argument("--use_text_completions", action="store_true",
                   help=(
                       "Use the /v1/completions endpoint with a raw text "
                       "prompt instead of /v1/chat/completions. The raw "
                       "prompt is built as 'User: {q}\\nAssistant:' to "
                       "match the exact base-model tokenization used by "
                       "hybrid_eval (base_prompt). Use this for base-model "
                       "rollouts so they share the same prompt format "
                       "regardless of which think variant consumes them."))
    p.add_argument("--preview_first_n", type=int, default=5,
                   help=(
                       "Print the first N completed responses (first 300 "
                       "chars each) to stdout so you can spot-check that "
                       "prompts are well-formed early. Set 0 to disable."))
    p.add_argument("--base_prompt_style", default="default",
                   choices=["default", "stepwise", "boxed", "legacy_task"],
                   help=(
                       "Only used when --use_text_completions and "
                       "--role=base.  'default' sends bare 'User: {q}"
                       "\\nAssistant:'. 'stepwise' (ff v1) sends 'User: "
                       "Answer the following question. Respond step by "
                       "step.\\n\\n{q}\\nAssistant:'. 'boxed' (ff v2) "
                       "sends 'User: {q}\\n\\nPlease reason step by "
                       "step, and put your final answer within "
                       "\\boxed{}.\\nAssistant:'. 'legacy_task' (ff v3) "
                       "sends 'Task: Answer the question below. Explain "
                       "your reasoning step by step.\\n\\n\\n\\n"
                       "Question:\\n{q}\\n\\nStep by step answer:\\n' "
                       "(the structured prompt used by origin/main's "
                       "hybrid_*.py + optimize_steering_vectors.py).  "
                       "MUST match --base_prompt_style in hybrid_eval.py "
                       "and optimize_correction_vectors.py."))
    args = p.parse_args()

    fmt_name = args.format or detect_format(args.model)
    if args.role == "base":
        fmt_name = "passthrough"
    formatter = FORMATTERS[fmt_name]
    # The family is independent of the response-formatting style: it
    # decides how we shape the *outgoing* user content (Table-5 / math
    # directive), whereas ``formatter`` cleans the *response* afterwards.
    # For base rollouts we never wrap the user content, hence "base" here.
    family = fmt_name if args.role == "thinking" else "base"
    # Text-completions mode is only meaningful for base rollouts where
    # we want to match the hybrid_eval base_prompt exactly.  For think
    # rollouts the chat template is required, so we ignore the flag
    # with a warning to avoid silent surprises.
    use_completions = bool(args.use_text_completions)
    if use_completions and args.role == "thinking":
        print(f"  WARN: --use_text_completions ignored for role=thinking",
              flush=True)
        use_completions = False
    print(f"Model: {args.model}")
    print(f"Role: {args.role}")
    print(f"Format: {fmt_name}")
    print(f"Family (user-content shaping): {family}")
    print(f"Math directive mode: {args.math_directive_mode}")
    print(f"Endpoint: {'/v1/completions (raw)' if use_completions else '/v1/chat/completions'}",
          flush=True)
    print(f"Dataset: {args.dataset} (n={args.n_examples})")
    print(f"Max tokens: {args.max_tokens}")

    out_dir = args.output_dir or os.path.join(
        os.path.dirname(__file__), "..", "hybrid", "results", "response_cache")
    os.makedirs(out_dir, exist_ok=True)
    ts = _temp_label(args.temperature)
    s_suffix = f"_s{args.sample_idx}" if args.sample_idx >= 0 else ""
    out_path = os.path.join(
        out_dir,
        f"{args.role}_{args.model_short}_{args.dataset}"
        f"_temp{ts}_max{args.max_tokens}{s_suffix}.jsonl")

    # Load existing entries for resume
    existing_idx = set()
    if args.resume and os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    existing_idx.add(int(json.loads(line)["dataset_idx"]))
                except (json.JSONDecodeError, KeyError):
                    pass
        print(f"Resume: {len(existing_idx)} existing entries in {out_path}")

    questions = load_dataset_questions(args.dataset, args.n_examples,
                                      dataset_file=args.dataset_file)
    todo = [q for q in questions if q["dataset_idx"] not in existing_idx]
    print(f"To generate: {len(todo)} (skipping {len(existing_idx)} existing)")

    if not todo:
        print("Nothing to do.")
        return

    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key="dummy")

    # Verify server is up
    models = client.models.list()
    print(f"Server models: {[m.id for m in models.data]}")

    t0 = time.time()
    done = 0
    errors = 0

    previewed = 0
    with open(out_path, "a") as out_f:
        with ThreadPoolExecutor(max_workers=args.max_concurrent) as pool:
            futures = {}
            for item in todo:
                # Per-item seed: when --seed is set we deterministically
                # offset by dataset_idx so each row gets a distinct yet
                # reproducible sample; otherwise vLLM picks freely.
                _seed = (args.seed + int(item["dataset_idx"])
                         if args.seed is not None else None)
                is_math = _is_math_question(
                    item.get("source"), args.math_directive_mode, args.dataset)
                user_content = shape_user_content(
                    item["question"], family, is_math_question=is_math)
                if use_completions:
                    # Match the exact base_prompt format used by
                    # hybrid_eval._build_task_prompts: "User: {q}\nAssistant:"
                    # with NO chat template applied.  ``user_content`` is
                    # the (already-shaped) question text; for base
                    # rollouts shape_user_content is a no-op because
                    # family="base", but we still pass it through so the
                    # plumbing is consistent if we ever shape base rows.
                    # --base_prompt_style chooses between the legacy
                    # bare-question prompt and the final_final variants.
                    if args.base_prompt_style == "stepwise":
                        raw_prompt = (
                            "User: Answer the following question. "
                            "Respond step by step.\n\n"
                            f"{user_content}\nAssistant:")
                    elif args.base_prompt_style == "boxed":
                        raw_prompt = (
                            f"User: {user_content}\n\nPlease reason "
                            "step by step, and put your final answer "
                            "within \\boxed{}.\nAssistant:")
                    elif args.base_prompt_style == "legacy_task":
                        raw_prompt = (
                            "Task: Answer the question below. Explain "
                            "your reasoning step by step.\n\n\n\n"
                            f"Question:\n{user_content}\n\n"
                            "Step by step answer:\n")
                    else:
                        raw_prompt = f"User: {user_content}\nAssistant:"
                    fut = pool.submit(
                        generate_one_completion, client, args.model,
                        raw_prompt, args.max_tokens, args.temperature,
                        args.top_p, _seed)
                else:
                    fut = pool.submit(
                        generate_one, client, args.model,
                        user_content, args.max_tokens, args.temperature,
                        args.top_p, _seed)
                futures[fut] = (item, user_content)

            for fut in as_completed(futures):
                item, user_content = futures[fut]
                try:
                    result = fut.result()
                    formatted_response = formatter(result["content"])
                    record = {
                        "dataset_idx": item["dataset_idx"],
                        "response": formatted_response,
                        "n_tokens": result["n_tokens"],
                        "eos": result["eos"],
                    }
                    # Rolling write: one JSON record per line, flushed
                    # immediately so external tooling can `tail -f` the
                    # file while generation is in progress.
                    out_f.write(json.dumps(record) + "\n")
                    out_f.flush()
                    done += 1

                    # Early-response preview: print the first few
                    # responses verbatim (truncated) so the user can
                    # spot-check prompt shaping from slurm logs without
                    # waiting for the full run to finish.
                    if previewed < int(args.preview_first_n):
                        previewed += 1
                        snippet = formatted_response.replace("\n", " ")[:300]
                        usr_snippet = user_content.replace("\n", " ")[:200]
                        print(f"  [preview {previewed}/"
                              f"{int(args.preview_first_n)}] "
                              f"idx={item['dataset_idx']} "
                              f"n_tok={result['n_tokens']} "
                              f"eos={result['eos']}",
                              flush=True)
                        print(f"    user: {usr_snippet}", flush=True)
                        print(f"    resp: {snippet}", flush=True)
                except Exception as e:
                    errors += 1
                    print(f"  ERROR idx={item['dataset_idx']}: {e}",
                          flush=True)

                if (done + errors) % 100 == 0:
                    elapsed = time.time() - t0
                    rate = done / max(elapsed, 0.01)
                    eta = (len(todo) - done - errors) / max(rate, 0.001)
                    print(f"  [{done+errors}/{len(todo)}] done={done} "
                          f"err={errors} rate={rate:.1f}/s "
                          f"eta={eta/60:.1f}min", flush=True)

    elapsed = time.time() - t0
    print(f"\nDone: {done} generated, {errors} errors, "
          f"{elapsed:.0f}s ({done/max(elapsed,1):.1f}/s)")
    print(f"Output: {out_path}")


if __name__ == "__main__":
    main()
