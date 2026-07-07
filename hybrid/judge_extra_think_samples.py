#!/usr/bin/env python3
"""Judge the *extra* think-model rollouts (sample_idx 1 and 2) that the
main hybrid_eval.py invocation does not see.

hybrid_eval.py is run **once** per (cfg, dataset) with the s0 think
rollout.  It produces ``judge_reps_<base>_<ds>_final.json`` covering
think_s0 / base / hybrid.  For the temperature-noise quantification we
need think_s1 and think_s2 judged the same way; this script handles
that, reading the additional cache files and writing
``judge_reps_extra_think_<base>_<ds>_final.json``.

Filename conventions assumed (matches the final-run slug):
  hybrid/results/response_cache_final/
    thinking_<short>_<ds>_temp0.6_max2048_s{0,1,2}.jsonl

Usage:
    python judge_extra_think_samples.py \
        --cache_dir hybrid/results/response_cache_final \
        --think_short open-reasoner-zero-32b \
        --base_id qwen2.5-32b \
        --dataset math500 \
        --temp_label 0.6 --max_tokens 2048 \
        --sample_ids 1,2 \
        --judge_repetitions 3 \
        --out_dir artifacts/mlp_eval_qa_instr_holdoutsel_h512/orz-32b
"""

from __future__ import annotations
import argparse
import json
import os
import re
import sys
from typing import Dict, List

# Reuse the prompt + safe-batch infrastructure already used by hybrid_eval.
_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(_HERE, "..", "..", "hybrid")))
from hybrid_eval import (   # noqa: E402
    _build_judge_prompt, judge_batch, MCQA_DATASETS,
    TEXT_CLASSIFICATION_DATASETS, CODING_DATASETS,
)


def _dataset_type(name: str) -> str:
    if name in CODING_DATASETS:
        return "coding"
    if name in MCQA_DATASETS:
        return "mcqa"
    if name in TEXT_CLASSIFICATION_DATASETS:
        return "classification"
    return "math"


def _load_examples(dataset: str, gold_file: str = None) -> List[dict]:
    """Mirror the dataset-loading code in hybrid_eval._build_task_prompts."""
    from datasets import load_dataset
    if dataset in ("natreason", "hendrycks_holdout"):
        if not gold_file or not os.path.exists(gold_file):
            raise ValueError(f"--gold_file required for {dataset} extra-think "
                             f"judging (got {gold_file})")
        out = []
        for line in open(gold_file):
            if not line.strip():
                continue
            r = json.loads(line)
            out.append({"question": r["question"],
                        "gold": str(r.get("reference_answer", ""))})
        return out
    if dataset == "math500":
        ds = load_dataset("HuggingFaceH4/MATH-500")["test"]
        out = [{"question": r["problem"], "gold": r["answer"]} for r in ds]
    elif dataset == "gsm8k":
        ds = load_dataset("openai/gsm8k", "main")["test"]
        # gsm8k golds are "<reasoning>\n#### <num>"; strip to numeric answer.
        def _ans(a: str) -> str:
            m = re.search(r"####\s*(.+)", a)
            return m.group(1).strip() if m else a.strip()
        out = [{"question": r["question"], "gold": _ans(r["answer"])} for r in ds]
    else:
        raise ValueError(f"Unsupported dataset for extra-think judging: {dataset}")
    return out


def _think_cache_path(cache_dir: str, think_short: str, dataset: str,
                      temp_label: str, max_tokens: int, sample_idx: int) -> str:
    s = f"_s{sample_idx}" if sample_idx >= 0 else ""
    return os.path.join(
        cache_dir,
        f"thinking_{think_short}_{dataset}_temp{temp_label}_max{max_tokens}{s}.jsonl")


def _load_think_responses(path: str) -> Dict[int, str]:
    out: Dict[int, str] = {}
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                e = json.loads(line)
            except json.JSONDecodeError:
                continue
            out[int(e["dataset_idx"])] = e["response"]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cache_dir", required=True,
                   help="response_cache_final directory")
    p.add_argument("--think_short", required=True,
                   help="Model short name used in cache filenames "
                        "(e.g. open-reasoner-zero-32b).")
    p.add_argument("--base_id", required=True,
                   help="Base model short name (used only to compose "
                        "the output filename so it pairs with hybrid_eval).")
    p.add_argument("--dataset", required=True,
                   choices=["math500", "gsm8k", "natreason",
                            "hendrycks_holdout"])
    p.add_argument("--gold_file", default=None,
                   help="Gold jsonl for natreason (fields: question, "
                        "reference_answer).")
    p.add_argument("--temp_label", default="0.6")
    p.add_argument("--max_tokens", type=int, default=2048)
    p.add_argument("--sample_ids", type=str, default="1,2",
                   help="Comma-separated sample indices to judge.")
    p.add_argument("--judge_repetitions", type=int, default=3)
    p.add_argument("--judge_model", default="openai/gpt-5.2")
    p.add_argument("--max_concurrent", type=int, default=40)
    p.add_argument("--out_dir", required=True,
                   help="Where judge_reps_extra_think_<base>_<ds>_final.json "
                        "is written.")
    p.add_argument("--results_suffix", default="final",
                   help="Substring appended to the output filename, after "
                        "'_final' / the dataset slug.")
    args = p.parse_args()

    examples = _load_examples(args.dataset, args.gold_file)
    ds_type = _dataset_type(args.dataset)
    print(f"[extra-judge] loaded {len(examples)} {args.dataset} examples",
          flush=True)

    sample_ids = [int(s) for s in args.sample_ids.split(",") if s.strip()]

    per_sample: Dict[int, List[dict]] = {}
    for s in sample_ids:
        cp = _think_cache_path(args.cache_dir, args.think_short,
                               args.dataset, args.temp_label,
                               args.max_tokens, s)
        if not os.path.exists(cp):
            print(f"[extra-judge] FATAL: missing cache file {cp}")
            sys.exit(2)
        resps = _load_think_responses(cp)
        items = []
        for idx, ex in enumerate(examples):
            r = resps.get(idx, "")
            if not r:
                items.append(None)
                continue
            items.append(dict(
                answer=re.sub(r"\s+", " ", r).strip(),
                gold=ex["gold"], question=ex["question"],
                ds_type=ds_type, test_list=None,
                label=f"think_s{s}_idx{idx}"))
        per_sample[s] = items
        print(f"[extra-judge] sample {s}: cache={cp}  "
              f"loaded={sum(1 for it in items if it is not None)}/"
              f"{len(examples)}", flush=True)

    out: Dict[str, dict] = {
        "dataset": args.dataset,
        "think_short": args.think_short,
        "base_id": args.base_id,
        "temp_label": args.temp_label,
        "max_tokens": args.max_tokens,
        "judge_repetitions": args.judge_repetitions,
        "judge_model": args.judge_model,
        "per_sample": {},
    }

    for s, items in per_sample.items():
        active = [(i, it) for i, it in enumerate(items) if it is not None]
        if not active:
            print(f"[extra-judge] sample {s}: no items to judge, skipping")
            continue
        flat = [it for _, it in active]
        print(f"\n[extra-judge] judging sample {s} ({len(flat)} items, "
              f"n_reps={args.judge_repetitions})...", flush=True)
        jr = judge_batch(flat, args.judge_model,
                         n_reps=args.judge_repetitions,
                         max_concurrent=args.max_concurrent)

        # jr is aligned with flat (active items). Spread back to full size.
        per_item = [None] * len(items)
        for (orig_i, _), verdict in zip(active, jr):
            per_item[orig_i] = verdict

        # Per-rep tallies and means.
        n_reps = args.judge_repetitions
        per_rep_correct = [0] * n_reps
        total_active = 0
        for v in per_item:
            if v is None:
                continue
            total_active += 1
            reps = v.get("repetitions") or []
            for i, r in enumerate(reps[:n_reps]):
                if r.get("correct"):
                    per_rep_correct[i] += 1

        out["per_sample"][str(s)] = {
            "n_items": len(items),
            "n_active": total_active,
            "per_rep_correct": per_rep_correct,
            "per_rep_acc_pct": [
                100.0 * c / max(total_active, 1) for c in per_rep_correct
            ],
            "verdicts": per_item,
        }

        print(f"[extra-judge] sample {s}: "
              + "  ".join(f"r{i}={100.0*c/max(total_active,1):.2f}%"
                          for i, c in enumerate(per_rep_correct)),
              flush=True)

    os.makedirs(args.out_dir, exist_ok=True)
    suffix = (f"_{args.results_suffix}"
              if args.results_suffix and not args.results_suffix.startswith("_")
              else (args.results_suffix or ""))
    out_path = os.path.join(
        args.out_dir,
        f"judge_reps_extra_think_{args.base_id}_{args.dataset}{suffix}.json")
    with open(out_path, "w") as f:
        json.dump(out, f, indent=2)
    print(f"\n[extra-judge] wrote -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
