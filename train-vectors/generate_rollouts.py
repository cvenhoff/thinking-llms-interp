"""Batched greedy thinking-model rollout generator for arbitrary HF datasets.

Writes one JSON record per line to ``--output``:

    {"dataset_idx": int, "response": str, "n_tokens": int, "eos": bool}

The output format is compatible with ``_load_oos_responses`` in
``optimize_correction_vectors.py`` (cache root: any directory, file name:
``thinking_<short>_<dataset_name>_temp0_max2000.jsonl``).

Resumable: indices already present in the output file are skipped on a
subsequent run.  ``--start`` / ``--end`` let multiple processes shard the
work across GPUs / nodes; each process appends to its own output file
(or to a shared file -- writes are flushed line-by-line and resume
support deduplicates by dataset_idx on load).

We mirror the prompt construction in ``_load_oos_responses``:

    text = tokenizer.apply_chat_template(
        [{"role": "user", "content": question}],
        tokenize=False, add_generation_prompt=True,
    )

so the rollouts produced here are byte-identical to what
``optimize_correction_vectors.py`` would produce on cache miss.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

import torch
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer


DTYPE_MAP = {
    "bfloat16": torch.bfloat16,
    "float16": torch.float16,
    "float32": torch.float32,
}


DATASET_SPECS = {
    # name: (hf_id, config_name_or_None, split, question_column)
    "math500":              ("HuggingFaceH4/MATH-500",         None,             "test",            "problem"),
    "gsm8k":                ("openai/gsm8k",                   "main",           "test",            "question"),
    "hendrycks_math":       ("nlile/hendrycks-MATH-benchmark", None,             "train",           "problem"),
    "mmlu_auxiliary_train": ("cais/mmlu",                      "all",            "auxiliary_train", "question"),
}


def load_done_indices(path: str) -> set[int]:
    done: set[int] = set()
    if not os.path.exists(path):
        return done
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                r = json.loads(line)
                # Only count records that contain a non-empty response.
                if r.get("response") and r["response"].strip():
                    done.add(int(r["dataset_idx"]))
            except Exception:
                continue
    return done


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True,
                   help="HF model id, e.g. Open-Reasoner-Zero/Open-Reasoner-Zero-7B")
    p.add_argument("--output", required=True,
                   help="JSONL output path (appended; resumable).")
    p.add_argument("--dataset", default="hendrycks_math",
                   choices=list(DATASET_SPECS.keys()),
                   help="One of the registered datasets.")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--max_new_tokens", type=int, default=2000)
    p.add_argument("--max_prompt_tokens", type=int, default=512)
    p.add_argument("--start", type=int, default=0,
                   help="First dataset index (inclusive).")
    p.add_argument("--end", type=int, default=-1,
                   help="One past last dataset index (exclusive); "
                        "-1 = full split length.")
    p.add_argument("--dtype", default="bfloat16",
                   choices=list(DTYPE_MAP.keys()))
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--log_every", type=int, default=10,
                   help="Print a progress line every N batches.")
    args = p.parse_args()

    hf_id, cfg, split, q_key = DATASET_SPECS[args.dataset]
    print(f"[gen] dataset={args.dataset} ({hf_id}, cfg={cfg}, "
          f"split={split}, q_key={q_key})", flush=True)
    print(f"[gen] model={args.model}  dtype={args.dtype}  "
          f"batch_size={args.batch_size}  max_new_tokens={args.max_new_tokens}",
          flush=True)

    if cfg is not None:
        ds = load_dataset(hf_id, cfg, split=split)
    else:
        ds = load_dataset(hf_id, split=split)
    n_total = len(ds)
    end = args.end if args.end > 0 else n_total
    start = max(0, args.start)
    if end > n_total:
        print(f"[gen] clipping --end {end} -> {n_total} (split size)",
              flush=True)
        end = n_total
    if start >= end:
        print(f"[gen] empty shard [{start}, {end}); nothing to do.", flush=True)
        return 0

    done = load_done_indices(args.output)
    if done:
        print(f"[gen] resume: {len(done)} indices already in {args.output}",
              flush=True)
    todo = [i for i in range(start, end) if i not in done]
    if not todo:
        print(f"[gen] all {end - start} indices in [{start}, {end}) already "
              f"generated.", flush=True)
        return 0
    print(f"[gen] to generate: {len(todo)} indices in [{start}, {end})",
          flush=True)

    tok = AutoTokenizer.from_pretrained(args.model)
    if tok.pad_token_id is None:
        tok.pad_token_id = tok.eos_token_id
    tok.padding_side = "left"

    dtype = DTYPE_MAP[args.dtype]
    print(f"[gen] loading model {args.model}...", flush=True)
    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(
        args.model, torch_dtype=dtype, device_map=args.device)
    model.eval()
    print(f"[gen] model loaded in {time.time() - t0:.1f}s", flush=True)

    eos_id = tok.eos_token_id
    pad_id = tok.pad_token_id

    n_done = 0
    n_eos = 0
    n_truncated = 0
    n_tokens_total = 0
    t_loop_start = time.time()

    with open(args.output, "a") as out_f:
        for batch_start in range(0, len(todo), args.batch_size):
            batch_idx = todo[batch_start: batch_start + args.batch_size]
            prompts: list[str] = []
            for didx in batch_idx:
                q = ds[didx][q_key]
                try:
                    p_text = tok.apply_chat_template(
                        [{"role": "user", "content": q}],
                        tokenize=False, add_generation_prompt=True,
                    )
                except Exception:
                    p_text = f"User: {q}\nAssistant:"
                prompts.append(p_text)

            enc = tok(
                prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=args.max_prompt_tokens,
            )
            input_ids = enc["input_ids"].to(args.device)
            attn = enc["attention_mask"].to(args.device)
            with torch.no_grad():
                gen = model.generate(
                    input_ids=input_ids,
                    attention_mask=attn,
                    max_new_tokens=args.max_new_tokens,
                    do_sample=False,
                    temperature=1.0,
                    pad_token_id=pad_id,
                    eos_token_id=eos_id,
                )

            prompt_len = input_ids.shape[1]
            for j, didx in enumerate(batch_idx):
                new_ids = gen[j, prompt_len:]
                eos_pos = (new_ids == eos_id).nonzero(as_tuple=True)[0]
                if len(eos_pos) > 0:
                    cut = int(eos_pos[0].item())
                    new_ids = new_ids[:cut]
                    ended_eos = True
                    n_eos += 1
                else:
                    ended_eos = False
                    n_truncated += 1
                resp = tok.decode(new_ids, skip_special_tokens=True)
                n_tok = int(new_ids.shape[0])
                n_tokens_total += n_tok
                rec = {
                    "dataset_idx": int(didx),
                    "response": resp,
                    "n_tokens": n_tok,
                    "eos": ended_eos,
                }
                out_f.write(json.dumps(rec) + "\n")
            out_f.flush()
            n_done += len(batch_idx)

            if (batch_start // args.batch_size) % args.log_every == 0:
                elapsed = time.time() - t_loop_start
                rate = n_done / max(elapsed, 1e-6)
                remaining = len(todo) - n_done
                eta = remaining / max(rate, 1e-6)
                avg_tok = n_tokens_total / max(n_done, 1)
                print(
                    f"[gen] {n_done}/{len(todo)} done "
                    f"({rate:.2f} ex/s, ~{avg_tok:.0f} new tok/ex, "
                    f"ETA {eta/60:.1f} min)  "
                    f"eos={n_eos} trunc={n_truncated}",
                    flush=True,
                )

    elapsed = time.time() - t_loop_start
    print(f"[gen] DONE  {n_done} generated in {elapsed/60:.1f} min  "
          f"(eos={n_eos}, truncated_at_max={n_truncated}, "
          f"avg_tok={n_tokens_total / max(n_done, 1):.0f})", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
