#!/usr/bin/env python3
"""Generate base-model rollouts using the 'qa_instr' prompt:

    Answer the following question:
    Q: {question}
    A: {model continues here}

No chat template, no math directive, no family shaping. Companion to
gen_base_qa_response.py but with an explicit "Answer the following
question:" instruction and Q:/A: labels.

Output: <output_dir>/base_qa_instr_<base_short>_<dataset>_temp<T>_max<M>.jsonl
"""

from __future__ import annotations
import argparse, json, os, sys, time
from concurrent.futures import ThreadPoolExecutor, as_completed

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(_HERE, "..", "..", "vllm-serve")))
from generate_rollouts import (  # noqa: E402
    load_dataset_questions, _temp_label, generate_one_completion,
)


def _build_qa_instr_prompt(question: str) -> str:
    return f"Answer the following question:\nQ: {question}\nA:"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_url", required=True)
    ap.add_argument("--base_model", required=True)
    ap.add_argument("--base_short", required=True)
    ap.add_argument("--dataset", required=True,
                    choices=["math500", "gsm8k", "holdoutmix",
                             "hendrycks_holdout"])
    ap.add_argument("--questions_file", default=None,
                    help="For --dataset holdoutmix/hendrycks_holdout: jsonl "
                         "with fields 'question' and 'idx' (dataset_idx).")
    ap.add_argument("--n_examples", type=int, default=20000)
    ap.add_argument("--max_tokens", type=int, default=2048)
    ap.add_argument("--temperature", type=float, default=0.0)
    ap.add_argument("--top_p", type=float, default=1.0)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--output_dir", required=True)
    ap.add_argument("--max_concurrent", type=int, default=64)
    ap.add_argument("--resume", action="store_true", default=True)
    ap.add_argument("--preview_first_n", type=int, default=3)
    args = ap.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)
    ts = _temp_label(args.temperature)
    out_path = os.path.join(
        args.output_dir,
        f"base_qa_instr_{args.base_short}_{args.dataset}_"
        f"temp{ts}_max{args.max_tokens}.jsonl")
    print(f"Output: {out_path}", flush=True)

    existing_idx = set()
    if args.resume and os.path.exists(out_path):
        with open(out_path) as f:
            for line in f:
                try:
                    existing_idx.add(int(json.loads(line)["dataset_idx"]))
                except (json.JSONDecodeError, KeyError):
                    pass
        print(f"Resume: {len(existing_idx)} existing entries", flush=True)

    if args.dataset in ("holdoutmix", "hendrycks_holdout"):
        if not args.questions_file or not os.path.exists(args.questions_file):
            raise SystemExit(f"--questions_file required for {args.dataset} "
                             f"(got {args.questions_file})")
        questions = [{"dataset_idx": int(r["idx"]), "question": r["question"]}
                     for r in (json.loads(l) for l in open(args.questions_file)
                               if l.strip())]
    else:
        questions = load_dataset_questions(args.dataset, args.n_examples)
    todo = [q for q in questions if q["dataset_idx"] not in existing_idx]
    print(f"To generate: {len(todo)}", flush=True)
    if not todo:
        print("Nothing to do.")
        return

    prompts = {int(item["dataset_idx"]): _build_qa_instr_prompt(item["question"])
               for item in todo}

    sample_idx = next(iter(prompts))
    print(f"\n=== Example prompt for idx={sample_idx} ===")
    print(prompts[sample_idx])
    print("=== End example ===\n", flush=True)

    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key="dummy")
    models = client.models.list()
    print(f"Server models: {[m.id for m in models.data]}", flush=True)

    t0 = time.time()
    done = errors = previewed = 0
    with open(out_path, "a") as out_f:
        with ThreadPoolExecutor(max_workers=args.max_concurrent) as pool:
            futures = {}
            for item in todo:
                idx = int(item["dataset_idx"])
                _seed = (args.seed + idx) if args.seed is not None else None
                fut = pool.submit(
                    generate_one_completion, client, args.base_model,
                    prompts[idx], args.max_tokens, args.temperature,
                    args.top_p, _seed)
                futures[fut] = item

            for fut in as_completed(futures):
                item = futures[fut]
                try:
                    result = fut.result()
                    record = {
                        "dataset_idx": item["dataset_idx"],
                        "response": result["content"],
                        "n_tokens": result["n_tokens"],
                        "eos": result["eos"],
                    }
                    out_f.write(json.dumps(record) + "\n")
                    out_f.flush()
                    done += 1
                    if previewed < args.preview_first_n:
                        previewed += 1
                        snip = result["content"].replace("\n", " ")[:300]
                        print(f"  [preview {previewed}/{args.preview_first_n}] "
                              f"idx={item['dataset_idx']} n_tok={result['n_tokens']} "
                              f"eos={result['eos']}", flush=True)
                        print(f"    resp: {snip}", flush=True)
                except Exception as e:
                    errors += 1
                    print(f"  ERROR idx={item['dataset_idx']}: {e}", flush=True)
                if (done + errors) % 100 == 0:
                    rate = done / max(time.time() - t0, 0.01)
                    eta = (len(todo) - done - errors) / max(rate, 0.001)
                    print(f"  [{done+errors}/{len(todo)}] done={done} err={errors} "
                          f"rate={rate:.1f}/s eta={eta/60:.1f}min", flush=True)

    print(f"\nDone: {done} generated, {errors} errors, "
          f"{(time.time()-t0):.0f}s ({done/max(time.time()-t0,1):.1f}/s)",
          flush=True)
    print(f"Output: {out_path}", flush=True)


if __name__ == "__main__":
    main()
