#!/usr/bin/env python3
"""Build a ~1000-question hendrycks-MATH holdout eval set that is DISJOINT from
all data used so far:
  - training_mix_v1 train.jsonl + val.jsonl (the hendrycks_math rows used for
    vector training / selection)
  - math500 (HuggingFaceH4/MATH-500, our OOD math benchmark)

Pool = full hendrycks MATH (EleutherAI/hendrycks_math, all 7 subjects, both
splits = 12500 problems) with a recoverable \\boxed{} gold, minus the exclusion
sets (matched on normalized problem text). Deterministically shuffled (seed 0)
and truncated to N.

Emits data/hendrycks_holdout_eval/eval.jsonl:
  {idx, question, reference_answer, source="hendrycks_math", subject, level}
`idx` is the row position -> think/base caches key on dataset_idx.
"""
import argparse, json, os, random
from collections import Counter
from datasets import load_dataset

ROOT = "/workspace-vast/constantinv/thinking-llms-interp"
OUT_DIR = f"{ROOT}/data/hendrycks_holdout_eval"
TRAIN = f"{ROOT}/data/training_mix_v1/train.jsonl"
VAL = f"{ROOT}/data/training_mix_v1/val.jsonl"
CFGS = ["algebra", "counting_and_probability", "geometry",
        "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]


def norm(s):
    return " ".join(str(s).split()).strip()


def last_boxed(sol):
    i = sol.rfind("\\boxed")
    if i < 0:
        return None
    j = sol.find("{", i)
    if j < 0:
        return None
    depth = 0
    for k in range(j, len(sol)):
        if sol[k] == "{":
            depth += 1
        elif sol[k] == "}":
            depth -= 1
            if depth == 0:
                return sol[j + 1:k].strip()
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=1000)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    os.makedirs(OUT_DIR, exist_ok=True)

    # --- exclusion sets (normalized problem text) ---
    excl = set()
    n_hm_train = 0
    for split_file in (TRAIN, VAL):
        for line in open(split_file):
            r = json.loads(line)
            if r.get("source") == "hendrycks_math":
                excl.add(norm(r["question"]))
                n_hm_train += 1
    print(f"train+val hendrycks_math rows excluded: {n_hm_train} "
          f"(unique norm: {len(excl)})")
    m500 = load_dataset("HuggingFaceH4/MATH-500")["test"]
    n_m500 = 0
    for r in m500:
        excl.add(norm(r["problem"]))
        n_m500 += 1
    print(f"math500 rows excluded: {n_m500}; total exclusion set: {len(excl)}")

    # --- candidate pool from full hendrycks MATH ---
    pool = {}  # norm_q -> record
    for c in CFGS:
        for sp in ("train", "test"):
            try:
                ds = load_dataset("EleutherAI/hendrycks_math", c, split=sp)
            except Exception as e:
                print(f"  load fail {c}/{sp}: {e}")
                continue
            for r in ds:
                q = r["problem"]
                nq = norm(q)
                if nq in excl or nq in pool:
                    continue
                gold = last_boxed(r.get("solution", "")) or ""
                if not gold:
                    continue
                pool[nq] = {"question": q, "reference_answer": gold,
                            "source": "hendrycks_math",
                            "subject": r.get("type", c),
                            "level": r.get("level", "")}
    print(f"eligible pool (disjoint, boxed gold): {len(pool)}")

    recs = list(pool.values())
    rng = random.Random(args.seed)
    rng.shuffle(recs)
    recs = recs[:args.n]
    for i, r in enumerate(recs):
        r["idx"] = i

    with open(f"{OUT_DIR}/eval.jsonl", "w") as f:
        for r in recs:
            f.write(json.dumps({"idx": r["idx"], "question": r["question"],
                                "reference_answer": r["reference_answer"],
                                "source": r["source"], "subject": r["subject"],
                                "level": r["level"]}) + "\n")
    print(f"emitted {len(recs)} rows -> {OUT_DIR}/eval.jsonl")
    print("by subject:", dict(Counter(r["subject"] for r in recs)))
    print("by level:", dict(Counter(str(r["level"]) for r in recs)))
    # sanity: assert disjoint
    bad = sum(1 for r in recs if norm(r["question"]) in excl)
    print(f"overlap with exclusion (should be 0): {bad}")


if __name__ == "__main__":
    main()
