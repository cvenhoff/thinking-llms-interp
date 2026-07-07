#!/usr/bin/env python3
"""Reconstruct golds for the training_mix_v1 VAL holdout so we can run the real
hybrid pipeline on it (answer-level gap recovery) as a vector-SELECTION signal.

Sources covered: hendrycks_math (boxed answer from solution) + natural_reasoning
(reference_answer). scibench/theoremqa are skipped (small, harder to map).

Emits data/trainmix_holdout_eval/eval.jsonl with rows:
  {idx (0..N-1), mix_idx (global trainmix index = 9794+val_pos), val_pos,
   question, reference_answer, source}
idx is the row's own position -> hybrid_eval matches think-cache by dataset_idx.
mix_idx lets us re-slice the cached trainmix think rollouts.
"""
import json, os, re
from datasets import load_dataset

ROOT = "/workspace-vast/constantinv/thinking-llms-interp"
VAL = f"{ROOT}/data/training_mix_v1/val.jsonl"
OUT_DIR = f"{ROOT}/data/trainmix_holdout_eval"
VAL_BASE = 9794  # global trainmix idx of first val row (train has 9794 rows)


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
    os.makedirs(OUT_DIR, exist_ok=True)
    val = [json.loads(l) for l in open(VAL)]
    print(f"val rows: {len(val)}")

    # --- hendrycks_math lookup (all configs, both splits) ---
    hmath = {}
    cfgs = ["algebra", "counting_and_probability", "geometry",
            "intermediate_algebra", "number_theory", "prealgebra", "precalculus"]
    for c in cfgs:
        for sp in ("train", "test"):
            try:
                ds = load_dataset("EleutherAI/hendrycks_math", c, split=sp)
            except Exception as e:
                print(f"  hendrycks {c}/{sp} load fail: {e}")
                continue
            for r in ds:
                g = last_boxed(r.get("solution", "")) or ""
                if g:
                    hmath[norm(r["problem"])] = g
    print(f"hendrycks lookup size: {len(hmath)}")

    # --- natural_reasoning lookup ---
    nr = {}
    ds = load_dataset("facebook/natural_reasoning", split="train")
    for r in ds:
        ra = r.get("reference_answer")
        if ra:
            nr[norm(r["question"])] = ra
    print(f"natural_reasoning lookup size: {len(nr)}")

    out = []
    miss = {"hendrycks_math": 0, "natural_reasoning": 0, "other": 0}
    for vp, v in enumerate(val):
        src = v["source"]; q = norm(v["question"]); gold = None
        if src == "hendrycks_math":
            gold = hmath.get(q)
        elif src == "natural_reasoning":
            gold = nr.get(q)
        else:
            miss["other"] += 1
            continue
        if not gold:
            miss[src] += 1
            continue
        out.append({"idx": len(out), "mix_idx": VAL_BASE + vp, "val_pos": vp,
                    "question": v["question"], "reference_answer": gold,
                    "source": src})
    with open(f"{OUT_DIR}/eval.jsonl", "w") as f:
        for r in out:
            f.write(json.dumps(r) + "\n")
    from collections import Counter
    print(f"emitted {len(out)} holdout rows -> {OUT_DIR}/eval.jsonl")
    print("by source:", dict(Counter(r["source"] for r in out)))
    print("misses:", miss)


if __name__ == "__main__":
    main()
