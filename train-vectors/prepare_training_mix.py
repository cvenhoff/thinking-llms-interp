#!/usr/bin/env python3
"""Compose a training and validation dataset from 4 HuggingFace datasets.

Sources:
  - Hendrycks MATH (EleutherAI/hendrycks_math): 5000 train / 500 val
  - NaturalReasoning (facebook/natural_reasoning): 3500 train / 350 val
  - SciBench (xw27/scibench): 90/10 split (~623 train / ~69 val)
  - TheoremQA (TIGER-Lab/TheoremQA): text-only, 90/10 split (~672 train / ~75 val)

Outputs train.jsonl, val.jsonl, and mix_config.json into --output_dir.
"""

import argparse
import json
import os
from pathlib import Path

from datasets import concatenate_datasets, load_dataset


MATH_SUBJECTS = [
    "algebra",
    "counting_and_probability",
    "geometry",
    "intermediate_algebra",
    "number_theory",
    "prealgebra",
    "precalculus",
]


def load_hendrycks_math(seed: int):
    """Load all 7 MATH subjects, concatenate, then sample train/val."""
    parts = []
    for subject in MATH_SUBJECTS:
        ds = load_dataset("EleutherAI/hendrycks_math", subject, split="train")
        parts.append(ds)
    full = concatenate_datasets(parts)
    print(f"  Hendrycks MATH total: {len(full)}")

    shuffled = full.shuffle(seed=seed)
    train_ds = shuffled.select(range(5000))
    val_ds = shuffled.select(range(5000, 5500))

    train_records = [
        {
            "question": row["problem"],
            "source": "hendrycks_math",
            "source_category": row["type"],
        }
        for row in train_ds
    ]
    val_records = [
        {
            "question": row["problem"],
            "source": "hendrycks_math",
            "source_category": row["type"],
        }
        for row in val_ds
    ]
    return train_records, val_records


def load_natural_reasoning(seed: int):
    """Load NaturalReasoning and randomly sample train/val."""
    ds = load_dataset("facebook/natural_reasoning", split="train")
    print(f"  NaturalReasoning total: {len(ds)}")

    shuffled = ds.shuffle(seed=seed)
    train_ds = shuffled.select(range(3500))
    val_ds = shuffled.select(range(3500, 3850))

    train_records = [
        {
            "question": row["question"],
            "source": "natural_reasoning",
            "source_category": "mixed",
        }
        for row in train_ds
    ]
    val_records = [
        {
            "question": row["question"],
            "source": "natural_reasoning",
            "source_category": "mixed",
        }
        for row in val_ds
    ]
    return train_records, val_records


def load_scibench(seed: int):
    """Load SciBench, 90/10 split."""
    ds = load_dataset("xw27/scibench", split="train")
    total = len(ds)
    print(f"  SciBench total: {total}")

    shuffled = ds.shuffle(seed=seed)
    n_train = int(total * 0.9)
    train_ds = shuffled.select(range(n_train))
    val_ds = shuffled.select(range(n_train, total))

    train_records = [
        {
            "question": row["problem_text"],
            "source": "scibench",
            "source_category": row["source"],
        }
        for row in train_ds
    ]
    val_records = [
        {
            "question": row["problem_text"],
            "source": "scibench",
            "source_category": row["source"],
        }
        for row in val_ds
    ]
    return train_records, val_records


def load_theoremqa(seed: int):
    """Load TheoremQA, filter out picture questions, 90/10 split."""
    ds = load_dataset("TIGER-Lab/TheoremQA", split="test")
    total_before = len(ds)
    ds = ds.filter(lambda x: x["Picture"] is None)
    total_after = len(ds)
    print(f"  TheoremQA total: {total_before}, text-only: {total_after} (filtered {total_before - total_after} with images)")

    shuffled = ds.shuffle(seed=seed)
    n_train = int(total_after * 0.9)
    train_ds = shuffled.select(range(n_train))
    val_ds = shuffled.select(range(n_train, total_after))

    train_records = [
        {
            "question": row["Question"],
            "source": "theoremqa",
            "source_category": row["Answer_type"],
        }
        for row in train_ds
    ]
    val_records = [
        {
            "question": row["Question"],
            "source": "theoremqa",
            "source_category": row["Answer_type"],
        }
        for row in val_ds
    ]
    return train_records, val_records


def write_jsonl(records: list[dict], path: Path):
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(description="Prepare training mix from 4 HF datasets")
    parser.add_argument("--output_dir", type=str, default="../data/training_mix_v1")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    seed = args.seed

    print(f"Seed: {seed}")
    print(f"Output dir: {output_dir.resolve()}\n")

    print("Loading datasets...")

    print("[1/4] Hendrycks MATH")
    math_train, math_val = load_hendrycks_math(seed)

    print("[2/4] NaturalReasoning")
    nr_train, nr_val = load_natural_reasoning(seed)

    print("[3/4] SciBench")
    sci_train, sci_val = load_scibench(seed)

    print("[4/4] TheoremQA")
    tqa_train, tqa_val = load_theoremqa(seed)

    all_train = math_train + nr_train + sci_train + tqa_train
    all_val = math_val + nr_val + sci_val + tqa_val

    for idx, rec in enumerate(all_train):
        rec["idx"] = idx
    val_offset = len(all_train)
    for idx, rec in enumerate(all_val):
        rec["idx"] = val_offset + idx

    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"
    write_jsonl(all_train, train_path)
    write_jsonl(all_val, val_path)

    source_counts_train = {}
    source_counts_val = {}
    for rec in all_train:
        source_counts_train[rec["source"]] = source_counts_train.get(rec["source"], 0) + 1
    for rec in all_val:
        source_counts_val[rec["source"]] = source_counts_val.get(rec["source"], 0) + 1

    config = {
        "seed": seed,
        "total_train": len(all_train),
        "total_val": len(all_val),
        "sources": {
            "hendrycks_math": {
                "train": source_counts_train.get("hendrycks_math", 0),
                "val": source_counts_val.get("hendrycks_math", 0),
            },
            "natural_reasoning": {
                "train": source_counts_train.get("natural_reasoning", 0),
                "val": source_counts_val.get("natural_reasoning", 0),
            },
            "scibench": {
                "train": source_counts_train.get("scibench", 0),
                "val": source_counts_val.get("scibench", 0),
            },
            "theoremqa": {
                "train": source_counts_train.get("theoremqa", 0),
                "val": source_counts_val.get("theoremqa", 0),
            },
        },
    }

    config_path = output_dir / "mix_config.json"
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)

    print(f"\n{'='*60}")
    print("Summary")
    print(f"{'='*60}")
    print(f"{'Source':<25} {'Train':>8} {'Val':>8}")
    print(f"{'-'*25} {'-'*8} {'-'*8}")
    for src in ["hendrycks_math", "natural_reasoning", "scibench", "theoremqa"]:
        t = source_counts_train.get(src, 0)
        v = source_counts_val.get(src, 0)
        print(f"{src:<25} {t:>8} {v:>8}")
    print(f"{'-'*25} {'-'*8} {'-'*8}")
    print(f"{'TOTAL':<25} {len(all_train):>8} {len(all_val):>8}")
    print(f"\nFiles written:")
    print(f"  {train_path.resolve()}")
    print(f"  {val_path.resolve()}")
    print(f"  {config_path.resolve()}")


if __name__ == "__main__":
    main()
