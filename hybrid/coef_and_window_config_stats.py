"""Analyze guardrail config usage from hybrid generation JSONL results.

Reads per-token details from rolling JSONL files and reports:
  - Joint (coefficient, window) selection percentages
  - Marginal distribution over coefficients
  - Marginal distribution over window sizes

Usage:
    uv run python hybrid/guardrail_config_stats.py <jsonl_file_or_glob> [--top N]

Examples:
    uv run python hybrid/guardrail_config_stats.py hybrid/results/rolling/rolling_qwen2.5-32b_math500.jsonl
    uv run python hybrid/guardrail_config_stats.py 'hybrid/results/rolling/rolling_qwen2.5-32b_math500*.jsonl'
"""
import argparse
import glob
import json
import sys
from collections import Counter


def load_steered_tokens(paths):
    """Yield (coefficient, window) for every steered token across all files."""
    for path in paths:
        with open(path) as f:
            for line in f:
                rec = json.loads(line)
                per_token = rec.get("hybrid_details", {}).get("per_token", [])
                for tok in per_token:
                    if tok.get("selection") != "steered":
                        continue
                    coef = tok.get("coefficient")
                    win = tok.get("window")
                    if coef is not None and win is not None:
                        yield (float(coef), int(win))


def print_table(title, counts, total):
    print(f"\n{title}")
    print("-" * len(title))
    for key, count in counts.most_common():
        pct = 100.0 * count / total if total else 0
        print(f"  {key!s:<30s}  {count:>8d}  ({pct:5.1f}%)")
    print(f"  {'TOTAL':<30s}  {total:>8d}")


def main():
    parser = argparse.ArgumentParser(description="Guardrail config usage stats")
    parser.add_argument("files", type=str, help="JSONL file path or glob pattern")
    parser.add_argument("--top", type=int, default=0, help="Show only top N entries per table (0 = all)")
    args = parser.parse_args()

    paths = sorted(glob.glob(args.files))
    if not paths:
        print(f"No files matched: {args.files}", file=sys.stderr)
        sys.exit(1)
    print(f"Reading {len(paths)} file(s):")
    for p in paths:
        print(f"  {p}")

    joint = Counter()
    coef_marginal = Counter()
    window_marginal = Counter()
    total = 0

    for coef, win in load_steered_tokens(paths):
        joint[(coef, win)] += 1
        coef_marginal[coef] += 1
        window_marginal[win] += 1
        total += 1

    if total == 0:
        print("\nNo steered tokens with (coefficient, window) data found.")
        sys.exit(0)

    # Joint
    title = f"Joint (coefficient, window) distribution  [n={total}]"
    print(f"\n{title}")
    print("-" * len(title))
    items = joint.most_common(args.top if args.top else None)
    for (c, w), count in items:
        pct = 100.0 * count / total
        print(f"  coef={c:<5g}  window={w:<6d}  {count:>8d}  ({pct:5.1f}%)")
    print(f"  {'TOTAL':<30s}  {total:>8d}")

    # Marginal by coefficient
    title = f"Marginal by coefficient  [n={total}]"
    print(f"\n{title}")
    print("-" * len(title))
    items = sorted(coef_marginal.items(), key=lambda x: -x[1])
    if args.top:
        items = items[:args.top]
    for c, count in items:
        pct = 100.0 * count / total
        print(f"  coef={c:<5g}  {count:>8d}  ({pct:5.1f}%)")
    print(f"  {'TOTAL':<30s}  {total:>8d}")

    # Marginal by window
    title = f"Marginal by window  [n={total}]"
    print(f"\n{title}")
    print("-" * len(title))
    items = sorted(window_marginal.items(), key=lambda x: -x[1])
    if args.top:
        items = items[:args.top]
    for w, count in items:
        pct = 100.0 * count / total
        print(f"  window={w:<6d}  {count:>8d}  ({pct:5.1f}%)")
    print(f"  {'TOTAL':<30s}  {total:>8d}")


if __name__ == "__main__":
    main()
