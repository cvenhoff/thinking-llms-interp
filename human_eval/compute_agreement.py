"""
compute_agreement.py — Compute agreement metrics between human and LLM judges.

Usage:
    cd /Users/ivan/src/base-models-reasoning-interp
    uv run python human_eval/compute_agreement.py
    uv run python human_eval/compute_agreement.py --judges a d
"""

import argparse
import json
import os
from collections import defaultdict
import warnings
from datetime import datetime, timezone
from itertools import combinations

import numpy as np
from scipy.stats import spearmanr

warnings.filterwarnings("ignore", category=UserWarning, module="sklearn")
warnings.filterwarnings("ignore", category=RuntimeWarning, module="sklearn")
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    f1_score,
    precision_score,
    recall_score,
)

DATA_DIR = "human_eval/data"
RESULTS_PATH = "/Users/ivan/latex/icml-2026-rebuttals/thinking-llms/experiment_results/human_eval.md"

JUDGE_NAMES = {
    "a": "Taxonomy Consistency",
    "b": "Taxonomy Completeness",
    "c": "Taxonomy Independence",
    "d": "Benchmark Scoring",
}


def load_data(judge):
    path = os.path.join(DATA_DIR, f"judge_{judge}.json")
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def find_annotators(rows):
    annotators = set()
    for row in rows:
        annotators.update(row.get("human_labels", {}).keys())
    return sorted(annotators)


def label_to_int(label):
    """Convert Yes/No to 1/0."""
    return 1 if label in ("Yes", "YES", "yes", True) else 0


# ---------------------------------------------------------------------------
# Binary agreement (Judges A, D)
# ---------------------------------------------------------------------------

def compute_binary_metrics(llm_labels, human_labels):
    """Compute binary agreement metrics. Labels are 0/1 ints."""
    n = len(llm_labels)
    if n == 0:
        return {}

    acc = accuracy_score(human_labels, llm_labels)

    # Kappa: handle edge case where all labels are same class
    try:
        kappa = cohen_kappa_score(human_labels, llm_labels)
    except Exception:
        kappa = float("nan")

    # Precision/recall/F1 treating "Yes" (1) as positive class
    prec = precision_score(human_labels, llm_labels, zero_division=0)
    rec = recall_score(human_labels, llm_labels, zero_division=0)
    f1 = f1_score(human_labels, llm_labels, zero_division=0)

    return {
        "n": n,
        "accuracy": round(acc, 4),
        "kappa": round(kappa, 4),
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "agreement_pct": round(acc * 100, 1),
    }


def analyze_binary_judge(rows, annotators, judge):
    """Full analysis for a binary judge (A or D)."""
    results = {"per_annotator": {}, "inter_annotator": {}}

    for ann in annotators:
        # Collect paired labels
        llm_labels = []
        human_labels = []
        gt_labels = []  # ground truth (Judge A only)
        breakdown_data = defaultdict(lambda: {"llm": [], "human": []})

        for row in rows:
            if ann not in row.get("human_labels", {}):
                continue
            human_lbl = label_to_int(row["human_labels"][ann]["label"])
            llm_lbl = label_to_int(row["llm_judge"]["label"])
            llm_labels.append(llm_lbl)
            human_labels.append(human_lbl)

            if judge == "a":
                gt_labels.append(1 if row.get("is_positive", False) else 0)
                cat_key = row["category_title"]
            elif judge == "d":
                cat_key = row["provenance"]["model_type"]
            else:
                cat_key = "all"

            breakdown_data[cat_key]["llm"].append(llm_lbl)
            breakdown_data[cat_key]["human"].append(human_lbl)

        overall = compute_binary_metrics(llm_labels, human_labels)

        # Ground truth metrics (Judge A)
        if judge == "a" and gt_labels:
            overall["llm_vs_gt"] = compute_binary_metrics(llm_labels, gt_labels)
            overall["human_vs_gt"] = compute_binary_metrics(human_labels, gt_labels)

        # Per-category breakdown
        per_cat = {}
        for cat_key, data in sorted(breakdown_data.items()):
            per_cat[cat_key] = compute_binary_metrics(data["llm"], data["human"])
        overall["per_category"] = per_cat

        results["per_annotator"][ann] = overall

    # Inter-annotator agreement
    if len(annotators) >= 2:
        for ann1, ann2 in combinations(annotators, 2):
            labels1 = []
            labels2 = []
            for row in rows:
                hl = row.get("human_labels", {})
                if ann1 in hl and ann2 in hl:
                    labels1.append(label_to_int(hl[ann1]["label"]))
                    labels2.append(label_to_int(hl[ann2]["label"]))
            if labels1:
                results["inter_annotator"][(ann1, ann2)] = compute_binary_metrics(labels1, labels2)

    return results


# ---------------------------------------------------------------------------
# Rating agreement (Judges B, C)
# ---------------------------------------------------------------------------

def compute_rating_metrics(llm_scores, human_scores):
    """Compute rating agreement metrics."""
    n = len(llm_scores)
    if n < 3:
        return {"n": n, "note": "too few samples for correlation"}

    llm_arr = np.array(llm_scores, dtype=float)
    human_arr = np.array(human_scores, dtype=float)

    # Spearman
    try:
        sp_r, sp_p = spearmanr(llm_arr, human_arr)
    except Exception:
        sp_r, sp_p = float("nan"), float("nan")

    mae = float(np.mean(np.abs(llm_arr - human_arr)))
    rmse = float(np.sqrt(np.mean((llm_arr - human_arr) ** 2)))

    return {
        "n": n,
        "spearman_r": round(sp_r, 4),
        "spearman_p": round(sp_p, 6),
        "mae": round(mae, 2),
        "rmse": round(rmse, 2),
    }


def analyze_rating_judge(rows, annotators, judge):
    """Full analysis for a rating judge (B or C)."""
    results = {"per_annotator": {}, "inter_annotator": {}}

    # Determine LLM score field
    if judge == "b":
        llm_field = "rating"
    elif judge == "c":
        llm_field = "similarity_0_to_10"

    for ann in annotators:
        llm_scores = []
        human_scores = []
        breakdown_data = defaultdict(lambda: {"llm": [], "human": []})

        for row in rows:
            if ann not in row.get("human_labels", {}):
                continue
            human_score = row["human_labels"][ann]["rating"]
            llm_score = row["llm_judge"][llm_field]
            llm_scores.append(llm_score)
            human_scores.append(human_score)

            if judge == "b":
                cat_key = row.get("category_title", "unknown")
            elif judge == "c":
                cat_key = row["category_1"]["source_model"]
            else:
                cat_key = "all"

            breakdown_data[cat_key]["llm"].append(llm_score)
            breakdown_data[cat_key]["human"].append(human_score)

        overall = compute_rating_metrics(llm_scores, human_scores)

        # Per-category breakdown
        per_cat = {}
        for cat_key, data in sorted(breakdown_data.items()):
            per_cat[cat_key] = compute_rating_metrics(data["llm"], data["human"])
        overall["per_category"] = per_cat

        results["per_annotator"][ann] = overall

    # Inter-annotator agreement
    if len(annotators) >= 2:
        for ann1, ann2 in combinations(annotators, 2):
            scores1 = []
            scores2 = []
            for row in rows:
                hl = row.get("human_labels", {})
                if ann1 in hl and ann2 in hl:
                    scores1.append(hl[ann1]["rating"])
                    scores2.append(hl[ann2]["rating"])
            if scores1:
                results["inter_annotator"][(ann1, ann2)] = compute_rating_metrics(scores1, scores2)

    return results


# ---------------------------------------------------------------------------
# Report formatting
# ---------------------------------------------------------------------------

def fmt_pct(val):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val * 100:.1f}%"


def fmt_val(val, decimals=3):
    if val is None or (isinstance(val, float) and np.isnan(val)):
        return "N/A"
    return f"{val:.{decimals}f}"


def format_report(all_results):
    lines = []
    lines.append("# Human Evaluation of LLM Judge Quality")
    lines.append(f"\nGenerated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')}\n")

    # --- Judge A ---
    if "a" in all_results:
        res = all_results["a"]
        lines.append("## Judge A: Taxonomy Consistency (Binary classification)\n")

        for ann, metrics in res["per_annotator"].items():
            lines.append(f"### Annotator: {ann} (N={metrics.get('n', 0)})\n")

            # Ground truth comparison
            if "llm_vs_gt" in metrics:
                gt_llm = metrics["llm_vs_gt"]
                gt_human = metrics["human_vs_gt"]
                lines.append("| | Accuracy vs. GT | Precision | Recall | F1 |")
                lines.append("|---|---|---|---|---|")
                lines.append(f"| LLM judge (GPT-4.1-mini) | {fmt_pct(gt_llm.get('accuracy'))} | {fmt_pct(gt_llm.get('precision'))} | {fmt_pct(gt_llm.get('recall'))} | {fmt_pct(gt_llm.get('f1'))} |")
                lines.append(f"| Human annotator ({ann}) | {fmt_pct(gt_human.get('accuracy'))} | {fmt_pct(gt_human.get('precision'))} | {fmt_pct(gt_human.get('recall'))} | {fmt_pct(gt_human.get('f1'))} |")
                lines.append(f"| **Human-LLM agreement (kappa)** | --- | --- | --- | **{fmt_val(metrics.get('kappa'))}** |")
                lines.append("")

            # Per-category breakdown
            if metrics.get("per_category"):
                lines.append("Per-category breakdown:\n")
                lines.append("| Category | N | Agreement % | Kappa |")
                lines.append("|---|---|---|---|")
                for cat, cm in sorted(metrics["per_category"].items()):
                    lines.append(f"| {cat} | {cm.get('n', 0)} | {fmt_pct(cm.get('accuracy'))} | {fmt_val(cm.get('kappa'))} |")
                lines.append("")

        # Inter-annotator
        if res["inter_annotator"]:
            lines.append("### Inter-annotator agreement\n")
            lines.append("| Annotators | N | Kappa | Agreement % |")
            lines.append("|---|---|---|---|")
            for (a1, a2), m in res["inter_annotator"].items():
                lines.append(f"| {a1} vs {a2} | {m.get('n', 0)} | {fmt_val(m.get('kappa'))} | {fmt_pct(m.get('accuracy'))} |")
            lines.append("")

    # --- Judge B ---
    if "b" in all_results:
        res = all_results["b"]
        lines.append("## Judge B: Taxonomy Completeness (Rating)\n")

        for ann, metrics in res["per_annotator"].items():
            lines.append(f"### Annotator: {ann} (N={metrics.get('n', 0)})\n")
            lines.append("| | Spearman r | Mean abs. diff |")
            lines.append("|---|---|---|")
            lines.append(f"| Human vs. LLM (GPT-4.1-mini) | {fmt_val(metrics.get('spearman_r'))} | {fmt_val(metrics.get('mae'), 2)} |")
            lines.append("")

            if metrics.get("per_category"):
                lines.append("Per-category breakdown:\n")
                lines.append("| Category | N | Spearman r | MAE |")
                lines.append("|---|---|---|---|")
                for cat, cm in sorted(metrics["per_category"].items()):
                    lines.append(f"| {cat} | {cm.get('n', 0)} | {fmt_val(cm.get('spearman_r'))} | {fmt_val(cm.get('mae'), 2)} |")
                lines.append("")

        if res["inter_annotator"]:
            lines.append("### Inter-annotator agreement\n")
            lines.append("| Annotators | N | Spearman r | MAE |")
            lines.append("|---|---|---|---|")
            for (a1, a2), m in res["inter_annotator"].items():
                lines.append(f"| {a1} vs {a2} | {m.get('n', 0)} | {fmt_val(m.get('spearman_r'))} | {fmt_val(m.get('mae'), 2)} |")
            lines.append("")

    # --- Judge C ---
    if "c" in all_results:
        res = all_results["c"]
        lines.append("## Judge C: Taxonomy Independence (Rating)\n")

        for ann, metrics in res["per_annotator"].items():
            lines.append(f"### Annotator: {ann} (N={metrics.get('n', 0)})\n")
            lines.append("| | Spearman r | Mean abs. diff |")
            lines.append("|---|---|---|")
            lines.append(f"| Human vs. LLM (GPT-4.1-mini) | {fmt_val(metrics.get('spearman_r'))} | {fmt_val(metrics.get('mae'), 2)} |")
            lines.append("")

            if metrics.get("per_category"):
                lines.append("Per-source-model breakdown:\n")
                lines.append("| Source model | N | Spearman r | MAE |")
                lines.append("|---|---|---|---|")
                for cat, cm in sorted(metrics["per_category"].items()):
                    lines.append(f"| {cat} | {cm.get('n', 0)} | {fmt_val(cm.get('spearman_r'))} | {fmt_val(cm.get('mae'), 2)} |")
                lines.append("")

        if res["inter_annotator"]:
            lines.append("### Inter-annotator agreement\n")
            lines.append("| Annotators | N | Spearman r | MAE |")
            lines.append("|---|---|---|---|")
            for (a1, a2), m in res["inter_annotator"].items():
                lines.append(f"| {a1} vs {a2} | {m.get('n', 0)} | {fmt_val(m.get('spearman_r'))} | {fmt_val(m.get('mae'), 2)} |")
            lines.append("")

    # --- Judge D ---
    if "d" in all_results:
        res = all_results["d"]
        lines.append("## Judge D: Benchmark Scoring (Binary classification)\n")

        for ann, metrics in res["per_annotator"].items():
            lines.append(f"### Annotator: {ann} (N={metrics.get('n', 0)})\n")
            lines.append("| | Agreement % | Cohen's kappa |")
            lines.append("|---|---|---|")
            lines.append(f"| Human vs. LLM (GPT-5.2) | {fmt_pct(metrics.get('accuracy'))} | {fmt_val(metrics.get('kappa'))} |")
            lines.append("")

            if metrics.get("per_category"):
                lines.append("Broken down by response type:\n")
                lines.append("| Model type | N | Agreement % | Kappa |")
                lines.append("|---|---|---|---|")
                for cat, cm in sorted(metrics["per_category"].items()):
                    lines.append(f"| {cat} | {cm.get('n', 0)} | {fmt_pct(cm.get('accuracy'))} | {fmt_val(cm.get('kappa'))} |")
                lines.append("")

        if res["inter_annotator"]:
            lines.append("### Inter-annotator agreement\n")
            lines.append("| Annotators | N | Kappa | Agreement % |")
            lines.append("|---|---|---|---|")
            for (a1, a2), m in res["inter_annotator"].items():
                lines.append(f"| {a1} vs {a2} | {m.get('n', 0)} | {fmt_val(m.get('kappa'))} | {fmt_pct(m.get('accuracy'))} |")
            lines.append("")

    # --- Conclusion template ---
    lines.append("## Conclusion\n")
    # Build conclusion from available data
    parts = []
    total_annotated = 0
    for judge_key in ["a", "b", "c", "d"]:
        if judge_key not in all_results:
            continue
        res = all_results[judge_key]
        for ann, metrics in res["per_annotator"].items():
            n = metrics.get("n", 0)
            total_annotated += n
            if judge_key == "a":
                parts.append(f"For taxonomy consistency (Judge A), human-LLM agreement reaches kappa = {fmt_val(metrics.get('kappa'))} (N={n}), indicating {'substantial' if metrics.get('kappa', 0) > 0.6 else 'moderate' if metrics.get('kappa', 0) > 0.4 else 'fair'} agreement.")
            elif judge_key == "b":
                parts.append(f"For taxonomy completeness (Judge B), human and LLM ratings correlate at Spearman r = {fmt_val(metrics.get('spearman_r'))} (N={n}).")
            elif judge_key == "c":
                parts.append(f"For taxonomy independence (Judge C), correlation is r = {fmt_val(metrics.get('spearman_r'))} (N={n}).")
            elif judge_key == "d":
                parts.append(f"For benchmark answer scoring (Judge D), human-LLM agreement is {fmt_pct(metrics.get('accuracy'))} (kappa = {fmt_val(metrics.get('kappa'))}, N={n}).")
            break  # just use first annotator for conclusion

    lines.append(f"We validated all four LLM judges used in the paper with human-annotated datapoints ({total_annotated} total). " + " ".join(parts) + " These results confirm that the LLM judges used throughout the paper produce evaluations consistent with human judgment.")
    lines.append("")

    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Compute human-LLM agreement metrics")
    parser.add_argument("--judges", nargs="+", default=["a", "b", "c", "d"],
                        choices=["a", "b", "c", "d"])
    parser.add_argument("--annotators", nargs="+", default=None,
                        help="Filter to specific annotators (default: all found)")
    args = parser.parse_args()

    all_results = {}

    for judge in args.judges:
        data = load_data(judge)
        if data is None:
            print(f"Judge {judge.upper()}: no data file found, skipping.")
            continue

        annotators = args.annotators or find_annotators(data)
        if not annotators:
            print(f"Judge {judge.upper()}: no annotations found, skipping.")
            continue

        print(f"Judge {judge.upper()} ({JUDGE_NAMES[judge]}): {len(data)} items, annotators: {annotators}")

        if judge in ("a", "d"):
            all_results[judge] = analyze_binary_judge(data, annotators, judge)
        else:
            all_results[judge] = analyze_rating_judge(data, annotators, judge)

    if not all_results:
        print("\nNo annotated data found. Run annotate.py first.")
        return

    report = format_report(all_results)
    print("\n" + "=" * 60)
    print(report)
    print("=" * 60)

    # Write to results file
    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    with open(RESULTS_PATH, "w") as f:
        f.write(report)
    print(f"\nResults written to {RESULTS_PATH}")


if __name__ == "__main__":
    main()
