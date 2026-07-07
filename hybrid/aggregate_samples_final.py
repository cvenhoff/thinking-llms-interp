#!/usr/bin/env python3
"""Aggregate the per-sample final-run judge artifacts into a single
``hybrid_summary_<base>_<dataset>_final.json``.

Inputs (all in --eval_dir):
  judge_reps_<base_id>_<ds>_final.json            (from hybrid_eval, covers
                                                    think_s0 / base / hybrid)
  judge_reps_extra_think_<base_id>_<ds>_final.json (from
                                                    judge_extra_think_samples.py,
                                                    covers think_s1, think_s2)

Output:
  hybrid_summary_<base_id>_<ds>_final.json

Aggregation policy
==================
For each role we collect a flat list of (sample, judge_rep) accuracy
values across all available (sample, rep) pairs and report the *mean*
(no std, per user request).

  - thinking : 3 samples x 3 judge reps = 9 acc points -> mean
  - base     : 1 rollout x 3 judge reps = 3 acc points -> mean
  - hybrid   : 1 rollout x 3 judge reps = 3 acc points -> mean

Gap-recovered is computed once on the role means:
  gap = |thinking_mean - base_mean|
  rec = (hybrid_mean - min(thinking_mean, base_mean)) / gap
Negative recovery (hybrid below the base) is reported as-is; it is not
floored to zero, mirroring that recovery above 100% is left uncapped.
"""

from __future__ import annotations
import argparse
import json
import os
import sys
from typing import Dict, List, Optional


def _safe_load(path: str) -> Optional[dict]:
    if not os.path.exists(path):
        return None
    with open(path) as f:
        return json.load(f)


def _main_per_rep(jr: dict, role: str) -> Optional[List[int]]:
    """Pull ``role + '_per_rep'`` (list of #correct per judge rep) out of
    the main hybrid_eval ``judge_reps_*.json``."""
    per = jr.get("per_rep") or {}
    if role not in per:
        return None
    return list(per[role].get("per_rep_correct") or [])


def _main_total(jr: dict) -> int:
    return int(jr.get("total") or 0)


def _extra_per_sample_accs(ej: dict) -> Dict[str, List[float]]:
    """Pull per-rep accuracies for each extra think sample as a list of
    percentages (one per judge rep)."""
    out: Dict[str, List[float]] = {}
    for s, blob in (ej.get("per_sample") or {}).items():
        accs = blob.get("per_rep_acc_pct")
        if accs is None:
            # back-compute from correct + active
            n = max(1, int(blob.get("n_active", 1)))
            accs = [100.0 * c / n
                    for c in (blob.get("per_rep_correct") or [])]
        out[s] = list(accs)
    return out


def _mean(xs: List[float]) -> Optional[float]:
    return (sum(xs) / len(xs)) if xs else None


def aggregate(eval_dir: str, base_id: str, dataset: str,
              suffix: str = "final") -> dict:
    main_path = os.path.join(
        eval_dir, f"judge_reps_{base_id}_{dataset}_{suffix}.json")
    extra_path = os.path.join(
        eval_dir,
        f"judge_reps_extra_think_{base_id}_{dataset}_{suffix}.json")

    main = _safe_load(main_path)
    extra = _safe_load(extra_path)

    out: Dict = {
        "base_id": base_id,
        "dataset": dataset,
        "suffix": suffix,
        "sources": {"main": main_path, "extra_think": extra_path},
        "roles": {},
    }

    # ---- Hybrid + base + think_s0 from main judge_reps ----
    if main is None:
        print(f"[aggregate] WARN: {main_path} missing")
    else:
        total = _main_total(main)
        out["total"] = total

        for role in ("thinking", "base", "hybrid"):
            per_rep_corr = _main_per_rep(main, role)
            if per_rep_corr is None:
                continue
            accs = [100.0 * c / max(total, 1) for c in per_rep_corr]
            if role == "thinking":
                # thinking from main covers ONLY sample s0
                out["roles"]["thinking_s0"] = {
                    "per_rep_acc_pct": accs,
                    "mean_pct": _mean(accs),
                    "n_samples": 1,
                    "n_reps_per_sample": len(accs),
                }
            else:
                out["roles"][role] = {
                    "per_rep_acc_pct": accs,
                    "mean_pct": _mean(accs),
                    "n_samples": 1,
                    "n_reps_per_sample": len(accs),
                }

    # ---- Extra think samples s1/s2 ----
    extra_accs_by_s: Dict[str, List[float]] = {}
    if extra is None:
        print(f"[aggregate] WARN: {extra_path} missing")
    else:
        extra_accs_by_s = _extra_per_sample_accs(extra)
        for s, accs in extra_accs_by_s.items():
            out["roles"][f"thinking_s{s}"] = {
                "per_rep_acc_pct": accs,
                "mean_pct": _mean(accs),
                "n_samples": 1,
                "n_reps_per_sample": len(accs),
            }

    # ---- Combined 'thinking' role (means over all samples x reps) ----
    all_thinking_accs: List[float] = []
    if "thinking_s0" in out["roles"]:
        all_thinking_accs.extend(
            out["roles"]["thinking_s0"]["per_rep_acc_pct"])
    for s, accs in extra_accs_by_s.items():
        all_thinking_accs.extend(accs)
    if all_thinking_accs:
        out["roles"]["thinking"] = {
            "per_rep_acc_pct": all_thinking_accs,
            "mean_pct": _mean(all_thinking_accs),
            "n_samples": 1 + len(extra_accs_by_s),
            "n_reps_per_sample": (
                len(out["roles"]["thinking_s0"]["per_rep_acc_pct"])
                if "thinking_s0" in out["roles"] else None),
        }

    # ---- Gap recovered (means only) ----
    t_mu = out["roles"].get("thinking", {}).get("mean_pct")
    b_mu = out["roles"].get("base", {}).get("mean_pct")
    h_mu = out["roles"].get("hybrid", {}).get("mean_pct")
    if t_mu is not None and b_mu is not None and h_mu is not None:
        gap = abs(t_mu - b_mu)
        if gap > 0:
            rec = (h_mu - min(t_mu, b_mu)) / gap * 100.0
        else:
            rec = None
        out["headline"] = {
            "thinking_mean_pct": t_mu, "base_mean_pct": b_mu,
            "hybrid_mean_pct": h_mu, "gap_pct": gap,
            "gap_recovered_pct": rec,
        }

    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--eval_dir", required=True,
                   help="mlp_eval_final/<cfg>/ directory")
    p.add_argument("--base_id", required=True,
                   help="Base model short, must match the slug used by "
                        "hybrid_eval to name its outputs (e.g. qwen2.5-32b).")
    p.add_argument("--dataset", required=True,
                   choices=["math500", "gsm8k", "natreason",
                            "hendrycks_holdout"])
    p.add_argument("--suffix", default="final",
                   help="Same string passed to hybrid_eval's "
                        "--results_suffix (default: 'final').")
    args = p.parse_args()

    summary = aggregate(args.eval_dir, args.base_id, args.dataset,
                        suffix=args.suffix)
    out_path = os.path.join(
        args.eval_dir,
        f"hybrid_summary_{args.base_id}_{args.dataset}_{args.suffix}.json")
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[aggregate] wrote -> {out_path}")

    hd = summary.get("headline") or {}
    if hd:
        print("[aggregate] headline (means only):")
        print(f"  thinking = {hd['thinking_mean_pct']:.2f}%")
        print(f"  base     = {hd['base_mean_pct']:.2f}%")
        print(f"  hybrid   = {hd['hybrid_mean_pct']:.2f}%")
        rec = hd.get('gap_recovered_pct')
        rec_str = f"{rec:.2f}%" if rec is not None else "n/a"
        print(f"  gap_rec  = {rec_str}")


if __name__ == "__main__":
    main()
