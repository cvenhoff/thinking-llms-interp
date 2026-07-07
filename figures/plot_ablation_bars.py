#!/usr/bin/env python3
"""Negative-control ablation bar plot (gap recovered, mean over MATH-500 + GSM8K).

For each ablation we plug the ablation's perturbed-hybrid accuracy into the SAME
gap-recovered formula used for the headline metric, keeping the canonical
3-sample base/think as the denominator:
    gap_rec = (hybrid_abl - base_canon) / (think_canon - base_canon) * 100
The 'Full pipeline' bar is the canonical headline gap recovered (reference).

Bars shown: Full pipeline, randcat, randV, mlponly, randpos  x  {orz-1.5b, orz-32b}.
Output: figures/figs/ablation_bars.{pdf,png}
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = os.environ.get("THINKING_LLMS_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CANON = f"{ROOT}/artifacts/mlp_eval_qa_instr_holdoutsel_h512"
ABL = f"{ROOT}/artifacts/mlp_eval_qa_instr_holdoutsel_ablations"
OUT = f"{ROOT}/figures/figs"
os.makedirs(OUT, exist_ok=True)

BS = {"orz-1.5b": "qwen2.5-1.5b", "orz-32b": "qwen2.5-32b"}
MODELS = [("orz-1.5b", "ORZ-1.5B"), ("orz-32b", "ORZ-32B")]
DATASETS = ["math500", "gsm8k"]
# (key, display)
ABLATIONS = [
    ("__full__", "Full\npipeline"),
    ("randcat",  "Random\ncategory"),
    ("randV",    "Random\nvectors"),
    ("mlponly",  "MLP\nmagnitude\nonly"),
    ("randpos",  "Random\npositions"),
]

BAR_COLOR = {"orz-1.5b": "#4C78A8", "orz-32b": "#C0392B"}


def canon_headline(cfg, ds):
    f = f"{CANON}/{cfg}/hybrid_summary_{BS[cfg]}_{ds}_final.json"
    return json.load(open(f))["headline"]


def abl_hybrid(cfg, ds, abl):
    f = f"{ABL}/{cfg}-{abl}/{ds}/judge_reps_{BS[cfg]}_{ds}_abl_{abl}.json"
    return json.load(open(f))["per_rep"]["hybrid"]["mean_pct"]


def gap_for(cfg, ds, abl):
    h = canon_headline(cfg, ds)
    base, think = h["base_mean_pct"], h["thinking_mean_pct"]
    denom = think - base
    if abs(denom) < 1e-9:
        return float("nan")
    if abl == "__full__":
        return h["gap_recovered_pct"]
    hyb = abl_hybrid(cfg, ds, abl)
    return (hyb - base) / denom * 100.0


def main():
    # value[cfg][abl] = mean over datasets
    vals = {cfg: [] for cfg, _ in MODELS}
    for cfg, _ in MODELS:
        for abl, _ in ABLATIONS:
            gs = [gap_for(cfg, ds, abl) for ds in DATASETS]
            vals[cfg].append(float(np.nanmean(gs)))

    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 14,
        "axes.spines.top": False, "axes.spines.right": False,
        "axes.linewidth": 0.9, "pdf.fonttype": 42, "ps.fonttype": 42,
    })
    x = np.arange(len(ABLATIONS))
    w = 0.38
    fig, ax = plt.subplots(figsize=(6.8, 5.0))

    vmax = max(max(vals[c]) for c in vals)
    vmin = min(min(vals[c]) for c in vals)
    ymax = vmax + 14
    ymin = min(0.0, vmin - 14)          # extra room so negative labels clear axis
    ax.set_ylim(ymin, ymax)
    ax.axhline(0, color="#666", lw=1.0, zorder=2)

    for i, (cfg, disp) in enumerate(MODELS):
        off = (i - 0.5) * w
        bars = ax.bar(x + off, vals[cfg], w, label=disp,
                      color=BAR_COLOR[cfg], edgecolor="white", linewidth=0.8,
                      zorder=3)
        bars[0].set_hatch("//")          # reference (full pipeline) bar
        bars[0].set_alpha(0.92)
        for b, v in zip(bars, vals[cfg]):
            ax.text(b.get_x() + b.get_width() / 2,
                    v + (2.5 if v >= 0 else -2.5),
                    f"{v:.0f}", ha="center",
                    va="bottom" if v >= 0 else "top",
                    fontsize=12, color="#333", fontweight="bold")

    ax.set_xticks(x)
    ax.set_xticklabels([d for _, d in ABLATIONS], fontsize=13)
    ax.set_ylabel("Gap recovered (%)", fontsize=15)
    ax.set_title("Ablations", fontsize=17, fontweight="bold", pad=10)
    ax.grid(axis="y", alpha=0.25, linewidth=0.5)
    ax.axvspan(-0.5, 0.5, color="#f2f2f2", zorder=0)
    ax.legend(frameon=False, fontsize=13, loc="upper right",
              title="model", title_fontsize=13)

    fig.tight_layout()
    fig.savefig(f"{OUT}/ablation_bars.pdf", bbox_inches="tight")
    fig.savefig(f"{OUT}/ablation_bars.png", dpi=200, bbox_inches="tight")
    plt.close(fig)

    print("=== gap recovered (mean m500+gsm8k) ===")
    hdr = "model      " + "".join(f"{d.replace(chr(10),' '):>16}" for _, d in ABLATIONS)
    print(hdr)
    for cfg, disp in MODELS:
        print(f"{disp:<11}" + "".join(f"{v:>16.1f}" for v in vals[cfg]))
    print(f"\nwrote {OUT}/ablation_bars.png")


if __name__ == "__main__":
    main()
