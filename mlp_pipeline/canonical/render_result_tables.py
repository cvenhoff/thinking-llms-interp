#!/usr/bin/env python3
"""Render the three main results tables as standalone PNG images.

  table1_train_mix.png     - stage-2 training mix composition
  table2_main_results.png  - base/think/hybrid/gap across MATH-500/GSM8K/Hendrycks
  table3_gap_recovered.png - gap recovered incl. mean across all datasets

Reads the canonical holdoutsel hybrid summaries; no external args.
"""
import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = "/workspace-vast/constantinv/thinking-llms-interp"
OUT = f"{ROOT}/mlp_pipeline/canonical/figs"
os.makedirs(OUT, exist_ok=True)

BS = {"orz-0.5b": "qwen2.5-0.5b", "orz-1.5b": "qwen2.5-1.5b", "orz-7b": "qwen2.5-7b",
      "orz-32b": "qwen2.5-32b", "r1-14b": "qwen2.5-14b", "r1-llama8b": "llama-3.1-8b",
      "qwq-32b": "qwen2.5-32b", "r1-32b": "qwen2.5-32b", "r1-math1.5b": "qwen2.5-math-1.5b"}
LAB = {"orz-0.5b": "ORZ-0.5B", "orz-1.5b": "ORZ-1.5B", "orz-7b": "ORZ-7B",
       "orz-32b": "ORZ-32B", "r1-math1.5b": "R1-Math-1.5B", "r1-llama8b": "R1-Llama-8B",
       "r1-14b": "R1-14B", "qwq-32b": "QwQ-32B", "r1-32b": "R1-32B"}
ORZ = ["orz-0.5b", "orz-1.5b", "orz-7b", "orz-32b"]
REST = ["r1-math1.5b", "r1-llama8b", "r1-14b", "qwq-32b", "r1-32b"]
DS = ["math500", "gsm8k", "hendrycks_holdout"]

HEADER_BG = "#2f4b7c"
HEADER_FG = "white"
MEAN_BG = "#ffe9c7"
ORZ_BG = "#fbeeee"
REST_BG = "#eef3fb"
GRID = "#c9ced6"


def headline(cfg, ds):
    f = (f"{ROOT}/mlp_eval_hendrycks_holdout_qa_instr_holdoutsel_h512/{cfg}/"
         f"hybrid_summary_{BS[cfg]}_hendrycks_holdout_final.json"
         if ds == "hendrycks_holdout" else
         f"{ROOT}/mlp_eval_qa_instr_holdoutsel_h512/{cfg}/"
         f"hybrid_summary_{BS[cfg]}_{ds}_final.json")
    return json.load(open(f))["headline"]


def _render(rows, col_labels, out, colw, row_h=0.52, fontsize=13,
            title=None, bold_rows=(), tinted=None, header_rows=1,
            col_align=None, footnote=None, group_header=None):
    """rows: list of list[str]; tinted: dict row_idx->color.
    group_header: optional list of (label, start_col, end_col) drawn as a band
    above the column header (columns are 0-indexed, inclusive)."""
    tinted = tinted or {}
    ncol = len(col_labels)
    fig_w = sum(colw)
    n = len(rows) + header_rows
    band = 0.6 if group_header else 0.0
    fig_h = n * row_h + (0.5 if title else 0.15) + band
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    ax.axis("off")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    if title:
        ax.set_title(title, fontsize=fontsize + 3, fontweight="bold",
                     color="#222", pad=10)

    fracs = [w / fig_w for w in colw]
    top = 1.0 - (band / fig_h)          # table top (leave room for group band)
    tbl = ax.table(cellText=rows, colLabels=col_labels,
                   cellLoc="center", colWidths=fracs,
                   bbox=[0, 0, 1, top])
    tbl.auto_set_font_size(False)
    tbl.set_fontsize(fontsize)

    col_align = col_align or (["left"] + ["center"] * (ncol - 1))
    for (r, c), cell in tbl.get_celld().items():
        cell.set_edgecolor(GRID)
        cell.set_linewidth(0.6)
        if r == 0:  # header
            cell.set_facecolor(HEADER_BG)
            cell.set_text_props(color=HEADER_FG, fontweight="bold")
        else:
            data_r = r - 1
            if data_r in tinted:
                cell.set_facecolor(tinted[data_r])
            if data_r in bold_rows:
                cell.set_text_props(fontweight="bold")
        al = col_align[c]
        cell.set_text_props(ha=al)
        cell.PAD = 0.04
        if al == "left":
            cell._loc = "left"

    if group_header:
        edges = [0.0]
        for f in fracs:
            edges.append(edges[-1] + f)
        y0, y1 = top, top + (band / fig_h) * 0.72
        for lab, cs, ce in group_header:
            xl, xr = edges[cs], edges[ce + 1]
            ax.add_patch(plt.Rectangle((xl, y0), xr - xl, y1 - y0,
                                       facecolor=HEADER_BG, edgecolor="white",
                                       linewidth=1.2, clip_on=False, zorder=2))
            ax.text((xl + xr) / 2, (y0 + y1) / 2, lab, ha="center", va="center",
                    color=HEADER_FG, fontweight="bold", fontsize=fontsize + 1,
                    zorder=3, clip_on=False)

    if footnote:
        ax.text(0.0, -0.04, footnote, ha="left", va="top",
                fontsize=fontsize - 1.5, color="#555", style="italic",
                transform=ax.transAxes)

    fig.savefig(out + ".png", dpi=200, bbox_inches="tight", pad_inches=0.12)
    fig.savefig(out + ".pdf", bbox_inches="tight", pad_inches=0.12)
    plt.close(fig)
    print("wrote", out + ".png")


# ---------------------------------------------------------------- table 1
def table_train_mix():
    data = [
        ["hendrycks_math *", "Competition math", "5,000", "500", "5,500"],
        ["natural_reasoning", "General reasoning (Facebook)", "3,500", "350", "3,850"],
        ["theoremqa", "Theorem-based STEM QA", "672", "75", "747"],
        ["scibench", "College science", "622", "70", "692"],
        ["Total", "", "9,794", "995", "10,789"],
    ]
    _render(
        data, ["Dataset", "Domain", "Train", "Val", "Total"],
        f"{OUT}/table1_train_mix",
        colw=[3.4, 4.0, 1.2, 1.0, 1.3], fontsize=13,
        title="Training mix (stage-2 category-vector training)",
        bold_rows=(4,), tinted={4: MEAN_BG},
        col_align=["left", "left", "right", "right", "right"],
        footnote="* hendrycks_math subset is non-overlapping with MATH-500 (0 shared problems).",
    )


# ---------------------------------------------------------------- table 2
def table_main_results():
    col_labels = ["Model",
                  "base", "think", "hyb", "gap",
                  "base", "think", "hyb", "gap",
                  "base", "think", "hyb", "gap"]
    rows, tint, bold = [], {}, []
    i = 0
    for grp, bg in ((ORZ, ORZ_BG), (REST, REST_BG)):
        for cfg in grp:
            row = [LAB[cfg]]
            for ds in DS:
                h = headline(cfg, ds)
                row += [f"{h['base_mean_pct']:.1f}", f"{h['thinking_mean_pct']:.1f}",
                        f"{h['hybrid_mean_pct']:.1f}", f"{h['gap_recovered_pct']:.1f}"]
            rows.append(row)
            tint[i] = bg
            i += 1
    _render(
        rows, col_labels, f"{OUT}/table2_main_results",
        colw=[2.2] + [0.95] * 12, fontsize=12.5,
        title="Main results  (accuracy %; gap = % of think\u2212base gap recovered)",
        tinted=tint,
        col_align=["left"] + ["right"] * 12,
        group_header=[("MATH-500", 1, 4), ("GSM8K", 5, 8),
                      ("Hendrycks-MATH *", 9, 12)],
        footnote="* Hendrycks-MATH holdout (1000 Q), disjoint from train/val + MATH-500.",
    )


# ---------------------------------------------------------------- table 3
def table_gap_recovered():
    col_labels = ["Model", "MATH-500", "GSM8K", "Hendrycks-MATH*", "All datasets"]
    rows, tint, bold = [], {}, []
    i = 0
    for grp, gname, bg in ((ORZ, "MEAN ORZ", ORZ_BG), (REST, "MEAN R1+QwQ", REST_BG)):
        store = {d: [] for d in DS}
        for cfg in grp:
            gs = [headline(cfg, d)["gap_recovered_pct"] for d in DS]
            for d, v in zip(DS, gs):
                store[d].append(v)
            rows.append([LAB[cfg]] + [f"{v:.1f}" for v in gs] + [f"{sum(gs) / 3:.1f}"])
            tint[i] = bg
            i += 1
        pm = [sum(store[d]) / len(store[d]) for d in DS]
        allv = [v for d in DS for v in store[d]]
        rows.append([gname] + [f"{v:.1f}" for v in pm] + [f"{sum(allv) / len(allv):.1f}"])
        tint[i] = MEAN_BG
        bold.append(i)
        i += 1
    _render(
        rows, col_labels, f"{OUT}/table3_gap_recovered",
        colw=[2.4, 1.9, 1.7, 2.6, 2.0], fontsize=13,
        title="Gap recovered (%)",
        tinted=tint, bold_rows=tuple(bold),
        col_align=["left", "right", "right", "right", "right"],
        footnote="* Hendrycks-MATH holdout, disjoint from train/val + MATH-500.",
    )


if __name__ == "__main__":
    table_train_mix()
    table_main_results()
    table_gap_recovered()
