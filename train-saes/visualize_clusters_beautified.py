#!/usr/bin/env python3
"""Beautified SAE grid-search taxonomy figure (stage-1: SAE train + scoring).

Keeps the exact layout of visualize_clusters.visualize_all_models():
  Row 1 (5 panels): DeepSeek 1.5B / Llama-8B / Qwen-14B / Qwen-32B / QwQ-32B
  Row 2 (4 panels, centred): ORZ 0.5B / 1.5B / 7B / 32B
Each panel is a heatmap of the per-model normalized final score over
(n_clusters x layer), with the top-3 configs outlined in green, the bottom-3
in red, dead-latent cells hatched, a gold star on the single best config, and
one shared colorbar. Only the visual design is refreshed.

Reuses the data loader from visualize_clusters.py.

Run from inside train-saes/:
    ../.venv/bin/python visualize_clusters_beautified.py
Outputs:
    results/figures/sae_grid_search_all_models_beautified.{pdf,png}
"""
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyBboxPatch

from visualize_clusters import load_sae_grid_search_results


# ----------------------------------------------------------------------------
# Layout / styling constants
# ----------------------------------------------------------------------------
OUTPUT_DIR = "results/figures"

# (model_id, display, family)
MODELS = [
    ("deepseek-r1-distill-qwen-1.5b", "DeepSeek-R1 1.5B", "R1"),
    ("deepseek-r1-distill-llama-8b",  "DeepSeek-R1 Llama-8B", "R1"),
    ("deepseek-r1-distill-qwen-14b",  "DeepSeek-R1 14B", "R1"),
    ("deepseek-r1-distill-qwen-32b",  "DeepSeek-R1 32B", "R1"),
    ("qwq-32b",                       "QwQ 32B", "QwQ"),
    ("open-reasoner-zero-0.5b",       "ORZ 0.5B", "ORZ"),
    ("open-reasoner-zero-1.5b",       "ORZ 1.5B", "ORZ"),
    ("open-reasoner-zero-7b",         "ORZ 7B", "ORZ"),
    ("open-reasoner-zero-32b",        "ORZ 32B", "ORZ"),
]

FAMILY_COLOR = {"R1": "#2C6FB5", "QwQ": "#2E8B57", "ORZ": "#C0392B"}
FAMILY_LABEL = {"R1": "SFT-distilled (DeepSeek-R1)",
                "QwQ": "Mixed SFT+RL (QwQ)",
                "ORZ": "RL-trained (Open-Reasoner-Zero)"}

INK = "#2b2b2b"
GRID_TEXT = "#f7f7f9"

# Refined muted diverging colormap: coral (low) -> off-white -> indigo (high)
SCORE_CMAP = LinearSegmentedColormap.from_list(
    "score_div",
    [(0.00, "#b5342b"), (0.22, "#dd8a6f"), (0.42, "#f3d9c9"),
     (0.50, "#f7f7f4"), (0.58, "#c9d6ea"), (0.78, "#6f93c6"),
     (1.00, "#284c86")],
    N=256,
)

BEST_EDGE = "#1e8449"   # green
WORST_EDGE = "#c0392b"  # red
STAR_COLOR = "#f2b705"  # gold


def _paper_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans",
        "font.size": 19,
        "axes.titlesize": 26,
        "axes.labelsize": 21,
        "xtick.labelsize": 18,
        "ytick.labelsize": 18,
        "axes.edgecolor": "#9aa0a6",
        "axes.linewidth": 0.8,
        "savefig.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


def _normalized_final_score(df):
    metrics = ["f1", "completeness", "semantic_orthogonality"]
    for m in metrics:
        lo, hi = df[m].min(), df[m].max()
        rng = (hi - lo) or 1.0
        df[f"{m}_norm"] = (df[m] - lo) / rng
    df["normalized_final_score"] = (
        df["f1_norm"] + df["completeness_norm"] + df["semantic_orthogonality_norm"]
    ) / len(metrics)
    return df


def _rounded_outline(ax, col, row, color, lw):
    """A soft rounded outline inset within cell (col,row)."""
    pad = 0.09
    patch = FancyBboxPatch(
        (col + pad, row + pad), 1 - 2 * pad, 1 - 2 * pad,
        boxstyle="round,pad=0.0,rounding_size=0.18",
        linewidth=lw, edgecolor=color, facecolor="none",
        mutation_aspect=1.0, zorder=6,
    )
    ax.add_patch(patch)


def build_figure(max_reps=None, seed=42):
    _paper_style()
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Load + normalize each model
    model_data = []
    for mid, disp, fam in MODELS:
        df = load_sae_grid_search_results(mid, max_reps=max_reps, seed=seed)
        if df is None:
            print(f"  [skip] no data for {mid}")
            continue
        model_data.append((mid, disp, fam, _normalized_final_score(df)))

    n_cols = 5
    fig = plt.figure(figsize=(6.4 * n_cols, 8.4 * 2))
    fig.patch.set_facecolor("white")
    gs = fig.add_gridspec(
        2, 10 + 1,
        width_ratios=10 * [1] + [0.16],
        height_ratios=[1, 1],
        top=0.945, bottom=0.10, left=0.045, right=0.97,
        wspace=0.45, hspace=0.30,
    )

    axes = []
    for i in range(5):                       # row 1: 5 R1/QwQ panels
        axes.append(fig.add_subplot(gs[0, i * 2:(i + 1) * 2]))
    for i in range(4):                       # row 2: 4 ORZ panels, centred
        axes.append(fig.add_subplot(gs[1, 1 + i * 2:1 + (i + 1) * 2]))
    cbar_ax = fig.add_subplot(gs[:, -1])

    for ax, (mid, disp, fam, df) in zip(axes, model_data):
        pivot = df.pivot_table(index="n_clusters", columns="layer",
                               values="normalized_final_score", aggfunc="mean")
        dead = df.pivot_table(index="n_clusters", columns="layer",
                              values="has_dead_latents", aggfunc=lambda x: any(x))
        pivot = pivot.sort_index(axis=1).sort_index(axis=0, ascending=False)
        dead = dead.sort_index(axis=1).sort_index(axis=0, ascending=False)

        sns.heatmap(pivot, ax=ax, cmap=SCORE_CMAP, center=0.5, vmin=0, vmax=1,
                    cbar=False, annot=False, linewidths=1.1, linecolor="white",
                    square=False)

        flat = pivot.values.flatten()
        order = np.argsort(flat)
        top = order[-3:][::-1]
        bottom = order[:3]
        top_pos = [np.unravel_index(i, pivot.shape) for i in top]
        bot_pos = [np.unravel_index(i, pivot.shape) for i in bottom]

        for rank, pos in enumerate(top_pos):
            _rounded_outline(ax, pos[1], pos[0], BEST_EDGE, lw=3.2 - 0.5 * rank)
            ax.text(pos[1] + 0.5, pos[0] + 0.5, f"{pivot.iloc[pos]:.2f}",
                    ha="center", va="center", fontsize=18, weight="bold",
                    color=INK, zorder=7)
        for rank, pos in enumerate(bot_pos):
            _rounded_outline(ax, pos[1], pos[0], WORST_EDGE, lw=3.2 - 0.5 * rank)
            ax.text(pos[1] + 0.5, pos[0] + 0.5, f"{pivot.iloc[pos]:.2f}",
                    ha="center", va="center", fontsize=18, weight="bold",
                    color=INK, zorder=7)

        # dead-latent hatch
        for r in range(pivot.shape[0]):
            for c in range(pivot.shape[1]):
                if (r < dead.shape[0] and c < dead.shape[1]
                        and bool(dead.iloc[r, c])):
                    ax.add_patch(plt.Rectangle((c, r), 1, 1, fill=True,
                                               color="#3a3a3a", alpha=0.16,
                                               hatch="////", edgecolor="#3a3a3a",
                                               linewidth=0.0, zorder=4))

        col = axes.index(ax)
        if col in (0, 5):
            ax.set_ylabel("Number of clusters", fontsize=20, color=INK)
        else:
            ax.set_ylabel("")
        ax.set_xlabel("Layer", fontsize=20, color=INK)
        ax.tick_params(axis="both", length=0, colors=INK)
        for lbl in ax.get_yticklabels():
            lbl.set_rotation(0)

        # family-accented title (colour-coded by training family)
        fam_c = FAMILY_COLOR[fam]
        ax.set_title(disp, fontsize=25, pad=9, color=fam_c, weight="bold")
        for spine in ax.spines.values():
            spine.set_edgecolor("#c4c8cc")
            spine.set_linewidth(0.8)

    # shared colorbar
    sm = plt.cm.ScalarMappable(cmap=SCORE_CMAP, norm=plt.Normalize(vmin=0, vmax=1))
    sm.set_array([])
    cbar = plt.colorbar(sm, cax=cbar_ax)
    cbar.set_label("Normalized taxonomy score  (F1 · completeness · orthogonality)",
                   fontsize=19, color=INK, labelpad=12)
    cbar.ax.tick_params(labelsize=16, colors=INK, length=0)
    cbar.outline.set_edgecolor("#c4c8cc")

    # no suptitle (kept title-free like the original figure)

    # legend strip (families + top/bottom markers) along the bottom
    from matplotlib.lines import Line2D
    handles = [
        Line2D([], [], marker="s", ls="", ms=16, mfc=FAMILY_COLOR["R1"],
               mec="white", label=FAMILY_LABEL["R1"]),
        Line2D([], [], marker="s", ls="", ms=16, mfc=FAMILY_COLOR["QwQ"],
               mec="white", label=FAMILY_LABEL["QwQ"]),
        Line2D([], [], marker="s", ls="", ms=16, mfc=FAMILY_COLOR["ORZ"],
               mec="white", label=FAMILY_LABEL["ORZ"]),
        Line2D([], [], marker="s", ls="", ms=16, mfc="none",
               mec=BEST_EDGE, mew=2.6, label="top-3 configs"),
        Line2D([], [], marker="s", ls="", ms=16, mfc="none",
               mec=WORST_EDGE, mew=2.6, label="bottom-3 configs"),
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, 0.018), frameon=False, fontsize=18,
               handletextpad=0.5, columnspacing=2.0)

    out = os.path.join(OUTPUT_DIR, "sae_grid_search_all_models_beautified")
    fig.savefig(out + ".pdf", bbox_inches="tight", pad_inches=0.15)
    fig.savefig(out + ".png", bbox_inches="tight", pad_inches=0.15, dpi=200)
    plt.close(fig)
    print(f"wrote {out}.pdf / .png")
    return out + ".png"


if __name__ == "__main__":
    build_figure()
