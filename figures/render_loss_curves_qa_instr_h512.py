#!/usr/bin/env python3
"""Loss-curve figures for the qa_instr h=512 pipeline (all 9 models).

Draws holdout loss curves for the selected best-of-3 run of each pair. When the
regenerable per-run train_log.jsonl files are present (after train-vectors/run.sh)
they are read directly; on a clean clone they are absent and the curves are
rebuilt from the committed data_dump.json.

Outputs (under figures/figs/loss_curves_qa_instr_h512/):
  fig1_holdout_val.{pdf,png}
  fig1_holdout_math500.{pdf,png}
  fig1_holdout_gsm8k.{pdf,png}
  fig1_holdout_combined.{pdf,png}
  data_dump.json
"""
from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(os.environ.get("THINKING_LLMS_ROOT") or Path(__file__).resolve().parent.parent)
VECT = ROOT / "artifacts" / "mlp_vectors_qa_instr_h512"
OUT  = ROOT / "figures/figs/loss_curves_qa_instr_h512"
OUT.mkdir(parents=True, exist_ok=True)

HOLDOUTSEL = VECT.parent / "mlp_vectors_qa_instr_holdoutsel_h512"


BO3 = VECT.parent / "mlp_vectors_qa_instr_h512_bo3"


def _canonical_dir(slug: str) -> Path:
    """Training dir of the selected best-of-3 run (from selection.json).

    The per-run training outputs are regenerable scratch (gitignored); when present
    the loss curves are drawn from the selected run's train_log.jsonl, otherwise the
    curves fall back to the committed data_dump.json (see _dump_series)."""
    sel = HOLDOUTSEL / slug / "selection.json"
    if sel.exists():
        run = json.loads(sel.read_text()).get("selected_run")
        cand = {"run1": VECT / slug,
                "run2": BO3 / slug / "run2",
                "run3": BO3 / slug / "run3"}.get(run)
        if cand and (cand / "train_log.jsonl").exists():
            return cand
    return VECT / slug


CUR_DIR = {p["slug"]: _canonical_dir(p["slug"]) for p in [
    {"slug": s} for s in (
        "orz-0.5b", "orz-1.5b", "orz-7b", "orz-32b", "r1-math1.5b",
        "r1-llama8b", "r1-14b", "r1-32b", "qwq-32b")
]}

PAIRS: List[dict] = [
    {"slug": "orz-0.5b",    "label": "ORZ-0.5B",     "family": "ORZ"},
    {"slug": "orz-1.5b",    "label": "ORZ-1.5B",     "family": "ORZ"},
    {"slug": "orz-7b",      "label": "ORZ-7B",       "family": "ORZ"},
    {"slug": "orz-32b",     "label": "ORZ-32B",      "family": "ORZ"},
    {"slug": "r1-math1.5b", "label": "R1-Math-1.5B", "family": "R1"},
    {"slug": "r1-llama8b",  "label": "R1-Llama-8B",  "family": "R1"},
    {"slug": "r1-14b",      "label": "R1-14B",       "family": "R1"},
    {"slug": "r1-32b",      "label": "R1-32B",       "family": "R1"},
    {"slug": "qwq-32b",     "label": "QwQ-32B",      "family": "QwQ"},
]

FAMILY_CMAP      = {"ORZ": plt.get_cmap("Reds"), "R1": plt.get_cmap("Blues"),
                    "QwQ": plt.get_cmap("Greens")}
FAMILY_LINESTYLE = {"ORZ": "--", "R1": "-",  "QwQ": ":"}
FAMILY_MARKER    = {"ORZ": "o",  "R1": "s",  "QwQ": "D"}
FAMILY_LABEL     = {"ORZ": "RL-trained", "R1": "SFT-distilled", "QwQ": "Mixed (SFT+RL)"}


def _assign_colours() -> Dict[str, Tuple[float, ...]]:
    by_fam: Dict[str, List[dict]] = {}
    for p in PAIRS:
        by_fam.setdefault(p["family"], []).append(p)
    out: Dict[str, Tuple[float, ...]] = {}
    for fam, ps in by_fam.items():
        for i, p in enumerate(ps):
            t = 0.40 + 0.50 * (i / max(len(ps) - 1, 1))
            out[p["slug"]] = FAMILY_CMAP[fam](t)
    return out


COLOURS = _assign_colours()


def _set_paper_style() -> None:
    plt.rcParams.update({
        "font.size": 9,
        "font.family": "DejaVu Sans",
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.linewidth":     0.7,
        "axes.labelsize":     9,
        "axes.titlesize":     9,
        "axes.grid":          True,
        "grid.linewidth":     0.35,
        "grid.alpha":         0.35,
        "grid.color":         "#cccccc",
        "legend.frameon":     True,
        "legend.framealpha":  0.92,
        "legend.edgecolor":   "#cccccc",
        "legend.borderpad":   0.4,
        "legend.labelspacing":0.25,
        "legend.handlelength":1.4,
        "legend.fontsize":    7.5,
        "xtick.labelsize":    8,
        "ytick.labelsize":    8,
        "lines.linewidth":    1.5,
        "lines.markersize":   3.5,
        "figure.dpi":         160,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "pdf.fonttype":       42,
        "ps.fonttype":        42,
    })


def _load_train_log(slug: str) -> Optional[List[dict]]:
    p = CUR_DIR.get(slug, VECT / slug) / "train_log.jsonl"
    if not p.exists() or p.stat().st_size == 0:
        return None
    try:
        return [json.loads(l) for l in p.open()]
    except Exception:
        return None


HOLDOUT_KEYS = [
    ("trainmix_holdout", "Validation",  "val"),
    ("math500_oos",      "MATH-500",    "math500"),
    ("gsm8k_oos",        "GSM8K",       "gsm8k"),
]


def _dump_series(slug: str) -> Dict[str, Tuple[np.ndarray, np.ndarray]]:
    """Fallback curves from the committed data_dump.json (shipped output).

    Used on a clean clone where the regenerable per-run train logs are absent, so
    figures/run.sh rebuilds the loss-curve PDFs from the committed data instead of
    silently emptying them."""
    p = OUT / "data_dump.json"
    if not p.exists():
        return {}
    fi = json.loads(p.read_text()).get("figure1_individual", {}).get(slug, {})
    out: Dict[str, Tuple[np.ndarray, np.ndarray]] = {}
    for hkey, kv in fi.items():
        if kv.get("epochs") and kv.get("ce"):
            out[hkey] = (np.asarray(kv["epochs"]), np.asarray(kv["ce"]))
    return out


def _gather_series() -> Tuple[
    Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]],
    List[float],
    int,
]:
    series: Dict[str, Dict[str, Tuple[np.ndarray, np.ndarray]]] = {}
    all_vals: List[float] = []
    max_ep = 0
    for pair in PAIRS:
        log = _load_train_log(pair["slug"])
        if not log:
            for hkey, (xa, ya) in _dump_series(pair["slug"]).items():
                series.setdefault(pair["slug"], {})[hkey] = (xa, ya)
                all_vals.extend(ya.tolist())
                if xa.size:
                    max_ep = max(max_ep, int(xa.max()))
            continue
        for hkey, _, _ in HOLDOUT_KEYS:
            xs, ys = [], []
            for r in log:
                ep = r.get("epoch")
                v  = (r.get("holdout_ce") or {}).get(hkey)
                if ep is not None and v is not None:
                    xs.append(int(ep)); ys.append(float(v))
            if xs:
                order = np.argsort(xs)
                xa = np.asarray(xs)[order]
                ya = np.asarray(ys)[order]
                series.setdefault(pair["slug"], {})[hkey] = (xa, ya)
                all_vals.extend(ya.tolist())
                max_ep = max(max_ep, int(xa.max()))
    return series, all_vals, max_ep


def figure_1_individual() -> dict:
    _set_paper_style()
    series, all_vals, max_ep = _gather_series()
    available = sorted(series.keys())
    print(f"[fig1] train logs available for: {available}")

    y_lo = max(0.0, min(all_vals) - 0.05) if all_vals else 0.0
    y_hi = max(all_vals) + 0.15            if all_vals else 3.5
    dump: Dict[str, dict] = {"figure1_individual": {}}

    fam_order = ["ORZ", "QwQ", "R1"]
    by_fam: Dict[str, List[dict]] = {}
    for p in PAIRS:
        by_fam.setdefault(p["family"], []).append(p)

    for hkey, hlabel, fname_key in HOLDOUT_KEYS:
        fig, ax = plt.subplots(figsize=(3.5, 2.9))
        for fam in fam_order:
            for pair in by_fam.get(fam, []):
                if pair["slug"] not in series or hkey not in series[pair["slug"]]:
                    continue
                xa, ya = series[pair["slug"]][hkey]
                ax.plot(xa, ya, linestyle=FAMILY_LINESTYLE[fam],
                        marker=FAMILY_MARKER[fam], color=COLOURS[pair["slug"]],
                        label=pair["label"], linewidth=1.6, markersize=3.5,
                        markeredgewidth=0.5, markeredgecolor="white", zorder=3)
                if hkey == "trainmix_holdout":
                    ib = int(np.argmin(ya))
                    ax.plot(xa[ib], ya[ib], "*", color=COLOURS[pair["slug"]],
                            markersize=9, markeredgecolor="white",
                            markeredgewidth=0.6, zorder=5)
                dump["figure1_individual"].setdefault(pair["slug"], {})[hkey] = {
                    "epochs": xa.tolist(), "ce": ya.tolist(),
                }

        ax.set_xlabel("Epoch", labelpad=3)
        ax.set_ylabel("CE (sample-weighted)", labelpad=3)
        ax.set_title(hlabel, pad=4, fontsize=9, fontweight="semibold")
        ax.set_xlim(-0.4, max_ep + 0.4)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xticks([t for t in range(0, max_ep + 1, 2)])
        ax.tick_params(axis="both", length=2.5, pad=2)

        handles, labels_leg = ax.get_legend_handles_labels()
        by_fam_leg: Dict[str, List[Tuple]] = {}
        for h, l in zip(handles, labels_leg):
            fam = next((p["family"] for p in PAIRS if p["label"] == l), "?")
            by_fam_leg.setdefault(fam, []).append((h, l))
        final_h, final_l = [], []
        for fam in fam_order:
            if fam not in by_fam_leg:
                continue
            proxy = plt.Line2D([], [], color="none")
            final_h.append(proxy)
            final_l.append(f"$\\bf{{{FAMILY_LABEL[fam]}}}$")
            for h, l in by_fam_leg[fam]:
                final_h.append(h); final_l.append(f"  {l}")
        leg = ax.legend(final_h, final_l, loc="upper right", fontsize=6.5,
                        handlelength=1.6, handletextpad=0.4,
                        labelspacing=0.18, borderpad=0.5,
                        framealpha=0.93, edgecolor="#cccccc", ncol=1)
        for txt in leg.get_texts():
            if txt.get_text().startswith("$"):
                txt.set_color("#444444")

        fig.tight_layout(pad=0.4)
        fout = OUT / f"fig1_holdout_{fname_key}"
        fig.savefig(fout.with_suffix(".pdf"))
        fig.savefig(fout.with_suffix(".png"))
        plt.close(fig)
        print(f"  wrote {fout}.(pdf,png)")
    return dump


def figure_1_combined() -> dict:
    _set_paper_style()
    series, all_vals, max_ep = _gather_series()
    if not all_vals:
        print("[fig1c] no series available; skip")
        return {}

    y_lo = max(0.0, min(all_vals) - 0.08)
    y_hi = max(all_vals) + 0.12

    fam_order = ["ORZ", "QwQ", "R1"]
    by_fam: Dict[str, List[dict]] = {}
    for p in PAIRS:
        by_fam.setdefault(p["family"], []).append(p)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.65), sharey=True)
    PANELS = [("trainmix_holdout", "Validation"),
              ("math500_oos",      "MATH-500"),
              ("gsm8k_oos",        "GSM8K")]
    for ax, (hkey, title) in zip(axes, PANELS):
        for fam in fam_order:
            for pair in by_fam.get(fam, []):
                if pair["slug"] not in series or hkey not in series[pair["slug"]]:
                    continue
                xa, ya = series[pair["slug"]][hkey]
                ax.plot(xa, ya, linestyle=FAMILY_LINESTYLE[fam],
                        marker=FAMILY_MARKER[fam], color=COLOURS[pair["slug"]],
                        linewidth=1.5, markersize=3.0, markeredgewidth=0.4,
                        markeredgecolor="white", zorder=3)
                if hkey == "trainmix_holdout":
                    ib = int(np.argmin(ya))
                    ax.plot(xa[ib], ya[ib], "*", color=COLOURS[pair["slug"]],
                            markersize=8, markeredgecolor="white",
                            markeredgewidth=0.5, zorder=5)
        ax.set_title(title, fontsize=8.5, pad=3)
        ax.set_xlabel("Training epoch", fontsize=8.5, labelpad=2)
        ax.set_xlim(-0.4, max_ep + 0.4)
        ax.set_ylim(y_lo, y_hi)
        ax.set_xticks([t for t in range(0, max_ep + 1, 2)])
        ax.tick_params(axis="both", length=2.5, pad=2, labelsize=7.5)
    axes[0].set_ylabel("Holdout CE", fontsize=8.5, labelpad=3)

    ordered: List[dict] = []
    for fam in fam_order:
        ordered.extend(by_fam.get(fam, []))

    def _proxy(p):
        return mlines.Line2D([], [], color=COLOURS[p["slug"]],
                             linestyle=FAMILY_LINESTYLE[p["family"]],
                             marker=FAMILY_MARKER[p["family"]],
                             linewidth=1.5, markersize=3.0,
                             markeredgecolor="white", markeredgewidth=0.4)

    handles = [_proxy(p) for p in ordered]
    labels  = [p["label"] for p in ordered]
    fig.legend(handles, labels, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.02), fontsize=7.0,
               handlelength=1.6, handletextpad=0.4,
               columnspacing=1.3, labelspacing=0.25,
               frameon=False)
    fig.tight_layout(rect=(0, 0.13, 1, 1.0), w_pad=1.0)
    fout = OUT / "fig1_holdout_combined"
    fig.savefig(fout.with_suffix(".pdf"))
    fig.savefig(fout.with_suffix(".png"))
    plt.close(fig)
    print(f"  wrote {fout}.(pdf,png)")
    return {}


def main() -> None:
    print(f"=== loss curves (qa_instr h=512, all 9 models) -> {OUT} ===")
    dump: dict = {}
    dump.update(figure_1_individual())
    dump.update(figure_1_combined())
    (OUT / "data_dump.json").write_text(json.dumps(dump, indent=2))
    print(f"wrote {OUT/'data_dump.json'}")


if __name__ == "__main__":
    main()
