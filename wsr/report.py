"""A readable report of an experiment: ``work/<experiment>/scores/report.md`` with figures next to it.

Figures (needs matplotlib): balanced accuracy of every reader-target cell per condition, the contrasts with their
95% intervals, and per-behavior accuracy. Without matplotlib the report is written without figures.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

from .config import Layout

INK, MUTED, GRID = "#1f2937", "#6b7280", "#e5e7eb"
COND_COLORS = {"weights": "#2563eb", "both": "#7c3aed", "text": "#9ca3af"}
COND_NAMES = {"weights": "Weights + Tools", "both": "Weights + Text", "text": "Text-Only"}


def _plt():
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.rcParams.update({"font.size": 10, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": INK,
                             "ytick.color": INK, "axes.spines.top": False, "axes.spines.right": False})
        return plt
    except ImportError:
        return None


def fig_cells(out: dict, path: Path) -> bool:
    plt = _plt()
    if plt is None or not out["cells"]:
        return False
    pairs = sorted({k.split("|")[0] for k in out["cells"]})
    conds = [c for c in ("weights", "both", "text") if any(k.endswith("|" + c) for k in out["cells"])]
    h = 0.8 / len(conds)
    fig, ax = plt.subplots(figsize=(7.5, 0.55 * len(pairs) * max(1, len(conds) / 1.6) + 1.2))
    for j, c in enumerate(conds):
        ys = [i + (j - (len(conds) - 1) / 2) * h for i in range(len(pairs))]
        xs = [out["cells"].get(f"{p}|{c}", {}).get("balanced", math.nan) for p in pairs]
        ax.barh(ys, xs, height=h * 0.9, color=COND_COLORS[c], label=COND_NAMES[c])
        for y, x in zip(ys, xs):
            if not math.isnan(x):
                ax.text(x + 0.008, y, f"{x:.3f}", va="center", fontsize=8, color=INK)
    ax.axvline(0.5, color=MUTED, ls="--", lw=1)
    ax.text(0.5, len(pairs) - 0.45, ' always "same"', color=MUTED, fontsize=8, va="bottom")
    ax.set_yticks(range(len(pairs)), [p.replace("->", " → ") for p in pairs])
    ax.invert_yaxis()
    ax.set_xlim(0.3, 1.0)
    ax.set_xlabel("balanced accuracy (reader → target)")
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def fig_contrasts(out: dict, path: Path) -> bool:
    plt = _plt()
    rows = [(k.replace(" (balanced)", ""), v) for k, v in out["contrasts"].items()
            if v and k.endswith("(balanced)") and not math.isnan(v["diff"])]
    if plt is None or not rows:
        return False
    fig, ax = plt.subplots(figsize=(8, 0.38 * len(rows) + 1.0))
    for i, (k, v) in enumerate(rows):
        lo, hi = v["ci95"]
        sig = lo > 0 or hi < 0
        col = "#2563eb" if sig else MUTED
        ax.plot([lo, hi], [i, i], color=col, lw=2)
        ax.plot([v["diff"]], [i], "o", color=col, ms=5)
    ax.axvline(0, color=INK, lw=1)
    ax.set_yticks(range(len(rows)), [k for k, _ in rows], fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("difference in balanced accuracy (95% paired bootstrap interval)")
    ax.grid(axis="x", color=GRID)
    ax.set_axisbelow(True)
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def fig_behaviors(out: dict, path: Path) -> bool:
    plt = _plt()
    pb = out.get("per_behavior") or {}
    if plt is None or not pb:
        return False
    import numpy as np
    cells = list(pb)
    keys = list(next(iter(pb.values())))
    M = np.array([[pb[c].get(k, math.nan) for k in keys] for c in cells], float)
    fig, ax = plt.subplots(figsize=(1.1 * len(keys) + 3.2, 0.42 * len(cells) + 1.4))
    im = ax.imshow(M, cmap="RdBu", vmin=0.0, vmax=1.0, aspect="auto")
    for i in range(len(cells)):
        for j in range(len(keys)):
            if not math.isnan(M[i, j]):
                ax.text(j, i, f"{M[i, j]:.2f}", ha="center", va="center", fontsize=8,
                        color="white" if abs(M[i, j] - 0.5) > 0.3 else INK)
    ax.set_xticks(range(len(keys)), keys, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(cells)), [c.replace("->", " → ").replace("|", "  ") for c in cells], fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.03, label="balanced accuracy")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return True


def _f(x, d=3):
    return "–" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{d}f}"


def build_report(cfg: dict, exp: dict) -> Path:
    from .score import cmd_score
    from .write import score_programs
    L = Layout(cfg, exp)
    out = cmd_score(cfg, exp, quiet=True)
    d = L.scores()
    d.mkdir(parents=True, exist_ok=True)
    md = [f"# {exp['name']}", "", f"Experiment `{exp['_source']}`, suite `{exp['suite'] if isinstance(exp['suite'], str) else exp['suite'].get('name')}`.", ""]
    if out["cells"]:
        if all(math.isnan(v["recall"]) for v in out["cells"].values()):
            md += ["> No selected edit changed a behavior, so change recall and balanced accuracy are undefined. "
                   "Use stronger edits, more probe items or a larger model.", ""]
        md += ["## Cells", "", "Balanced accuracy is the mean of change recall and specificity over all behavior labels "
               "of a cell; always answering \"same\" scores 0.500.", "",
               "| reader → target | condition | balanced | recall | specificity | change AUROC | n |", "|---|---|---|---|---|---|---|"]
        for k, v in out["cells"].items():
            p, c = k.split("|")
            cd = out["change_detection"].get(k, {})
            md.append(f"| {p.replace('->', ' → ')} | {COND_NAMES.get(c, c)} | {_f(v['balanced'])} | {_f(v['recall'])} | "
                      f"{_f(v['specificity'])} | {_f(cd.get('auroc'))} | {v['n_interventions']} |")
        md.append("")
        if fig_cells(out, d / "cells.png"):
            md += ["![cells](cells.png)", ""]
    shown = [(k, v) for k, v in out["contrasts"].items() if v and k.endswith("(balanced)") and not math.isnan(v["diff"])]
    if shown:
        md += ["## Contrasts", "", "| contrast | difference | 95% CI |", "|---|---|---|"]
        md += [f"| {k.replace(' (balanced)', '')} | {v['diff']:+.3f} | [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}] |"
               for k, v in shown]
        md.append("")
        if fig_contrasts(out, d / "contrasts.png"):
            md += ["![contrasts](contrasts.png)", ""]
    if out.get("per_behavior") and fig_behaviors(out, d / "behaviors.png"):
        md += ["## Per behavior", "", "![behaviors](behaviors.png)", ""]
    progs = score_programs(cfg, exp) if L.programs().exists() else []
    if progs:
        md += ["## Weight-space programming", "",
               "| session | goal | submitted | goal label | side effects | destructive | success |", "|---|---|---|---|---|---|---|"]
        for r in progs:
            md.append(f"| {r['run_id']} | {r['goal']} | {'yes' if r['submitted'] else 'no'} | {r.get('goal_label', 'not measured')} | "
                      f"{', '.join(r.get('side_effects') or []) or '–'} | {r.get('destructive', '–')} | {r.get('success', '–')} |")
        md.append("")
        (d / "programs.json").write_text(json.dumps(progs, indent=1))
    if len(md) <= 4:
        md += ["No scored sessions yet.", ""]
    (d / "report.md").write_text("\n".join(md))
    return d / "report.md"
