"""Draw the diagrams of the README (docs/assets/*.svg). Deterministic; the output is committed.

    python scripts/make_figures.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

OUT = Path(__file__).resolve().parent.parent / "docs" / "assets"
INK, MUTED, LINE = "#111827", "#4b5563", "#cbd5e1"
BLUE, BLUE_BG = "#1d4ed8", "#eff6ff"
TEAL, TEAL_BG = "#0f766e", "#ecfdf5"
AMBER, AMBER_BG = "#b45309", "#fffbeb"
GRAY_BG = "#f8fafc"
plt.rcParams.update({"font.family": "DejaVu Sans", "svg.fonttype": "none"})


def box(ax, x, y, w, h, title, body="", color=BLUE, bg=BLUE_BG, title_size=10.5, body_size=8.6):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=bg, ec=color, lw=1.4))
    ax.text(x + w / 2, y + h - 0.2, title, ha="center", va="top", fontsize=title_size, weight="bold", color=color)
    if body:
        ax.text(x + w / 2, y + h - 0.55, body, ha="center", va="top", fontsize=body_size, color=INK, linespacing=1.45)


def arrow(ax, a, b, color=MUTED, text="", dy=0.12, rad=0.0, ls="-"):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=13, lw=1.4, color=color,
                                 connectionstyle=f"arc3,rad={rad}", linestyle=ls))
    if text:
        ax.text((a[0] + b[0]) / 2, (a[1] + b[1]) / 2 + dy, text, ha="center", va="bottom", fontsize=8, color=color)


def canvas(w, h):
    fig, ax = plt.subplots(figsize=(w, h))
    ax.set_xlim(0, w)
    ax.set_ylim(0, h)
    ax.axis("off")
    return fig, ax


def save(fig, name):
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / name, bbox_inches="tight", pad_inches=0.05, transparent=False, facecolor="white")
    plt.close(fig)
    print(OUT / name)


def overview():
    fig, ax = canvas(12, 4.6)
    box(ax, 0.1, 1.75, 2.5, 1.75, "Target model  $W_0$", "any decoder-only LM\nlocal or Hugging Face\nTransformers or vLLM")
    box(ax, 3.3, 2.6, 2.7, 1.7, "Edit  $\\Delta W$", "scale a block · rescale a row\nremove a direction · add noise\nrescale a norm · your own ops",
        AMBER, AMBER_BG)
    box(ax, 3.3, 0.3, 2.7, 1.8, "Ground truth", "run $W_0 + \\Delta W$ on GPU\nvs. 5 base runs (Welch test)\nup / down / same per behavior",
        TEAL, TEAL_BG)
    box(ax, 6.7, 2.6, 2.8, 1.7, "Reader agent", "any model · any API\nCPU sandbox, no GPU,\nno internet, never runs $W$",
        BLUE, BLUE_BG)
    box(ax, 6.7, 0.3, 2.8, 1.8, "Prediction", "up / down / same\nfor each behavior,\nwith a confidence", BLUE, BLUE_BG)
    box(ax, 10.1, 1.0, 1.8, 2.6, "Score", "balanced\naccuracy\n\ncontrasts\nwith 95% CI", INK, GRAY_BG)
    arrow(ax, (2.6, 3.0), (3.3, 3.4))
    arrow(ax, (2.6, 2.2), (3.3, 1.4))
    arrow(ax, (4.65, 2.6), (4.65, 2.1), AMBER)
    arrow(ax, (6.0, 3.45), (6.7, 3.45), AMBER)
    ax.text(6.35, 4.42, "weights and/or a text description", ha="center", fontsize=8, color=AMBER)
    arrow(ax, (8.1, 2.6), (8.1, 2.1), BLUE)
    arrow(ax, (4.65, 0.3), (10.6, 1.0), TEAL, rad=0.18)
    ax.text(8.1, -0.12, "measured labels", ha="center", fontsize=8, color=TEAL)
    arrow(ax, (9.5, 1.2), (10.1, 2.0), BLUE)
    ax.text(6, 4.75, "Weight-space reading: predict what an edit does from the weights alone", ha="center",
            fontsize=12.5, weight="bold", color=INK)
    save(fig, "overview.svg")


def pipeline():
    steps = [("pool", "generate edits", AMBER, AMBER_BG), ("measure", "GPU: probe sets", TEAL, TEAL_BG),
             ("label", "judge + Welch", TEAL, TEAL_BG), ("select", "balanced set", INK, GRAY_BG),
             ("materialize", "reader files", INK, GRAY_BG), ("read", "agents, sandbox", BLUE, BLUE_BG),
             ("score / report", "tables + figures", INK, GRAY_BG)]
    w, gap = 1.55, 0.28
    fig, ax = canvas(len(steps) * (w + gap) + 0.1, 1.9)
    for i, (name, sub, c, bg) in enumerate(steps):
        x = 0.1 + i * (w + gap)
        box(ax, x, 0.3, w, 1.25, name, sub, c, bg, title_size=10, body_size=8.2)
        if i:
            arrow(ax, (x - gap, 0.92), (x, 0.92))
    ax.text(0.1, 1.75, "wsr run <experiment>   =", fontsize=9.5, family="DejaVu Sans Mono", color=MUTED, va="center")
    save(fig, "pipeline.svg")


def modes():
    fig, ax = canvas(12, 4.4)
    for x0, title, color, bg, lines in [
        (0.1, "Reading  (wsr read)", BLUE, BLUE_BG,
         ["given: target weights + an edit", "task: predict how 5 behaviors move", "no forward pass, no GPU",
          "conditions: weights · text · both", "framing: self · other · neutral", "scored against measured labels"]),
        (6.1, "Programming  (wsr write)", TEAL, TEAL_BG,
         ["given: target weights + a goal", "task: write an edit that reaches it", "no training, no gradients",
          "optional budget of forward passes", "on practice items only", "scored: goal reached, nothing broken"])]:
        ax.add_patch(FancyBboxPatch((x0, 0.2), 5.8, 3.6, boxstyle="round,pad=0.02,rounding_size=0.1", fc=bg, ec=color, lw=1.6))
        ax.text(x0 + 2.9, 3.55, title, ha="center", va="top", fontsize=12.5, weight="bold", color=color)
        for i, t in enumerate(lines):
            ax.text(x0 + 0.45, 2.85 - i * 0.43, "•  " + t, fontsize=10, color=INK, va="center")
    ax.text(6, 4.25, "Two tasks, one harness: the same models, agents, sandbox and suites", ha="center", fontsize=12,
            weight="bold", color=INK)
    save(fig, "modes.svg")


def crossed():
    fig, ax = canvas(6.2, 4.0)
    ax.text(3.6, 3.75, "target (whose weights)", ha="center", fontsize=10, color=MUTED)
    ax.text(0.25, 1.55, "reader", ha="center", rotation=90, fontsize=10, color=MUTED)
    for j, t in enumerate(["A", "B"]):
        ax.text(2.4 + j * 2.4, 3.35, f"model {t}", ha="center", fontsize=10.5, weight="bold", color=INK)
    for i, r in enumerate(["A", "B"]):
        ax.text(0.95, 2.35 - i * 1.55, f"model {r}", ha="center", va="center", fontsize=10.5, weight="bold", color=INK)
        for j, t in enumerate(["A", "B"]):
            own = r == t
            c, bg = (BLUE, BLUE_BG) if own else (MUTED, GRAY_BG)
            ax.add_patch(FancyBboxPatch((1.55 + j * 2.4, 1.7 - i * 1.55), 1.7, 1.25, boxstyle="round,pad=0.02,rounding_size=0.08",
                                        fc=bg, ec=c, lw=1.4))
            ax.text(2.4 + j * 2.4, 2.33 - i * 1.55, "self" if own else "other", ha="center", va="center", fontsize=11,
                    color=c, weight="bold")
    ax.text(3.6, -0.25, "interaction = (A→A − B→A) + (B→B − A→B): self-advantage net of reader skill",
            ha="center", fontsize=8.3, color=MUTED)
    save(fig, "crossed.svg")


if __name__ == "__main__":
    overview()
    pipeline()
    modes()
    crossed()
