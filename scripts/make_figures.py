"""Draw the banner and the diagrams of the README (docs/assets/*.svg). Deterministic; the output is committed.

    python scripts/make_figures.py

Plain SVG, no dependencies. Every figure sits on its own white panel so it reads well on GitHub's light and dark
themes alike.
"""
from __future__ import annotations

import random
from pathlib import Path
from xml.sax.saxutils import escape

OUT = Path(__file__).resolve().parent.parent / "docs" / "assets"
FONT = "Inter, 'Segoe UI', -apple-system, 'Helvetica Neue', Helvetica, Arial, sans-serif"
MONO = "'JetBrains Mono', 'SFMono-Regular', Menlo, Consolas, monospace"
INK, MUTED, FAINT, LINE, PANEL = "#0f172a", "#475569", "#94a3b8", "#e2e8f0", "#ffffff"
C = {  # accent, soft fill
    "blue": ("#2563eb", "#eff6ff"), "teal": ("#0d9488", "#f0fdfa"), "amber": ("#d97706", "#fffbeb"),
    "violet": ("#7c3aed", "#f5f3ff"), "slate": ("#334155", "#f8fafc"), "rose": ("#e11d48", "#fff1f2"),
}


class SVG:
    def __init__(self, w: int, h: int, panel: bool = True):
        self.w, self.h, self.parts = w, h, []
        self.defs = [
            '<filter id="sh" x="-10%" y="-10%" width="120%" height="140%">'
            '<feDropShadow dx="0" dy="2" stdDeviation="3" flood-color="#0f172a" flood-opacity="0.08"/></filter>']
        for name, (acc, _) in C.items():
            self.defs.append(f'<marker id="ar-{name}" viewBox="0 0 10 10" refX="8.5" refY="5" markerWidth="7" '
                             f'markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" fill="{acc}"/></marker>')
        if panel:
            self.parts.append(f'<rect x="0.5" y="0.5" width="{w - 1}" height="{h - 1}" rx="18" fill="{PANEL}" stroke="{LINE}"/>')

    def add(self, s: str):
        self.parts.append(s)

    def text(self, x, y, s, size=14, color=INK, weight=400, anchor="start", family=FONT, extra=""):
        self.add(f'<text x="{x}" y="{y}" font-family="{family}" font-size="{size}" font-weight="{weight}" fill="{color}" '
                 f'text-anchor="{anchor}" {extra}>{escape(s)}</text>')

    def card(self, x, y, w, h, color="slate", title=None, lines=(), badge=None, tag=None, title_size=16, line_size=13):
        acc, soft = C[color]
        self.add(f'<g filter="url(#sh)"><rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="#ffffff" stroke="{LINE}"/></g>')
        self.add(f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="14" fill="{soft}" stroke="{acc}" stroke-opacity="0.35"/>')
        self.add(f'<rect x="{x}" y="{y + 16}" width="4" height="{h - 32}" rx="2" fill="{acc}"/>')
        ty = y + 32
        if badge:
            self.add(f'<circle cx="{x + 30}" cy="{ty - 5}" r="12" fill="{acc}"/>')
            self.text(x + 30, ty, str(badge), 12, "#ffffff", 700, "middle")
        if title:
            self.text(x + (50 if badge else 22), ty, title, title_size, INK, 700)
        if tag:
            tw = 9 + 7.2 * len(tag)
            self.add(f'<rect x="{x + w - tw - 14}" y="{ty - 16}" width="{tw}" height="22" rx="11" fill="#ffffff" stroke="{acc}" stroke-opacity="0.5"/>')
            self.text(x + w - tw / 2 - 14, ty - 1, tag, 11, acc, 600, "middle", MONO)
        for i, ln in enumerate(lines):
            self.text(x + 22, ty + 28 + i * (line_size + 8), ln, line_size, MUTED)

    def arrow(self, pts, color="slate", dash=False, width=2):
        d = "M" + " L".join(f"{a},{b}" for a, b in pts)
        da = ' stroke-dasharray="6 5"' if dash else ""
        self.add(f'<path d="{d}" fill="none" stroke="{C[color][0]}" stroke-width="{width}" stroke-linejoin="round"'
                 f'{da} marker-end="url(#ar-{color})"/>')

    def curve(self, d, color="slate", dash=False):
        da = ' stroke-dasharray="6 5"' if dash else ""
        self.add(f'<path d="{d}" fill="none" stroke="{C[color][0]}" stroke-width="2"{da} marker-end="url(#ar-{color})"/>')

    def label(self, x, y, s, color="slate", size=12):
        w = 12 + 6.6 * len(s)
        self.add(f'<rect x="{x - w / 2}" y="{y - 14}" width="{w}" height="21" rx="10.5" fill="#ffffff" stroke="{LINE}"/>')
        self.text(x, y + 1, s, size, C[color][0], 600, "middle")

    def save(self, name):
        OUT.mkdir(parents=True, exist_ok=True)
        body = "\n".join(self.parts)
        (OUT / name).write_text(f'<svg xmlns="http://www.w3.org/2000/svg" width="{self.w}" height="{self.h}" '
                                f'viewBox="0 0 {self.w} {self.h}"><defs>{"".join(self.defs)}</defs>\n{body}\n</svg>\n')
        print(OUT / name)


# ---------------------------------------------------------------- banner
def banner():
    W, H = 1280, 320
    s = SVG(W, H, panel=False)
    s.defs += ['<linearGradient id="bg" x1="0" y1="0" x2="1" y2="1"><stop offset="0" stop-color="#0b1220"/>'
               '<stop offset="1" stop-color="#1e293b"/></linearGradient>',
               '<linearGradient id="glow" x1="0" y1="0" x2="1" y2="0"><stop offset="0" stop-color="#60a5fa"/>'
               '<stop offset="1" stop-color="#a78bfa"/></linearGradient>',
               '<clipPath id="lens"><circle cx="1010" cy="169" r="74"/></clipPath>']
    s.add(f'<rect width="{W}" height="{H}" rx="24" fill="url(#bg)"/>')
    # weight matrix
    rng = random.Random(7)
    x0, y0, cw, ch, cols, rows = 760, 46, 21, 19, 22, 12
    edit_row = 6
    cells = []
    for r in range(rows):
        for c in range(cols):
            v = rng.random()
            if r == edit_row:
                cells.append(f'<rect x="{x0 + c * cw}" y="{y0 + r * ch}" width="{cw - 3}" height="{ch - 3}" rx="3" '
                             f'fill="#f59e0b" fill-opacity="{0.45 + 0.55 * v:.2f}"/>')
            else:
                col = "#60a5fa" if v > 0.5 else "#a78bfa"
                cells.append(f'<rect x="{x0 + c * cw}" y="{y0 + r * ch}" width="{cw - 3}" height="{ch - 3}" rx="3" '
                             f'fill="{col}" fill-opacity="{0.08 + 0.42 * abs(v - 0.5) * 2:.2f}"/>')
    s.add("".join(cells))
    # magnified view inside the lens
    s.add('<g clip-path="url(#lens)"><rect x="930" y="89" width="160" height="160" fill="#0b1220"/>')
    lens = []
    row_mid = y0 + edit_row * ch + (ch - 3) / 2                      # the lens is centred on the edited row
    for r in range(5):
        for c in range(4):
            v = rng.random()
            fill, op = ("#f59e0b", 0.5 + 0.5 * v) if r == 2 else ("#60a5fa", 0.15 + 0.5 * v)
            lens.append(f'<rect x="{944 + c * 34}" y="{row_mid - 15 + (r - 2) * 34:.1f}" width="30" height="30" rx="6" '
                        f'fill="{fill}" fill-opacity="{op:.2f}"/>')
    s.add("".join(lens) + "</g>")
    s.add('<circle cx="1010" cy="169" r="74" fill="none" stroke="#e2e8f0" stroke-width="6"/>')
    s.add('<path d="M1063,222 L1104,263" stroke="#e2e8f0" stroke-width="16" stroke-linecap="round"/>')
    s.text(x0, y0 + rows * ch + 30, "W[6, :]  ←  W[6, :] × 0.5", 15, "#fbbf24", 600, family=MONO)
    # wordmark
    s.text(72, 168, "wsr", 132, "#ffffff", 800, extra='letter-spacing="-4"')
    s.add('<rect x="76" y="190" width="214" height="6" rx="3" fill="url(#glow)"/>')
    s.text(74, 236, "weight-space reading & programming", 30, "#e2e8f0", 600)
    s.text(74, 272, "Can a model tell what an edit to its weights will do, without running it? Can it write one?", 17, "#94a3b8")
    s.save("banner.svg")


# ---------------------------------------------------------------- overview
def overview():
    s = SVG(1180, 500)
    s.text(40, 52, "Weight-space reading", 24, INK, 800)
    s.text(40, 80, "An edit is applied to a model's weights. The edited model is run to get the truth; a reader agent predicts it from the weights alone.", 14, MUTED)
    top, h = 120, 170
    s.card(40, top, 220, h, "slate", "Target model", ["any decoder-only LM", "local or Hugging Face", "Transformers or vLLM"], tag="W₀")
    s.card(320, top, 240, h, "amber", "Edit", ["rescale a block or a row", "remove a direction", "noise · norm gains", "or your own ops"], tag="ΔW")
    s.card(620, top, 250, h, "blue", "Reader agent", ["any model, any API or CLI", "sees the weights and/or a", "description of ΔW", "CPU sandbox, never runs W"], tag="read")
    s.card(620, 330, 250, 140, "teal", "Ground truth", ["run W₀ + ΔW on the probes", "Welch test vs. 5 base runs", "up / down / same"], tag="GPU")
    s.card(930, top, 210, 350, "violet", "Score", ["prediction vs. truth,", "per behavior", "", "balanced accuracy", "recall · specificity", "", "contrasts with", "95% bootstrap CIs"])
    mid = top + h / 2
    s.arrow([(260, mid), (320, mid)])
    s.arrow([(560, mid), (620, mid)], "amber")
    s.arrow([(440, top + h), (440, 400), (620, 400)], "amber")
    s.arrow([(870, mid), (930, mid)], "blue")
    s.label(900, mid - 30, "ŷ", "blue")
    s.arrow([(870, 400), (930, 400)], "teal")
    s.label(900, 370, "y", "teal")
    s.save("overview.svg")


# ---------------------------------------------------------------- pipeline
def pipeline():
    steps = [("pool", "generate edits", "amber", "CPU"), ("measure", "run the probes", "teal", "GPU"),
             ("label", "judge + Welch", "teal", "CPU"), ("select", "balanced set", "slate", "CPU"),
             ("materialize", "reader files", "slate", "CPU"), ("read", "agents in sandbox", "blue", "agent"),
             ("report", "tables + figures", "violet", "CPU")]
    w, gap, x0 = 150, 22, 40
    s = SVG(x0 * 2 + len(steps) * w + (len(steps) - 1) * gap, 210)
    s.text(40, 46, "wsr run <experiment>", 18, INK, 700, family=MONO)
    s.text(300, 46, "every step is resumable: finished work is skipped", 14, MUTED)
    for i, (name, sub, col, hw) in enumerate(steps):
        x = x0 + i * (w + gap)
        acc, soft = C[col]
        s.add(f'<g filter="url(#sh)"><rect x="{x}" y="76" width="{w}" height="104" rx="14" fill="#ffffff" stroke="{LINE}"/></g>')
        s.add(f'<rect x="{x}" y="76" width="{w}" height="104" rx="14" fill="{soft}" stroke="{acc}" stroke-opacity="0.35"/>')
        s.add(f'<circle cx="{x + 24}" cy="100" r="11" fill="{acc}"/>')
        s.text(x + 24, 104, str(i + 1), 11, "#ffffff", 700, "middle")
        s.text(x + 42, 105, name, 15, INK, 700)
        s.text(x + 16, 135, sub, 12.5, MUTED)
        s.add(f'<rect x="{x + 16}" y="149" width="{14 + 7 * len(hw)}" height="20" rx="10" fill="#ffffff" stroke="{acc}" stroke-opacity="0.5"/>')
        s.text(x + 23 + 3.5 * len(hw), 163, hw, 11, acc, 600, "middle", MONO)
        if i:
            s.arrow([(x - gap + 3, 128), (x - 3, 128)])
    s.save("pipeline.svg")


# ---------------------------------------------------------------- the three modes
def modes():
    s = SVG(1180, 420)
    s.text(40, 52, "One toolkit, three ways to use it", 24, INK, 800)
    s.text(40, 80, "The same models, agents, sandbox, suites and measurement underneath.", 14, MUTED)
    cols = [("blue", "Read", "wsr read · wsr run", ["given the weights and an edit", "predict how each behavior moves",
                                                    "no forward pass, no GPU", "weights · text · both", "self · other framing",
                                                    "scored against measured labels"]),
            ("teal", "Write", "wsr write", ["given the weights and a goal", "write an edit, by reasoning",
                                           "no training, no gradients", "optional forward-pass budget",
                                           "practice items only", "scored: goal met, nothing broken"]),
            ("violet", "Program", "wsr program · wsr.program()", ["a fine-tuning-style config", "goal defined by a dataset",
                                                                  "attempts × rounds, best edit wins", "held-out + guard checks",
                                                                  "saves a new checkpoint", "optional push to the Hub"])]
    for i, (col, title, cmd, lines) in enumerate(cols):
        x = 40 + i * 375
        s.card(x, 110, 350, 280, col, title, [])
        s.text(x + 22, 168, cmd, 13, C[col][0], 600, family=MONO)
        for j, ln in enumerate(lines):
            s.add(f'<circle cx="{x + 28}" cy="{203 + j * 30}" r="3.5" fill="{C[col][0]}"/>')
            s.text(x + 42, 208 + j * 30, ln, 14, MUTED)
    s.save("modes.svg")


# ---------------------------------------------------------------- crossed design
def crossed():
    s = SVG(640, 440)
    s.text(40, 50, "Crossed 2×2: is self-reading special?", 20, INK, 800)
    s.text(400, 104, "target: whose weights", 13, MUTED, 600, "middle")
    s.text(56, 255, "reader", 13, MUTED, 600, "middle", extra='transform="rotate(-90 56 255)"')
    for j, t in enumerate("AB"):
        s.text(290 + j * 220, 138, f"model {t}", 15, INK, 700, "middle")
    for i, r in enumerate("AB"):
        s.text(118, 208 + i * 120, f"model {r}", 15, INK, 700, "middle")
        for j, t in enumerate("AB"):
            own = r == t
            acc, soft = C["blue"] if own else C["slate"]
            x, y = 190 + j * 220, 154 + i * 120
            s.add(f'<rect x="{x}" y="{y}" width="200" height="104" rx="14" fill="{soft}" stroke="{acc}" '
                  f'stroke-opacity="{0.6 if own else 0.25}" stroke-width="{2 if own else 1}"/>')
            s.text(x + 100, y + 48, "self" if own else "other", 18, acc if own else MUTED, 700, "middle")
            s.text(x + 100, y + 74, f"{r} reads {t}", 13, MUTED, 400, "middle", MONO)
    s.text(320, 412, "interaction = (A→A − B→A) + (B→B − A→B): a self-advantage net of reader skill", 13, MUTED, 400, "middle")
    s.save("crossed.svg")


# ---------------------------------------------------------------- wsr program loop
def program_loop():
    s = SVG(1180, 400)
    s.text(40, 52, "wsr program: from a goal to an edited checkpoint", 24, INK, 800)
    s.text(40, 80, "Like a fine-tuning run, without the training: the agent writes the edit; wsr decides whether to keep it.", 14, MUTED)
    s.card(40, 120, 230, 230, "slate", "Inputs", ["base model", "goal + dataset", "  (practice | held-out)", "guard suites",
                                                  "agent(s)", "acceptance rule"], line_size=13)
    s.card(320, 120, 250, 230, "teal", "Attempts", ["agent reads the weights", "in the sandbox", "run_model on practice",
                                                    "check_edit (free)", "submits edit.safetensors", "+ rationale.md"], tag="×N")
    s.card(620, 120, 250, 230, "amber", "Accept?", ["held-out gain > 0,", "bootstrap CI excludes 0", "guards unchanged (Welch)",
                                                    "generation not broken", "size cap (optional)", "best accepted edit wins"])
    s.card(920, 120, 220, 230, "violet", "Output", ["model/ checkpoint", "  + model card", "edit.safetensors", "report.md",
                                                    "sessions/ (all logs)", "→ Hugging Face Hub"])
    s.arrow([(270, 235), (320, 235)])
    s.arrow([(570, 235), (620, 235)], "teal")
    s.arrow([(870, 235), (920, 235)], "amber")
    s.label(895, 205, "yes", "amber")
    s.curve("M745,350 C745,392 445,392 445,354", "amber", dash=True)
    s.label(595, 386, "next round starts from the accepted model", "amber")
    s.save("program.svg")


if __name__ == "__main__":
    banner()
    overview()
    pipeline()
    modes()
    crossed()
    program_loop()
