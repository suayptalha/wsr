"""Collect every program run and baseline result under $WM_ROOT into one table.

    python -m wsr.wm.summarize            # prints markdown, writes $WM_ROOT/results/summary.{md,json}
"""
from __future__ import annotations

import json
from pathlib import Path


def collect_runs(root: Path) -> list[dict]:
    rows = []
    for p in sorted((root / "runs").glob("*/result.json")):
        r = json.loads(p.read_text())
        h = r["history"]
        usage = [x.get("usage") or {} for x in h]
        rows.append({"run": p.parent.name, "shift": r["shift"], "info": r["info"], "passes": r["passes"],
                     "agent": ",".join(r.get("agents") or []), "text_from": r.get("text_from"),
                     "latents": r["latents"], "base_new": r["base_new"], "final_new": r["final_new"],
                     "rel_gain": 1 - r["final_new"] / r["base_new"], "ci": r["final_gain_vs_base"]["ci"],
                     "base_old": r["base_old"], "final_old": r["final_old"], "accepted": r["accepted_rounds"],
                     "best_attempt_new": min([x.get("heldout_new", float("inf")) for x in h] or [float("inf")]),
                     "tokens_in": sum(u.get("input_tokens", 0) for u in usage),
                     "tokens_out": sum(u.get("output_tokens", 0) for u in usage),
                     "passes_used": sum(x.get("passes_used") or 0 for x in h),
                     "audit": {k: sum((x.get("audit") or {}).get(k, 0) for x in h) for k in ("snippets", "autograd", "optimizer", "lstsq", "forward_reimpl")},
                     "tensors": sorted({t for x in h for t in x.get("tensors", [])})[:8]})
    return rows


def collect_baselines(root: Path) -> list[dict]:
    rows = []
    for p in sorted((root / "results" / "baselines").glob("*.json")):
        r = json.loads(p.read_text())
        for run in r["runs"]:
            c = {x["step"]: x for x in run["curve"]}
            rows.append({"shift": r["shift"], "data": run["data"], "n": run["n_segments"], "method": run["method"],
                         "lr": run["lr"], "n_params": run["n_params"], "base_new": r["base_new"],
                         **{f"new@{s}": c[s]["heldout_new"] for s in sorted(c)},
                         "old_final": run["final"]["heldout_old"]})
    return rows


def main():
    import os
    root = Path(os.environ.get("WM_ROOT", "wm-work"))
    runs, bls = collect_runs(root), collect_baselines(root)
    L = ["## Agent runs", "", "| run | shift | info | passes | latents | base | final | rel. reduction | CI (abs) | old physics | tokens in/out | audit |",
         "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for r in runs:
        a = r["audit"]
        L.append(f"| {r['run']} | {r['shift']} | {r['info']} | {r['passes']} | {r['latents']} | {r['base_new']:.4f} | "
                 f"{r['final_new']:.4f} | {100 * r['rel_gain']:.1f}% | [{r['ci'][0]:.4f}, {r['ci'][1]:.4f}] | "
                 f"{r['base_old']:.4f}->{r['final_old']:.4f} | {r['tokens_in'] / 1e6:.2f}M/{r['tokens_out'] / 1e3:.0f}k | "
                 f"{a['snippets']} sn, lstsq {a['lstsq']}, grad {a['autograd']}, opt {a['optimizer']}, fwd {a['forward_reimpl']} |")
    L += ["", "## Gradient baselines (held-out changed-physics rollout MSE after k Adam steps)", ""]
    if bls:
        steps = sorted({int(k.split("@")[1]) for b in bls for k in b if k.startswith("new@")})
        L.append("| shift | data (n) | method | lr | params | base | " + " | ".join(f"@{s}" for s in steps) + " | old@end |")
        L.append("|" + "---|" * (7 + len(steps)))
        for b in bls:
            L.append(f"| {b['shift']} | {b['data']} ({b['n']}) | {b['method']} | {b['lr']:g} | {b['n_params']} | {b['base_new']:.4f} | "
                     + " | ".join(f"{b.get(f'new@{s}', float('nan')):.4f}" for s in steps) + f" | {b['old_final']:.4f} |")
    md = "\n".join(L) + "\n"
    out = root / "results"
    out.mkdir(parents=True, exist_ok=True)
    (out / "summary.md").write_text(md)
    (out / "summary.json").write_text(json.dumps({"runs": runs, "baselines": bls}, indent=1))
    print(md)


if __name__ == "__main__":
    main()
