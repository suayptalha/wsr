"""Scoring. Every metric is computed over all behavior labels pooled across the interventions of a cell:
  balanced accuracy = mean of accuracy on changed labels (change recall) and on unchanged labels (specificity).
A contrast's point estimate is the difference of pooled cell scores; 95% intervals come from a paired bootstrap over
interventions (both cells always cover the same interventions)."""
from __future__ import annotations

import json

import numpy as np

from .common import ROUNDS, Layout
from .models import METRICS

N_BOOT = 4000


def load_rows(L: Layout) -> list[dict]:
    """One row per (session, intervention, question)."""
    labels = {}
    for t in ROUNDS[L.rnd]["targets"]:
        if L.labels(t).exists():
            labels[t] = json.loads(L.labels(t).read_text())["labels"]
    rows = []
    for rd in sorted(L.sessions().glob("*")):
        if not (rd / "meta.json").exists():
            continue
        m = json.loads((rd / "meta.json").read_text())
        try:
            ans = json.loads((rd / "answers.json").read_text())
        except Exception:  # noqa: BLE001
            ans = {}
        lab = labels.get(m["target"], {})
        for i in m["ids"]:
            if i not in lab:
                continue
            for q in METRICS + ["changes_behavior"]:
                pr = ans.get(i, {}).get(q) if isinstance(ans.get(i), dict) else None
                pred, conf = (pr[0], float(pr[1])) if isinstance(pr, list) and len(pr) == 2 else (None, 50.0)
                truth = lab[i]["labels"][q]
                rows.append({"reader": m["reader"], "target": m["target"], "cond": m["cond"], "framing": m["framing"],
                             "batch": m["batch"], "id": i, "q": q, "truth": truth, "pred": pred, "conf": conf,
                             "correct": pred == truth})
    return rows


def cell(rows, reader, target, cond, framing=None):
    framing = framing or ("self" if reader == target else "other")
    return [r for r in rows if r["reader"] == reader and r["target"] == target and r["cond"] == cond
            and (cond == "text" or r["framing"] == framing) and r["q"] != "changes_behavior"]


def by_id(rows):
    d = {}
    for r in rows:
        d.setdefault(r["id"], {"ch": [], "sm": []})["ch" if r["truth"] != "same" else "sm"].append(r["correct"])
    return d


def metric(d, ids, m):
    ch = [x for i in ids for x in d[i]["ch"]]
    sm = [x for i in ids for x in d[i]["sm"]]
    rec = np.mean(ch) if ch else np.nan
    spec = np.mean(sm) if sm else np.nan
    return {"balanced": (rec + spec) / 2, "recall": rec, "specificity": spec}[m]


def contrast(pairs, m, n=N_BOOT, seed=0):
    """Sum over (cellA, cellB) pairs of m(A) - m(B), each pair on its shared interventions."""
    rng = np.random.default_rng(seed)
    pt, bs = 0.0, np.zeros(n)
    for A, B in pairs:
        a, b = by_id(A), by_id(B)
        ids = sorted(set(a) & set(b))
        if not ids:
            return None
        pt += metric(a, ids, m) - metric(b, ids, m)
        for k in range(n):
            s = [ids[j] for j in rng.integers(0, len(ids), len(ids))]
            bs[k] += metric(a, s, m) - metric(b, s, m)
    return {"diff": float(pt), "ci95": [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))]}


def auroc(scores, y):
    s, y = np.asarray(scores, float), np.asarray(y)
    P, N = s[y == 1], s[y == 0]
    if not len(P) or not len(N):
        return float("nan")
    return float(((P[:, None] > N[None, :]).sum() + 0.5 * (P[:, None] == N[None, :]).sum()) / (len(P) * len(N)))


def change_detection(rows, reader, target, cond):
    """AUROC of the reader's confidence that an intervention changes behavior at all."""
    framing = "self" if reader == target else "other"
    x = [r for r in rows if r["reader"] == reader and r["target"] == target and r["cond"] == cond
         and (cond == "text" or r["framing"] == framing) and r["q"] == "changes_behavior"]
    s = [(r["conf"] if r["pred"] == "yes" else 100 - r["conf"]) if r["pred"] in ("yes", "no") else 50.0 for r in x]
    return {"auroc": auroc(s, [int(r["truth"] == "yes") for r in x]),
            "predicted_change_rate": float(np.mean([r["pred"] == "yes" for r in x])) if x else float("nan")}


def cmd_score(cfg: dict, a) -> None:
    L = Layout(cfg, a.round)
    R = ROUNDS[a.round]
    rows = load_rows(L)
    out = {"cells": {}, "contrasts": {}, "change_detection": {}}
    for tgt in R["targets"]:
        for rdr in R["readers"][tgt]:
            for cond in ("weights", "text"):
                c = cell(rows, rdr, tgt, cond)
                if c:
                    d = by_id(c)
                    out["cells"][f"{rdr}->{tgt}|{cond}"] = {m: float(metric(d, list(d), m)) for m in ("balanced", "recall", "specificity")}
                    out["cells"][f"{rdr}->{tgt}|{cond}"]["n_interventions"] = len(d)
                    out["change_detection"][f"{rdr}->{tgt}|{cond}"] = change_detection(rows, rdr, tgt, cond)
            for m in ("balanced", "recall"):
                out["contrasts"][f"weight access {rdr}->{tgt} ({m})"] = contrast(
                    [(cell(rows, rdr, tgt, "weights"), cell(rows, rdr, tgt, "text"))], m)
    halves = {}
    for tgt in R["targets"]:
        own, others = R["readers"][tgt][0], R["readers"][tgt][1:]
        for oth in others:
            for cond in ("weights", "text"):
                for m in ("balanced", "recall"):
                    out["contrasts"][f"self - other: {own}->{tgt} - {oth}->{tgt} [{cond}] ({m})"] = contrast(
                        [(cell(rows, own, tgt, cond), cell(rows, oth, tgt, cond))], m)
            halves[(tgt, oth)] = (cell(rows, own, tgt, "weights"), cell(rows, oth, tgt, "weights"))
        if R["framing_control"]:
            out["contrasts"][f"framing self - other {own}->{tgt} (balanced)"] = contrast(
                [(cell(rows, own, tgt, "weights", "self"), cell(rows, own, tgt, "weights", "other"))], "balanced")
    if set(R["targets"]) == {"27b", "9b"}:          # crossed 2x2 interaction on the {27B, 9B} grid
        for m in ("balanced", "recall"):
            out["contrasts"][f"crossed 2x2 interaction ({m})"] = contrast([halves[("27b", "9b")], halves[("9b", "27b")]], m)
    (L.scores()).mkdir(parents=True, exist_ok=True)
    (L.scores() / "summary.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))
