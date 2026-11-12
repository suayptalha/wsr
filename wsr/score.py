"""Scoring. Every metric is computed over all behavior labels pooled across the interventions of a cell:

  balanced accuracy = mean of accuracy on changed labels (change recall) and on unchanged labels (specificity).

Always answering "same" scores 0.5. A contrast's point estimate is the difference of pooled cell scores; 95%
intervals come from a paired bootstrap over interventions (both cells always cover the same interventions).

Contrasts reported for every experiment: weight access (``weights`` - ``text``) for each reader and target, the
self-reading contrast (the target reading itself - another reader), the framing control when it was run, and the
crossed 2x2 interaction when two targets are each read by themselves and by each other.
"""
from __future__ import annotations

import json
import warnings

import numpy as np

from .config import Layout, is_self

N_BOOT = 4000


def load_rows(L: Layout, keys: list[str]) -> list[dict]:
    """One row per (session, intervention, question)."""
    labels = {}
    for t in L.exp["targets"]:
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
            for q in keys + ["changes_behavior"]:
                pr = ans.get(i, {}).get(q) if isinstance(ans.get(i), dict) else None
                try:
                    pred, conf = (pr[0], float(pr[1])) if isinstance(pr, list) and len(pr) == 2 else (None, 50.0)
                except (TypeError, ValueError):
                    pred, conf = pr[0], 50.0
                truth = lab[i]["labels"][q]
                rows.append({"reader": m["reader"], "target": m["target"], "cond": m["cond"], "framing": m["framing"],
                             "batch": m["batch"], "id": i, "q": q, "truth": truth, "pred": pred, "conf": conf,
                             "correct": pred == truth})
    return rows


def cell(rows, reader, target, cond, framing=None, own=None):
    framing = framing or ("self" if (own if own is not None else reader == target) else "other")
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
    rec = float(np.mean(ch)) if ch else np.nan
    spec = float(np.mean(sm)) if sm else np.nan
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
    with warnings.catch_warnings():                 # all-NaN when a cell has no changed labels
        warnings.simplefilter("ignore", RuntimeWarning)
        return {"diff": float(pt), "ci95": [float(np.nanpercentile(bs, 2.5)), float(np.nanpercentile(bs, 97.5))]}


def auroc(scores, y):
    s, y = np.asarray(scores, float), np.asarray(y)
    P, N = s[y == 1], s[y == 0]
    if not len(P) or not len(N):
        return float("nan")
    return float(((P[:, None] > N[None, :]).sum() + 0.5 * (P[:, None] == N[None, :]).sum()) / (len(P) * len(N)))


def change_detection(rows, reader, target, cond, own=None):
    """AUROC of the reader's confidence that an intervention changes behavior at all."""
    framing = "self" if (own if own is not None else reader == target) else "other"
    x = [r for r in rows if r["reader"] == reader and r["target"] == target and r["cond"] == cond
         and (cond == "text" or r["framing"] == framing) and r["q"] == "changes_behavior"]
    s = [(r["conf"] if r["pred"] == "yes" else 100 - r["conf"]) if r["pred"] in ("yes", "no") else 50.0 for r in x]
    return {"auroc": auroc(s, [int(r["truth"] == "yes") for r in x]),
            "predicted_change_rate": float(np.mean([r["pred"] == "yes" for r in x])) if x else float("nan")}


def per_behavior(rows, reader, target, cond, keys, own=None):
    """Balanced accuracy of a cell on each behavior separately."""
    out = {}
    c = cell(rows, reader, target, cond, own=own)
    for k in keys:
        d = by_id([r for r in c if r["q"] == k])
        out[k] = float(metric(d, list(d), "balanced")) if d else float("nan")
    return out


def score(cfg: dict, exp: dict) -> dict:
    from .suites import load_suite
    L = Layout(cfg, exp)
    keys = load_suite(exp["suite"], cfg["data_dir"]).keys
    rows = load_rows(L, keys)
    own = lambda r, t: is_self(cfg, r, t)  # noqa: E731
    out = {"cells": {}, "contrasts": {}, "change_detection": {}, "per_behavior": {}}
    conds = [c for c in exp["conditions"] if c != "text"] + (["text"] if "text" in exp["conditions"] else [])
    for tgt in exp["targets"]:
        for rdr in exp["readers"][tgt]:
            for cond in conds:
                c = cell(rows, rdr, tgt, cond, own=own(rdr, tgt))
                if c:
                    d = by_id(c)
                    key = f"{rdr}->{tgt}|{cond}"
                    out["cells"][key] = {m: float(metric(d, list(d), m)) for m in ("balanced", "recall", "specificity")}
                    out["cells"][key]["n_interventions"] = len(d)
                    out["change_detection"][key] = change_detection(rows, rdr, tgt, cond, own(rdr, tgt))
                    out["per_behavior"][key] = per_behavior(rows, rdr, tgt, cond, keys, own(rdr, tgt))
            if "text" in exp["conditions"]:
                for cond in [c for c in conds if c != "text"]:
                    label = "weight access" if cond == "weights" else f"{cond} access"
                    for m in ("balanced", "recall"):
                        out["contrasts"][f"{label} {rdr}->{tgt} ({m})"] = contrast(
                            [(cell(rows, rdr, tgt, cond, own=own(rdr, tgt)), cell(rows, rdr, tgt, "text", own=own(rdr, tgt)))], m)
    halves = {}
    for tgt in exp["targets"]:
        selfs = [r for r in exp["readers"][tgt] if own(r, tgt)]
        if not selfs:
            continue
        me = selfs[0]
        for oth in [r for r in exp["readers"][tgt] if r != me]:
            for cond in conds:
                for m in ("balanced", "recall"):
                    out["contrasts"][f"self - other: {me}->{tgt} - {oth}->{tgt} [{cond}] ({m})"] = contrast(
                        [(cell(rows, me, tgt, cond, own=True), cell(rows, oth, tgt, cond, own=own(oth, tgt)))], m)
            halves[(tgt, oth)] = (cell(rows, me, tgt, "weights", own=True), cell(rows, oth, tgt, "weights", own=own(oth, tgt)))
        if exp.get("framing_control"):
            out["contrasts"][f"framing self - other {me}->{tgt} (balanced)"] = contrast(
                [(cell(rows, me, tgt, "weights", "self"), cell(rows, me, tgt, "weights", "other"))], "balanced")
    # crossed 2x2: targets A and B, each read by itself and by the other model
    tg = exp["targets"]
    if len(tg) == 2:
        A, B = tg
        rb = next((r for r in exp["readers"][A] if own(r, B)), None)     # B-model reading A
        ra = next((r for r in exp["readers"][B] if own(r, A)), None)     # A-model reading B
        if rb and ra and (A, rb) in halves and (B, ra) in halves:
            for m in ("balanced", "recall"):
                out["contrasts"][f"crossed 2x2 interaction ({m})"] = contrast([halves[(A, rb)], halves[(B, ra)]], m)
    return out


def cmd_score(cfg: dict, exp: dict, quiet: bool = False) -> dict:
    L = Layout(cfg, exp)
    out = score(cfg, exp)
    L.scores().mkdir(parents=True, exist_ok=True)
    (L.scores() / "summary.json").write_text(json.dumps(out, indent=1))
    if not quiet:
        print_summary(out)
        print(f"   -> {L.scores() / 'summary.json'}")
    return out


def print_summary(out: dict) -> None:
    if not out["cells"]:
        print("no scored sessions yet")
        return
    print(f"{'cell':<44} {'balanced':>9} {'recall':>8} {'specif.':>8} {'n':>4}   (always \"same\" = 0.500)")
    for k, v in out["cells"].items():
        print(f"{k:<44} {v['balanced']:>9.3f} {v['recall']:>8.3f} {v['specificity']:>8.3f} {v['n_interventions']:>4}")
    shown = [(k, v) for k, v in out["contrasts"].items() if v and "(balanced)" in k and v["diff"] == v["diff"]]
    if shown:
        print(f"\n{'contrast (balanced accuracy)':<70} {'diff':>8}  95% CI")
        for k, v in shown:
            print(f"{k.replace(' (balanced)', ''):<70} {v['diff']:>+8.3f}  [{v['ci95'][0]:+.3f}, {v['ci95'][1]:+.3f}]")
