"""The selection rule, reader batches, and the files a reader is given.

Selection: up to ``n_changed`` interventions labeled changed (no-op controls excluded; with
``non_destructive_only`` destructive edits excluded), drawn round-robin over families shuffled with a fixed seed,
at most ``family_cap`` per family, plus ``n_same`` unchanged ones (up to half of them no-op controls). With
``select_all`` every intervention of the pool is read.
"""
from __future__ import annotations

import json
import math
import random
from collections import Counter

from .config import Layout, model_dir, read_jsonl, write_jsonl


def select_ids(pool: dict, lab: dict, sel: dict) -> tuple[list[str], int]:
    rng = random.Random(sel.get("seed", 11))
    ch: dict[str, list[str]] = {}
    for rid in sorted(lab):
        if (lab[rid]["labels"]["changes_behavior"] == "yes" and pool[rid]["type"] != "F"
                and not (sel.get("non_destructive_only") and lab[rid]["destructive"])):
            ch.setdefault(pool[rid]["type"], []).append(rid)
    for v in ch.values():
        rng.shuffle(v)
    picked, fams, cap = [], sorted(ch), sel.get("family_cap") or 10 ** 9
    n_of = lambda f: sum(1 for p in picked if pool[p]["type"] == f)  # noqa: E731
    while len(picked) < sel["n_changed"] and any(ch.values()):
        for f in fams:
            if ch[f] and n_of(f) < cap and len(picked) < sel["n_changed"]:
                picked.append(ch[f].pop())
        if all(not ch[f] or n_of(f) >= cap for f in fams):
            break
    n_changed = len(picked)
    same_f = sorted(r for r in lab if pool[r]["type"] == "F" and lab[r]["labels"]["changes_behavior"] == "no")
    same_o = sorted(r for r in lab if pool[r]["type"] != "F" and lab[r]["labels"]["changes_behavior"] == "no")
    rng.shuffle(same_f)
    rng.shuffle(same_o)
    nf = min(len(same_f), (sel["n_same"] + 1) // 2)
    picked += same_f[:nf] + same_o[:sel["n_same"] - nf]
    return picked, n_changed


def cmd_select(cfg: dict, exp: dict, targets: list[str] | None = None) -> None:
    L = Layout(cfg, exp)
    sel = exp["selection"]
    for t in targets or exp["targets"]:
        pool = {s["id"]: s for s in read_jsonl(L.pool(t))}
        if sel.get("select_all"):
            write_jsonl(L.bank(t), list(pool.values()))
            print(f"{t}: selected all {len(pool)} interventions\n   -> {L.bank(t)}")
            continue
        if not L.labels(t).exists():
            raise SystemExit(f"wsr: {t}: no labels yet; run `wsr label {exp['name']}` first")
        lab = json.loads(L.labels(t).read_text())["labels"]
        picked, n_changed = select_ids(pool, lab, sel)
        write_jsonl(L.bank(t), [pool[r] for r in picked])
        note = " (underpowered: fewer than 10 changed)" if n_changed < 10 else ""
        print(f"{t}: selected {len(picked)} | changed {n_changed}{note} | families "
              f"{dict(Counter(pool[r]['type'] for r in picked))}\n   -> {L.bank(t)}")


def batches(L: Layout, t: str) -> list[list[str]]:
    """The selected set shuffled with a fixed seed and split into batches; identical for all readers and conditions."""
    n = L.exp["selection"].get("n_batches", 4)
    ids = [s["id"] for s in read_jsonl(L.bank(t))]
    random.Random(L.exp["selection"].get("batch_seed", 7)).shuffle(ids)
    k = math.ceil(len(ids) / n) if ids else 1
    return [ids[i:i + k] for i in range(0, len(ids), k)]


def cmd_materialize(cfg: dict, exp: dict, targets: list[str] | None = None) -> None:
    """Write the compact reader file of every selected intervention and add its Text-Only description to the bank."""
    from safetensors.torch import save_file

    from .checkpoint import Checkpoint
    from .measure import anatomy_for
    from .ops import apply_ops
    L = Layout(cfg, exp)
    for t in targets or exp["targets"]:
        ck = Checkpoint(model_dir(cfg, t))
        anat = anatomy_for(cfg, t, ck)
        out = L.reader_files(t)
        out.mkdir(parents=True, exist_ok=True)
        specs = read_jsonl(L.bank(t))
        for s in specs:
            full, compact, desc = apply_ops(ck, s, anat)
            s["desc"], s["tensors"] = desc, sorted(full)
            p = out / f"{s['id']}.safetensors"
            if not p.exists():
                save_file(compact, str(p))
            print(s["id"], s["type"], f"{p.stat().st_size / 1e6:.1f} MB", flush=True)
        write_jsonl(L.bank(t), specs)
        print(f"   -> {out}")
