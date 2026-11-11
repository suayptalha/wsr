"""Pool construction, the selection rule, reader batches and materialization of the reader files."""
from __future__ import annotations

import json
import math
import random

from .common import BASE_RUNS, ROUNDS, Layout, read_jsonl, write_jsonl
from .labels import changed_without_refusal, item_values
from .models import METRICS
from .pools import full_dose_pool, muse_pool, muse_standard_edits, subtle_pool


def cmd_pool(cfg: dict, a) -> None:
    from collections import Counter
    L = Layout(cfg, a.round)
    for t in (a.targets or ROUNDS[a.round]["targets"]):
        if a.round == "full_dose":
            S = full_dose_pool(t, cfg.get("agent_edits_dir"))
        elif a.round == "subtle":
            F = Layout(cfg, "full_dose")
            full = read_jsonl(F.pool(t))
            nr = [k for k in METRICS if k != "refusal"]
            vals = lambda rid: item_values(read_jsonl(F.measure(t) / f"{rid}.jsonl"), {})  # noqa: E731
            base = {k: sum((vals(b)[k] for b in BASE_RUNS), []) for k in nr}
            parents = {s["id"] for s in full if s["type"] != "F" and (F.measure(t) / f"{s['id']}.jsonl").exists()
                       and changed_without_refusal(vals(s["id"]), base)}
            S = subtle_pool(t, full, parents)
        elif a.round == "muse":
            S = muse_pool()
        else:
            S = muse_standard_edits()
        write_jsonl(L.pool(t), S)
        print(t, len(S), dict(Counter(s["type"] for s in S)))


def cmd_select(cfg: dict, a) -> None:
    """Up to n_changed interventions labeled changed (no-op controls excluded; in the Subtle round destructive edits
    excluded), taken round-robin over families shuffled with seed 11, at most `family_cap` per family, plus n_same
    unchanged ones (up to half no-op controls)."""
    from collections import Counter
    L = Layout(cfg, a.round)
    R = ROUNDS[a.round]
    for t in (a.targets or R["targets"]):
        pool = {s["id"]: s for s in read_jsonl(L.pool(t))}
        if R["family_cap"] is None:                     # standard-edit set on Muse-30B: everything is read
            write_jsonl(L.bank(t), list(pool.values()))
            continue
        lab = json.loads(L.labels(t).read_text())["labels"]
        rng = random.Random(11)
        ch: dict[str, list[str]] = {}
        for rid in sorted(lab):
            if (lab[rid]["labels"]["changes_behavior"] == "yes" and pool[rid]["type"] != "F"
                    and not (R["non_destructive_only"] and lab[rid]["destructive"])):
                ch.setdefault(pool[rid]["type"], []).append(rid)
        for v in ch.values():
            rng.shuffle(v)
        picked, fams, cap = [], sorted(ch), R["family_cap"]
        n_of = lambda f: sum(1 for p in picked if pool[p]["type"] == f)  # noqa: E731
        while len(picked) < a.n_changed and any(ch.values()):
            for f in fams:
                if ch[f] and n_of(f) < cap and len(picked) < a.n_changed:
                    picked.append(ch[f].pop())
            if all(not ch[f] or n_of(f) >= cap for f in fams):
                break
        n_changed = len(picked)
        same_f = sorted(r for r in lab if pool[r]["type"] == "F" and lab[r]["labels"]["changes_behavior"] == "no")
        same_o = sorted(r for r in lab if pool[r]["type"] != "F" and lab[r]["labels"]["changes_behavior"] == "no")
        rng.shuffle(same_f)
        rng.shuffle(same_o)
        nf = min(len(same_f), (a.n_same + 1) // 2)
        picked += same_f[:nf] + same_o[:a.n_same - nf]
        write_jsonl(L.bank(t), [pool[r] for r in picked])
        note = " (underpowered: fewer than 10 changed)" if n_changed < 10 else ""
        print(f"{t}: selected {len(picked)} | changed {n_changed}{note} | families {dict(Counter(pool[r]['type'] for r in picked))}")


def batches(L: Layout, t: str, n: int = 4) -> list[list[str]]:
    """The selected set shuffled with seed 7 and split into four batches; identical for all readers and conditions."""
    ids = [s["id"] for s in read_jsonl(L.bank(t))]
    random.Random(7).shuffle(ids)
    k = math.ceil(len(ids) / n)
    return [ids[i:i + k] for i in range(0, len(ids), k)]


def cmd_materialize(cfg: dict, a) -> None:
    """Write the compact reader file of every selected intervention and add its Text-Only description to the bank."""
    from safetensors.torch import save_file
    from .checkpoint import Checkpoint
    from .edits import apply_ops
    L = Layout(cfg, a.round)
    for t in (a.targets or ROUNDS[a.round]["targets"]):
        ck = Checkpoint(cfg["models"][t])
        out = L.reader_files(t)
        out.mkdir(parents=True, exist_ok=True)
        specs = read_jsonl(L.bank(t))
        for s in specs:
            full, compact, desc = apply_ops(ck, s)
            s["desc"], s["tensors"] = desc, sorted(full)
            p = out / f"{s['id']}.safetensors"
            if not p.exists():
                save_file(compact, str(p))
            print(s["id"], s["type"], f"{p.stat().st_size / 1e6:.1f} MB", flush=True)
        write_jsonl(L.bank(t), specs)
