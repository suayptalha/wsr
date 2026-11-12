"""Ground truth: run the base model and every edited model of the pool on the suite's probe sets.

The base model is measured ``base_runs`` times with different seeds (the noise floor); each edited model once.
An experiment with a ``parent`` reuses the parent's base runs and the measurement of any edit identical to one the
parent already measured (``same_as``).
"""
from __future__ import annotations

import json
import os
import time
from pathlib import Path

from .config import Layout, base_runs, model_dir, read_jsonl


def run_ids(L: Layout, t: str) -> list[str]:
    """Base runs plus every intervention of the pool whose measurement is not inherited from the parent."""
    bases = [] if L.parent() else list(base_runs(L.exp))
    return bases + [s["id"] for s in read_jsonl(L.pool(t)) if "same_as" not in s]


def measurement_file(L: Layout, t: str, rid: str, spec: dict | None = None) -> Path:
    P = L.parent()
    if P is not None and (rid in base_runs(P.exp) or (spec and "same_as" in spec)):
        return P.measure(t) / f"{spec['same_as'] if spec and 'same_as' in spec else rid}.jsonl"
    return L.measure(t) / f"{rid}.jsonl"


def base_ids(L: Layout) -> list[str]:
    P = L.parent()
    return list(base_runs(P.exp if P is not None else L.exp))


def anatomy_for(cfg: dict, t: str, ck):
    """The anatomy used to materialize edits: the paper's fixed names for its presets, detection otherwise."""
    from .anatomy import Anatomy
    from .ops import StaticAnatomy
    m = cfg["models"][t]
    if m.get("preset"):
        return StaticAnatomy(**(m.get("anatomy") or {}))
    return Anatomy(ck, m.get("anatomy"))


def program_runs(L: Layout, t: str) -> dict[str, Path]:
    """Submitted edits of programming sessions on target t: run id -> edit file."""
    out = {}
    for d in sorted(L.programs().glob("*")):
        meta = d / "meta.json"
        if not meta.exists():
            continue
        m = json.loads(meta.read_text())
        if m.get("target") != t or not m.get("submission"):
            continue
        out[f"prog_{m['run_id']}"] = d / m["submission"]
    return out


def measure_runs(cfg: dict, exp: dict, t: str, ids: list[str] | None = None, programs: bool = False,
                 log=print) -> int:
    from .checkpoint import Checkpoint
    from .backends import final_text, make_backend
    from .ops import apply_ops, load_submission
    from .suites import load_suite
    L = Layout(cfg, exp)
    suite = load_suite(exp["suite"], cfg["data_dir"])
    items = suite.items(cfg["data_dir"])
    msgs = [suite.messages(it) for it in items]
    pool = {s["id"]: s for s in read_jsonl(L.pool(t))} if L.pool(t).exists() else {}
    progs = program_runs(L, t) if programs else {}
    todo = ids or (list(progs) if programs else run_ids(L, t))
    out = L.measure(t)
    out.mkdir(parents=True, exist_ok=True)
    if programs and L.parent() is None:          # programs are labeled against the base runs: measure them first
        todo = [b for b in base_runs(exp) if b not in todo] + todo
    todo = [r for r in todo if not (out / f"{r}.jsonl").exists()]
    if not todo:
        log(f"{t}: nothing to measure ({out})")
        return 0
    mcfg = cfg["models"][t]
    d = model_dir(cfg, t)
    ck = Checkpoint(d)
    anat = anatomy_for(cfg, t, ck)
    fmt = mcfg.get("answer_format", "think")
    log(f"{t}: measuring {len(todo)} runs x {len(items)} probe items with backend `{mcfg['backend']}`")
    backend = make_backend(d, mcfg, suite)
    seeds = base_runs(L.parent().exp if L.parent() else exp)
    done = 0
    try:
        for rid in todo:
            f = out / f"{rid}.jsonl"
            if f.exists():
                continue
            t0 = time.time()
            if rid in seeds:
                tensors, seed0 = {}, seeds[rid]
            elif rid in progs:
                tensors, seed0 = load_submission(ck, progs[rid], anat), 0
            else:
                if rid not in pool:
                    raise SystemExit(f"wsr: `{rid}` is not in the pool of {t}")
                tensors, seed0 = apply_ops(ck, pool[rid], anat)[0], 0
            backend.patch(tensors)
            outs = backend.generate(msgs, seed0)
            with open(str(f) + ".tmp", "w") as fh:
                for it, o in zip(items, outs):
                    rec = suite.grade(it, final_text(fmt, o.text), o.n_tokens, o.hit_limit)
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            os.replace(str(f) + ".tmp", f)
            done += 1
            log(f"{rid} done in {time.time() - t0:.0f}s ({done}/{len(todo)})")
    finally:
        backend.close()
    return done


def cmd_measure(cfg: dict, exp: dict, targets: list[str] | None, ids: list[str] | None = None,
                programs: bool = False) -> None:
    for t in targets or exp["targets"]:
        measure_runs(cfg, exp, t, ids, programs)
