"""Intervention pools: the edits whose effects are measured and then read.

Pool kinds (``pool.kind`` in an experiment):

* ``standard``: a bank of surgical edits generated for **any** decoder-only checkpoint from its own tensor names
  (see ``wsr.anatomy``): block ablation and scaling, vocabulary-row scaling, direction removal, norm-matched noise,
  norm-gain scaling and no-op controls;
* ``file``: edits written by hand in a YAML or JSON file (``pool.path``), in the op vocabulary of ``wsr.ops``;
* ``doses``: weaker versions (``pool.doses``) of the edits of the ``parent`` experiment that changed a behavior;
* ``paper.full_dose``, ``paper.subtle``, ``paper.muse``, ``paper.muse_standard``: the pools of the paper, as released.

Family letters: B block ablation/scaling, C vocabulary rows, D direction removal, E noise, F no-op controls,
G norm gains, A edits proposed by an agent (paper), M gate_proj scaling, H/P/T further norm gains (paper, Muse-30B).
"""
from __future__ import annotations

import json
from collections import Counter

from .config import ConfigError, Layout, load_file, read_jsonl, write_jsonl
from .ops import dose_ops, is_centered

FAMILY_NAMES = {"A": "agent-proposed edit", "B": "block ablation / scaling", "C": "vocabulary-row scaling",
                "D": "direction removal", "E": "norm-matched noise", "F": "no-op control", "G": "norm-gain scaling",
                "H": "post-attention norm gain", "M": "gate_proj scaling", "P": "pre-FFN norm gain", "T": "final norm gain"}
STANDARD_FAMILIES = ("B", "C", "D", "E", "G", "F")


def _assign_ids(S: list[dict], target: str, tag: str) -> list[dict]:
    width = max(2, len(str(len(S) - 1)))
    for i, s in enumerate(S):
        s.setdefault("id", f"{target}_{tag}{i:0{width}d}{s['type']}")
        s["target"] = target
    return S


def _spread(n_layers: int, k: int, lo: float = 0.0, hi: float = 1.0) -> list[int]:
    """k distinct layer indices spread over the fraction [lo, hi] of the stack."""
    if n_layers <= 0:
        return []
    idx = [round(lo * (n_layers - 1) + (hi - lo) * (n_layers - 1) * i / max(1, k - 1)) for i in range(k)]
    return sorted(set(min(n_layers - 1, max(0, i)) for i in idx))


# ---------------------------------------------------------------- the standard bank for any model
def standard_pool(ck, anat, roles: dict, target: str, opts: dict | None = None) -> list[dict]:
    """A bank of edits built from the checkpoint's own names. ``opts``: ``families`` (subset of BCDEGF),
    ``size`` (``small``: about 40 edits, ``medium``: about 80, default), ``seed`` (noise seeds)."""
    o = {"families": list(STANDARD_FAMILIES), "size": "medium", "seed": 0, **(opts or {})}
    big = o["size"] != "small"
    fam = set(o["families"])
    L = anat.n_layers
    layers = sorted(anat.layers)
    lay = lambda f: layers[min(L - 1, max(0, round(f * (L - 1))))]  # noqa: E731
    U = anat.unembed
    S: list[dict] = []
    add = lambda typ, ops: S.append({"type": typ, "ops": ops})  # noqa: E731
    sc = lambda n, f: {"op": "scale", "tensor": n, "factor": f}  # noqa: E731
    rows = lambda r, f: {"op": "scale_rows", "tensor": U, "rows": r, "factor": f, **({"tied": True} if anat.tied else {})}  # noqa: E731
    mlp = [anat.mlp_out(l) for l in layers]
    if None in mlp and fam & {"B", "D", "E"}:
        raise ConfigError(f"{target}: could not find the MLP output matrices; set models.{target}.anatomy.mlp_out "
                          "(e.g. 'model.layers.{l}.mlp.down_proj.weight')")
    seed = int(o["seed"])

    if "B" in fam:
        for l in _spread(L, 8 if big else 5):
            add("B", [sc(mlp[l], 0.0)])
        for f in ((0.25, 0.5, 0.75, 1.0) if big else (0.5, 1.0)):
            add("B", [sc(mlp[layers.index(lay(f))], 0.5)])
        attn = [(l, anat.attn_out(l)) for l in layers]
        attn = [(l, n) for l, n in attn if n]
        if attn:
            for i in _spread(len(attn), 4 if big else 2, 0.1, 1.0):
                add("B", [sc(attn[i][1], 0.0)])
    if "C" in fam:
        if "close" in roles:
            for f in ((0.0, 0.5, 1.5, 3.0) if big else (0.0, 2.0)):
                add("C", [rows([roles["close"]], f)])
        if "eos" in roles:
            for f in ((0.5, 1.5) if big else (1.5,)):
                add("C", [rows([roles["eos"]], f)])
        if "letters" in roles:
            first = roles["letters"][0]
            for f in ((0.0, 1.5) if big else (0.0,)):
                add("C", [rows([first], f)])
            for f in ((0.7, 1.3) if big else (1.3,)):
                add("C", [rows(roles["letters"][:4], f)])
        if "digits" in roles and big:
            add("C", [rows(roles["digits"], 0.8)])
            add("C", [rows([roles["digits"][7]], 2.0)])
    if "D" in fam:
        dirs = [("close", [roles["close"]])] if "close" in roles else []
        if "letters" in roles:
            dirs.append(("letters", roles["letters"][:4]))
        if "eos" in roles and not dirs:
            dirs.append(("eos", [roles["eos"]]))
        for _name, u in dirs:
            for start in ((0.5, 0.75) if big else (0.75,)):
                add("D", [{"op": "orth", "tensors": [mlp[i] for i in range(layers.index(lay(start)), L)], "u_rows": u}])
    if "E" in fam:
        for i, f in enumerate((0.25, 0.5, 0.75, 1.0) if big else (0.5, 1.0)):
            n = mlp[layers.index(lay(f))]
            add("E", [{"op": "noise", "tensor": n, "rel": 0.5, "seed": seed + 100 + i}])
            if big:
                add("E", [{"op": "noise", "tensor": n, "rel": 1.0, "seed": seed + 200 + i}])
    if "G" in fam:
        kinds = anat.norm_kinds()
        groups = [layers[: L // 2], layers[L // 2:]] if L > 1 else [layers]
        for kind in kinds[: (3 if big else 2)]:
            for g in groups:
                ns = [anat.norms(l)[kind] for l in g]
                c = is_centered(ck, ns[0])
                for f in ((0.5, 2.0) if big else (2.0,)):
                    add("G", [{"op": "gain", "tensors": ns, "factor": f, "centered": c}])
        if anat.final_norm:
            for f in ((0.5, 2.0) if big else (2.0,)):
                add("G", [{"op": "gain", "tensors": [anat.final_norm], "factor": f,
                           "centered": is_centered(ck, anat.final_norm)}])
    if "F" in fam:
        ref = roles.get("close", roles.get("eos", (roles.get("letters") or [0])[0]))
        for f in ((1.01, 0.99) if big else (1.01,)):
            add("F", [rows([ref], f)])
        for r in (roles.get("unused") or [])[: (2 if big else 1)]:
            add("F", [rows([r], 0.0)])
        mid = mlp[layers.index(lay(0.5))]
        add("F", [{"op": "noise", "tensor": mid, "rel": 1e-4, "seed": seed + 300}])
        add("F", [sc(mid, 1.001)])
        if big:
            add("F", [sc(mlp[layers.index(lay(0.9))], 1.001)])
    return _assign_ids(S, target, "S")


# ---------------------------------------------------------------- edits from a file
def file_pool(path: str, target: str) -> list[dict]:
    """A list of ``{type?, ops}`` (or ``{edits: [...]}``); ``{target}`` in a tensor name is not expanded."""
    raw = load_file(path)
    edits = raw.get("edits", raw) if isinstance(raw, dict) else raw
    if not isinstance(edits, list):
        raise ConfigError(f"{path}: expected a list of edits (or a mapping with `edits:`)")
    S = []
    for e in edits:
        if "ops" not in e:
            raise ConfigError(f"{path}: every edit needs `ops`")
        if isinstance(e.get("target"), str) and e["target"] != target:
            continue
        S.append({"type": e.get("type", "X"), "ops": e["ops"], **({"id": e["id"]} if "id" in e else {}),
                  **({"note": e["note"]} if "note" in e else {})})
    return _assign_ids(S, target, "U")


# ---------------------------------------------------------------- dose variants of a parent experiment
def dose_pool(parent_pool: list[dict], changed: set[str], target: str, doses=(0.5, 0.25)) -> list[dict]:
    key = lambda ops: json.dumps(ops, sort_keys=True)  # noqa: E731
    by_key = {key(s["ops"]): s["id"] for s in parent_pool}
    out, seen = [], set()
    for s in parent_pool:
        if s["type"] == "F":
            out.append({"type": "F", "ops": s["ops"], "parent": s["id"], "same_as": s["id"]})
            continue
        if s["id"] not in changed:
            continue
        for d in doses:
            ops = dose_ops(s["ops"], d)
            if key(ops) in seen:
                continue
            seen.add(key(ops))
            out.append({"type": s["type"], "ops": ops, "parent": s["id"], "dose": d,
                        **({"same_as": by_key[key(ops)]} if key(ops) in by_key else {})})
    return _assign_ids(out, target, "M")


def changed_parents(PL: Layout, t: str, suite) -> set[str]:
    """Edits of the parent experiment that changed a behavior that needs no judge (the judge has not run yet)."""
    from .labels import changed_without_judge
    base = PL.measure(t)
    from .config import base_runs
    bvals = [suite.values(read_jsonl(base / f"{b}.jsonl")) for b in base_runs(PL.exp) if (base / f"{b}.jsonl").exists()]
    if not bvals:
        raise ConfigError(f"{PL.name}: no base runs measured for {t}; measure the parent experiment first")
    pooled = {k: sum((v[k] for v in bvals), []) for k in suite.keys}
    out = set()
    for s in read_jsonl(PL.pool(t)):
        f = base / f"{s['id']}.jsonl"
        if s["type"] != "F" and f.exists() and changed_without_judge(suite, suite.values(read_jsonl(f)), pooled):
            out.add(s["id"])
    return out


# ---------------------------------------------------------------- command
def build_pool(cfg: dict, exp: dict, t: str) -> list[dict]:
    from .config import model_dir
    kind = exp["pool"]["kind"]
    if kind == "paper.full_dose":
        from .paper.pools import full_dose_pool
        return full_dose_pool(t, exp["pool"].get("agent_edits_dir") or cfg.get("agent_edits_dir"))
    if kind == "paper.muse":
        from .paper.pools import muse_pool
        return muse_pool()
    if kind == "paper.muse_standard":
        from .paper.pools import muse_standard_edits
        return muse_standard_edits()
    if kind == "paper.subtle":
        from .paper.pools import subtle_pool
        from .suites import load_suite
        PL = Layout(cfg, exp).parent()
        if PL is None:
            raise ConfigError(f"{exp['name']}: pool `paper.subtle` needs a `parent` experiment")
        return subtle_pool(t, read_jsonl(PL.pool(t)), changed_parents(PL, t, load_suite(PL.exp["suite"])))
    if kind == "doses":
        from .suites import load_suite
        PL = Layout(cfg, exp).parent()
        if PL is None:
            raise ConfigError(f"{exp['name']}: pool `doses` needs a `parent` experiment")
        return dose_pool(read_jsonl(PL.pool(t)), changed_parents(PL, t, load_suite(PL.exp["suite"])), t,
                         tuple(exp["pool"].get("doses", (0.5, 0.25))))
    if kind == "file":
        return file_pool(exp["pool"]["path"], t)
    if kind == "standard":
        from .anatomy import Anatomy, token_roles
        from .checkpoint import Checkpoint
        d = model_dir(cfg, t)
        ck = Checkpoint(d)
        m = cfg["models"][t]
        anat = Anatomy(ck, m.get("anatomy"))
        return standard_pool(ck, anat, token_roles(d, m.get("tokens")), t, exp["pool"])
    raise ConfigError(f"{exp['_source']}: unknown pool kind `{kind}`")


def cmd_pool(cfg: dict, exp: dict, targets: list[str] | None = None) -> None:
    L = Layout(cfg, exp)
    for t in targets or exp["targets"]:
        S = build_pool(cfg, exp, t)
        ids = [s["id"] for s in S]
        if len(ids) != len(set(ids)):
            raise ConfigError(f"{t}: duplicate intervention ids in the pool")
        write_jsonl(L.pool(t), S)
        fams = Counter(s["type"] for s in S)
        print(f"{t}: {len(S)} interventions | " + ", ".join(f"{k} {FAMILY_NAMES.get(k, k)}: {v}" for k, v in sorted(fams.items())))
        print(f"   -> {L.pool(t)}")


__all__ = ["standard_pool", "file_pool", "dose_pool", "build_pool", "cmd_pool", "FAMILY_NAMES"]

