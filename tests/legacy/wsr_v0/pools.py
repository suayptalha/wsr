"""Intervention pools (deterministic specs). An intervention is {"id", "target", "type", "ops": [...]}.

Families (type letters): A agent-proposed edits, B block ablation/scaling, C vocabulary-row scaling, D direction removal,
E norm-matched noise, F no-op controls; Muse-30B only: G post-FFN norm gain, H post-attention norm gain,
P pre-FFN norm gain, M MLP gate_proj scaling, T final norm gain.

Ops:
  scale       {"tensor", "factor"}                       W <- factor * W
  scale_rows  {"tensor", "rows", "factor"}               W[rows] <- factor * W[rows]
  orth        {"tensors", "u_rows", "alpha"?}            W <- W - alpha * u u^T W  (u from the unembedding rows)
  noise       {"tensor", "rel", "seed"}                  W <- W + N, ||N||_F = rel * ||W||_F
  gain        {"tensors", "factor", "centered"}          effective RMSNorm gain (1 + w or w) * factor
  agent_edit  {"source", "scale"?}                       W0 + scale * (W_edit - W0)
"""
from __future__ import annotations

import json
from pathlib import Path

from .models import FINAL_NORM, LM_HEAD, MODELS, TOKENS, attn_out, mlp, norm, oproj

# Agent-proposed edits (27B only): eight edits proposed by Qwen3.8-27B in the self-editing pilot.
# They are not distributed; place them as agent_edit_1.safetensors ... agent_edit_8.safetensors in the
# directory given by "agent_edits_dir" in the config to include this family.
N_AGENT_EDITS = 8

sc = lambda n, f: {"op": "scale", "tensor": n, "factor": f}                       # noqa: E731
sr = lambda rows, f: {"op": "scale_rows", "tensor": LM_HEAD, "rows": rows, "factor": f}  # noqa: E731
nz = lambda n, r, s: {"op": "noise", "tensor": n, "rel": r, "seed": s}            # noqa: E731
gain = lambda ns, f, c=True: {"op": "gain", "tensors": ns, "factor": f, "centered": c}  # noqa: E731


def _ids(S: list[dict], target: str, tag: str, width: int) -> list[dict]:
    for i, s in enumerate(S):
        s["id"] = f"{target}_{tag}{i:0{width}d}{s['type']}"
        s["target"] = target
    return S


# ---------------------------------------------------------------- standard edits on Muse-30B
def muse_standard_edits() -> list[dict]:
    """34 standard edits and 6 no-op controls applied to Muse-30B (all read; no selection)."""
    k = TOKENS["muse30b"]
    S = []
    add = lambda typ, ops: S.append({"type": typ, "ops": ops})  # noqa: E731
    for op in [sc(mlp(1), 0.0), sc(mlp(13), 0.0), sc(mlp(26), 0.0), sc(mlp(51), 0.0), sc(mlp(51), 0.5), sc(mlp(25), 0.5),
               sc(oproj(3), 0.0), sc(oproj(51), 0.0), sc(oproj(10), 0.0), sc(oproj(40), 0.0), sc(mlp(5), 0.0), sc(mlp(45), 0.0)]:
        add("B", [op])
    top = [mlp(l) for l in range(48, 52)]
    for ops in [[{"op": "orth", "tensors": top, "u_rows": [k["close"]]}],
                [{"op": "orth", "tensors": [mlp(l) for l in range(44, 52)], "u_rows": [k["close"]]}],
                [{"op": "orth", "tensors": top, "u_rows": k["letters"]}],
                [{"op": "orth", "tensors": [mlp(51)], "u_rows": [k["close"]]}],
                [{"op": "orth", "tensors": top + [oproj(l) for l in range(48, 52)], "u_rows": [k["close"]]}],
                [{"op": "orth", "tensors": [mlp(l) for l in range(44, 52)], "u_rows": k["letters"]}]]:
        add("D", ops)
    for n, r, s in [(mlp(26), .01, 1), (mlp(26), .1, 2), (mlp(26), .3, 3), (mlp(51), .1, 4), (mlp(51), .5, 5), (mlp(3), .3, 6)]:
        add("E", [nz(n, r, s)])
    for rows, f in [([k["close"]], 0.0), ([k["close"]], 0.7), ([k["close"]], 1.3), ([k["close"]], 2.0),
                    ([k["letters"][0]], 0.0), ([k["letters"][0]], 1.3), (k["letters"], 0.9), ([k["wait_cap"]], 0.0),
                    ([k["wait_low"]], 0.0), ([k["letters"][1]], 1.3)]:
        add("C", [sr(rows, f)])
    for ops in [[sr([k["close"]], 1.02)], [sr([k["close"]], 0.98)], [sr([k["unused"][0]], 0.0)], [sr([k["unused"][1]], 0.0)],
                [nz(mlp(26), 1e-4, 7)], [sc(mlp(26), 1.001)]]:
        add("F", ops)
    return _ids(S, "muse30b", "S", 2)


# ---------------------------------------------------------------- Full-Dose pools (27B, 9B)
def full_dose_pool(t: str, agent_edits_dir: str | None = None) -> list[dict]:
    k = TOKENS[t]
    L = MODELS[t]["layers"]
    S = []
    add = lambda typ, ops: S.append({"type": typ, "ops": ops})  # noqa: E731
    for f in [0.0, 0.2, 0.4, 0.6, 0.8, 1.2, 1.5, 2.0, 3.0]:
        add("C", [sr([k["close"]], f)])
    for row in k["letters"][:2]:
        for f in [0.0, 0.5, 1.5, 2.5]:
            add("C", [sr([row], f)])
    for f in [0.6, 0.8, 1.2, 1.4]:
        add("C", [sr(k["letters"], f)])
    for row in [k["wait_cap"], k["wait_low"]]:
        for f in [0.0, 2.0, 4.0]:
            add("C", [sr([row], f)])
    starts = [24, 32, 40, 44, 48, 52, 56, 60] if t == "27b" else [12, 16, 20, 22, 24, 26, 28, 30]
    for s0 in starts:
        add("D", [{"op": "orth", "tensors": [mlp(l) for l in range(s0, L)], "u_rows": [k["close"]]}])
    for s0 in starts[3:7]:
        add("D", [{"op": "orth", "tensors": [mlp(l) for l in range(s0, L)], "u_rows": k["letters"]}])
    for s0 in starts[4:6]:
        add("D", [{"op": "orth", "tensors": [mlp(l) for l in range(s0, L)] + [attn_out(l) for l in range(s0, L)],
                   "u_rows": [k["close"]]}])
    step = L // 16
    for l in list(range(0, L, step)) + [L - 1]:
        add("B", [sc(mlp(l), 0.0)])
    for l in [L // 8, L // 4, L // 2, 3 * L // 4, L - 3, L - 1]:
        add("B", [sc(mlp(l), 0.5)])
    for l in [3, L // 4 + 1, L // 2 - 1, L // 2 + 2, 3 * L // 4 - 1, L - 1]:
        add("B", [sc(attn_out(l), 0.0)])
    for i, l in enumerate([L // 4, L // 2, 3 * L // 4, L - 2]):
        add("E", [nz(mlp(l), 0.5, 100 + i)])
        add("E", [nz(mlp(l), 1.0, 200 + i)])
    if t == "27b":
        for j in range(1, N_AGENT_EDITS + 1):
            src = str(Path(agent_edits_dir or ".") / f"agent_edit_{j}.safetensors")
            for s_ in (1.0, 0.5, 2.0):
                add("A", [{"op": "agent_edit", "source": src, "scale": s_}])
    for f in [1.01, 0.99, 1.03]:
        add("F", [sr([k["close"]], f)])
    for row in [k["unused"][0], k["unused"][1], k["unused"][0] + 7]:
        add("F", [sr([row], 0.0)])
    for i, l in enumerate([L // 2, L - 2]):
        add("F", [nz(mlp(l), 1e-4, 300 + i)])
        add("F", [sc(mlp(l), 1.001)])
    S = _ids(S, t, "P", 3)
    # agent-proposed edits are kept only if their files are available
    return [s for s in S if all(op["op"] != "agent_edit" or Path(op["source"]).exists() for op in s["ops"])]


# ---------------------------------------------------------------- Subtle pools (doses of effective Full-Dose edits)
def dose_ops(ops: list[dict], d: float) -> list[dict]:
    """Dose d in (0, 1]: scaling factors 1 + (f - 1) d, removed fraction and noise norm times d, agent-edit scale times d."""
    out = []
    for op in ops:
        op = dict(op)
        if op["op"] in ("scale", "scale_rows"):
            op["factor"] = round(1 + (op["factor"] - 1) * d, 4)
        elif op["op"] == "orth":
            op["alpha"] = round(op.get("alpha", 1.0) * d, 4)
        elif op["op"] == "noise":
            op["rel"] = round(op["rel"] * d, 5)
        elif op["op"] == "agent_edit":
            op["scale"] = round(op.get("scale", 1.0) * d, 4)
        out.append(op)
    return out


SUBTLE_DOSES = (0.5, 0.25)

# Intermediate doses on 27B, per Full-Dose parent (added before any reader session).
SUBTLE_EXTRA_DOSES_27B = {
    "27b_P000C": (0.3, 0.375, 0.45), "27b_P012C": (0.3, 0.375, 0.45), "27b_P016C": (0.3, 0.375, 0.45),
    "27b_P008C": (0.2, 0.3), "27b_P023C": (0.1, 0.15), "27b_P026C": (0.35,), "27b_P022C": (0.4, 0.6),
    "27b_P027D": (0.75, 0.9), "27b_P028D": (0.75, 0.9), "27b_P029D": (0.75, 0.9), "27b_P030D": (0.75, 0.9),
    "27b_P031D": (0.75, 0.9), "27b_P032D": (0.75, 0.9), "27b_P039D": (0.75, 0.9), "27b_P040D": (0.75, 0.9),
    "27b_P080A": (0.75,), "27b_P083A": (0.75,), "27b_P086A": (0.75,), "27b_P087A": (0.75,), "27b_P092A": (0.75,),
    "27b_P099A": (0.1,), "27b_P100A": (0.1,), "27b_P101A": (0.1,), "27b_P089A": (0.35,),
}


def subtle_pool(t: str, full_pool: list[dict], changed_parents: set[str]) -> list[dict]:
    """Doses d in {0.5, 0.25} of every Full-Dose edit that changed a behavior other than refusal; no-op controls are
    carried over; on 27B the intermediate doses above are appended. Each spec records its Full-Dose parent."""
    key = lambda ops: json.dumps(ops, sort_keys=True)  # noqa: E731
    full_by_key = {key(s["ops"]): s["id"] for s in full_pool}
    out, seen = [], set()
    for s in full_pool:
        if s["type"] == "F":
            out.append({"type": "F", "ops": s["ops"], "parent": s["id"], "same_as": s["id"]})
            continue
        if s["id"] not in changed_parents:
            continue
        for d in SUBTLE_DOSES:
            ops = dose_ops(s["ops"], d)
            if key(ops) in seen:
                continue
            seen.add(key(ops))
            out.append({"type": s["type"], "ops": ops, "parent": s["id"], "dose": d,
                        **({"same_as": full_by_key[key(ops)]} if key(ops) in full_by_key else {})})
    out = _ids(out, t, "M", 3)
    if t == "27b":
        by_id = {s["id"]: s for s in full_pool}
        for p, ds in SUBTLE_EXTRA_DOSES_27B.items():
            if p not in by_id:
                continue
            for d in ds:
                ops = dose_ops(by_id[p]["ops"], d)
                if key(ops) in seen:
                    continue
                seen.add(key(ops))
                s = {"type": by_id[p]["type"], "ops": ops, "parent": p, "dose": d}
                s.update(id=f"{t}_M{len(out):03d}{s['type']}", target=t)
                out.append(s)
    return out


# ---------------------------------------------------------------- Muse-30B pool (norm gains and gate)
MUSE_GROUPS = {"0-12": range(0, 13), "13-25": range(13, 26), "26-38": range(26, 39), "39-51": range(39, 52)}


def muse_pool() -> list[dict]:
    """129 edits: post-FFN / post-attention gains (centered, 1 + w), pre-FFN gains, gate_proj, final-norm gain (w),
    noise on post-FFN gains, and 10 no-op controls."""
    L = MODELS["muse30b"]["layers"]
    S = []
    add = lambda typ, ops: S.append({"type": typ, "ops": ops})  # noqa: E731
    for kind, typ in (("post_feedforward_layernorm", "G"), ("post_attention_layernorm", "H")):
        for ls in MUSE_GROUPS.values():
            for f in (0.5, 0.8, 1.25, 2.0):
                add(typ, [gain([norm(l, kind) for l in ls], f)])
        for f in (0.9, 1.1):
            add(typ, [gain([norm(l, kind) for l in range(L)], f)])
    for ls in MUSE_GROUPS.values():
        for f in (0.7, 1.4):
            add("P", [gain([norm(l, "pre_feedforward_layernorm") for l in ls], f)])
        for f in (0.7, 1.4):
            add("M", [sc(f"model.language_model.layers.{l}.mlp.gate_proj.weight", f) for l in ls])
    for f in (0.6, 0.8, 1.25, 1.6):
        add("T", [gain([FINAL_NORM], f, c=False)])
    for i, (g, r) in enumerate([("13-25", 0.5), ("13-25", 1.0), ("39-51", 0.5), ("39-51", 1.0)]):
        add("E", [nz(norm(l, "post_feedforward_layernorm"), r, 400 + 13 * i + l) for l in MUSE_GROUPS[g]])
    for ops in [[gain([norm(10, "post_feedforward_layernorm")], 1.001)], [gain([norm(40, "post_feedforward_layernorm")], 1.001)],
                [gain([FINAL_NORM], 1.001, c=False)], [gain([norm(26, "post_attention_layernorm")], 0.999)],
                [sr([201000], 0.0)], [sr([201500], 0.0)], [sr([201007], 0.0)],
                [nz(mlp(26), 1e-4, 501)], [nz(mlp(45), 1e-4, 502)], [sc(mlp(13), 1.001)]]:
        add("F", ops)
    S = _ids(S, "muse30b", "Q", 3)
    # additional factors (added before any reader session)
    ext = []
    for kind, typ in (("post_feedforward_layernorm", "G"), ("post_attention_layernorm", "H")):
        for ls in MUSE_GROUPS.values():
            for f in (0.3, 0.4, 0.6, 1.5, 1.7):
                ext.append({"type": typ, "ops": [gain([norm(l, kind) for l in ls], f)]})
    for f in (0.4, 2.0, 2.5):
        ext.append({"type": "T", "ops": [gain([FINAL_NORM], f, c=False)]})
    for ls in MUSE_GROUPS.values():
        for f in (0.4, 2.0):
            ext.append({"type": "P", "ops": [gain([norm(l, "pre_feedforward_layernorm") for l in ls], f)]})
            ext.append({"type": "M", "ops": [sc(f"model.language_model.layers.{l}.mlp.gate_proj.weight", f) for l in ls]})
    seen = {json.dumps(s["ops"], sort_keys=True) for s in S}
    for s in ext:
        k = json.dumps(s["ops"], sort_keys=True)
        if k not in seen:
            seen.add(k)
            s.update(id=f"muse30b_Q{len(S):03d}{s['type']}", target="muse30b")
            S.append(s)
    return S


__all__ = ["muse_standard_edits", "full_dose_pool", "subtle_pool", "muse_pool", "dose_ops"]
