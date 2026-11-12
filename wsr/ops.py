"""The edit vocabulary: turn an intervention spec into tensors, a reader file and a text description.

An intervention is ``{"id", "target", "type", "ops": [...]}``. Each op is a small, exactly specified change:

  scale       {"tensor", "factor"}                      W <- factor * W
  scale_rows  {"tensor", "rows", "factor"}              W[rows] <- factor * W[rows]
  orth        {"tensors", "u_rows", "alpha"?}           W <- W - alpha * u u^T W, u from unembedding rows
  noise       {"tensor", "rel", "seed"}                 W <- W + N, ||N||_F = rel * ||W||_F
  gain        {"tensors", "factor", "centered"?}        effective norm gain (1 + w if centered, else w) * factor
  delta       {"source", "scale"?}                      W0 + scale * (W_file - W0), tensors read from a safetensors file

``apply_ops`` returns three views of the same edit:

* **full**: ``{name: tensor}`` replacement tensors, used to measure the edited model;
* **compact**: what a reader gets. A tensor in which only a few rows changed is stored as ``name::rows`` and
  ``name::row_index`` so that the file stays small; otherwise the full tensor;
* **description**: the Text-Only condition's account of the edit, built from fixed templates.
"""
from __future__ import annotations

import os

import torch

from .checkpoint import Checkpoint
from .paper.models import FINAL_NORM, LM_HEAD

MAX_COMPACT_ROWS = 256
DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32, "float64": torch.float64}


class StaticAnatomy:
    """The few facts ``apply_ops`` needs, given explicitly instead of detected (used for the paper's models)."""

    def __init__(self, unembed: str = LM_HEAD, final_norm: str | None = FINAL_NORM, norm_label: str = "RMSNorm",
                 tied: bool = False, **_):
        self.unembed, self.final_norm, self.norm_label, self.tied = unembed, final_norm, norm_label, tied

    def out_axis(self, name: str) -> int:
        return 0


def effective_gain(ck: Checkpoint, name: str | None) -> torch.Tensor | float:
    """Effective gain of a norm (some models store the offset w of a 1 + w gain; then |w| is small)."""
    if not name:
        return 1.0
    g = ck.get(name).float()
    return (1.0 + g) if g.abs().mean() < 0.5 else g


def is_centered(ck: Checkpoint, name: str) -> bool:
    return bool(ck.get(name).float().abs().mean() < 0.5)


def changed_rows(new: torch.Tensor, old: torch.Tensor) -> torch.Tensor:
    return torch.nonzero((new != old).reshape(new.shape[0], -1).any(1)).flatten()


def describe_replacement(name: str, new: torch.Tensor, old: torch.Tensor) -> str:
    """Template for an arbitrary replacement tensor (agent-written edits): how far it moved, and where."""
    new, old = new.float(), old.float()
    rows = changed_rows(new, old).tolist() if new.dim() > 1 else None
    if rows is not None and len(rows) <= MAX_COMPACT_ROWS:
        d = new[rows] - old[rows]
        rel = (d.norm() / old[rows].norm()).item()
        cos = torch.nn.functional.cosine_similarity(d.flatten(), old[rows].flatten(), dim=0).item()
        return (f"Row(s) {rows} of `{name}` replaced by new values: |new - old| / |old| = {rel:.3f}, "
                f"cosine(new - old, old) = {cos:.3f}.")
    d = new - old
    return (f"`{name}` replaced by a new tensor: |new - old| / |old| = {(d.norm() / old.norm()).item():.4f}, "
            f"{int((d != 0).sum())} values changed.")


def compact_view(ck: Checkpoint, full: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Reader view: only the changed rows of a tensor when few rows changed."""
    compact = {}
    for n, T in full.items():
        B = ck.get(n)
        if T.dim() > 1:
            rows = changed_rows(T, B)
            if len(rows) == 0 or (len(rows) <= MAX_COMPACT_ROWS and len(rows) < T.shape[0]):
                compact[n + "::rows"] = T[rows].contiguous()
                compact[n + "::row_index"] = rows.to(torch.int64)
                continue
        compact[n] = T
    return compact


def expand_compact(ck: Checkpoint, tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Inverse of ``compact_view``: ``name::rows`` + ``name::row_index`` pairs become full tensors."""
    full = {}
    for k, T in tensors.items():
        if k.endswith("::row_index"):
            continue
        if k.endswith("::rows"):
            n = k[:-len("::rows")]
            idx = tensors.get(n + "::row_index")
            if idx is None:
                raise ValueError(f"`{k}` has no matching `{n}::row_index`")
            if n not in ck:
                raise ValueError(f"`{n}` is not a parameter of the model")
            B = ck.get(n).clone()
            B[idx.long()] = T.to(B.dtype)
            full[n] = B
        else:
            full[k] = T
    return full


def apply_ops(ck: Checkpoint, spec: dict, anat=None, dtype: str | None = None) -> tuple[dict, dict, str]:
    """-> (full {name: tensor}, compact {key: tensor}, description).

    ``anat`` supplies the unembedding, the final norm and matrix orientation; without it the names of the paper's
    models are assumed. ``dtype`` forces the stored dtype of the edit (default: each tensor keeps its own).
    """
    anat = anat or StaticAnatomy()
    label = getattr(anat, "norm_label", "RMSNorm")
    full, desc = {}, []
    for op in spec["ops"]:
        k = op["op"]
        if k == "scale":
            full[op["tensor"]] = full.get(op["tensor"], ck.get(op["tensor"]).float()) * op["factor"]
            desc.append(f"All values of `{op['tensor']}` multiplied by {op['factor']}.")
        elif k == "scale_rows":
            n = op["tensor"]
            T = full.get(n, ck.get(n).float()).clone()
            T[op["rows"]] *= op["factor"]
            full[n] = T
            tied = " (this matrix is also the input embedding)" if op.get("tied") else ""
            desc.append(f"Row(s) {op['rows']} of `{n}`{tied} multiplied by {op['factor']}.")
        elif k == "gain":
            for n in op["tensors"]:
                Wn = full.get(n, ck.get(n).float())
                full[n] = (op["factor"] * (1.0 + Wn) - 1.0) if op.get("centered") else op["factor"] * Wn
            form = "(1 + w)" if op.get("centered") else "w"
            desc.append(f"The effective {label} gain {form} of each of {op['tensors']} multiplied by {op['factor']}.")
        elif k == "noise":
            T = ck.get(op["tensor"]).float()
            g = torch.Generator().manual_seed(op["seed"])
            N = torch.randn(T.shape, generator=g)
            full[op["tensor"]] = T + op["rel"] * T.norm() / N.norm() * N
            desc.append(f"Gaussian noise added to `{op['tensor']}` (noise Frobenius norm = {op['rel']} x the tensor's "
                        f"Frobenius norm).")
        elif k == "orth":
            unembed = op.get("unembed") or anat.unembed
            fnorm = op["final_norm"] if "final_norm" in op else anat.final_norm
            u = effective_gain(ck, fnorm) * ck.get(unembed)[op["u_rows"]].float().mean(0)
            u = u / u.norm()
            al = op.get("alpha", 1.0)
            axis = op.get("axis")
            for n in op["tensors"]:
                W = ck.get(n).float()
                ax = anat.out_axis(n) if axis is None else axis
                # axis 0: W is [hidden, in], the output dimension is the residual stream; axis 1: W is [in, hidden]
                full[n] = W - al * (torch.outer(u, u @ W) if ax == 0 else torch.outer(W @ u, u))
            ax = axis if axis is not None else anat.out_axis(op["tensors"][0])
            where = "output dimension = residual stream" + ("" if ax == 0 else ", stored as [in, out]")
            form = "u u^T W" if ax == 0 else "W u u^T"
            udef = (f"normalized (final {label} gain * mean of row(s) {op['u_rows']} of `{unembed}`)" if fnorm
                    else f"normalized (mean of row(s) {op['u_rows']} of `{unembed}`)")
            if al == 1.0:
                desc.append(f"From each of {op['tensors']} ({where}) the component along a unit "
                            f"vector u was projected out (W <- W - {form}), where u = {udef}.")
            else:
                desc.append(f"From each of {op['tensors']} ({where}) a fraction {al} of the component "
                            f"along a unit vector u was removed (W <- W - {al} {form}), where u = {udef}.")
        elif k in ("agent_edit", "delta"):
            from safetensors.torch import load_file
            for n, Tn in expand_compact(ck, load_file(op["source"])).items():
                B, Tn = ck.get(n).float(), Tn.float()
                if op.get("scale", 1.0) != 1.0:
                    Tn = B + op["scale"] * (Tn - B)
                full[n] = Tn
                desc.append(describe_replacement(n, Tn, B))
        else:
            raise ValueError(f"unknown op `{k}`")
    force = DTYPES[dtype] if dtype else None
    full = {n: T.to(force or ck.get(n).dtype).contiguous() for n, T in full.items()}
    return full, compact_view(ck, full), " ".join(desc)


def dose_ops(ops: list[dict], d: float) -> list[dict]:
    """The same edit at dose d in (0, 1]: scaling factors become 1 + (f - 1) d; the removed fraction, the noise norm
    and a delta's scale are multiplied by d."""
    out = []
    for op in ops:
        op = dict(op)
        if op["op"] in ("scale", "scale_rows", "gain"):
            op["factor"] = round(1 + (op["factor"] - 1) * d, 4)
        elif op["op"] == "orth":
            op["alpha"] = round(op.get("alpha", 1.0) * d, 4)
        elif op["op"] == "noise":
            op["rel"] = round(op["rel"] * d, 5)
        elif op["op"] in ("agent_edit", "delta"):
            op["scale"] = round(op.get("scale", 1.0) * d, 4)
        out.append(op)
    return out


# ---------------------------------------------------------------- edits written by an agent (weight-space programming)
def load_submission(ck: Checkpoint, path: str | os.PathLike, anat=None, dtype: str | None = None) -> dict:
    """Read an agent's edit: a ``.safetensors`` file (full tensors, or the ``::rows`` form) or a ``.json`` op list."""
    import json
    from pathlib import Path

    from safetensors.torch import load_file
    p = Path(path)
    if p.suffix == ".json":
        ops = json.loads(p.read_text())
        ops = ops["ops"] if isinstance(ops, dict) else ops
        return apply_ops(ck, {"ops": ops}, anat, dtype)[0]
    full = expand_compact(ck, load_file(str(p)))
    return {n: T.to(DTYPES[dtype] if dtype else ck.get(n).dtype).contiguous() for n, T in full.items() if n in ck}


def validate_edit(ck: Checkpoint, full: dict[str, torch.Tensor], frozen: list[str] | None = None) -> dict:
    """Check an edit against the checkpoint and describe it. -> {"ok", "errors", "tensors": [...], "rel_norm"}"""
    import re
    errors, rows, num, den = [], [], 0.0, 0.0
    for n, T in full.items():
        if n not in ck:
            errors.append(f"`{n}` is not a parameter of the model")
            continue
        if tuple(T.shape) != ck.shape(n):
            errors.append(f"`{n}` has shape {tuple(T.shape)}, expected {ck.shape(n)}")
            continue
        if any(re.search(rx, n) for rx in (frozen or [])):
            errors.append(f"`{n}` is frozen and may not be edited")
            continue
        if not torch.isfinite(T.float()).all():
            errors.append(f"`{n}` contains non-finite values")
            continue
        B = ck.get(n).float()
        d = T.float() - B
        num += float(d.pow(2).sum())
        den += float(B.pow(2).sum())
        rows.append({"tensor": n, "rel_norm": float(d.norm() / B.norm().clamp_min(1e-12)), "values_changed": int((d != 0).sum()),
                     "values": int(d.numel()), "description": describe_replacement(n, T, B)})
    if not full:
        errors.append("the edit is empty")
    return {"ok": not errors, "errors": errors, "tensors": rows,
            "rel_norm": (num / den) ** 0.5 if den else 0.0,
            "values_changed": sum(r["values_changed"] for r in rows)}
