"""Materialize an intervention: full replacement tensors (for measurement), the compact reader file, and the
Text-Only description."""
from __future__ import annotations

import torch

from .checkpoint import Checkpoint
from .models import FINAL_NORM, LM_HEAD


def final_norm_gain(ck: Checkpoint) -> torch.Tensor:
    """Effective gain of the final RMSNorm (the Qwen models store the offset w of a 1 + w gain)."""
    g = ck.get(FINAL_NORM).float()
    return (1.0 + g) if g.abs().mean() < 0.5 else g


def apply_ops(ck: Checkpoint, spec: dict) -> tuple[dict, dict, str]:
    """-> (full {name: bf16 tensor}, compact {key: tensor}, description)"""
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
            desc.append(f"Row(s) {op['rows']} of `{n}` multiplied by {op['factor']}.")
        elif k == "gain":
            for n in op["tensors"]:
                Wn = full.get(n, ck.get(n).float())
                full[n] = (op["factor"] * (1.0 + Wn) - 1.0) if op.get("centered") else op["factor"] * Wn
            form = "(1 + w)" if op.get("centered") else "w"
            desc.append(f"The effective RMSNorm gain {form} of each of {op['tensors']} multiplied by {op['factor']}.")
        elif k == "noise":
            T = ck.get(op["tensor"]).float()
            g = torch.Generator().manual_seed(op["seed"])
            N = torch.randn(T.shape, generator=g)
            full[op["tensor"]] = T + op["rel"] * T.norm() / N.norm() * N
            desc.append(f"Gaussian noise added to `{op['tensor']}` (noise Frobenius norm = {op['rel']} x the tensor's "
                        f"Frobenius norm).")
        elif k == "orth":
            u = final_norm_gain(ck) * ck.get(LM_HEAD)[op["u_rows"]].float().mean(0)
            u = u / u.norm()
            al = op.get("alpha", 1.0)
            for n in op["tensors"]:
                W = ck.get(n).float()                       # [hidden, in]: output dimension = residual stream
                full[n] = W - al * torch.outer(u, u @ W)
            if al == 1.0:
                desc.append(f"From each of {op['tensors']} (output dimension = residual stream) the component along a unit "
                            f"vector u was projected out (W <- W - u u^T W), where u = normalized (final RMSNorm gain * mean of "
                            f"row(s) {op['u_rows']} of `{LM_HEAD}`).")
            else:
                desc.append(f"From each of {op['tensors']} (output dimension = residual stream) a fraction {al} of the component "
                            f"along a unit vector u was removed (W <- W - {al} u u^T W), where u = normalized (final RMSNorm gain "
                            f"* mean of row(s) {op['u_rows']} of `{LM_HEAD}`).")
        elif k == "agent_edit":
            from safetensors.torch import load_file
            for n, Tn in load_file(op["source"]).items():
                B, Tn = ck.get(n).float(), Tn.float()
                if op.get("scale", 1.0) != 1.0:
                    Tn = B + op["scale"] * (Tn - B)
                rows = torch.nonzero((Tn != B).reshape(Tn.shape[0], -1).any(1)).flatten().tolist() if Tn.dim() > 1 else None
                full[n] = Tn
                if rows is not None and len(rows) <= 256:
                    d = Tn[rows] - B[rows]
                    rel = (d.norm() / B[rows].norm()).item()
                    cos = torch.nn.functional.cosine_similarity(d.flatten(), B[rows].flatten(), dim=0).item()
                    desc.append(f"Row(s) {rows} of `{n}` replaced by new values: |new - old| / |old| = {rel:.3f}, "
                                f"cosine(new - old, old) = {cos:.3f}.")
                else:
                    d = Tn - B
                    desc.append(f"`{n}` replaced by a new tensor: |new - old| / |old| = {(d.norm() / B.norm()).item():.4f}, "
                                f"{int((d != 0).sum())} values changed.")
        else:
            raise ValueError(k)
    full = {n: T.to(torch.bfloat16).contiguous() for n, T in full.items()}
    compact = {}
    for n, T in full.items():                       # reader view: only the changed rows if few rows changed
        B = ck.get(n)
        if T.dim() > 1:
            rows = torch.nonzero((T != B).reshape(T.shape[0], -1).any(1)).flatten()
            if len(rows) == 0 or (len(rows) <= 256 and len(rows) < T.shape[0]):
                compact[n + "::rows"] = T[rows].contiguous()
                compact[n + "::row_index"] = rows.to(torch.int64)
                continue
        compact[n] = T
    return full, compact, " ".join(desc)
