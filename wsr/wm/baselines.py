"""Gradient-based adaptation baselines on the same segments the agent's forward passes use.

Each method fine-tunes a subset of the dynamics tensors with Adam on the LeWM prediction loss (teacher-forced
next-embedding MSE over every 4-frame window of the practice segments; the encoder is frozen, so the targets are
the cached embeddings) and is scored exactly like an agent's edit: held-out autoregressive rollout MSE in the
changed physics (and in the old physics). ``data`` chooses the practice set: the agent's ``practice`` segments, or
a large ``train``-split set (an oracle with far more data than the agent ever sees).
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

SCOPES = {
    "full": [r"^predictor\.", r"^action_encoder\.", r"^pred_proj\."],
    "predictor": [r"^predictor\."],
    "adaln": [r"adaLN_modulation"],
    "action_enc": [r"^action_encoder\."],
    "last_block": [r"^predictor\.transformer\.layers\.5\.", r"^predictor\.transformer\.norm\."],
    "pred_proj": [r"^pred_proj\."],
}


def windows(emb: torch.Tensor, act: torch.Tensor, history: int = 3):
    """All (history+1)-frame windows of segments: ctx emb (N, h, D), act (N, h, A), targets (N, h, D)."""
    F = emb.shape[1]
    E, A, T = [], [], []
    for s in range(F - history):
        E.append(emb[:, s:s + history])
        A.append(act[:, s:s + history])
        T.append(emb[:, s + 1:s + history + 1])
    return torch.cat(E), torch.cat(A), torch.cat(T)


def finetune(be, scope: list[str], data: dict, steps: int, lr: float, batch: int = 256, seed: int = 0,
             checkpoints: list[int] | None = None, eval_fn=None):
    import re
    torch.manual_seed(seed)
    m = be.model
    params = [p for n, p in m.named_parameters() if any(re.search(s, n) for s in scope)]
    names = [n for n, p in m.named_parameters() if any(re.search(s, n) for s in scope)]
    for p in params:
        p.requires_grad_(True)
    opt = torch.optim.AdamW(params, lr=lr, weight_decay=1e-3)
    E, A, T = (x.to(be.device).float() for x in windows(data["emb"], data["act"]))
    curve = []
    m.eval()                                   # BatchNorm uses running stats, dropout off (as at inference)
    for step in range(1, steps + 1):
        idx = torch.randint(0, len(E), (min(batch, len(E)),), device=be.device)
        pred = m.predict(E[idx], m.action_encoder(A[idx]))
        loss = (pred - T[idx]).pow(2).mean()
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if checkpoints and step in checkpoints and eval_fn:
            with torch.no_grad():
                curve.append({"step": step, "train_loss": float(loss), **eval_fn()})
    for p in params:
        p.requires_grad_(False)
    return names, curve


def run_baselines(env: str, shift: str, methods: list[str], n_practice: int, n_heldout: int, horizon: int, seed: int,
                  out: str | None = None, lrs=(1e-4, 1e-3), steps: int = 2000, train_n: int = 2048):
    from .cli import ensure_ckpt, root
    from .envs import collect, load
    from .model import WMBackend
    from .program import Scorer, paired
    base_dir = ensure_ckpt(env)
    be = WMBackend(base_dir)
    sc = Scorer(be, horizon)
    data = root() / "data"
    P = collect(be, env, [shift], "practice", n_practice, horizon, 101, data)
    H = collect(be, env, [shift], "heldout", n_heldout, horizon, seed, data)
    hn, ho = load(H[shift]), load(H["base"])
    sets = {"practice": load(P[shift])}
    if train_n:
        Tr = collect(be, env, [shift], "train", train_n, horizon, 7, data)
        sets["train"] = load(Tr[shift])
    base = sc.score({}, hn)
    base_old = sc.score({}, ho)
    orig = be.state()
    res = {"env": env, "shift": shift, "base_new": base["mean"], "base_old": base_old["mean"], "runs": []}
    ckpts = sorted({10, 30, 100, 300, 1000, steps})

    def eval_fn():
        be.patch({})        # the live (fine-tuned) weights are scored as they are
        e_new = be.errors(hn["emb"], hn["act"], horizon)["roll"].numpy().mean(1)
        e_old = be.errors(ho["emb"], ho["act"], horizon)["roll"].numpy().mean(1)
        g = paired(e_new, base["per_segment"])
        return {"heldout_new": float(e_new.mean()), "heldout_old": float(e_old.mean()), "gain": g}

    for dname, dset in sets.items():
        for meth in methods:
            for lr in lrs:
                t0 = time.time()
                with torch.no_grad():
                    for k, v in be.model.state_dict().items():
                        v.copy_(orig[k].to(v.device))
                names, curve = finetune(be, SCOPES[meth], dset, steps, lr, checkpoints=ckpts, eval_fn=eval_fn)
                best = min(curve, key=lambda c: c["heldout_new"])
                row = {"data": dname, "n_segments": len(dset["emb"]), "method": meth, "lr": lr, "n_params":
                       int(sum(be.slots[n].numel() for n in names)), "curve": curve, "best_step_oracle": best["step"],
                       "final": curve[-1], "wall_s": round(time.time() - t0, 1)}
                res["runs"].append(row)
                print(f"{shift} {dname} {meth} lr={lr}: base {base['mean']:.5f} -> final {curve[-1]['heldout_new']:.5f} "
                      f"(best {best['heldout_new']:.5f} @ {best['step']}); old physics {curve[-1]['heldout_old']:.5f}", flush=True)
    with torch.no_grad():
        for k, v in be.model.state_dict().items():
            v.copy_(orig[k].to(v.device))
    p = Path(out or root() / "results" / "baselines" / f"{shift.replace('/', '_')}.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(res, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o)))
    print("wrote", p)
    return res


# ---------------------------------------------------------------- v2: no peeking at held-out data
def run_baselines_v2(env: str, shift: str, methods: list[str], horizon: int = 5, seed: int = 1,
                     lrs=(1e-4, 3e-4, 1e-3), steps: int = 2000, data_sets=("practice", "train"), val_frac: float = 0.25,
                     n_practice: int = 64, n_heldout: int = 256, train_n: int = 2048):
    """Like ``run_baselines`` but every choice (stopping step, method, learning rate) is made on a validation
    split of the fine-tuning data itself, never on the held-out segments. The selected models are saved
    (dynamics tensors only) under results/baselines_v2/models/ for planning evaluation."""
    from safetensors.torch import save_file

    from .cli import ensure_ckpt, root
    from .envs import collect, load
    from .model import WMBackend
    from .program import Scorer, paired
    be = WMBackend(ensure_ckpt(env))
    sc = Scorer(be, horizon)
    data = root() / "data"
    H = collect(be, env, [shift], "heldout", n_heldout, horizon, seed, data)
    hn, ho = load(H[shift]), load(H["base"])
    sets = {}
    if "practice" in data_sets:
        sets["practice"] = load(collect(be, env, [shift], "practice", n_practice, horizon, 101, data)[shift])
    if "train" in data_sets:
        sets["train"] = load(collect(be, env, [shift], "train", train_n, horizon, 7, data)[shift])
    base = sc.score({}, hn)
    orig = be.state()
    ckpts = sorted({10, 30, 100, 200, 300, 500, 1000, 1500, steps})
    outdir = root() / "results" / "baselines_v2"
    (outdir / "models").mkdir(parents=True, exist_ok=True)
    res = {"env": env, "shift": shift, "base_new": base["mean"], "base_old": sc.score({}, ho)["mean"], "runs": [], "selected": {}}
    g = torch.Generator().manual_seed(0)

    def restore():
        with torch.no_grad():
            for k, v in be.model.state_dict().items():
                v.copy_(orig[k].to(v.device))

    for dname, dset in sets.items():
        n = len(dset["emb"])
        perm = torch.randperm(n, generator=g)
        nv = max(4, int(round(n * val_frac)))
        val_idx, tr_idx = perm[:nv], perm[nv:]
        tr = {"emb": dset["emb"][tr_idx], "act": dset["act"][tr_idx]}
        va = {"emb": dset["emb"][val_idx], "act": dset["act"][val_idx]}
        best_overall = None
        for meth in methods:
            for lr in lrs:
                restore()
                t0 = time.time()
                snap = {"val": float("inf")}

                def eval_fn():
                    be.patch({})
                    v = be.errors(va["emb"], va["act"], horizon)["roll"].mean().item()
                    e_new = be.errors(hn["emb"], hn["act"], horizon)["roll"].numpy().mean(1)
                    e_old = be.errors(ho["emb"], ho["act"], horizon)["roll"].numpy().mean(1)
                    return {"val": v, "heldout_new": float(e_new.mean()), "heldout_old": float(e_old.mean()),
                            "_per": e_new}

                import re
                names = [nm for nm, _ in be.model.named_parameters() if any(re.search(s, nm) for s in SCOPES[meth])]

                def eval_and_snap():
                    r = eval_fn()
                    if r["val"] < snap["val"]:
                        snap.update(val=r["val"], step=None, per=r.pop("_per"), heldout_new=r["heldout_new"],
                                    heldout_old=r["heldout_old"],
                                    tensors={nm: be.slots[nm].detach().cpu().clone() for nm in names})
                    else:
                        r.pop("_per")
                    return r

                # step 0 (no training) is a valid choice too
                r0 = eval_and_snap()
                snap["step"] = 0
                _, curve = finetune(be, SCOPES[meth], tr, steps, lr, checkpoints=ckpts, eval_fn=eval_and_snap)
                curve = [{"step": 0, **r0}] + curve
                for c in curve:
                    if abs(c["val"] - snap["val"]) < 1e-12:
                        snap["step"] = c["step"]
                        break
                row = {"data": dname, "n_train": len(tr_idx), "n_val": nv, "method": meth, "lr": lr,
                       "n_params": int(sum(be.slots[nm].numel() for nm in names)), "curve": curve,
                       "selected_step": snap["step"], "selected_val": snap["val"], "selected_heldout_new": snap["heldout_new"],
                       "selected_heldout_old": snap["heldout_old"], "gain": paired(snap["per"], base["per_segment"]),
                       "wall_s": round(time.time() - t0, 1)}
                res["runs"].append(row)
                print(f"{shift} {dname} {meth} lr={lr}: val-selected step {snap['step']} -> held-out {snap['heldout_new']:.5f} "
                      f"(base {base['mean']:.5f}); old physics {snap['heldout_old']:.5f}", flush=True)
                if best_overall is None or snap["val"] < best_overall[0]:
                    best_overall = (snap["val"], row, snap["tensors"])
        _, row, tensors = best_overall
        tag = f"{shift.replace('/', '_')}__{dname}"
        save_file({k: v.contiguous() for k, v in tensors.items()}, str(outdir / "models" / f"{tag}.safetensors"))
        res["selected"][dname] = {k: row[k] for k in ("method", "lr", "selected_step", "selected_val",
                                                       "selected_heldout_new", "selected_heldout_old", "gain")}
        print(f"SELECTED {shift} {dname}: {row['method']} lr={row['lr']} step {row['selected_step']} -> held-out "
              f"{row['selected_heldout_new']:.5f}", flush=True)
    restore()
    p = outdir / f"{shift.replace('/', '_')}.json"
    p.write_text(json.dumps(res, indent=1, default=lambda o: o.tolist() if isinstance(o, np.ndarray) else str(o)))
    print("wrote", p)
    return res
