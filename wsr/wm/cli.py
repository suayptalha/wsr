"""Command line for world-model weight-space programming.

    python -m wsr.wm convert --env pusht                         # LeWM release -> $WM_ROOT/ckpt/pusht
    python -m wsr.wm e0 --env pusht --shifts all                 # how much each shift hurts the frozen model
    python -m wsr.wm program programs/wm/pusht_slip.yaml         # an agent edits the predictor (see program.py)
    python -m wsr.wm baseline --env pusht --shift pusht/slip0.5  # gradient baselines on the same data

$WM_ROOT (default ./wm-work) holds converted checkpoints, cached segment sets and results.
"""
from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path


def root() -> Path:
    return Path(os.environ.get("WM_ROOT", "wm-work"))


def ckpt_dir(env: str) -> Path:
    return root() / "ckpt" / env


def ensure_ckpt(env: str) -> Path:
    from .envs import ENVS, stablewm_home
    from .model import convert
    d = ckpt_dir(env)
    if not (d / "model.safetensors").exists():
        spec = ENVS[env]
        convert(stablewm_home() / spec.hf_dir, d, {"env": env, "frameskip": spec.frameskip})
    return d


def summarize(err: dict) -> dict:
    import torch
    r = err["roll"]
    return {"tf": float(err["tf"].mean()), "roll1": float(r[:, 0].mean()), "rollH": float(r[:, -1].mean()),
            "roll_mean": float(r.mean()), "roll_mean_se": float(r.mean(1).std() / len(r) ** 0.5),
            "n": len(r), "per_h": [round(float(x), 5) for x in r.mean(0)], "tf_se": float(err["tf"].std() / len(r) ** 0.5),
            "_seg": r.mean(1).tolist() if isinstance(r, torch.Tensor) else None}


def cmd_e0(a):
    """Frozen model under every shift, on the same held-out segments; plus sanity checks."""
    import numpy as np
    import torch
    from .envs import SHIFTS, Episodes, collect
    from .model import WMBackend
    d = ensure_ckpt(a.env)
    be = WMBackend(d)
    keys = sorted(k for k in SHIFTS if k.startswith(a.env + "/")) if a.shifts == "all" else a.shifts.split(",")
    t0 = time.time()
    paths = collect(be, a.env, keys, "heldout", a.n, a.horizon, a.seed, root() / "data", a.workers, keep_frames=4)
    out = {"env": a.env, "n": a.n, "horizon": a.horizon, "collect_s": round(time.time() - t0), "results": {}}
    from .envs import load
    base = load(paths["base"])
    # sanity 1: replayed first frames vs the dataset's own pixels
    eps = Episodes(a.env)
    rows = [int(eps.ep_offset[e] + s) for e, s in base["segments"][:64]]
    emb_ds = be.encode(eps.pixels(rows))
    out["sanity_frame0_mse"] = float((emb_ds - base["emb"][:64, 0]).pow(2).mean())
    out["sanity_emb_var"] = float(base["emb"][:, 0].var(0).mean())
    # sanity 2: action layout (time-major concat vs dim-major)
    act = base["act"]
    n, T, A = act.shape
    fs = eps.spec.frameskip
    alt = act.reshape(n, T, fs, A // fs).transpose(2, 3).reshape(n, T, A)
    out["sanity_act_layout"] = {"time_major": summarize(be.errors(base["emb"], act, a.horizon))["tf"],
                                "dim_major": summarize(be.errors(base["emb"], alt, a.horizon))["tf"],
                                "zero_action": summarize(be.errors(base["emb"], torch.zeros_like(act), a.horizon))["tf"]}
    for k, p in paths.items():
        s = load(p)
        e = be.errors(s["emb"], s["act"], a.horizon)
        summ = summarize(e)
        seg = summ.pop("_seg")
        # how different the shifted world's latents are from the base world's (same actions, same start)
        summ["latent_shift"] = float((s["emb"] - base["emb"]).pow(2).mean())
        summ["moved_block_frac"] = float(((s["states"][:, -1, 2:4] - s["states"][:, 0, 2:4]).norm(dim=1) > 1.0).float().mean())
        if k != "base":
            bseg = torch.tensor(out["results"]["base"]["_seg"])
            d_ = torch.tensor(seg) - bseg
            summ["delta_vs_base"] = float(d_.mean())
            summ["ratio_vs_base"] = float(torch.tensor(seg).mean() / bseg.mean())
        summ["_seg"] = seg
        out["results"][k] = summ
        print(k, {kk: vv for kk, vv in summ.items() if kk not in ("_seg",)}, flush=True)
    res = root() / "results" / "e0" / f"{a.env}_n{a.n}_h{a.horizon}_s{a.seed}.json"
    res.parent.mkdir(parents=True, exist_ok=True)
    for v in out["results"].values():
        v.pop("_seg", None)
    res.write_text(json.dumps(out, indent=1))
    print("wrote", res)


def cmd_prep(a):
    """Cache practice and held-out segment sets for every shift (run once before parallel programs)."""
    from .envs import SHIFTS, collect
    from .model import WMBackend
    be = WMBackend(ensure_ckpt(a.env))
    keys = sorted(k for k in SHIFTS if k.startswith(a.env + "/")) if a.shifts == "all" else a.shifts.split(",")
    sets = {"practice": (a.n_practice, a.practice_seed), "heldout": (a.n, a.seed), "train": (a.n_train, 7)}
    for split in a.splits.split(","):
        n, seed = sets[split]
        collect(be, a.env, keys, split, n, a.horizon, seed, root() / "data", a.workers)
    print("PREP_DONE")


def cmd_convert(a):
    print(ensure_ckpt(a.env))


def cmd_program(a):
    from .program import run
    over = {}
    for kv in a.set or []:
        k, v = kv.split("=", 1)
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        over[k] = v
    run(a.config, over)


def cmd_baseline(a):
    from .baselines import run_baselines, run_baselines_v2
    if a.v2:
        run_baselines_v2(a.env, a.shift, a.methods.split(","), a.horizon, a.seed)
    else:
        run_baselines(a.env, a.shift, a.methods.split(","), a.n_practice, a.n_heldout, a.horizon, a.seed, a.out)


def cmd_reading_models(a):
    from .read import make_models
    make_models(a.env, a.controls)


def cmd_read(a):
    from .read import run_session
    over = {}
    for kv in a.set or []:
        k, v = kv.split("=", 1)
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        over[k] = v
    run_session(a.config, a.env, a.model, over)


def cmd_read_score(a):
    from .read import score
    score(a.env)


def cmd_plan(a):
    from .plan import cmd
    cmd(a.env, a.shift, a.models.split(","), a.n, a.seed, a.budget, a.goal_offset, a.with_base_env)


def cmd_audit_judge(a):
    from .audit_judge import run
    params = {"reasoning_effort": a.effort} if a.effort else {}
    run(a.model, provider=a.provider, params=params, base_url=a.base_url, workers=a.workers)


def cmd_param(a):
    from .param import run
    over = {}
    for kv in a.set or []:
        k, v = kv.split("=", 1)
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        over[k] = v
    run(a.config, a.env, a.family, [float(x) for x in a.values.split(",")], a.example, over)


def main(argv=None):
    p = argparse.ArgumentParser(prog="python -m wsr.wm")
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("convert")
    s.add_argument("--env", default="pusht")
    s.set_defaults(fn=cmd_convert)
    s = sub.add_parser("e0")
    s.add_argument("--env", default="pusht")
    s.add_argument("--shifts", default="all")
    s.add_argument("--n", type=int, default=256)
    s.add_argument("--horizon", type=int, default=5)
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--workers", type=int, default=16)
    s.set_defaults(fn=cmd_e0)
    s = sub.add_parser("prep")
    s.add_argument("--env", default="pusht")
    s.add_argument("--shifts", default="all")
    s.add_argument("--n", type=int, default=256)
    s.add_argument("--n-practice", type=int, default=64)
    s.add_argument("--horizon", type=int, default=5)
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--practice-seed", type=int, default=101)
    s.add_argument("--n-train", type=int, default=2048)
    s.add_argument("--splits", default="practice,heldout")
    s.add_argument("--workers", type=int, default=40)
    s.set_defaults(fn=cmd_prep)
    s = sub.add_parser("program")
    s.add_argument("config")
    s.add_argument("--set", action="append")
    s.set_defaults(fn=cmd_program)
    s = sub.add_parser("baseline")
    s.add_argument("--env", default="pusht")
    s.add_argument("--shift", required=True)
    s.add_argument("--methods", default="full,adaln,action_enc,last_block")
    s.add_argument("--n-practice", type=int, default=64)
    s.add_argument("--n-heldout", type=int, default=256)
    s.add_argument("--horizon", type=int, default=5)
    s.add_argument("--seed", type=int, default=1)
    s.add_argument("--out", default=None)
    s.add_argument("--v2", action="store_true", help="validation-selected stopping/method/lr; saves the models")
    s.set_defaults(fn=cmd_baseline)
    s = sub.add_parser("reading-models")
    s.add_argument("--env", default="pusht")
    s.add_argument("--controls", type=int, default=2)
    s.set_defaults(fn=cmd_reading_models)
    s = sub.add_parser("read")
    s.add_argument("config")
    s.add_argument("--env", default="pusht")
    s.add_argument("--model", required=True)
    s.add_argument("--set", action="append")
    s.set_defaults(fn=cmd_read)
    s = sub.add_parser("read-score")
    s.add_argument("--env", default="pusht")
    s.set_defaults(fn=cmd_read_score)
    s = sub.add_parser("plan")
    s.add_argument("--env", default="pusht")
    s.add_argument("--shift", required=True)
    s.add_argument("--models", default="base")
    s.add_argument("--n", type=int, default=50)
    s.add_argument("--seed", type=int, default=0)
    s.add_argument("--budget", type=int, default=50)
    s.add_argument("--goal-offset", type=int, default=25)
    s.add_argument("--with-base-env", action="store_true")
    s.set_defaults(fn=cmd_plan)
    s = sub.add_parser("audit-judge")
    s.add_argument("--provider", default="gemini", help="any provider of wsr.providers (openai, anthropic, gemini, ...)")
    s.add_argument("--model", default="gemini-3.8-flash")
    s.add_argument("--effort", default="high", help="reasoning_effort sent to the judge ('' to omit)")
    s.add_argument("--base-url", default=None)
    s.add_argument("--workers", type=int, default=16)
    s.set_defaults(fn=cmd_audit_judge)
    s = sub.add_parser("param")
    s.add_argument("config")
    s.add_argument("--env", default="pusht")
    s.add_argument("--family", required=True)
    s.add_argument("--values", required=True)
    s.add_argument("--example", type=float, required=True)
    s.add_argument("--set", action="append")
    s.set_defaults(fn=cmd_param)
    a = p.parse_args(argv)
    a.fn(a)


if __name__ == "__main__":
    main()
