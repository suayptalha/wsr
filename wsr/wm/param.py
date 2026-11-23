"""Parametric weight-space programs: one program for a whole family of physics changes.

    python -m wsr.wm param <agent.yaml> --env pusht --family kv --values 5,10,30,40,60,80,120

The agent is told about a one-parameter family of changes (for example the agent's velocity gain k_v = X) and
writes ``edit_program.py`` with ``make_edit(x) -> {tensor name: new tensor}``. It never sees data and cannot run the
model (zero-shot). Afterwards the program is executed in the sandbox for every test value, including values far
from any example in the prompt, and each edit is scored on held-out segments of that environment.
"""
from __future__ import annotations

import json
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np

FAMILIES = {
    ("pusht", "kv"): ("k_v", 20.0, "the agent's PD velocity gain k_v (originally 20) becomes X"),
    ("pusht", "ascale"): ("action_scale", 100.0, "the action scale (originally 100) becomes X: target = agent_position + X * action"),
    ("pusht", "slip"): ("damping", 0.0, "pymunk's space.damping (originally 0, i.e. the block stops instantly) becomes X in (0, 1)"),
    ("reacher", "gear"): ("gear_scale", 1.0, "the gear of both actuators is multiplied by X (X = 1: original)"),
    ("reacher", "mass"): ("mass_scale", 1.0, "the mass and inertia of both moving links are multiplied by X (X = 1: original)"),
}


def prompt(env: str, fam: str, example: float, paths: dict, minutes: int) -> str:
    from .envs import env_description, get_shift
    W, w = paths["target"], paths["work"]
    name, orig, desc = FAMILIES[(env, fam)]
    ex = get_shift(f"{env}/{fam}{example:g}")
    return "\n".join([
        "You are a researcher doing **weight-space programming**: you change what a neural network does by reading its "
        "weights and editing them directly, using your own reasoning and knowledge. You do not train it.", "",
        "## The model", "",
        f"A pretrained JEPA world model (LeWorldModel) is at `{W}` (read-only): `model.safetensors`, `config.json`, "
        "`README.md` (architecture and forward pass; read it first) and source code `lewm.py`, `module.py`.", "",
        "## The environment it was trained on", "", env_description(env), "",
        "## A family of changes", "",
        f"The environment will change in one parameter X: {desc}. For example, for X = {example:g}: {ex.text}", "",
        "## Your goal", "",
        f"Write `{w}/edit_program.py` defining `make_edit(x: float) -> dict[str, torch.Tensor]`: for any value X of the "
        f"parameter (original value {orig:g}), it returns the NEW values of the tensors to change (full tensors, keys = "
        "tensor names of model.safetensors; only `predictor.*`, `action_encoder.*`, `pred_proj.*`) so that the edited "
        "model predicts the changed environment. It may read the weights with `from wsr.sandbox_lib import W` (W.get). "
        "It will be called for several values of X, most of which are not mentioned here, and each edit is scored by "
        "the error of 5-step latent rollouts on held-out episodes of that changed environment (lower is better). Also "
        f"explain the program in `{w}/rationale.md`.", "",
        "## Tools and rules", "",
        "A shell and Python (CPU only, no GPU, no internet); numpy and torch (CPU). You have no data and cannot run the "
        "model. 1. No training or fitting of weights (no gradients, optimizers, search or solvers whose unknowns are "
        "model weights). 2. Do not run or re-implement the model's forward pass on inputs. 3. `make_edit` must be "
        "deterministic and run in under a minute on CPU. "
        f"4. You have about {minutes} minutes."])


def run(config: str, env: str, fam: str, values: list[float], example: float, overrides: dict | None = None,
        n_heldout: int = 128, horizon: int = 5, log=print) -> dict:
    import torch
    from safetensors.torch import load_file

    from ..config import normalize_config
    from ..harness import nudge, run_builtin
    from ..sandbox import Sandbox, install_helper, install_python_logger
    from .cli import ensure_ckpt, root
    from .envs import collect, load
    from .model import FROZEN, WMBackend
    from .program import Scorer, audit, load_spec, paired, scope_errors
    spec = load_spec(config, {"shift": "x", **(overrides or {})})
    cfg = normalize_config({"agents": spec["agents"], "sandbox": spec["sandbox"], "models": {}}, spec["_source"])
    akey = spec["agent"]
    rd = root() / "param" / f"{env}_{fam}_{akey}{spec['tag']}"
    rescore = bool(spec.get("rescore")) and (rd / "edit_program.py").exists()
    if rd.exists() and not rescore:
        shutil.rmtree(rd)
    (rd / "logs").mkdir(parents=True, exist_ok=True)
    base_dir = ensure_ckpt(env)
    pkg = install_helper(rd.parent / "_sandbox_pkg")
    home = tempfile.mkdtemp(prefix="agenthome_")
    sb = Sandbox(cfg, {"work": (str(rd), False), "pkg": (str(pkg), True), "home": (home, False), "target": (str(base_dir), True)})
    install_python_logger(rd, sb)
    text = prompt(env, fam, example, sb.paths, int(spec["hours"] * 60))
    (rd / "prompt.md").write_text(text)
    done = lambda: (rd / "edit_program.py").exists() and (rd / "rationale.md").exists()  # noqa: E731
    t0 = time.time()
    with sb:
        if rescore:
            old = json.loads((rd / "result.json").read_text()) if (rd / "result.json").exists() else {}
            status, usage = old.get("status", "done(rescored)"), old.get("usage")
        else:
          status, _, usage = run_builtin(cfg["agents"][akey], text, sb, t0 + spec["hours"] * 3600, done,
                                       nudge(f"{sb.paths['work']}/edit_program.py and rationale.md are written"), rd / "logs",
                                       warn_secs=min(900, int(spec["hours"] * 900)),
                                       warn_msg="TIME IS ALMOST UP: make sure edit_program.py and rationale.md are final.")
        (rd / "out").mkdir(exist_ok=True)
        gen = {}
        for v in values:
            code = (f"import sys; sys.path.insert(0, '{sb.paths['work']}'); import torch; from edit_program import make_edit; "
                    f"from safetensors.torch import save_file; t = make_edit({v!r}); "
                    f"save_file({{k: torch.as_tensor(x).float().contiguous() for k, x in t.items()}}, "
                    f"'{sb.paths['work']}/out/edit_{v:g}.safetensors')")
            rc, out = sb.exec(f"python -c {json.dumps(code)}", 300) if done() else (1, "no program")
            gen[v] = {"rc": rc, "out": out[-500:]}
    shutil.rmtree(home, ignore_errors=True)
    be = WMBackend(base_dir)
    sc = Scorer(be, horizon)
    results = {}
    for v in values:
        key = f"{env}/{fam}{v:g}"
        try:
            H = collect(be, env, [key], "heldout", n_heldout, horizon, 1, root() / "data")
        except Exception as e:  # noqa: BLE001  (e.g. MuJoCo instability for extreme parameters)
            results[f"{v:g}"] = {"value": v, "error": f"environment failed: {str(e)[:200]}"}
            log(f"{key}: environment failed ({str(e)[:100]})")
            continue
        hn = load(H[key])
        base = sc.score({}, hn)
        p = rd / "out" / f"edit_{v:g}.safetensors"
        row = {"value": v, "base": base["mean"], "program_rc": gen[v]["rc"]}
        if p.exists():
            from ..checkpoint import Checkpoint
            ck = Checkpoint(base_dir)
            t = {k: x.to(ck.get(k).dtype) for k, x in load_file(str(p)).items() if k in ck}
            bad = scope_errors(t, [r"^predictor\.", r"^action_encoder\.", r"^pred_proj\."]) + \
                [f"{k} frozen" for k in t if any(__import__("re").search(f, k) for f in FROZEN)]
            if bad or any(tuple(x.shape) != ck.shape(k) for k, x in t.items()):
                row["error"] = "; ".join(bad) or "shape mismatch"
            else:
                e = sc.score(t, hn)
                row.update(edited=e["mean"], rel_gain=1 - e["mean"] / base["mean"], gain=paired(e["per_segment"], base["per_segment"]))
        else:
            row["error"] = gen[v]["out"][-300:]
        results[f"{v:g}"] = row
        log(f"{key}: base {row['base']:.4f} -> {row.get('edited', float('nan')):.4f} {row.get('error', '')[:120]}")
    be.close()
    res = {"env": env, "family": fam, "agent": akey, "example": example, "status": status, "usage": usage,
           "wall_s": round(time.time() - t0), "audit": audit(rd), "results": results}
    (rd / "result.json").write_text(json.dumps(res, indent=1, default=str))
    return res
