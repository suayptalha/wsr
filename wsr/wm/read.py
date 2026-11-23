"""Weight-space reading of world models: which physics change does a weight update encode?

    python -m wsr.wm reading-models --env pusht           # GPU: one fine-tuned predictor per shift (+ controls)
    python -m wsr.wm read <agent.yaml> --env pusht --model <id>   # one reading session
    python -m wsr.wm read-score --env pusht

For every shift of an environment, the frozen LeWM is fine-tuned with gradients (all dynamics tensors, the same
recipe for every shift) on trajectories of the shifted environment. Controls are fine-tuned the same way on
trajectories of the ORIGINAL environment (more data, no change). The models get opaque ids; the key is kept out
of the sandbox. A reader agent gets the original weights and one fine-tuned checkpoint, the description of the
environment and the list of candidate changes (all shifts of the environment plus "no change"), and must say
which change the fine-tuning data came from, without running either model.
"""
from __future__ import annotations

import json
import random
import time
from pathlib import Path

RECIPE = {"method": "full", "lr": 1e-4, "steps": 1000, "data": "train"}


def models_dir(env: str) -> Path:
    from .cli import root
    return root() / "reading" / env


def make_models(env: str, n_controls: int = 2, horizon: int = 5) -> None:
    import torch

    from ..saving import export_checkpoint
    from .baselines import SCOPES, finetune
    from .cli import ensure_ckpt, root
    from .envs import SHIFTS, collect, load
    from .model import WMBackend
    base_dir = ensure_ckpt(env)
    be = WMBackend(base_dir)
    orig = be.state()
    keys = sorted(k for k in SHIFTS if k.startswith(env + "/"))
    paths = collect(be, env, keys, "train", 2048, horizon, 7, root() / "data")
    jobs = [(k, paths[k], 0) for k in keys] + [("none", paths["base"], s) for s in range(n_controls)]
    out = models_dir(env)
    (out / "models").mkdir(parents=True, exist_ok=True)
    rng = random.Random(f"reading-{env}")
    key = {}
    for shift, path, seed in jobs:
        with torch.no_grad():
            for k, v in be.model.state_dict().items():
                v.copy_(orig[k].to(v.device))
        d = load(path)
        if shift == "none":          # a disjoint half per control, so the controls differ
            half = len(d["emb"]) // n_controls
            d = {"emb": d["emb"][seed * half:(seed + 1) * half], "act": d["act"][seed * half:(seed + 1) * half]}
        names, _ = finetune(be, SCOPES[RECIPE["method"]], d, RECIPE["steps"], RECIPE["lr"], seed=seed)
        tensors = {n: be.slots[n].detach().cpu().clone() for n in names}
        mid = "m" + "".join(rng.choice("abcdefghjkmnpqrstuvwxyz23456789") for _ in range(6))
        export_checkpoint(base_dir, tensors, out / "models" / mid, overwrite=True)
        for f in ("wsr_edit.json",):
            p = out / "models" / mid / f
            if p.exists():
                p.unlink()
        key[mid] = {"shift": shift, "seed": seed}
        print(f"{mid}: {shift} (seed {seed})", flush=True)
    (out / "key.json").write_text(json.dumps({"recipe": RECIPE, "models": key}, indent=1))
    print("wrote", out / "key.json")


def candidates(env: str) -> list[tuple[str, str]]:
    from .envs import SHIFTS
    c = [(k, SHIFTS[k].text) for k in sorted(SHIFTS) if k.startswith(env + "/")]
    c.append(("none", "Nothing changed: the extra training data came from the ORIGINAL environment (same physics)."))
    return c


def prompt(env: str, paths: dict, letters: dict, minutes: int, reference: bool = False) -> str:
    from .envs import env_description
    T, M, w = paths["target"], paths["interventions"], paths["work"]
    L = ["You are a researcher doing **weight-space reading**: you work out what a change to a neural network's weights "
         "does, or where it came from, by reading the weights, without running the network.", "",
         "## The models", "",
         f"`{T}` (read-only) is a pretrained JEPA world model (LeWorldModel): `model.safetensors`, `config.json`, "
         "`README.md` (architecture and forward pass; read it first) and source code `lewm.py`, `module.py`. It predicts, "
         "in a learned 192-d latent space, the embedding of the next frame from the last three frame embeddings and the "
         "actions.", "",
         f"`{M}` (read-only) is the SAME model after further gradient training: all dynamics tensors (`predictor.*`, "
         "`action_encoder.*`, `pred_proj.*`) were trained with the usual next-embedding loss on trajectories from an "
         "environment that may differ from the original one (the encoder and projector were frozen). Every candidate "
         "below was trained with exactly the same recipe and amount of data.", ""] + ([
         f"`{paths.get('reference')}` (read-only) is a REFERENCE: the same model fine-tuned with exactly the same recipe "
         "and amount of data on trajectories from the ORIGINAL environment (no change). Fine-tuning moves the weights "
         "even when nothing changed; compare against this reference to separate that generic drift from what is "
         "specific to the change.", ""] if reference else []) + [
         "## The environment it was originally trained on", "", env_description(env), "",
         "## Candidates: where did the extra training data come from?", ""]
    for letter, (_, text) in letters.items():
        L.append(f"- **{letter}**: {text}")
    L += ["", "## Your task", "",
          "Decide which candidate the fine-tuning data came from, by comparing the two sets of weights. Use what you know "
          "about the physics, the action encoding and the architecture (for example: which tensors moved, in which "
          "directions, what the change does to how actions are encoded or to how motion is extrapolated).", "",
          "## Tools", "",
          "A shell and Python (CPU only, no GPU, no internet). `from wsr.sandbox_lib import W` reads the original model "
          f"(W.names, W.get, W.info); `from wsr.checkpoint import Checkpoint; C = Checkpoint('{M}')` reads the trained "
          "one (C.names(), C.get(name)). numpy and torch (CPU) are available.", "",
          "## Answer", "",
          f"Write `{w}/answer.json`: {{\"choice\": \"<letter>\", \"ranking\": [all letters, most to least likely], "
          "\"confidence\": <0-100>}} and explain your reasoning in "
          f"`{w}/rationale.md`.", "",
          "## Rules", "",
          "1. Do not run either model (or re-implement its forward pass) on inputs; read the weights.",
          "2. No internet.",
          f"3. You have about {minutes} minutes. Write a first answer early; you may overwrite it."]
    return "\n".join(L)


def run_session(config: str, env: str, mid: str, overrides: dict | None = None, log=print) -> dict:
    import shutil
    import tempfile

    from ..config import normalize_config
    from ..harness import nudge, run_builtin
    from ..sandbox import Sandbox, install_helper, install_python_logger
    from .cli import ensure_ckpt
    from .program import audit, load_spec
    over = dict(overrides or {})
    reference = bool(over.pop("reference", False))
    spec = load_spec(config, {"shift": "x", **over})
    cfg = normalize_config({"agents": spec["agents"], "sandbox": spec["sandbox"], "models": {}}, spec["_source"])
    akey = spec["agent"]
    agent = cfg["agents"][akey]
    cands = candidates(env)
    rng = random.Random(f"{mid}-{akey}")
    order = cands[:]
    rng.shuffle(order)
    letters = {chr(65 + i): c for i, c in enumerate(order)}
    rd = models_dir(env) / "sessions" / f"{mid}_{akey}{spec['tag']}"
    if rd.exists():
        shutil.rmtree(rd)
    (rd / "logs").mkdir(parents=True)
    pkg = install_helper(rd.parent / "_sandbox_pkg")
    home = tempfile.mkdtemp(prefix="agenthome_")
    mounts = {"work": (str(rd), False), "pkg": (str(pkg), True), "home": (home, False),
              "target": (str(ensure_ckpt(env)), True), "interventions": (str(models_dir(env) / "models" / mid), True)}
    if reference:
        key = json.loads((models_dir(env) / "key.json").read_text())["models"]
        ref = next(m for m, v in key.items() if v["shift"] == "none" and v["seed"] == 0)
        if ref == mid:
            raise SystemExit("the reference model cannot be read against itself")
        mounts["reference"] = (str(models_dir(env) / "models" / ref), True)
    sb = Sandbox(cfg, mounts)
    install_python_logger(rd, sb)
    text = prompt(env, sb.paths, letters, int(spec["hours"] * 60), reference)
    (rd / "prompt.md").write_text(text)
    (rd / "letters.json").write_text(json.dumps({k: v[0] for k, v in letters.items()}))

    def done():
        try:
            a = json.loads((rd / "answer.json").read_text())
            return a.get("choice") in letters and (rd / "rationale.md").exists()
        except Exception:  # noqa: BLE001
            return False
    t0 = time.time()
    deadline = t0 + spec["hours"] * 3600
    warn = f"TIME IS ALMOST UP. Make sure {sb.paths['work']}/answer.json and rationale.md hold your final answer."
    with sb:
        status, nudges, usage = run_builtin(agent, text, sb, deadline, done,
                                            nudge(f"{sb.paths['work']}/answer.json and rationale.md are written"),
                                            rd / "logs", warn_secs=min(900, int(spec["hours"] * 900)), warn_msg=warn)
    shutil.rmtree(home, ignore_errors=True)
    meta = {"model": mid, "agent": akey, "reference": reference, "status": status, "usage": usage, "wall_s": round(time.time() - t0),
            "audit": audit(rd)}
    (rd / "meta.json").write_text(json.dumps(meta, indent=1))
    log(json.dumps({k: meta[k] for k in ("model", "agent", "status", "wall_s")}))
    return meta


def score(env: str) -> dict:
    key = json.loads((models_dir(env) / "key.json").read_text())["models"]
    fam = lambda s: "none" if s == "none" else "".join(ch for ch in s.split("/")[-1] if ch.isalpha())  # noqa: E731
    rows = []
    for sd in sorted((models_dir(env) / "sessions").glob("m*")):
        mid = sd.name.split("_")[0]
        letters = json.loads((sd / "letters.json").read_text())
        meta = json.loads((sd / "meta.json").read_text()) if (sd / "meta.json").exists() else {}
        truth = key[mid]["shift"]
        try:
            a = json.loads((sd / "answer.json").read_text())
            choice = letters.get(a.get("choice"))
            ranking = [letters.get(x) for x in a.get("ranking", []) if x in letters]
        except Exception:  # noqa: BLE001
            choice, ranking, a = None, [], {}
        rank = ranking.index(truth) + 1 if truth in ranking else None
        rows.append({"session": sd.name, "model": mid, "agent": (meta.get("agent") or "") + ("+ref" if meta.get("reference") else ""), "truth": truth, "choice": choice,
                     "correct": choice == truth, "family_correct": choice is not None and fam(choice) == fam(truth),
                     "rank": rank, "rr": 1 / rank if rank else 0.0, "confidence": a.get("confidence"),
                     "status": meta.get("status"), "tokens_in": (meta.get("usage") or {}).get("input_tokens")})
    n_c = len(candidates(env))
    by_agent = {}
    for r in rows:
        g = by_agent.setdefault(r["agent"], [])
        g.append(r)
    summ = {ag: {"n": len(g), "top1": sum(r["correct"] for r in g) / len(g), "family": sum(r["family_correct"] for r in g) / len(g),
                 "mrr": sum(r["rr"] for r in g) / len(g)} for ag, g in by_agent.items()}
    out = {"env": env, "chance_top1": 1 / n_c, "summary": summ, "rows": rows}
    from .cli import root
    p = root() / "results" / "reading" / f"{env}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, indent=1))
    for r in rows:
        print(f"{r['session']:32s} truth={r['truth']:18s} choice={str(r['choice']):18s} {'OK' if r['correct'] else '  '} rank={r['rank']}")
    print(json.dumps(summ, indent=1), "chance", round(1 / n_c, 3))
    return out
