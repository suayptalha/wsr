"""Weight-space programming of a world model's predictor after a physics change.

    python -m wsr.wm program programs/wm/pusht_slip.yaml [--set passes=0 --set info=none]

The frozen LeWM checkpoint was trained in the original environment. The environment's physics change (``shift``).
The agent is told about the change in words (``info: text``) or only that something changed (``info: none``), reads
the weights in a CPU sandbox with no network, and writes an edit of the dynamics tensors. It never trains: no
gradients, no optimizers, no solving for weights against data. With ``passes > 0`` it may spend forward passes
(``run_model``) that report the latent prediction error of its edit on practice segments of the changed
environment. With ``latents: true`` it also gets the practice segments' encoder embeddings and true states (in
both the old and the new physics) to analyse.

Every submission is scored on held-out segments of the changed environment (episodes the agent never sees):
mean squared error of ``horizon``-step autoregressive latent rollouts from three context frames. An edit is
accepted when the paired reduction of that error is positive with a bootstrap interval excluding zero (and the
edit is not larger than ``accept.max_rel_norm``). The old-physics error on the same segments is reported too.

Program file (YAML):

    env: pusht
    shift: pusht/slip0.5
    agent: gemini
    agents: {gemini: {provider: gemini, model: gemini-4-hydrogen, params: {reasoning_effort: high}}}
    info: text            # text | none
    latents: false
    passes: 10
    hours: 1.0
    rounds: 1
    attempts: 1
    horizon: 5
    n_practice: 64
    n_heldout: 256
    seed: 1
    accept: {min_gain: 0.0, significant: true, confidence: 0.95, max_rel_norm: null}
    output_dir: runs/pusht_slip0.5_text_p10
    sandbox: {runtime: docker, image: wsr-sandbox}   # or userns (see wsr.sandbox), apptainer, local
"""
from __future__ import annotations

import copy
import json
import re
import time
from pathlib import Path

import numpy as np

DEFAULTS = {
    "env": "pusht", "shift": None, "agent": None, "agents": {}, "info": "text", "latents": False, "passes": 10,
    "hours": 1.0, "rounds": 1, "attempts": 1, "horizon": 5, "n_practice": 64, "n_heldout": 256, "seed": 1,
    "practice_seed": 101, "accept": {"min_gain": 0.0, "significant": True, "confidence": 0.95, "max_rel_norm": None},
    "output_dir": None, "overwrite": False, "sandbox": {"runtime": "local"}, "max_output_tokens": None,
    "edit_scope": None, "tag": "", "text_from": None, "rescore": False,
}

# Python constructs that would mean fitting weights instead of writing them; flagged in the audit, not blocked
AUDIT_PATTERNS = {
    "autograd": r"\.backward\(|torch\.autograd|requires_grad_?\(True\)|requires_grad\s*=\s*True",
    "optimizer": r"torch\.optim|optax|\bAdam\(|\bSGD\(|scipy\.optimize|minimize\(|least_squares\(",
    "lstsq": r"lstsq|\bpinv\(|linalg\.solve\(|\bsolve\(|LinearRegression|Ridge\(|curve_fit",
    "forward_reimpl": r"scaled_dot_product_attention|\.predict\(|ConditionalBlock\(|Predictor\(|import lewm|from lewm|"
                      r"import module|from module",
}


def load_spec(path: str | dict, overrides: dict | None = None) -> dict:
    from ..config import deep_merge, load_file
    raw = load_file(path) if not isinstance(path, dict) else copy.deepcopy(path)
    for k, v in (overrides or {}).items():
        cur = raw
        parts = k.split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = v
    spec = deep_merge(DEFAULTS, raw)
    if not spec["shift"] or not spec["agent"]:
        raise SystemExit("wm program: `shift` and `agent` are required")
    if spec["info"] not in ("text", "none"):
        raise SystemExit("wm program: info must be `text` or `none`")
    spec["_source"] = str(path) if not isinstance(path, dict) else "<program>"
    return spec


# ---------------------------------------------------------------- scoring
def paired(new: np.ndarray, old: np.ndarray, conf: float = 0.95, n: int = 4000, seed: int = 0) -> dict:
    """Mean reduction old - new over paired segments with a bootstrap interval."""
    d = np.asarray(old, dtype=np.float64) - np.asarray(new, dtype=np.float64)
    rng = np.random.default_rng(seed)
    boots = np.sort(d[rng.integers(0, len(d), size=(n, len(d)))].mean(1))
    lo, hi = boots[int((1 - conf) / 2 * n)], boots[min(n - 1, int((1 + conf) / 2 * n))]
    return {"gain": float(d.mean()), "ci": [float(lo), float(hi)], "rel": float(d.mean() / max(1e-12, np.mean(old)))}


class Scorer:
    """Errors of the backend (with any tensors patched in) on cached segment sets."""

    def __init__(self, backend, horizon: int):
        self.be, self.h = backend, horizon

    def score(self, tensors: dict, seg: dict, n: int | None = None) -> dict:
        self.be.patch(tensors)
        emb, act = seg["emb"], seg["act"]
        if n:
            emb, act = emb[:n], act[:n]
        e = self.be.errors(emb, act, self.h)
        roll = e["roll"].numpy()
        per = roll.mean(1)
        finite = bool(np.isfinite(per).all())
        return {"per_segment": per, "mean": float(per.mean()) if finite else float("inf"), "per_h": roll.mean(0).tolist(),
                "tf": float(e["tf"].mean()), "finite": finite}


# ---------------------------------------------------------------- the agent's view
def build_prompt(spec: dict, shift, paths: dict, passes: int, practice_n: int, has_latents: bool, history: list[str],
                 env_text: str) -> str:
    from .envs import ENVS, get_shift
    state_desc = ENVS[spec["env"]].state_desc
    W, w = paths["target"], paths["work"]
    L = ["You are a researcher doing **weight-space programming**: you change what a neural network does by reading "
         "its weights and editing them directly, using your own reasoning and knowledge. You do not train it.", "",
         "## The model", "",
         f"A pretrained JEPA world model (LeWorldModel) is at `{W}` (read-only): `model.safetensors` (weights), "
         f"`config.json`, `README.md` (architecture and the forward pass; read it first) and its source code "
         f"`lewm.py`, `module.py`. It was trained on the environment below and predicts, in a learned 192-d latent "
         "space, the embedding of the next frame from the last three frame embeddings and the actions.", "",
         "## The environment it was trained on", "", env_text, ""]
    if history:
        L += ["Earlier rounds already changed the model (the weights above include those changes):"] + [f"- {h}" for h in history] + [""]
    L += ["## What changed", ""]
    if spec["info"] == "text":
        told = get_shift(spec["text_from"]) if spec.get("text_from") else shift
        L += [told.text, ""]
    else:
        L += ["The environment's dynamics have changed. You are not told how; images look the same as before.", ""]
    L += ["## Your goal", "",
          "Edit the weights so that the model predicts the **changed** environment well. The edit is scored on "
          "held-out episodes of the changed environment that you never see: from three true context frames and the "
          f"true actions, the model is rolled out autoregressively for {spec['horizon']} model steps "
          f"({spec['horizon'] * 0.5:g} s), and the score is the mean squared error between predicted and true "
          "latent embeddings (lower is better). The model does not need to keep predicting the old physics. "
          "Your edit is accepted only if it reduces this error significantly; smaller, targeted, well-understood "
          "edits are preferred.", "",
          "## Your tools", "",
          "- A shell and Python (CPU only, no GPU, no internet). `from wsr.sandbox_lib import W` gives "
          "W.names(regex), W.get(name), W.info(name) for the weights; numpy and torch (CPU) are available. "
          f"Keep notes in `{w}/NOTES.md`.",
          "- `check_edit(edit)`: validates an edit file and reports how much each tensor changed. Free."]
    if passes:
        L += [f"- `run_model(edit, n)`: runs the model with your edit (or `base` for the current model) on the first n "
              f"(max {practice_n}) practice segments of the CHANGED environment on GPU, and reports the rollout error "
              "(mean and per step), the teacher-forced one-step error, the current model's error on the same "
              "segments, and the error of the edited model on the same segments under the OLD physics. You have "
              f"**{passes} forward passes**; each call costs one."]
    else:
        L += ["- You cannot run the model in this session: design the edit from the weights, the code and the "
              "description of the change."]
    if has_latents:
        L += [f"- `{w}/practice.npz`: {practice_n} practice segments, each 3 + {spec['horizon']} frames: `act` "
              "(n, frames-1, action_dim) the normalized model-step actions (act[:, k] moves frame k to k+1); "
              "`emb_new` / `emb_old` (n, frames, 192) the encoder embeddings of the true frames in the changed / old "
              "physics (same start state, same actions); `state_new` / `state_old` (n, frames, s) true states "
              f"{state_desc}. You may analyse these "
              "(statistics, differences, even linear probes to understand what latent directions encode)."]
    L += ["", "## How to submit", "",
          f"Write `{w}/edit.safetensors`: a safetensors file holding the NEW values of every tensor you change "
          "(key = tensor name as in model.safetensors; full tensors, or for row edits `<name>::rows` with int64 "
          "`<name>::row_index`). Only `predictor.*`, `action_encoder.*` and `pred_proj.*` may change. Whatever is in "
          f"that file when the session ends is your submission. Then explain in `{w}/rationale.md` what you found, "
          "what you changed and why.", "",
          ] + ([f"**In this session only tensors whose names match {spec['edit_scope']} (regular expressions) may "
                "change**; an edit touching any other tensor is rejected.", ""] if spec.get("edit_scope") else []) + [
          "## Rules (checked from your logs afterwards)", "",
          "1. No training or fitting of weights: no gradients or autograd, no optimizers, no evolutionary or random "
          "search over weights, no least-squares/regression or any solver whose unknowns are model weights. You may "
          "compute statistics of the weights (norms, SVD, projections, products) and design the edit by reasoning.",
          "2. Do not run the model or re-implement its forward pass on inputs" + (" (use `run_model`)." if passes else "."),
          "3. No internet.",
          f"4. You have about {int(spec['hours'] * 60)} minutes of wall-clock time. Write a first edit early; you may "
          "overwrite it."]
    return "\n".join(L)


class WMRunner:
    """Host-side ``run_model`` tool sharing the loaded backend."""

    def __init__(self, scorer: Scorer, practice_new: dict, practice_old: dict, budget: int, sb, ck, accum: dict,
                 frozen, scope):
        self.sc, self.pn, self.po = scorer, practice_new, practice_old
        self.left = self.budget = budget
        self.sb, self.ck, self.accum, self.frozen, self.scope = sb, ck, accum, frozen, scope
        self.log: list[dict] = []
        self._cur: dict = {}

    def run(self, args: dict) -> tuple[str, bool]:
        from ..ops import load_submission, validate_edit
        if self.left <= 0:
            return "no forward passes left", True
        n = max(1, min(int(args.get("n") or len(self.pn["emb"])), len(self.pn["emb"])))
        if args["edit"] == "base":
            tensors = {}
        else:
            p = self.sb.readable_host_path(args["edit"])
            if p is None or not p.exists():
                return f"no such file: {args['edit']}", True
            try:
                tensors = load_submission(self.ck, p)
            except Exception as e:  # noqa: BLE001
                return f"could not read the edit: {e}", True
            v = validate_edit(self.ck, tensors, self.frozen)
            bad = scope_errors(tensors, self.scope)
            if not v["ok"] or bad:
                return "invalid edit: " + "; ".join(v["errors"] + bad), True
        self.left -= 1
        full = {**self.accum, **tensors}
        r = self.sc.score(full, self.pn, n)
        out = {"n": n, "rollout_mse": round(r["mean"], 6), "rollout_mse_per_step": [round(x, 6) for x in r["per_h"]],
               "teacher_forced_mse": round(r["tf"], 6)}
        if n not in self._cur:
            c = self.sc.score(self.accum, self.pn, n)
            self._cur[n] = {"rollout_mse": round(c["mean"], 6), "per_step": [round(x, 6) for x in c["per_h"]],
                            "teacher_forced_mse": round(c["tf"], 6)}
        out["current_model"] = self._cur[n]
        o = self.sc.score(full, self.po, n)
        out["old_physics_rollout_mse"] = round(o["mean"], 6)
        out["forward_passes_left"] = self.left
        self.log.append({"t": time.time(), "edit": args["edit"], **out})
        return json.dumps(out, indent=1), False

    def close(self):
        pass


def scope_errors(tensors: dict, scope: list[str] | None) -> list[str]:
    if not scope:
        return []
    return [f"`{n}` is outside the editable scope {scope}" for n in tensors if not any(re.search(s, n) for s in scope)]


def audit(session: Path) -> dict:
    """Count Python snippets of a session that match patterns of fitting or of running the model."""
    files = list((session / "logs" / "snippets").glob("*.py")) + list((session / "logs" / "code").glob("*.py"))
    hits = {k: [] for k in AUDIT_PATTERNS}
    for f in files:
        try:
            t = f.read_text(errors="replace")
        except OSError:
            continue
        for k, rx in AUDIT_PATTERNS.items():
            if re.search(rx, t):
                hits[k].append(f.name)
    return {"snippets": len(files), **{k: len(v) for k, v in hits.items()}, "files": {k: v[:20] for k, v in hits.items() if v}}


def write_latents(path: Path, new: dict, old: dict, n: int) -> None:
    np.savez_compressed(path, act=new["act"][:n].numpy(), emb_new=new["emb"][:n].numpy(), emb_old=old["emb"][:n].numpy(),
                        state_new=new["states"][:n].numpy(), state_old=old["states"][:n].numpy())


# ---------------------------------------------------------------- the program
def run(config, overrides: dict | None = None, log=print, clients=None) -> dict:
    from safetensors.torch import save_file

    from ..checkpoint import Checkpoint
    from ..config import normalize_config
    from ..ops import compact_view, load_submission, validate_edit
    from ..saving import export_checkpoint
    from ..write import RUN_MODEL_SCHEMA, edit_session
    from .cli import ensure_ckpt, root
    from .envs import collect, env_description, get_shift, load
    from .model import FROZEN, WMBackend

    spec = load_spec(config, overrides)
    shift = get_shift(spec["shift"])
    cfg = normalize_config({"agents": spec["agents"], "sandbox": spec["sandbox"], "models": {}}, spec["_source"])
    agent_keys = spec["agent"] if isinstance(spec["agent"], list) else [spec["agent"]]
    for k in agent_keys:
        if spec["max_output_tokens"]:
            cfg["agents"][k].setdefault("max_output_tokens", spec["max_output_tokens"])
    out = Path(spec["output_dir"] or root() / "runs" / f"{shift.env}_{shift.name}_{spec['info']}_p{spec['passes']}{spec['tag']}")
    if out.exists() and any(out.iterdir()):
        if not spec["overwrite"]:
            raise SystemExit(f"{out} is not empty")
        import shutil
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "program.json").write_text(json.dumps({k: v for k, v in spec.items() if not k.startswith("_")}, indent=1, default=str))
    base_dir = ensure_ckpt(spec["env"])
    be = WMBackend(base_dir)
    sc = Scorer(be, spec["horizon"])
    data = root() / "data"
    P = collect(be, spec["env"], [spec["shift"]], "practice", spec["n_practice"], spec["horizon"], spec["practice_seed"], data)
    H = collect(be, spec["env"], [spec["shift"]], "heldout", spec["n_heldout"], spec["horizon"], spec["seed"], data)
    pn, po, hn, ho = load(P[spec["shift"]]), load(P["base"]), load(H[spec["shift"]]), load(H["base"])
    env_text = env_description(spec["env"])
    accum: dict = {}
    base_new, base_old = sc.score({}, hn), sc.score({}, ho)
    cur = base_new
    log(f"[{shift.key()}] base model held-out rollout MSE: changed physics {base_new['mean']:.5f}, old physics {base_old['mean']:.5f}")
    history, notes = [], []
    current_dir = base_dir
    clients = list(clients or [])
    t_start = time.time()
    for r in range(1, int(spec["rounds"]) + 1):
        ck = Checkpoint(current_dir)
        cands = []
        for k in range(1, int(spec["attempts"]) + 1):
            akey = agent_keys[(k - 1) % len(agent_keys)]
            agent = cfg["agents"][akey]
            rd = out / "sessions" / f"r{r}_a{k}_{akey}"
            rd.mkdir(parents=True)
            if spec["latents"]:
                write_latents(rd / "practice.npz", pn, po, spec["n_practice"])

            def tools_fn(sb, ck=ck):
                if not int(spec["passes"]):
                    return {}, None
                runner = WMRunner(sc, pn, po, int(spec["passes"]), sb, ck, accum, FROZEN, spec["edit_scope"])
                schema = copy.deepcopy(RUN_MODEL_SCHEMA)
                schema["properties"].pop("task")
                schema["properties"]["n"]["description"] = f"number of practice segments (default and max {spec['n_practice']})"
                return {"run_model": ({"name": "run_model", "description": "Run the world model with an edit (or 'base') "
                                       "on the first n practice segments of the changed environment and report its "
                                       "latent prediction errors. Costs one forward pass.", "schema": schema},
                                      runner.run)}, runner

            meta = edit_session(cfg, agent, rd, Path(current_dir), ck, None, FROZEN,
                                lambda paths, has_runner: build_prompt(spec, shift, paths, int(spec["passes"]) if has_runner else 0,
                                                                       spec["n_practice"], bool(spec["latents"]), notes, env_text),
                                float(spec["hours"]), tools_fn, clients.pop(0) if clients else None, log,
                                {"run_id": rd.name, "round": r, "attempt": k, "agent": akey})
            row = {"round": r, "attempt": k, "agent": akey, "session": str(rd), "status": meta["status"],
                   "passes_used": meta.get("passes_used", 0), "usage": meta.get("usage"), "wall_s": meta.get("wall_s"),
                   "audit": audit(rd)}
            if not meta["submission"]:
                row.update(accepted=False, reason="no valid edit submitted")
                history.append(row)
                log(f"  {rd.name}: no valid edit ({meta.get('validation')})")
                continue
            tensors = load_submission(ck, rd / meta["submission"])
            bad = scope_errors(tensors, spec["edit_scope"])
            if bad:
                row.update(accepted=False, reason="; ".join(bad))
                history.append(row)
                continue
            trial = {**accum, **tensors}
            sn, so = sc.score(trial, hn), sc.score(trial, ho)
            g = paired(sn["per_segment"], cur["per_segment"], spec["accept"]["confidence"])
            g0 = paired(sn["per_segment"], base_new["per_segment"], spec["accept"]["confidence"])
            v = validate_edit(Checkpoint(base_dir), trial, FROZEN)
            a = spec["accept"]
            if not sn["finite"]:
                ok, why = False, "non-finite predictions"
            elif g["gain"] <= a["min_gain"]:
                ok, why = False, f"error reduction {g['gain']:+.5f} not above {a['min_gain']}"
            elif a["significant"] and g["ci"][0] <= 0:
                ok, why = False, f"error reduction {g['gain']:+.5f} not significant [{g['ci'][0]:+.5f}, {g['ci'][1]:+.5f}]"
            elif a["max_rel_norm"] is not None and v["rel_norm"] > a["max_rel_norm"]:
                ok, why = False, f"edit too large (rel norm {v['rel_norm']:.4f})"
            else:
                ok, why = True, f"error {cur['mean']:.5f} -> {sn['mean']:.5f} ({100 * g['rel']:+.1f}%)"
            row.update(heldout_new=sn["mean"], heldout_old=so["mean"], heldout_new_per_h=sn["per_h"], tf_new=sn["tf"],
                       gain=g, gain_vs_base=g0, rel_norm=v["rel_norm"], values_changed=v["values_changed"],
                       tensors=sorted(tensors), accepted=ok, reason=why)
            history.append(row)
            log(f"  {rd.name}: held-out {sn['mean']:.5f} (current {cur['mean']:.5f}, old physics {so['mean']:.5f}) -> "
                f"{'ACCEPTED' if ok else 'rejected'}: {why}")
            if ok:
                cands.append((g["gain"], row, tensors, sn))
        if not cands:
            log(f"round {r}: nothing accepted")
            break
        _, row, tensors, sn = max(cands, key=lambda c: c[0])
        row["chosen"] = True
        accum.update(tensors)
        cur = sn
        rat = (Path(row["session"]) / "rationale.md")
        notes.append(f"round {r}: changed {', '.join(sorted(tensors))[:300]}. " +
                     (rat.read_text()[:400].replace("\n", " ") if rat.exists() else ""))
        current_dir = export_checkpoint(base_dir, accum, out / "rounds" / f"r{r}")
    final_old = sc.score(accum, ho)
    res = {"shift": shift.key(), "info": spec["info"], "text_from": spec.get("text_from"), "passes": spec["passes"], "latents": spec["latents"],
           "agents": agent_keys, "horizon": spec["horizon"], "n_heldout": spec["n_heldout"],
           "base_new": base_new["mean"], "base_old": base_old["mean"], "final_new": cur["mean"], "final_old": final_old["mean"],
           "final_gain_vs_base": paired(cur["per_segment"], base_new["per_segment"]),
           "accepted_rounds": sum(1 for h in history if h.get("chosen")), "wall_s": round(time.time() - t_start),
           "history": history}
    if accum:
        ck0 = Checkpoint(base_dir)
        save_file(compact_view(ck0, accum), str(out / "edit.safetensors"))
        res["edit"] = {k: validate_edit(ck0, accum, FROZEN)[k] for k in ("rel_norm", "values_changed")}
    np.save(out / "heldout_per_segment.npy", np.stack([base_new["per_segment"], cur["per_segment"]]))
    (out / "result.json").write_text(json.dumps(res, indent=1, default=str))
    (out / "report.md").write_text(report(res))
    be.close()
    log(f"done: {shift.key()} base {base_new['mean']:.5f} -> final {cur['mean']:.5f}   ({out / 'result.json'})")
    return res


def report(res: dict) -> str:
    L = [f"# World-model weight-space program: {res['shift']}", "",
         f"info={res['info']} passes={res['passes']} latents={res['latents']} agents={res['agents']}", "",
         f"Held-out rollout MSE (changed physics): base {res['base_new']:.5f} -> final {res['final_new']:.5f}  "
         f"(gain {res['final_gain_vs_base']['gain']:+.5f}, CI {res['final_gain_vs_base']['ci']})", "",
         f"Old-physics MSE: base {res['base_old']:.5f} -> final {res['final_old']:.5f}", "",
         "| round | attempt | agent | status | held-out new | held-out old | gain [CI] | passes | rel norm | accepted | reason |",
         "|---|---|---|---|---|---|---|---|---|---|---|"]
    for h in res["history"]:
        g = h.get("gain")
        gs = f"{g['gain']:+.5f} [{g['ci'][0]:+.5f}, {g['ci'][1]:+.5f}]" if g else "–"
        L.append(f"| {h['round']} | {h['attempt']} | {h['agent']} | {h['status']} | {h.get('heldout_new', '–')} | "
                 f"{h.get('heldout_old', '–')} | {gs} | {h.get('passes_used')} | {h.get('rel_norm', '–')} | "
                 f"{'**yes**' if h.get('chosen') else ('yes' if h.get('accepted') else 'no')} | {h.get('reason', '')} |")
    return "\n".join(L) + "\n"
