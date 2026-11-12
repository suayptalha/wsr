"""Weight-space programming as a tool: give a model, a goal and an agent; get back an edited model.

    wsr program program.yaml              # command line
    wsr.program("program.yaml")           # Python

Like a fine-tuning config, a program file names the base model, the goal (described in words and defined by a
dataset), the agent that writes the edit, and how the result is checked. Nothing is trained: the agent reads the
weights in a sandbox and writes the new values of the parameters it changes, by reasoning.

Each **round** runs ``attempts`` independent sessions on the current model. Every submitted edit is scored on a
held-out split of the goal data that no agent sees, and on the **guards** (behavior suites that must not change).
The best edit that passes the **acceptance rule** is applied, and the next round starts from the edited model.
At the end the accepted edits are written as a new checkpoint (``output_dir/model``) with a model card, and
optionally pushed to the Hugging Face Hub.

Goal data (JSONL), one example per line; ``prompt`` is a string or a list of chat messages:

  {"prompt": "...", "target": "..."}                       metrics contains | exact | regex | numeric | judge | logprob
  {"prompt": "...", "chosen": "...", "rejected": "..."}    metric preference
  {"prompt": "..."}                                        metric length (number of generated tokens)
"""
from __future__ import annotations

import copy
import json
import random
import re
import statistics
import time
from dataclasses import dataclass, field
from pathlib import Path

from .config import (ConfigError, agent_cfg, deep_merge, find_config, is_self, load_file, model_dir, normalize_config,
                     read_jsonl, write_jsonl)

METRICS = {
    "contains": "the answer contains the target text (case-insensitive)",
    "exact": "the answer equals the target text (ignoring case and surrounding whitespace)",
    "regex": "the target regular expression matches the answer",
    "numeric": "the last number in the answer equals the target",
    "judge": "a judge decides whether the answer meets the target description",
    "logprob": "mean log-probability per token of the target text after the prompt (no sampling)",
    "preference": "the chosen answer is more likely than the rejected one (per-token log-probability)",
    "length": "number of tokens the model generates",
}
GENERATIVE = {"contains", "exact", "regex", "numeric", "judge", "length"}
JUDGE_GOAL = """You are checking whether a model's answer meets a requirement.
Prompt:
<<<{prompt}>>>
Requirement:
<<<{target}>>>
Answer:
<<<{output}>>>
Does the answer meet the requirement? Reply with exactly one word: YES or NO."""

DEFAULTS = {
    "base_model": None, "agent": None,
    "goal": {"text": None, "data": None, "metric": "contains", "direction": None, "eval_fraction": 0.5,
             "max_eval": 200, "max_practice": 64, "seed": 0, "target_score": None, "max_tokens": 256,
             "sampling": {"temperature": 0.0}, "thinking": None, "judge_prompt": JUDGE_GOAL, "system": None},
    "guards": [],
    "rounds": 1, "attempts": 1, "hours": 1.0, "passes": 10,
    "accept": {"min_gain": 0.0, "significant": True, "confidence": 0.95, "max_rel_norm": None, "max_hit_limit_rise": 0.25},
    "output_dir": "wsr-out", "overwrite": False, "push_to_hub": None, "private": True, "framing": None,
}


class GenSettings:
    """What a backend needs to generate: max_tokens, sampling, thinking."""

    def __init__(self, max_tokens: int, sampling: dict, thinking):
        self.max_tokens, self.sampling, self.thinking = max_tokens, dict(sampling), thinking


# ---------------------------------------------------------------- loading
def load_program(config: str | dict, overrides: dict | None = None, wsr_config: str | None = None) -> tuple[dict, dict]:
    """-> (wsr config, program spec). The program file may carry its own models/agents/judge/sandbox sections,
    which are merged over wsr.yaml when one exists."""
    raw = load_file(config) if not isinstance(config, dict) else copy.deepcopy(config)
    src = str(config) if not isinstance(config, dict) else "<program>"
    for k, v in (overrides or {}).items():
        cur = raw
        parts = k.split(".")
        for p in parts[:-1]:
            cur = cur.setdefault(p, {})
        cur[parts[-1]] = v
    base_cfg = {}
    try:
        base_cfg = load_file(find_config(wsr_config))
    except SystemExit:
        if wsr_config:
            raise
    sections = ("work_dir", "data_dir", "models", "agents", "judge", "sandbox", "backend", "reader_files_dir")
    cfg_raw = deep_merge(base_cfg, {k: raw[k] for k in sections if k in raw})
    cfg_raw.setdefault("models", {})
    cfg_raw.setdefault("agents", {})
    spec = deep_merge(DEFAULTS, {k: v for k, v in raw.items() if k not in sections})
    spec["_source"] = src
    # base model: a key of the config, or a path / repo id / mapping given here
    bm = spec["base_model"]
    if bm is None:
        raise ConfigError(f"{src}: `base_model` is required")
    if isinstance(bm, dict) or bm not in cfg_raw["models"]:
        cfg_raw["models"]["base"] = bm if isinstance(bm, dict) else {"path": bm}
        spec["model_key"] = "base"
    else:
        spec["model_key"] = bm
    # agents: keys of the config or inline mappings; attempts cycle through the list
    ags = spec["agent"]
    if ags is None:
        raise ConfigError(f"{src}: `agent` is required (a key of wsr.yaml's agents, or a mapping)")
    keys = []
    for i, a in enumerate(ags if isinstance(ags, list) else [ags]):
        if isinstance(a, dict):
            k = a.get("name", f"agent{i}")
            cfg_raw["agents"][k] = {kk: vv for kk, vv in a.items() if kk != "name"}
            keys.append(k)
        else:
            keys.append(a)
    spec["agents"] = keys
    cfg = normalize_config(cfg_raw, src)
    for k in keys:
        agent_cfg(cfg, k)
        if cfg["agents"][k]["harness"] in ("direct", "baseline"):
            raise ConfigError(f"{src}: agent `{k}` has harness `{cfg['agents'][k]['harness']}`; programming needs tools")
    g = spec["goal"]
    if g["metric"] not in METRICS:
        raise ConfigError(f"{src}: goal.metric must be one of {sorted(METRICS)}")
    if not g.get("data"):
        raise ConfigError(f"{src}: goal.data is required (a JSONL file of examples)")
    if g["direction"] is None:
        g["direction"] = "down" if g["metric"] == "length" else "up"
    if g["direction"] not in ("up", "down"):
        raise ConfigError(f"{src}: goal.direction must be up or down")
    if not g.get("text"):
        g["text"] = {"up": "Increase", "down": "Decrease"}[g["direction"]] + f" this score: {METRICS[g['metric']]}."
    if g["metric"] == "judge" and cfg["judge"] in (None, "heuristic"):
        raise ConfigError(f"{src}: metric `judge` needs a model judge (judge: <agent name> or an endpoint)")
    spec["guards"] = [{"suite": x} if isinstance(x, str) else dict(x) for x in spec["guards"]]
    return cfg, spec


def _resolve(path: str, src: str) -> Path:
    p = Path(path)
    if not p.is_absolute() and not p.exists() and src not in ("<program>",):
        q = Path(src).parent / p
        if q.exists():
            return q
    return p


def load_goal_data(spec: dict) -> tuple[list[dict], list[dict]]:
    """-> (practice examples the agent sees, held-out examples used for acceptance)."""
    g = spec["goal"]
    p = _resolve(g["data"], spec["_source"])
    if not p.exists():
        raise ConfigError(f"goal data {p} not found")
    rows = read_jsonl(p)
    need = {"preference": ("chosen", "rejected"), "length": ()}.get(g["metric"], ("target",))
    for i, r in enumerate(rows):
        if "prompt" not in r or any(k not in r for k in need):
            raise ConfigError(f"{p}: line {i + 1} needs `prompt`" + (f" and {', '.join(need)}" if need else ""))
        r.setdefault("id", f"g{i}")
    if g.get("eval_data"):
        ev = read_jsonl(_resolve(g["eval_data"], spec["_source"]))
        for i, r in enumerate(ev):
            r.setdefault("id", f"e{i}")
        practice = rows
    else:
        rows = rows[:]
        random.Random(g["seed"]).shuffle(rows)
        k = max(1, round(len(rows) * g["eval_fraction"]))
        if len(rows) < 2:
            raise ConfigError(f"{p}: need at least 2 examples (some are held out)")
        ev, practice = rows[:k], rows[k:]
    return practice[: g["max_practice"]], ev[: g["max_eval"]]


def messages_of(ex: dict, system: str | None) -> list[dict]:
    m = ex["prompt"] if isinstance(ex["prompt"], list) else [{"role": "user", "content": ex["prompt"]}]
    return ([{"role": "system", "content": system}] if system else []) + m


# ---------------------------------------------------------------- scoring
def item_score(metric: str, ex: dict, text: str, n_tokens: int) -> float:
    from .suites.grading import last_number
    t = ex.get("target")
    if metric == "contains":
        golds = t if isinstance(t, list) else [t]
        return float(any(str(x).lower() in text.lower() for x in golds))
    if metric == "exact":
        golds = t if isinstance(t, list) else [t]
        return float(any(text.strip().casefold() == str(x).strip().casefold() for x in golds))
    if metric == "regex":
        return float(re.search(t, text) is not None)
    if metric == "numeric":
        n = last_number(text)
        try:
            return float(n is not None and abs(float(n) - float(t)) < 1e-9)
        except ValueError:
            return 0.0
    if metric == "length":
        return float(n_tokens)
    raise ValueError(metric)


def paired_gain(new: list[float], old: list[float], sign: int, conf: float = 0.95, n: int = 2000, seed: int = 0) -> dict:
    """Mean of sign * (new - old) over paired examples, with a bootstrap interval."""
    d = [sign * (a - b) for a, b in zip(new, old)]
    if not d:
        return {"gain": 0.0, "ci": [0.0, 0.0]}
    rng = random.Random(seed)
    boots = sorted(statistics.mean(rng.choices(d, k=len(d))) for _ in range(n))
    lo, hi = boots[int((1 - conf) / 2 * n)], boots[min(n - 1, int((1 + conf) / 2 * n))]
    return {"gain": statistics.mean(d), "ci": [lo, hi]}


class Evaluator:
    """Holds the base model loaded once; scores the goal and the guards for any accumulated edit."""

    def __init__(self, cfg: dict, spec: dict, log=print):
        from .backends import make_backend
        from .suites import load_suite
        self.cfg, self.spec, self.log = cfg, spec, log
        self.key = spec["model_key"]
        self.mcfg = cfg["models"][self.key]
        g = spec["goal"]
        self.metric, self.sign = g["metric"], 1 if g["direction"] == "up" else -1
        self.goal_gen = GenSettings(g["max_tokens"], g["sampling"], g["thinking"])
        self.guards = []
        for gd in spec["guards"]:
            s = load_suite(gd["suite"], cfg["data_dir"])
            keep = gd.get("keep") or [b.key for b in s.behaviors if not b.judge or cfg["judge"] not in (None,)]
            self.guards.append({"name": gd.get("name", s.name), "suite": s, "keep": keep,
                                "base_runs": int(gd.get("base_runs", 3)), "items": s.items(cfg["data_dir"])})
        widest = max([g["max_tokens"]] + [x["suite"].max_tokens for x in self.guards])
        self.backend = make_backend(model_dir(cfg, self.key), self.mcfg, GenSettings(widest, g["sampling"], g["thinking"]))
        self.judge = None
        if self.metric == "judge" or any(x["suite"].judged for x in self.guards):
            from .labels import Judge
            self.judge = Judge(cfg)
        self.fmt = self.mcfg.get("answer_format", "think")
        self.system = g.get("system")
        self._guard_base: dict = {}

    def close(self):
        self.backend.close()

    # ---- goal
    def goal(self, tensors: dict, examples: list[dict]) -> dict:
        """Per-example scores and outputs of the model with ``tensors`` patched in."""
        from .backends import final_text
        self.backend.patch(tensors)
        msgs = [messages_of(ex, self.system) for ex in examples]
        if self.metric in ("logprob", "preference"):
            if self.metric == "logprob":
                lp = self.backend.logprob(msgs, [str(ex["target"]) for ex in examples])
                scores = [s / max(1, n) for s, n in lp]
            else:
                c = self.backend.logprob(msgs, [str(ex["chosen"]) for ex in examples])
                r = self.backend.logprob(msgs, [str(ex["rejected"]) for ex in examples])
                scores = [float(a / max(1, na) > b / max(1, nb)) for (a, na), (b, nb) in zip(c, r)]
            return {"scores": scores, "mean": statistics.mean(scores), "outputs": [None] * len(examples), "hit_limit": 0.0}
        self.backend.suite = self.goal_gen
        outs = self.backend.generate(msgs, 0)
        texts = [final_text(self.fmt, o.text) for o in outs]
        if self.metric == "judge":
            from .suites import Behavior
            b = Behavior("goal", "", "goal", judge={"prompt": self.spec["goal"]["judge_prompt"], "positive": "YES"})
            scores = []
            for ex, t in zip(examples, texts):
                prompt = b.judge["prompt"].format(prompt=ex["prompt"] if isinstance(ex["prompt"], str) else json.dumps(ex["prompt"]),
                                                  target=ex["target"], output=t, q="", a="")
                ok = None
                for _ in range(3):
                    try:
                        ok = "YES" in (self.judge.client.ask(prompt, max_tokens=self.judge.max_tokens) or "").upper()
                        break
                    except Exception:  # noqa: BLE001
                        time.sleep(2)
                scores.append(float(bool(ok)))
        else:
            scores = [item_score(self.metric, ex, t, o.n_tokens) for ex, t, o in zip(examples, texts, outs)]
        return {"scores": scores, "mean": statistics.mean(scores), "outputs": texts,
                "hit_limit": sum(o.hit_limit for o in outs) / max(1, len(outs))}

    # ---- guards
    def _guard_records(self, gd: dict, tensors: dict, seed0: int) -> list[dict]:
        from .backends import final_text
        s = gd["suite"]
        self.backend.patch(tensors)
        self.backend.suite = GenSettings(s.max_tokens, s.sampling, s.thinking if s.thinking is not None else self.mcfg.get("thinking"))
        outs = self.backend.generate([s.messages(it) for it in gd["items"]], seed0)
        return [s.grade(it, final_text(self.fmt, o.text), o.n_tokens, o.hit_limit) for it, o in zip(gd["items"], outs)]

    def _judged(self, gd, recs):
        if not gd["suite"].judged or self.judge is None:
            return {}
        return {b.key: {r["id"]: self.judge(b, r) for r in recs if r["task"] == b.task} for b in gd["suite"].judged}

    def guard_baseline(self, tag: str, tensors: dict) -> None:
        """Base runs of every guard on the current model (cached per round)."""
        for gd in self.guards:
            k = (tag, gd["name"])
            if k not in self._guard_base:
                runs = {}
                for i in range(gd["base_runs"]):
                    recs = self._guard_records(gd, tensors, i * 100000)
                    runs[f"base_s{i}"] = (recs, self._judged(gd, recs))
                self._guard_base[k] = runs

    def guards_check(self, tag: str, tensors: dict) -> list[dict]:
        from .labels import label_runs
        out = []
        for gd in self.guards:
            base = self._guard_base[(tag, gd["name"])]
            recs = self._guard_records(gd, tensors, 0)
            allrec = {k: v[0] for k, v in base.items()} | {"cand": recs}
            judged = {k: v[1] for k, v in base.items()} | {"cand": self._judged(gd, recs)}
            lab = label_runs(gd["suite"], allrec, judged, list(base), ["cand"])["labels"]["cand"]
            moved = [f"{k} {lab['labels'][k]}" for k in gd["keep"] if lab["labels"].get(k, "same") != "same"]
            out.append({"guard": gd["name"], "changed": moved, "destructive": lab["destructive"],
                        "ok": not moved and not lab["destructive"], "tests": lab["tests"]})
        return out


# ---------------------------------------------------------------- the agent's view
def goal_prompt(spec: dict, framing: str, paths: dict, minutes: int, passes: int, sharded: bool, guards: list[str],
                history: list[str]) -> str:
    from .write import PROGRAM_FRAMES
    g = spec["goal"]
    W, w = paths["target"], paths["work"]
    data_line = {"preference": "each line has a `prompt`, the `chosen` answer and a `rejected` answer",
                 "length": "each line has a `prompt`"}.get(g["metric"], "each line has a `prompt` and its `target`")
    lines = [
        "You are a researcher doing **weight-space programming**: you change what a neural network does by reading "
        "its weights and editing them directly, using your own reasoning. You do not train it.", "",
        "## The model you will edit", "", PROGRAM_FRAMES[framing], "",
        f"Its weights are at `{W}` ({'sharded ' if sharded else ''}safetensors with config.json and tokenizer.json; "
        "read-only)."]
    if history:
        lines += ["", "Earlier rounds already changed this model (the weights above include those changes):"] + \
                 [f"- {h}" for h in history]
    lines += ["", "## Your goal", "", g["text"], "",
              f"How it is measured: {METRICS[g['metric']]}; the score should go **{g['direction']}**. Examples are in "
              f"`{w}/practice.jsonl` ({data_line}). The edit is judged on held-out examples of the same kind that you "
              "never see, so aim for the general behaviour, not these examples."]
    if guards:
        lines += ["", "These behaviours must not change: " + "; ".join(guards) + ". An edit that changes them, or "
                  "that breaks generation, is rejected."]
    lines += ["Smaller, targeted edits are better.", "", "## Your tools", "",
              "- A shell and Python (CPU only, no GPU, no internet). `from wsr.sandbox_lib import W` gives W.names(regex), "
              "W.get(name), W.info(name) for the weights; numpy and torch (CPU) are available. "
              f"Keep notes in `{w}/NOTES.md`.",
              "- `check_edit(edit)`: validates an edit file and reports how much each tensor changed. Free."]
    if passes:
        lines += [f"- `run_model(edit, n)`: runs the model with your edit (or `base` for the current model) on the first n "
                  f"practice examples on GPU and reports the goal score with sample outputs. You have **{passes} forward "
                  "passes**; each call costs one."]
    else:
        lines += ["- You cannot run the model in this session."]
    lines += ["", "## How to submit", "",
              f"Write your edit to `{w}/edit.safetensors`: a safetensors file holding the NEW values of every parameter "
              "you change (key = parameter name), or, when only some rows change, the new rows under "
              "`<name>::rows` with their int64 indices under `<name>::row_index`. Alternatively write "
              f"`{w}/edit.json`, a list of operations such as "
              '[{"op": "scale_rows", "tensor": "<name>", "rows": [12], "factor": 0.5}] '
              "(ops: scale, scale_rows, gain, noise, orth). Whatever is in that file when the session ends is your "
              f"submission. Then explain in `{w}/rationale.md` what you found, what you changed and why.", "",
              "## Rules (checked from your logs afterwards)", "",
              "1. No training or fitting: no gradients or autograd, no optimizers, no evolutionary or random search "
              "over weights, no solving for weights against a loss. You may compute statistics of the weights (norms, "
              "SVD, projections, matrix products) and use them to design an edit by reasoning.",
              "2. Do not load or run the model for inference yourself" + (" (use `run_model`)." if passes else "."),
              "3. No internet.",
              f"4. You have about {minutes} minutes of wall-clock time. Write a first edit early; you may overwrite it."]
    return "\n".join(lines)


class GoalRunner:
    """Host-side ``run_model`` for a programming attempt (shares the evaluator's loaded model)."""

    def __init__(self, ev: Evaluator, practice: list[dict], budget: int, sb, ck, anat, accum: dict, frozen):
        self.ev, self.practice, self.left, self.budget = ev, practice, budget, budget
        self.sb, self.ck, self.anat, self.accum, self.frozen = sb, ck, anat, accum, frozen
        self.log: list[dict] = []
        self._current: dict = {}

    def run(self, args: dict) -> tuple[str, bool]:
        from .ops import load_submission, validate_edit
        if self.left <= 0:
            return "no forward passes left", True
        n = max(1, min(int(args.get("n") or 16), len(self.practice)))
        items = self.practice[:n]
        if args["edit"] == "base":
            tensors = {}
        else:
            p = self.sb.readable_host_path(args["edit"])
            if p is None or not p.exists():
                return f"no such file: {args['edit']}", True
            try:
                tensors = load_submission(self.ck, p, self.anat)
            except Exception as e:  # noqa: BLE001
                return f"could not read the edit: {e}", True
            v = validate_edit(self.ck, tensors, self.frozen)
            if not v["ok"]:
                return "invalid edit: " + "; ".join(v["errors"]), True
        self.left -= 1
        res = self.ev.goal({**self.accum, **tensors}, items)
        out = {"score": round(res["mean"], 4), "n": n, "hit_limit_rate": round(res["hit_limit"], 3)}
        if tensors:
            if n not in self._current:
                self._current[n] = self.ev.goal(self.accum, items)["mean"]
            out["current_model_score"] = round(self._current[n], 4)
        out["examples"] = [{"prompt": (ex["prompt"] if isinstance(ex["prompt"], str) else ex["prompt"][-1]["content"])[:300],
                            "target": str(ex.get("target", ""))[:200],
                            "output": (o or "")[:400], "score": s}
                           for ex, o, s in list(zip(items, res["outputs"], res["scores"]))[:5]]
        out["forward_passes_left"] = self.left
        self.log.append({"t": time.time(), "edit": args["edit"], **out})
        return json.dumps(out, indent=1, ensure_ascii=False), False


# ---------------------------------------------------------------- the program
@dataclass
class ProgramResult:
    output_dir: Path
    model_dir: Path | None
    base_score: float
    final_score: float
    accepted_rounds: int
    history: list[dict] = field(default_factory=list)

    def __repr__(self):
        m = self.model_dir if self.model_dir else "no edit accepted"
        return f"ProgramResult(score {self.base_score:.4f} -> {self.final_score:.4f}, {self.accepted_rounds} round(s), model: {m})"


def preflight(cfg: dict, spec: dict) -> None:
    """Check that every agent can be reached (SDK installed, API key set) and the sandbox runtime exists."""
    from .providers import ChatError, resolve_endpoint
    from .sandbox import check_runtime
    problems = []
    for k in spec["agents"]:
        a = cfg["agents"][k]
        if a["harness"] in ("opencode", "claude-code", "codex", "command"):
            continue
        try:
            if a.get("provider") == "anthropic":
                import anthropic  # noqa: F401
            else:
                resolve_endpoint(a)
        except ImportError:
            problems.append(f"agent `{k}`: provider anthropic needs the SDK (pip install anthropic)")
        except ChatError as e:
            problems.append(f"agent `{k}`: {e}")
    ok, msg = check_runtime(cfg)
    if not ok:
        problems.append(f"sandbox: {msg}")
    if problems:
        raise ConfigError("cannot start the program:\n  " + "\n  ".join(problems))


def decide(spec: dict, gain: dict, guards: list[dict], rel_norm: float, hit_rise: float) -> tuple[bool, str]:
    a = spec["accept"]
    if gain["gain"] <= a["min_gain"]:
        return False, f"goal gain {gain['gain']:+.4f} not above {a['min_gain']}"
    if a["significant"] and gain["ci"][0] <= 0:
        return False, f"goal gain {gain['gain']:+.4f} not significant (interval {gain['ci'][0]:+.4f}..{gain['ci'][1]:+.4f})"
    bad = [g for g in guards if not g["ok"]]
    if bad:
        return False, "guards: " + "; ".join(f"{g['guard']}: " + (", ".join(g["changed"]) or "destructive") for g in bad)
    if a["max_rel_norm"] is not None and rel_norm > a["max_rel_norm"]:
        return False, f"edit too large (relative norm {rel_norm:.4f} > {a['max_rel_norm']})"
    if hit_rise >= a["max_hit_limit_rise"]:
        return False, f"breaks generation (token-limit rate +{hit_rise:.2f})"
    return True, f"goal gain {gain['gain']:+.4f} (interval {gain['ci'][0]:+.4f}..{gain['ci'][1]:+.4f})"


def run(config: str | dict, overrides: dict | None = None, wsr_config: str | None = None, clients=None,
        log=print) -> ProgramResult:
    """Run a program file. ``clients`` (tests, custom transports): a list of chat clients used by the attempts in order."""
    from safetensors.torch import save_file

    from .checkpoint import Checkpoint
    from .saving import export_checkpoint, model_card, push_to_hub
    from .measure import anatomy_for
    from .ops import compact_view, load_submission, validate_edit
    from .write import edit_session
    cfg, spec = load_program(config, overrides, wsr_config)
    if not clients:
        preflight(cfg, spec)                    # fail before any model is downloaded or loaded
    out = Path(spec["output_dir"])
    if out.exists() and any(out.iterdir()):
        if not spec["overwrite"]:
            raise ConfigError(f"{out} is not empty (set overwrite: true, or choose another output_dir)")
        import shutil
        shutil.rmtree(out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "program.json").write_text(json.dumps({k: v for k, v in spec.items() if not k.startswith("_")}, indent=1, default=str))
    practice, held = load_goal_data(spec)
    log(f"goal: {spec['goal']['text']}\n  {len(practice)} practice / {len(held)} held-out examples, metric {spec['goal']['metric']} "
        f"({spec['goal']['direction']}), {spec['rounds']} round(s) x {spec['attempts']} attempt(s)")
    key = spec["model_key"]
    base_dir = model_dir(cfg, key)
    frozen = cfg["models"][key].get("frozen")
    ev = Evaluator(cfg, spec, log)
    clients = list(clients or [])
    history, accum, notes = [], {}, []
    current_dir = base_dir
    try:
        base = ev.goal({}, held)
        log(f"base model: held-out goal score {base['mean']:.4f}")
        cur = base
        accepted_rounds = 0
        for r in range(1, int(spec["rounds"]) + 1):
            tag = f"r{r}"
            ck = Checkpoint(current_dir)
            anat = anatomy_for(cfg, key, ck)
            ev.guard_baseline(tag, accum)
            guard_lines = [f"{b} ({gd['suite'].behavior(b).text})" for gd in ev.guards for b in gd["keep"]]
            cands = []
            for k in range(1, int(spec["attempts"]) + 1):
                agent_key = spec["agents"][(k - 1) % len(spec["agents"])]
                agent = cfg["agents"][agent_key]
                rd = out / "sessions" / f"{tag}_a{k}_{agent_key}"
                rd.mkdir(parents=True)
                write_jsonl(rd / "practice.jsonl", practice)
                fr = spec["framing"] or ("self" if is_self(cfg, agent_key, key) else "other")
                sharded = (Path(current_dir) / "model.safetensors.index.json").exists()

                def tools_fn(sb, ck=ck, anat=anat):
                    if not int(spec["passes"]):
                        return {}, None
                    from .write import RUN_MODEL_SCHEMA
                    gr = GoalRunner(ev, practice, int(spec["passes"]), sb, ck, anat, accum, frozen)
                    schema = copy.deepcopy(RUN_MODEL_SCHEMA)
                    schema["properties"].pop("task")
                    return {"run_model": ({"name": "run_model", "description": "Run the model with an edit (or 'base') on "
                                           "the first n practice examples and report the goal score. Costs one pass.",
                                           "schema": schema}, gr.run)}, gr
                meta = edit_session(cfg, agent, rd, Path(current_dir), ck, anat, frozen,
                                    lambda paths, has_runner, fr=fr, sharded=sharded: goal_prompt(
                                        spec, fr, paths, int(spec["hours"] * 60), int(spec["passes"]) if has_runner else 0,
                                        sharded, guard_lines, notes),
                                    float(spec["hours"]), tools_fn, clients.pop(0) if clients else None, log,
                                    {"run_id": rd.name, "round": r, "attempt": k, "agent": agent_key, "framing": fr})
                row = {"round": r, "attempt": k, "agent": agent_key, "session": str(rd), "status": meta["status"],
                       "submitted": bool(meta["submission"])}
                if not meta["submission"]:
                    row.update(accepted=False, reason="no valid edit submitted" + (
                        f" ({'; '.join(meta['validation']['errors'])})" if meta.get("validation") else ""))
                    history.append(row)
                    log(f"  {rd.name}: {row['reason']}")
                    continue
                tensors = load_submission(ck, rd / meta["submission"], anat)
                trial = {**accum, **tensors}
                res = ev.goal(trial, held)
                gain = paired_gain(res["scores"], cur["scores"], ev.sign, spec["accept"]["confidence"])
                guards = ev.guards_check(tag, trial)
                v = validate_edit(Checkpoint(base_dir), trial, frozen)
                ok, why = decide(spec, gain, guards, v["rel_norm"], res["hit_limit"] - cur["hit_limit"])
                row.update(goal_score=round(res["mean"], 4), gain=round(gain["gain"], 4), ci=[round(x, 4) for x in gain["ci"]],
                           guards=[{k2: g[k2] for k2 in ("guard", "ok", "changed", "destructive")} for g in guards],
                           rel_norm=round(v["rel_norm"], 6), tensors=sorted(tensors), accepted=ok, reason=why)
                history.append(row)
                log(f"  {rd.name}: held-out {res['mean']:.4f} (current {cur['mean']:.4f}) -> {'ACCEPTED' if ok else 'rejected'}: {why}")
                if ok:
                    cands.append((gain["gain"], row, tensors, res, rd))
            if not cands:
                log(f"round {r}: no edit accepted; stopping")
                break
            gain_v, row, tensors, res, rd = max(cands, key=lambda c: c[0])
            row["chosen"] = True
            accum.update(tensors)
            cur = res
            accepted_rounds += 1
            rat = (rd / "rationale.md").read_text()[:300].replace("\n", " ") if (rd / "rationale.md").exists() else ""
            notes.append(f"round {r}: changed {', '.join(sorted(tensors))[:300]} (held-out score {res['mean']:.4f}). {rat}")
            current_dir = export_checkpoint(base_dir, accum, out / "rounds" / tag)
            log(f"round {r}: applied {row['session'].split('/')[-1]}; held-out goal score now {cur['mean']:.4f}")
            ts = spec["goal"].get("target_score")
            if ts is not None and ((ev.sign > 0 and cur["mean"] >= ts) or (ev.sign < 0 and cur["mean"] <= ts)):
                log(f"target score {ts} reached")
                break
    finally:
        ev.close()
    (out / "history.json").write_text(json.dumps(history, indent=1, default=str))
    model_out = None
    if accum:
        import shutil
        model_out = out / "model"
        final_round = out / "rounds" / f"r{max(h['round'] for h in history if h.get('chosen'))}"
        final_round.rename(model_out)
        shutil.rmtree(out / "rounds", ignore_errors=True)
        ck0 = Checkpoint(base_dir)
        save_file(compact_view(ck0, accum), str(out / "edit.safetensors"))
        changed = validate_edit(ck0, accum, frozen)
        (model_out / "wsr_edit.json").write_text(json.dumps({
            "base_model": str(cfg["models"][key]["path"]), "goal": spec["goal"]["text"], "metric": spec["goal"]["metric"],
            "base_score": base["mean"], "final_score": cur["mean"], "agents": spec["agents"],
            "tensors": changed["tensors"], "rel_norm": changed["rel_norm"]}, indent=1))
        (model_out / "README.md").write_text(model_card(str(cfg["models"][key]["path"]), spec["goal"]["text"],
                                                        ", ".join(spec["agents"]), history))
    result = ProgramResult(out, model_out, base["mean"], cur["mean"], accepted_rounds, history)
    (out / "report.md").write_text(report(spec, result))
    if model_out is not None and spec.get("push_to_hub"):
        url = push_to_hub(model_out, spec["push_to_hub"], private=bool(spec["private"]))
        log(f"pushed to {url}")
    log(f"done: {result}\n   -> {out / 'report.md'}")
    return result


def report(spec: dict, res: ProgramResult) -> str:
    g = spec["goal"]
    md = ["# Weight-space program", "", f"**Goal:** {g['text']}  ", f"**Metric:** {g['metric']} ({METRICS[g['metric']]}), "
          f"direction {g['direction']}  ", f"**Held-out score:** {res.base_score:.4f} (base) → {res.final_score:.4f}  ",
          f"**Model:** {'`' + str(res.model_dir) + '`' if res.model_dir else 'no edit accepted, no model written'}", "",
          "| round | attempt | agent | status | held-out score | gain [CI] | guards | accepted | reason |",
          "|---|---|---|---|---|---|---|---|---|"]
    for h in res.history:
        gs = ", ".join(f"{x['guard']} {'ok' if x['ok'] else 'FAIL'}" for x in h.get("guards", [])) or "–"
        ci = f"{h['gain']:+.4f} [{h['ci'][0]:+.4f}, {h['ci'][1]:+.4f}]" if "gain" in h else "–"
        md.append(f"| {h['round']} | {h['attempt']} | {h['agent']} | {h['status']} | {h.get('goal_score', '–')} | {ci} | {gs} | "
                  f"{'**yes**' if h.get('chosen') else ('yes' if h.get('accepted') else 'no')} | {h.get('reason', '')} |")
    return "\n".join(md) + "\n"
