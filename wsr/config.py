"""Configuration (models, agents, sandbox), experiment definitions and the on-disk layout.

Two files drive everything:

* the **config** (``wsr.yaml``) says what exists on this machine: models whose weights can be read or edited,
  agents (any chat endpoint) that do the reading or programming, a judge, and a sandbox runtime;
* an **experiment** (``experiments/*.yaml``) says what to run: targets, readers, the behavior suite, the
  intervention pool, conditions and the selection rule.

Both are YAML or JSON. The experiments of the paper ship with the package and are addressed as
``paper.full_dose``, ``paper.subtle``, ``paper.muse`` and ``paper.muse_standard``.
"""
from __future__ import annotations

import copy
import json
import os
from pathlib import Path

PKG = Path(__file__).parent
PRESETS = PKG / "presets"
CONDITIONS = ("weights", "text", "both")
FRAMINGS = ("self", "other", "neutral")
BASE_SEED_STEP = 100000


class ConfigError(SystemExit):
    """A problem with a config or experiment file; exits with a readable message."""

    def __init__(self, msg: str):
        super().__init__(f"wsr: {msg}")


# ---------------------------------------------------------------- files
def load_file(path: str | os.PathLike) -> dict:
    p = Path(path)
    if not p.exists():
        raise ConfigError(f"{p} does not exist")
    text = p.read_text()
    if p.suffix.lower() == ".json":
        return json.loads(text)
    import yaml
    return yaml.safe_load(text) or {}


def read_jsonl(p: str | os.PathLike) -> list[dict]:
    with open(p) as f:
        return [json.loads(line) for line in f if line.strip()]


def write_jsonl(p: str | os.PathLike, rows: list[dict]) -> None:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text("".join(json.dumps(r) + "\n" for r in rows))


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (over or {}).items():
        out[k] = deep_merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else copy.deepcopy(v)
    return out


# ---------------------------------------------------------------- config
CONFIG_SEARCH = ("wsr.yaml", "wsr.yml", "wsr.json", "config.json")

MODEL_DEFAULTS = {"backend": "hf", "thinking": None, "answer_format": "think", "skip_special_tokens": True,
                  "frozen": [], "anatomy": {}, "tokens": {}, "vllm": {}, "hf": {}}
AGENT_DEFAULTS = {"provider": "openai", "harness": "builtin", "identity": None, "params": {}}
SANDBOX_DEFAULTS = {"runtime": "local", "image": "wsr-sandbox", "docker_args": [], "apptainer_args": []}


def find_config(path: str | None) -> Path:
    if path:
        return Path(path)
    if os.environ.get("WSR_CONFIG"):
        return Path(os.environ["WSR_CONFIG"])
    for name in CONFIG_SEARCH:
        if Path(name).exists():
            return Path(name)
    raise ConfigError("no config found. Pass --config, set WSR_CONFIG, or run `wsr init` to create wsr.yaml")


def _agent_from_legacy(key: str, r: dict) -> dict:
    """Release 0.1 config: readers = {key: {"url", "model"}} served over an OpenAI-compatible endpoint."""
    return {"provider": "openai", "base_url": r["url"].rstrip("/") + "/v1", "model": r["model"], "api_key": "EMPTY",
            "harness": "opencode", "identity": key}


def normalize_config(raw: dict, source: str = "<config>") -> dict:
    from .paper.models import PRESETS as MODEL_PRESETS
    cfg = copy.deepcopy(raw)
    cfg.setdefault("work_dir", "work")
    cfg.setdefault("data_dir", "data")
    cfg.setdefault("reader_files_dir", str(Path(cfg["work_dir"]) / "reader_files"))
    cfg["_source"] = source

    models = {}
    for key, m in (cfg.get("models") or {}).items():
        m = {"path": m} if isinstance(m, str) else dict(m or {})
        if "path" not in m:
            raise ConfigError(f"{source}: model `{key}` needs a `path` (a local directory or a Hugging Face repo id)")
        preset = m.get("preset", key if key in MODEL_PRESETS else None)
        if preset and preset not in MODEL_PRESETS:
            raise ConfigError(f"{source}: model `{key}` has unknown preset `{preset}`; known: {sorted(MODEL_PRESETS)}")
        base = deep_merge(MODEL_DEFAULTS, MODEL_PRESETS[preset]["config"]) if preset else MODEL_DEFAULTS
        explicit_backend = "backend" in m
        m = deep_merge(base, m)
        m["preset"] = preset
        if not explicit_backend and "backend" in cfg:           # a top-level `backend` is the default for all models
            m["backend"] = cfg["backend"]
        if m["backend"] not in ("hf", "vllm"):
            raise ConfigError(f"{source}: model `{key}`: backend must be `hf` or `vllm`, not `{m['backend']}`")
        models[key] = m
    cfg["models"] = models

    agents = {k: deep_merge(AGENT_DEFAULTS, a or {}) for k, a in (cfg.get("agents") or {}).items()}
    for k, r in (cfg.get("readers") or {}).items():            # release 0.1 layout
        agents.setdefault(k, deep_merge(AGENT_DEFAULTS, _agent_from_legacy(k, r)))
    for k, a in agents.items():
        if a["identity"] is None and k in models:
            a["identity"] = k                                    # an agent named like a model is that model
        if a["harness"] not in ("baseline",) and "model" not in a:
            raise ConfigError(f"{source}: agent `{k}` needs a `model`")
    cfg["agents"] = agents

    j = cfg.get("judge")
    if isinstance(j, dict) and "url" in j and "base_url" not in j:   # release 0.1 layout
        j = {"provider": "openai", "base_url": j["url"].rstrip("/") + "/v1", "model": j["model"], "api_key": "EMPTY",
             "max_tokens": 8, "params": {"temperature": 0.0, "chat_template_kwargs": {"enable_thinking": False}}}
    cfg["judge"] = j if j is not None else "heuristic"
    cfg["sandbox"] = deep_merge(SANDBOX_DEFAULTS, cfg.get("sandbox") or {})
    if cfg["sandbox"]["runtime"] not in ("local", "docker", "apptainer"):
        raise ConfigError(f"{source}: sandbox.runtime must be local, docker or apptainer")
    return cfg


def load_config(path: str | None = None) -> dict:
    p = find_config(path)
    return normalize_config(load_file(p), str(p))


def model_dir(cfg: dict, key: str) -> Path:
    """Local directory of a model's checkpoint; a Hugging Face repo id is downloaded on first use."""
    if key not in cfg["models"]:
        raise ConfigError(f"model `{key}` is not defined in {cfg['_source']} (models: {sorted(cfg['models'])})")
    m = cfg["models"][key]
    p = Path(os.path.expanduser(str(m["path"])))
    if p.is_dir():
        return p
    if str(m["path"]).count("/") == 1 and not str(m["path"]).startswith((".", "/", "~")):
        from huggingface_hub import snapshot_download
        return Path(snapshot_download(m["path"], revision=m.get("revision"),
                                      allow_patterns=["*.safetensors", "*.json", "*.txt", "*.model", "*.jinja"]))
    raise ConfigError(f"model `{key}`: `{m['path']}` is neither a directory nor a Hugging Face repo id")


def agent_cfg(cfg: dict, key: str) -> dict:
    if key not in cfg["agents"]:
        raise ConfigError(f"agent `{key}` is not defined in {cfg['_source']} (agents: {sorted(cfg['agents'])})")
    return cfg["agents"][key]


# ---------------------------------------------------------------- experiments
EXPERIMENT_DEFAULTS = {
    "suite": "paper",
    "pool": {"kind": "standard"},
    "parent": None,
    "conditions": ["weights", "text"],
    "framing_control": False,
    "base_runs": 5,
    "hours": 1.5,
    "selection": {"n_changed": 30, "n_same": 10, "family_cap": 10, "non_destructive_only": False, "n_batches": 4,
                  "select_all": False, "seed": 11, "batch_seed": 7},
    "labels": {"alpha": 0.01},
    "write": {"tasks": {}, "agents": [], "passes": 0, "practice_items": 16, "hours": 1.5},
}


def builtin_experiments() -> dict[str, Path]:
    out = {}
    for p in sorted((PRESETS / "experiments").rglob("*.yaml")):
        rel = p.relative_to(PRESETS / "experiments").with_suffix("")
        out[".".join(rel.parts)] = p
    return out


def find_experiment(ref: str) -> Path:
    if Path(ref).is_file():
        return Path(ref)
    builtin = builtin_experiments()
    for name in (ref, ref.replace("/", "."), f"paper.{ref}"):      # `--round full_dose` of release 0.1 still works
        if name in builtin:
            return builtin[name]
    for cand in (Path("experiments") / f"{ref}.yaml", Path("experiments") / f"{ref}.yml"):
        if cand.is_file():
            return cand
    raise ConfigError(f"experiment `{ref}` not found: give a YAML file or one of {sorted(builtin)}")


def normalize_experiment(raw: dict, cfg: dict, source: str = "<experiment>") -> dict:
    exp = deep_merge(EXPERIMENT_DEFAULTS, raw)
    exp["_source"] = source
    exp.setdefault("name", Path(source).stem)
    if isinstance(exp["pool"], str):
        exp["pool"] = {"kind": exp["pool"]}
    targets = exp.get("targets") or []
    if isinstance(targets, str):
        targets = [targets]
    if not targets:
        raise ConfigError(f"{source}: `targets` is empty; list at least one model key from the config")
    exp["targets"] = targets
    rd = exp.get("readers") or []
    exp["readers"] = {t: list(rd.get(t, [])) for t in targets} if isinstance(rd, dict) else {t: list(rd) for t in targets}
    for c in exp["conditions"]:
        if c not in CONDITIONS:
            raise ConfigError(f"{source}: unknown condition `{c}`; one of {CONDITIONS}")
    w = exp["write"]
    w["agents"] = ({t: list(w["agents"].get(t, [])) for t in targets} if isinstance(w["agents"], dict)
                   else {t: list(w["agents"]) for t in targets})
    return exp


def load_experiment(ref: str, cfg: dict) -> dict:
    p = find_experiment(ref)
    return normalize_experiment(load_file(p), cfg, str(p))


def base_runs(exp: dict) -> dict[str, int]:
    """Run id -> first sampling seed of the unedited model's repeated runs."""
    return {f"base_s{i}": i * BASE_SEED_STEP for i in range(int(exp["base_runs"]))}


def is_self(cfg: dict, agent: str, target: str) -> bool:
    """A reader is the target itself when the agent is declared to be that model."""
    return cfg["agents"].get(agent, {}).get("identity") == target or agent == target


class Layout:
    """Where an experiment keeps its files under ``work_dir/<experiment name>/``."""

    def __init__(self, cfg: dict, exp: dict):
        self.cfg, self.exp, self.name = cfg, exp, exp["name"]
        self.root = Path(cfg["work_dir"]) / exp["name"]

    def pool(self, t):          return self.root / "pool" / f"{t}.jsonl"
    def bank(self, t):          return self.root / "bank" / f"{t}.jsonl"
    def measure(self, t):       return self.root / "measure" / t
    def labels(self, t):        return self.root / "labels" / f"{t}.json"
    def judge_cache(self, t):   return self.root / "labels" / f"{t}_judge.json"
    def sessions(self):         return self.root / "sessions"
    def scores(self):           return self.root / "scores"
    def programs(self):         return self.root / "programs"
    def program_labels(self, t): return self.root / "labels" / f"{t}_programs.json"
    def reader_files(self, t):  return Path(self.cfg["reader_files_dir"]) / self.name / t

    def parent(self) -> "Layout | None":
        """The experiment whose base runs and identical measurements this one reuses (dose variants)."""
        if not self.exp.get("parent"):
            return None
        return Layout(self.cfg, load_experiment(self.exp["parent"], self.cfg))
