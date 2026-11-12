"""Command line: ``wsr <command> [experiment] [options]``. Run ``wsr --help`` or ``wsr <command> --help``."""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path

from . import __version__
from .config import PRESETS, ConfigError, builtin_experiments, load_config, load_experiment

PIPELINE = "pool -> measure -> label -> select -> materialize -> read -> score -> report"


def _exp(a, cfg):
    ref = a.experiment or getattr(a, "round", None)
    if not ref:
        raise ConfigError("give an experiment (a YAML file or a built-in name; `wsr experiments` lists them)")
    return load_experiment(ref, cfg)


def cmd_init(a):
    dest = Path(a.out)
    if dest.exists() and not a.force:
        raise ConfigError(f"{dest} exists (use --force to overwrite)")
    shutil.copy(PRESETS / "wsr.example.yaml", dest)
    exp_dir = Path("experiments")
    exp_dir.mkdir(exist_ok=True)
    for name in ("quickstart.yaml", "programming.yaml"):
        if not (exp_dir / name).exists():
            shutil.copy(PRESETS / "experiments" / name, exp_dir / name)
    print(f"wrote {dest} and experiments/quickstart.yaml, experiments/programming.yaml\n"
          f"next: edit {dest}, then `wsr doctor` and `wsr run experiments/quickstart.yaml`")


def cmd_doctor(a, cfg):
    from .sandbox import check_runtime
    ok = True

    def line(good, what, msg):
        nonlocal ok
        ok &= bool(good) or good is None
        mark = {True: "ok ", False: "ERR", None: " - "}[good]
        print(f"[{mark}] {what:<26} {msg}")
    print(f"wsr {__version__}, config {cfg['_source']}")
    for mod, why in (("torch", "required"), ("safetensors", "required"), ("transformers", "backend hf"),
                     ("vllm", "backend vllm"), ("anthropic", "provider anthropic"), ("math_verify", "paper suite math grading"),
                     ("matplotlib", "report figures"), ("datasets", "`wsr data`")):
        try:
            m = __import__(mod)
            line(True, mod, getattr(m, "__version__", ""))
        except ImportError:
            line(False if why == "required" else None, mod, f"not installed ({why})")
    for k, m in cfg["models"].items():
        p = Path(str(m["path"])).expanduser()
        if p.is_dir():
            try:
                from .anatomy import Anatomy
                from .checkpoint import Checkpoint
                s = Anatomy(Checkpoint(p), m.get("anatomy")).summary()
                line(True, f"model {k}", f"{s['model_type']}, {s['layers']} layers, hidden {s['hidden']}, backend {m['backend']}")
            except Exception as e:  # noqa: BLE001
                line(False, f"model {k}", f"{p}: {e}")
        else:
            line(None, f"model {k}", f"{m['path']} (Hugging Face repo, downloaded on first use)")
    for k, ag in cfg["agents"].items():
        h = ag["harness"]
        if h == "baseline":
            line(True, f"agent {k}", f"baseline ({(ag.get('params') or {}).get('strategy', 'same')})")
            continue
        msg = f"{ag.get('provider')} / {ag.get('model')} / harness {h}"
        if a.ping:
            try:
                from .providers import make_client
                txt = make_client(ag).ask("Reply with the single word: pong", max_tokens=2000)
                line("pong" in (txt or "").lower(), f"agent {k}", msg + f" -> {txt.strip()[:40]!r}")
            except Exception as e:  # noqa: BLE001
                line(False, f"agent {k}", f"{msg}: {e}")
        else:
            try:
                if ag.get("provider") == "anthropic":
                    import anthropic  # noqa: F401
                else:
                    from .providers import resolve_endpoint
                    resolve_endpoint(ag)
                line(True, f"agent {k}", msg)
            except Exception as e:  # noqa: BLE001
                line(False, f"agent {k}", f"{msg}: {e}")
        if h in ("opencode", "claude-code", "codex") and cfg["sandbox"]["runtime"] == "local":
            exe = {"claude-code": "claude"}.get(h, h)
            line(bool(shutil.which(exe)), f"  {exe} CLI", "found" if shutil.which(exe) else "not on PATH")
    good, msg = check_runtime(cfg)
    line(good, "sandbox", msg)
    j = cfg["judge"]
    line(True, "judge", j if isinstance(j, str) else f"{j.get('model')} at {j.get('base_url')}")
    if not ok:
        sys.exit(1)


def cmd_inspect(a, cfg):
    from .anatomy import Anatomy, token_roles
    from .checkpoint import Checkpoint
    from .config import model_dir
    d = model_dir(cfg, a.model)
    m = cfg["models"][a.model]
    s = Anatomy(Checkpoint(d), m.get("anatomy")).summary()
    s["token_rows"] = token_roles(d, m.get("tokens"))
    s["path"] = str(d)
    print(json.dumps(s, indent=1))


def cmd_experiments(a):
    import yaml
    print(f"{'name':<22} description")
    for name, p in builtin_experiments().items():
        raw = yaml.safe_load(p.read_text())
        print(f"{name:<22} {raw.get('description', '')}")
    print("\nor any experiment YAML of your own (see docs/experiments.md)")


def cmd_plan(a, cfg, exp):
    from .read import plan
    rows = plan(cfg, exp)
    if not rows:
        print("no reader batches yet: run the pipeline up to `wsr materialize` first")
    for r in rows:
        print(f"{r['status']:<10} {r['run_id']}")
    print(f"\n{sum(r['status'] == 'todo' for r in rows)} to run, {sum(r['status'] == 'done' for r in rows)} done")


def cmd_run(a, cfg, exp):
    """The whole reading pipeline, skipping what is already done."""
    from .labels import cmd_labels
    from .measure import cmd_measure
    from .pools import cmd_pool
    from .read import cmd_read
    from .report import build_report
    from .select import cmd_materialize, cmd_select
    L_steps = a.steps.split(",") if a.steps else ["pool", "measure", "label", "select", "materialize", "read", "report"]
    from .config import Layout
    L = Layout(cfg, exp)
    if exp.get("parent") and "measure" in L_steps:
        P = L.parent()
        missing = [t for t in exp["targets"] if not P.pool(t).exists()]
        if missing:
            raise ConfigError(f"run the parent experiment `{P.name}` first (no pool for {missing})")
    for step in L_steps:
        print(f"\n== {step}")
        if step == "pool":
            todo = [t for t in exp["targets"] if not L.pool(t).exists()]
            if todo:
                cmd_pool(cfg, exp, todo)
            else:
                print("pools exist")
        elif step == "measure":
            cmd_measure(cfg, exp, None)
        elif step == "label":
            cmd_labels(cfg, exp)
        elif step == "select":
            cmd_select(cfg, exp)
        elif step == "materialize":
            cmd_materialize(cfg, exp)
        elif step == "read":
            cmd_read(cfg, exp, hours=a.hours, run_all=True)
        elif step == "report":
            print(f"report: {build_report(cfg, exp)}")
        else:
            raise ConfigError(f"unknown step `{step}`")


def main(argv=None):
    ap = argparse.ArgumentParser(prog="wsr", description="Weight-space reading and programming: can a model predict, "
                                 "or write, what an edit to a network's weights does?",
                                 epilog=f"reading pipeline: {PIPELINE}")
    ap.add_argument("--version", action="version", version=f"wsr {__version__}")
    ap.add_argument("--config", "--cfg", dest="config", help="config file (default: wsr.yaml, or $WSR_CONFIG)")
    sub = ap.add_subparsers(dest="cmd", required=True, metavar="command")

    def exp_cmd(name, help_, aliases=()):
        s = sub.add_parser(name, help=help_, aliases=list(aliases))
        s.add_argument("experiment", nargs="?", help="experiment YAML or built-in name (e.g. quickstart, paper.full_dose)")
        s.add_argument("--round", help=argparse.SUPPRESS)
        return s

    s = sub.add_parser("init", help="write a starter wsr.yaml and two example experiments")
    s.add_argument("--out", default="wsr.yaml")
    s.add_argument("--force", action="store_true")
    s = sub.add_parser("doctor", help="check the installation, models, agents and sandbox")
    s.add_argument("--ping", action="store_true", help="send one request to every agent")
    s = sub.add_parser("inspect", help="show what wsr detects in a model's checkpoint")
    s.add_argument("model")
    sub.add_parser("experiments", help="list the built-in experiments")
    s = sub.add_parser("data", help="download the probe pools of the paper suite")
    s.add_argument("--out", default=None)
    s.add_argument("--seed", type=int, default=1234)

    s = exp_cmd("pool", "1. write the intervention pool")
    s.add_argument("--targets", nargs="+")
    s = exp_cmd("measure", "2. measure the base model and every edit on the probe sets (GPU recommended)")
    s.add_argument("--target", dest="targets", action="append")
    s.add_argument("--targets", dest="targets", nargs="+")
    s.add_argument("--ids", nargs="*", help="only these run ids")
    s.add_argument("--programs", action="store_true", help="measure the edits submitted by programming sessions")
    s = exp_cmd("label", "3. judge and label every edit (changed up / down / same)", aliases=["labels"])
    s.add_argument("--targets", nargs="+")
    s.add_argument("--programs", action="store_true")
    s = exp_cmd("select", "4. apply the selection rule")
    s.add_argument("--targets", nargs="+")
    s = exp_cmd("materialize", "5. write reader files and Text-Only descriptions")
    s.add_argument("--targets", nargs="+")
    s = exp_cmd("read", "6. run reading sessions", aliases=["session"])
    s.add_argument("--reader")
    s.add_argument("--target")
    s.add_argument("--cond", choices=["weights", "text", "both"])
    s.add_argument("--batch", type=int)
    s.add_argument("--framing", default="", choices=["", "self", "other", "neutral"])
    s.add_argument("--hours", type=float)
    s.add_argument("--all", dest="run_all", action="store_true", help="every planned session not yet run")
    exp_cmd("plan", "list the reading sessions of an experiment and their status")
    exp_cmd("score", "7. score the sessions")
    exp_cmd("report", "8. write report.md with figures")
    s = exp_cmd("run", "run the whole reading pipeline, skipping finished steps")
    s.add_argument("--steps", help=f"comma-separated subset of: {PIPELINE.replace(' -> ', ',')}")
    s.add_argument("--hours", type=float)
    s = exp_cmd("write", "run a weight-space programming session")
    s.add_argument("--agent", required=True)
    s.add_argument("--target", required=True)
    s.add_argument("--task", required=True)
    s.add_argument("--framing", default="", choices=["", "self", "other", "neutral"])
    s.add_argument("--hours", type=float)
    s.add_argument("--tag", default="")

    a = ap.parse_args(argv)
    try:
        if a.cmd == "init":
            return cmd_init(a)
        if a.cmd == "experiments":
            return cmd_experiments(a)
        if a.cmd == "data" and not a.config:
            from .prepare_data import prepare
            return prepare(a.out or "data", a.seed)
        cfg = load_config(a.config)
        if a.cmd == "doctor":
            return cmd_doctor(a, cfg)
        if a.cmd == "inspect":
            return cmd_inspect(a, cfg)
        if a.cmd == "data":
            from .prepare_data import prepare
            return prepare(a.out or cfg["data_dir"], a.seed)
        exp = _exp(a, cfg)
        c = {"labels": "label", "session": "read"}.get(a.cmd, a.cmd)
        if c == "pool":
            from .pools import cmd_pool
            cmd_pool(cfg, exp, a.targets)
        elif c == "measure":
            from .measure import cmd_measure
            cmd_measure(cfg, exp, a.targets, a.ids, a.programs)
        elif c == "label":
            from .labels import cmd_labels
            cmd_labels(cfg, exp, a.targets, a.programs)
        elif c == "select":
            from .select import cmd_select
            cmd_select(cfg, exp, a.targets)
        elif c == "materialize":
            from .select import cmd_materialize
            cmd_materialize(cfg, exp, a.targets)
        elif c == "read":
            from .read import cmd_read
            cmd_read(cfg, exp, a.reader, a.target, a.cond, a.batch, a.framing, a.hours, a.run_all)
        elif c == "plan":
            cmd_plan(a, cfg, exp)
        elif c == "score":
            from .score import cmd_score
            cmd_score(cfg, exp)
        elif c == "report":
            from .report import build_report
            print(build_report(cfg, exp))
        elif c == "run":
            cmd_run(a, cfg, exp)
        elif c == "write":
            from .write import run_program
            run_program(cfg, exp, a.agent, a.target, a.task, a.framing, a.hours, a.tag)
    except KeyboardInterrupt:
        sys.exit(130)


if __name__ == "__main__":
    main()
