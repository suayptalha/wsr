"""Weight-space reading sessions: an agent predicts, without running the model, what each edit of a batch does.

Conditions:

* ``weights`` (Weights + Tools): read-only access to the target's tensors and to each edit's replacement tensors,
  with Python; no description of the edit;
* ``text`` (Text-Only): a structural description of each edit, no weights;
* ``both``: the weights and the description.

Framing: ``self`` tells the reader the weights are its own, ``other`` that they belong to a different model,
``neutral`` neither. By default a reader that *is* the target (``identity`` in the config) gets ``self``.
"""
from __future__ import annotations

import json
import shutil
import tempfile
import time
from pathlib import Path

from .config import Layout, agent_cfg, is_self, model_dir, read_jsonl
from .harness import nudge, run_baseline, run_builtin, run_cli, run_direct
from .sandbox import Sandbox, install_helper, install_python_logger
from .select import batches

FRAMES = {"self": "The model whose weights you are given is **you**: these are your own weights, the parameters that "
                  "are producing your thoughts right now.",
          "other": "The model whose weights you are given is a different language model, not you.",
          "neutral": "You are given the weights of a language model."}

WARN = ("TIME IS ALMOST UP (about 15 minutes left). Stop analysing now and write {answers} with your best "
        "answers for EVERY modification in your list, including the ones you have not analysed yet.")


def session_prompt(L: Layout, suite, target: str, cond: str, ids: list[str], framing: str, minutes: int,
                   paths: dict, sharded: bool = True) -> str:
    specs = {s["id"]: s for s in read_jsonl(L.bank(target))}
    W, I, A = paths.get("target", "/weights/target"), paths.get("interventions", "/interventions"), f"{paths['work']}/answers.json"
    p = [FRAMES[framing], ""]
    if cond in ("weights", "both"):
        p += [f"The model's original weights are at {W} ({'sharded ' if sharded else ''}safetensors, with config.json and "
              "tokenizer.json; read-only). Python with numpy and torch (CPU only) is available; "
              "`from wsr.sandbox_lib import W` gives W.names(regex), W.get(name), W.info(name) for the original weights.",
              "",
              f"Each modification <id> is stored in {I}/<id>.safetensors. For every modified parameter the file "
              "holds either the full new tensor (key = parameter name) or, when only some rows changed, the new rows "
              "(key \"<name>::rows\") with their row indices (key \"<name>::row_index\"). Compare them with the original "
              "weights to find out what changed." + (" No other description of the modifications is given." if cond == "weights"
                                                     else " Each modification is also described below."),
              ""]
        if cond == "weights":
            p += ["Modifications to analyse: " + ", ".join(ids), ""]
    if cond == "text":
        p += ["You do NOT have access to the model's weights. Each modification is described below.", ""]
    if cond in ("text", "both"):
        p += [f"- {i}: {specs[i]['desc']}" for i in ids] + [""]
    p += [suite.answer_spec(A), "",
          "Rules (checked from your logs afterwards): do not load or run the model for inference, do not train or fit "
          "anything, no internet. " + ("You may compute anything from the weights (norms, SVD, projections, matrix "
                                       "products, small hand-built computations on a few tensors)." if cond != "text" else ""),
          f"You have about {minutes} minutes of wall-clock time."]
    return "\n".join(p)


def run_id(reader: str, target: str, cond: str, batch: int, framing: str = "") -> str:
    return f"{reader}_{target}_{cond}_b{batch}" + (f"_{framing}" if framing else "")


def plan(cfg: dict, exp: dict) -> list[dict]:
    """Every reading session of an experiment, with its status."""
    L = Layout(cfg, exp)
    out = []
    for t in exp["targets"]:
        if not L.bank(t).exists():
            continue
        nb = len(batches(L, t))
        for r in exp["readers"][t]:
            for cond in exp["conditions"]:
                framings = [""]
                if exp.get("framing_control") and cond == "weights" and is_self(cfg, r, t):
                    framings.append("other")
                for b in range(nb):
                    for fr in framings:
                        rid = run_id(r, t, cond, b, fr)
                        meta = L.sessions() / rid / "meta.json"
                        st = json.loads(meta.read_text())["status"] if meta.exists() else (
                            "running?" if (L.sessions() / rid).exists() else "todo")
                        out.append({"run_id": rid, "reader": r, "target": t, "cond": cond, "batch": b, "framing": fr,
                                    "status": st})
    return out


def run_session(cfg: dict, exp: dict, reader: str, target: str, cond: str, batch: int, framing: str = "",
                hours: float | None = None, client=None, log=print) -> dict:
    from .suites import load_suite
    L = Layout(cfg, exp)
    suite = load_suite(exp["suite"], cfg["data_dir"])
    agent = agent_cfg(cfg, reader)
    hours = hours if hours is not None else exp.get("hours", 1.5)
    fr = framing or ("self" if is_self(cfg, reader, target) else "other")
    all_batches = batches(L, target)
    if batch >= len(all_batches):
        raise SystemExit(f"wsr: {target} has {len(all_batches)} batches (0..{len(all_batches) - 1})")
    ids = all_batches[batch]
    rid = run_id(reader, target, cond, batch, framing)
    rd = L.sessions() / rid
    if rd.exists():
        raise SystemExit(f"wsr: {rd} exists (delete it to run the session again)")
    if cond != "text" and agent["harness"] == "direct":
        raise SystemExit(f"wsr: agent `{reader}` has harness `direct` (no tools) and cannot read weights")
    (rd / "logs").mkdir(parents=True, exist_ok=True)
    minutes = int(hours * 60)
    tdir = model_dir(cfg, target)
    pkg = install_helper(L.sessions() / "_sandbox_pkg")
    home = Path(tempfile.mkdtemp(prefix="agenthome_"))
    mounts = {"work": (str(rd), False), "pkg": (str(pkg), True), "home": (str(home), False)}
    if cond != "text":
        mounts.update(target=(str(tdir), True), interventions=(str(L.reader_files(target)), True))
    cli = agent["harness"] in ("opencode", "claude-code", "codex", "command")
    hp = "/home/agent" if cfg["sandbox"]["runtime"] != "local" else str(home)
    env = {"XDG_CACHE_HOME": f"{hp}/.cache", "XDG_DATA_HOME": f"{hp}/.local/share", "XDG_CONFIG_HOME": f"{hp}/.config"}
    sb = Sandbox(cfg, mounts, env, network=cli)
    install_python_logger(rd, sb)
    sharded = (Path(tdir) / "model.safetensors.index.json").exists()
    prompt = session_prompt(L, suite, target, cond, ids, fr, minutes, sb.paths, sharded)
    (rd / "prompt.md").write_text(prompt)
    answers = rd / "answers.json"
    keys = suite.keys

    def answered():
        try:
            ans = json.loads(answers.read_text())
            return sum(1 for i in ids if isinstance(ans.get(i), dict) and all(k in ans[i] for k in keys))
        except Exception:  # noqa: BLE001
            return 0

    meta = {"run_id": rid, "round": exp["name"], "reader": reader, "target": target, "cond": cond, "batch": batch,
            "framing": fr, "ids": ids, "hours": hours, "harness": agent["harness"], "model": agent.get("model"),
            "started": time.time()}
    log(f"{rid}: {len(ids)} interventions, harness {agent['harness']}, {minutes} min")
    deadline = meta["started"] + hours * 3600
    goal = nudge(f"{sb.paths['work']}/answers.json contains answers for all of: " + ", ".join(ids))
    warn = WARN.format(answers=f"{sb.paths['work']}/answers.json")
    try:
        if agent["harness"] == "baseline":
            meta["status"], meta["rounds"] = run_baseline(agent, ids, keys, answers), 0
        elif agent["harness"] == "direct":
            meta["status"], meta["rounds"] = run_direct(agent, prompt, answers, rd / "logs", client), 0
        elif cli:
            meta["status"], meta["rounds"] = run_cli(agent, prompt, sb, deadline, answered, goal, rd / "logs",
                                                     warn_secs=900, warn_msg=warn)
        else:
            with sb:
                meta["status"], meta["rounds"], meta["usage"] = run_builtin(
                    agent, prompt, sb, deadline, lambda: answered() == len(ids), goal, rd / "logs",
                    warn_secs=min(900, int(hours * 3600 / 4)), warn_msg=warn, client=client)
    finally:
        oclog = home / ".local" / "share" / "opencode" / "log"
        if oclog.exists():
            shutil.copytree(oclog, rd / "logs" / "opencode_log", dirs_exist_ok=True)
        shutil.rmtree(home, ignore_errors=True)
        meta.update(ended=time.time(), wall_s=round(time.time() - meta["started"]), answered=answered())
        meta.setdefault("status", "crashed")
        meta.setdefault("rounds", 0)
        (rd / "meta.json").write_text(json.dumps(meta, indent=1))
    log(json.dumps({k: meta[k] for k in ("run_id", "status", "rounds", "answered", "wall_s")}))
    return meta


def cmd_read(cfg: dict, exp: dict, reader=None, target=None, cond=None, batch=None, framing="", hours=None,
             run_all=False) -> None:
    if not run_all:
        if None in (reader, target, cond, batch):
            raise SystemExit("wsr read: give --reader, --target, --cond and --batch, or --all")
        run_session(cfg, exp, reader, target, cond, batch, framing, hours)
        return
    todo = [s for s in plan(cfg, exp) if s["status"] == "todo"
            and (reader is None or s["reader"] == reader) and (target is None or s["target"] == target)
            and (cond is None or s["cond"] == cond)]
    print(f"{len(todo)} sessions to run")
    for s in todo:
        run_session(cfg, exp, s["reader"], s["target"], s["cond"], s["batch"], s["framing"], hours)
