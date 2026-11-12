"""Agent harnesses: how a model works on a task inside the sandbox until it is done or its time is up.

``harness:`` of an agent in the config:

* ``builtin`` (default): wsr's own tool loop. It works with every provider in ``wsr.providers`` (OpenAI-compatible
  APIs, local servers, Claude). The model gets ``bash``, ``python``, ``read_file`` and ``write_file`` tools that run
  in the sandbox; the model itself is called from the host, so the sandbox has no network at all.
* ``opencode``, ``claude-code``, ``codex``: an agent CLI runs inside the sandbox (the image must contain it, and the
  sandbox needs network access to reach the model). ``opencode`` is the harness of the paper.
* ``command``: any other agent CLI, given as ``command: [...]`` and ``continue_command: [...]`` with ``{prompt}``.
* ``direct``: no tools, one request and one answer (Text-Only reading only).
* ``baseline``: no model; ``params.strategy`` is ``same`` (always "no change"), ``change`` or ``random``.

All harnesses share the same contract: a first message, then nudges whenever the agent stops before ``done()``,
a warning ``warn_secs`` before the deadline, and a hard stop at the deadline.
"""
from __future__ import annotations

import json
import os
import random
import subprocess
import time
from pathlib import Path

from .providers import ChatError, ContextOverflow, make_client, validate_arguments
from .sandbox import Sandbox, clip

SYSTEM = ("You are a careful research assistant working autonomously in a sandboxed Linux machine. Use the tools to "
          "inspect files and run code; do not just describe what you would do. The shell has no internet access "
          "and no GPU. Keep tool outputs small (print summaries, not whole tensors).")

TOOLS = [
    {"name": "bash", "description": "Run a shell command in the sandbox (working directory = your session directory). "
                                    "Returns the exit code and the combined output (truncated if long).",
     "schema": {"type": "object", "properties": {"command": {"type": "string", "description": "the bash command"},
                                                 "timeout": {"type": "integer", "description": "seconds (default 600)"}},
                "required": ["command"], "additionalProperties": False}},
    {"name": "python", "description": "Run a Python program in the sandbox and return its output. NumPy and CPU "
                                      "PyTorch are available. Each call is a fresh process.",
     "schema": {"type": "object", "properties": {"code": {"type": "string", "description": "the program"},
                                                 "timeout": {"type": "integer", "description": "seconds (default 600)"}},
                "required": ["code"], "additionalProperties": False}},
    {"name": "read_file", "description": "Read a text file (or a range of its lines).",
     "schema": {"type": "object", "properties": {"path": {"type": "string"},
                                                 "offset": {"type": "integer", "description": "first line, from 1"},
                                                 "limit": {"type": "integer", "description": "number of lines"}},
                "required": ["path"], "additionalProperties": False}},
    {"name": "write_file", "description": "Create or overwrite a text file in your session directory.",
     "schema": {"type": "object", "properties": {"path": {"type": "string"}, "content": {"type": "string"}},
                "required": ["path", "content"], "additionalProperties": False}},
]


def nudge(goal: str):
    return lambda left: (f"Continue working on the task from where you left off. About {int(left // 60)} minutes of "
                         f"wall-clock time remain. Do not just describe what you will do next: do it, using your tools. "
                         f"Keep going until {goal}.")


class Transcript:
    def __init__(self, path: Path):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)

    def __call__(self, kind: str, **kw) -> None:
        with open(self.path, "a") as f:
            f.write(json.dumps({"t": round(time.time(), 2), "kind": kind, **kw}, ensure_ascii=False) + "\n")


# ---------------------------------------------------------------- built-in tool loop
class ToolBox:
    """The sandbox tools plus optional host-side tools: {name: (schema dict, fn(args) -> (text, is_error))}."""

    def __init__(self, sb: Sandbox, extra: dict | None = None):
        self.sb = sb
        self.extra = extra or {}
        self.n_snippet = 0

    @property
    def specs(self) -> list[dict]:
        return TOOLS + [spec for spec, _ in self.extra.values()]

    def run(self, name: str, args: dict, left: float) -> tuple[str, bool]:
        cap = max(5, min(int(args.get("timeout") or 600), int(left) - 30))
        if name == "bash":
            rc, out = self.sb.exec(args["command"], cap)
            return f"exit code {rc}\n{clip(out)}", rc != 0
        if name == "python":
            self.n_snippet += 1
            rel = f"logs/snippets/{self.n_snippet:04d}.py"
            host = self.sb.host_path(rel)
            host.parent.mkdir(parents=True, exist_ok=True)
            host.write_text(args["code"])
            rc, out = self.sb.exec(f"python {rel}", cap)
            return f"exit code {rc}\n{clip(out)}", rc != 0
        if name == "read_file":
            p = self.sb.readable_host_path(args["path"])
            if p is None or not p.is_file():
                return f"no such file: {args['path']}", True
            try:
                lines = p.read_text(errors="replace").splitlines()
            except OSError as e:
                return str(e), True
            a = max(1, int(args.get("offset") or 1))
            n = int(args.get("limit") or 400)
            body = "\n".join(f"{i:6d}\t{s}" for i, s in enumerate(lines[a - 1:a - 1 + n], a))
            return clip(body or "(empty)"), False
        if name == "write_file":
            p = self.sb.host_path(args["path"])
            if p is None:
                return f"{args['path']} is outside your session directory ({self.sb.paths['work']})", True
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(args["content"])
            return f"wrote {len(args['content'])} characters to {args['path']}", False
        if name in self.extra:
            return self.extra[name][1](args)
        return f"unknown tool `{name}`", True


def run_builtin(agent: dict, prompt: str, sb: Sandbox, deadline: float, done, nudge_fn, log_dir: Path,
                warn_secs: int = 900, warn_msg: str | None = None, max_nudges: int = 40, extra_tools: dict | None = None,
                client=None) -> tuple[str, int, dict]:
    """-> (status, nudges, usage totals)"""
    client = client or make_client(agent)
    box = ToolBox(sb, extra_tools)
    specs = box.specs
    schema = {t["name"]: t["schema"] for t in specs}
    log = Transcript(log_dir / "transcript.jsonl")
    system = agent.get("system", SYSTEM)
    history = client.start(system, prompt)
    log("prompt", system=system, text=prompt)
    nudges, warned, usage = 0, warn_msg is None, {"input_tokens": 0, "output_tokens": 0, "requests": 0}
    while True:
        left = deadline - time.time()
        if left < 90:
            return "timeout", nudges, usage
        if not warned and left <= warn_secs + 60:
            warned = True
            client.add_user(history, warn_msg)
            log("user", text=warn_msg)
        try:
            reply = client.chat(history, specs)
        except ContextOverflow:
            if client.trim(history):
                log("trim")
                continue
            return "context_full", nudges, usage
        except ChatError as e:
            log("error", text=str(e))
            return "error", nudges, usage
        usage["requests"] += 1
        for k in ("input_tokens", "output_tokens", "prompt_tokens", "completion_tokens"):
            if k in reply.usage and isinstance(reply.usage[k], int):
                usage[k.replace("prompt", "input").replace("completion", "output")] += reply.usage[k]
        client.add_reply(history, reply)
        log("assistant", text=reply.text, tool_calls=[{"name": c.name, "arguments": c.arguments} for c in reply.tool_calls],
            stop=reply.stop)
        if reply.stop == "refusal":
            return "refusal", nudges, usage
        if reply.tool_calls:
            results = []
            for c in reply.tool_calls:
                err = validate_arguments(c, schema.get(c.name, {"properties": {}})) if c.name in schema else f"unknown tool `{c.name}`"
                text, is_err = (f"error: {err}", True) if err else box.run(c.name, c.arguments, deadline - time.time())
                results.append((c, text, is_err))
                log("tool", name=c.name, output=text[:4000], error=is_err)
            client.add_tool_results(history, results)
            continue
        if done():
            return "done", nudges, usage
        if nudges >= max_nudges:
            return "max_rounds", nudges, usage
        msg = ("Your last reply was cut off by the output limit. Continue, and keep each reply shorter."
               if reply.stop == "length" else nudge_fn(deadline - time.time()))
        client.add_user(history, msg)
        log("user", text=msg)
        nudges += 1


# ---------------------------------------------------------------- agent CLIs inside the sandbox
def run_agent_loop(base_cmd, first_msg, nudge_fn, env, deadline, done, log_path, max_rounds=40, warn_secs=900,
                   warn_msg=None, continue_cmd=None):
    """The agent CLI ends a turn when the model answers without a tool call. Re-enter the same session (--continue)
    with a nudge until done() or the deadline; `warn_secs` before the deadline the agent is told that time is almost up.
    Returns (status, rounds)."""
    rounds, warned = 0, warn_msg is None
    with open(log_path, "a") as out:
        while True:
            left = deadline - time.time()
            if left < 90:
                return "timeout", rounds
            if rounds == 0:
                msg = first_msg
            elif not warned and left <= warn_secs + 60:
                msg, warned = warn_msg, True
            else:
                msg = nudge_fn(left)
            if continue_cmd is None:
                args = base_cmd + (["--continue"] if rounds else []) + [msg]
            else:
                args = [a.replace("{prompt}", msg) for a in (continue_cmd if rounds else base_cmd)]
                if not any("{prompt}" in a for a in (continue_cmd if rounds else base_cmd)):
                    args.append(msg)
            budget = left if warned else max(60, left - warn_secs)
            out.write(f"\n===== round {rounds} ({int(left)} s left) =====\n")
            out.flush()
            try:
                subprocess.run(args, stdout=out, stderr=subprocess.STDOUT, timeout=budget, env=env)
            except subprocess.TimeoutExpired:
                if warned:
                    return "timeout", rounds + 1
                out.write("\n===== interrupted for the final-minutes warning =====\n")
            rounds += 1
            if done():
                return "done", rounds
            if rounds >= max_rounds:
                return "max_rounds", rounds


def opencode_config(agent: dict) -> dict:
    """opencode configuration for an OpenAI-compatible endpoint (the paper's setup)."""
    from .providers import resolve_endpoint
    base, key = resolve_endpoint(agent)
    return {"$schema": "https://opencode.ai/config.json",
            "provider": {"local": {"npm": "@ai-sdk/openai-compatible", "name": "Reader",
                                   "options": {"baseURL": base, "apiKey": key, "timeout": False, "headerTimeout": False},
                                   "models": {agent["model"]: {"name": agent["model"],
                                                               "limit": {"context": agent.get("context", 262144),
                                                                         "output": agent.get("max_output_tokens", 100000)}}}}},
            "model": f"local/{agent['model']}", "autoupdate": False, "share": "disabled",
            "permission": {"edit": "allow", "bash": "allow", "webfetch": "deny", "external_directory": "allow"}}


def cli_commands(agent: dict, work_host: Path) -> tuple[list[str], list[str] | None, dict]:
    """(first command, continue command or None for `--continue`, extra environment) of an agent CLI."""
    h = agent["harness"]
    env = {k: os.environ[k] for k in agent.get("env_passthrough", []) if k in os.environ}
    if h == "opencode":
        if agent.get("opencode_model"):
            m = agent["opencode_model"]
        else:
            (work_host / "opencode.json").write_text(json.dumps(opencode_config(agent), indent=1))
            m = f"local/{agent['model']}"
        env.update({"OPENCODE_DISABLE_AUTOUPDATE": "1", "OPENCODE_EXPERIMENTAL_OUTPUT_TOKEN_MAX": "100000"})
        return ["opencode", "run", "--model", m], None, env
    if h == "claude-code":
        for k in ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "ANTHROPIC_BASE_URL"):
            if k in os.environ:
                env.setdefault(k, os.environ[k])
        base = ["claude", "-p", "--permission-mode", "bypassPermissions", "--model", agent["model"]]
        return base, base + ["--continue"], env
    if h == "codex":
        for k in ("OPENAI_API_KEY", "CODEX_HOME"):
            if k in os.environ:
                env.setdefault(k, os.environ[k])
        flags = ["--skip-git-repo-check", "--dangerously-bypass-approvals-and-sandbox", "-m", agent["model"]]
        return ["codex", "exec"] + flags, ["codex", "exec", "resume", "--last"] + flags, env
    if h == "command":
        if "command" not in agent:
            raise SystemExit("wsr: harness `command` needs `command: [...]`")
        return list(agent["command"]), list(agent.get("continue_command") or agent["command"]), env
    raise SystemExit(f"wsr: unknown harness `{h}`")


def run_cli(agent: dict, prompt: str, sb: Sandbox, deadline: float, done, nudge_fn, log_dir: Path, warn_secs=900,
            warn_msg=None) -> tuple[str, int]:
    work_host = Path(sb.mounts["work"][0])
    first, cont, env = cli_commands(agent, work_host)
    base_cmd, agent_env = sb.cli(first, env)
    cont_cmd = sb.cli(cont, env)[0] if cont else None     # None: the same command with --continue (opencode)
    return run_agent_loop(base_cmd, prompt, nudge_fn, agent_env, deadline, done, log_dir / "agent_stdout.log",
                          warn_secs=warn_secs, warn_msg=warn_msg, continue_cmd=cont_cmd)


# ---------------------------------------------------------------- no-tool readers
def extract_json(text: str) -> dict | None:
    """The last JSON object in a reply (fenced or bare)."""
    import re
    blocks = re.findall(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    cands = blocks[::-1] + [text[text.find("{"): text.rfind("}") + 1]] if "{" in text else blocks[::-1]
    for c in cands:
        try:
            v = json.loads(c)
            if isinstance(v, dict):
                return v
        except json.JSONDecodeError:
            continue
    return None


def run_direct(agent: dict, prompt: str, answers: Path, log_dir: Path, client=None) -> str:
    client = client or make_client(agent)
    log = Transcript(log_dir / "transcript.jsonl")
    msg = prompt + "\n\nYou have no tools in this session: reply with the JSON object itself, in a ```json block."
    log("prompt", text=msg)
    text = client.ask(msg, max_tokens=agent.get("max_output_tokens", 16000))
    log("assistant", text=text)
    ans = extract_json(text)
    if ans is None:
        return "no_answer"
    answers.write_text(json.dumps(ans, indent=1))
    return "done"


def run_baseline(agent: dict, ids: list[str], keys: list[str], answers: Path) -> str:
    strat = (agent.get("params") or {}).get("strategy", "same")
    rng = random.Random((agent.get("params") or {}).get("seed", 0))
    out = {}
    for i in ids:
        if strat == "same":
            a = {k: ["same", 50] for k in keys}
        elif strat == "change":
            a = {k: ["down", 50] for k in keys}
        elif strat == "random":
            a = {k: [rng.choice(["up", "down", "same"]), 50] for k in keys}
        else:
            raise SystemExit(f"wsr: baseline strategy must be same, change or random, not `{strat}`")
        a["changes_behavior"] = ["no" if strat == "same" else "yes", 50]
        out[i] = a
    answers.write_text(json.dumps(out, indent=1))
    return "done"
