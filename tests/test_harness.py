"""The built-in agent loop, the sandbox tools and the provider clients (no network: scripted models)."""
from __future__ import annotations

import json
import time

import httpx

from wsr.config import normalize_config
from wsr.harness import extract_json, run_builtin, run_direct
from wsr.providers import OpenAIClient, Reply, ToolCall, validate_arguments
from wsr.sandbox import Sandbox, install_python_logger


class Scripted:
    """A fake chat client that plays back a list of replies (OpenAI-style history)."""

    def __init__(self, replies):
        self.replies, self.seen = list(replies), []

    def start(self, system, user):
        return [{"role": "system", "content": system}, {"role": "user", "content": user}]

    def add_user(self, h, text):
        h.append({"role": "user", "content": text})

    def add_reply(self, h, r):
        h.append({"role": "assistant", "content": r.text})

    def add_tool_results(self, h, results):
        for c, text, _ in results:
            h.append({"role": "tool", "tool_call_id": c.id, "content": text})

    def trim(self, h, keep_last=12):
        return False

    def chat(self, h, tools=None, max_tokens=None):
        self.seen.append([dict(m) for m in h])
        return self.replies.pop(0) if self.replies else Reply("I am done.")

    def ask(self, prompt, max_tokens=512):
        return self.chat([{"role": "user", "content": prompt}]).text


def _sandbox(tmp_path):
    cfg = normalize_config({"models": {}, "sandbox": {"runtime": "local"}})
    (tmp_path / "work").mkdir()
    sb = Sandbox(cfg, {"work": (str(tmp_path / "work"), False)})
    install_python_logger(tmp_path / "work", sb)
    return sb


def test_builtin_loop_uses_tools_and_nudges(tmp_path):
    sb = _sandbox(tmp_path)
    ans = tmp_path / "work" / "answers.json"
    client = Scripted([
        Reply("Let me look.", [ToolCall("1", "python", {"code": "print(6 * 7)"})], "tool"),
        Reply("I will now write the answers."),                                   # stops early -> nudged
        Reply("", [ToolCall("2", "write_file", {"path": "answers.json", "content": '{"e1": {}}'}),
                   ToolCall("3", "bash", {"command": "cat answers.json; exit 3"})], "tool"),
        Reply("", [ToolCall("4", "bash", {"wrong": 1})], "tool"),               # invalid arguments
        Reply("Done."),
    ])
    status, nudges, usage = run_builtin({}, "task", sb, time.time() + 3600, lambda: ans.exists(),
                                        lambda left: "keep going", tmp_path / "logs", warn_msg=None, client=client)
    assert status == "done" and nudges == 1 and usage["requests"] == 5
    assert json.loads(ans.read_text()) == {"e1": {}}
    tool_msgs = [m["content"] for m in client.seen[-1] if m["role"] == "tool"]
    assert tool_msgs[0].startswith("exit code 0\n42")
    assert "exit code 3" in tool_msgs[2] and '{"e1": {}}' in tool_msgs[2]
    assert "missing required argument `command`" in tool_msgs[3]
    assert (tmp_path / "work" / "logs" / "code" / "argv.log").exists()      # python calls are logged
    log = [json.loads(l) for l in open(tmp_path / "logs" / "transcript.jsonl")]
    assert [e["kind"] for e in log].count("tool") == 4


def test_builtin_loop_stops_at_deadline_and_warns(tmp_path):
    sb = _sandbox(tmp_path)
    client = Scripted([Reply("thinking")] * 3)
    status, _, _ = run_builtin({}, "task", sb, time.time() + 100, lambda: False, lambda left: "go on", tmp_path / "logs",
                               warn_secs=900, warn_msg="TIME IS ALMOST UP", client=client)
    assert status == "timeout" or any("TIME IS ALMOST UP" in str(m) for m in client.seen[0])


def test_write_file_cannot_escape(tmp_path):
    from wsr.harness import ToolBox
    sb = _sandbox(tmp_path)
    text, err = ToolBox(sb).run("write_file", {"path": "/etc/passwd_x", "content": "x"}, 100)
    assert err and "outside" in text
    text, err = ToolBox(sb).run("write_file", {"path": "../escape.txt", "content": "x"}, 100)
    assert err and not (tmp_path / "escape.txt").exists()


def test_sandbox_timeout(tmp_path):
    sb = _sandbox(tmp_path)
    rc, out = sb.exec("sleep 30", timeout=5)
    assert rc != 0 and "killed" in out


def test_direct_reader(tmp_path):
    client = Scripted([Reply('Here:\n```json\n{"e1": {"refusal": ["same", 60]}}\n```')])
    assert run_direct({}, "prompt", tmp_path / "a.json", tmp_path, client) == "done"
    assert json.loads((tmp_path / "a.json").read_text()) == {"e1": {"refusal": ["same", 60]}}
    assert extract_json("no json") is None


def test_openai_client_tool_calls():
    seen = {}

    def handler(req):
        seen["url"], seen["body"] = str(req.url), json.loads(req.content)
        return httpx.Response(200, json={"choices": [{"finish_reason": "tool_calls", "message": {
            "content": None, "tool_calls": [
                {"id": "c1", "type": "function", "function": {"name": "bash", "arguments": '{"command": "ls"}'}},
                {"id": "c2", "type": "function", "function": {"name": "bash", "arguments": '{"command": '}}]}}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 3}})
    c = OpenAIClient({"provider": "openai", "base_url": "http://x/v1", "model": "m", "api_key": "k",
                      "params": {"temperature": 0.2}})
    c.http = httpx.Client(base_url=c.base, transport=httpx.MockTransport(handler))
    h = c.start("sys", "hi")
    r = c.chat(h, [{"name": "bash", "description": "d", "schema": {"type": "object", "properties": {}}}])
    assert seen["url"] == "http://x/v1/chat/completions"
    assert seen["body"]["tools"][0]["function"]["name"] == "bash" and seen["body"]["temperature"] == 0.2
    assert r.stop == "tool" and r.tool_calls[0].arguments == {"command": "ls"}
    assert r.tool_calls[1].error and "not valid JSON" in r.tool_calls[1].error
    c.add_reply(h, r)
    c.add_tool_results(h, [(r.tool_calls[0], "out", False)])
    assert h[-2]["tool_calls"] and h[-1] == {"role": "tool", "tool_call_id": "c1", "content": "out"}


def test_openai_client_retries_then_fails():
    n = {"calls": 0}

    def handler(req):
        n["calls"] += 1
        return httpx.Response(429, headers={"retry-after": "0"}, json={"error": "slow down"})
    c = OpenAIClient({"base_url": "http://x/v1", "model": "m"})
    c.http = httpx.Client(base_url=c.base, transport=httpx.MockTransport(handler))
    try:
        c._post({}, tries=3)
        raise AssertionError("expected ChatError")
    except Exception as e:  # noqa: BLE001
        assert "429" in str(e) and n["calls"] == 3


def test_validate_arguments():
    schema = {"type": "object", "properties": {"n": {"type": "integer"}}, "required": ["n"]}
    assert validate_arguments(ToolCall("1", "t", {"n": 3}), schema) is None
    assert "integer" in validate_arguments(ToolCall("1", "t", {"n": "3"}), schema)
    assert "unknown" in validate_arguments(ToolCall("1", "t", {"n": 1, "x": 2}), schema)


def test_known_providers_need_keys(monkeypatch):
    from wsr.providers import ChatError, resolve_endpoint
    monkeypatch.delenv("OPENROUTER_API_KEY", raising=False)
    try:
        resolve_endpoint({"provider": "openrouter", "model": "x"})
        raise AssertionError
    except ChatError as e:
        assert "OPENROUTER_API_KEY" in str(e)
    monkeypatch.setenv("OPENROUTER_API_KEY", "k")
    assert resolve_endpoint({"provider": "openrouter"}) == ("https://openrouter.ai/api/v1", "k")
    assert resolve_endpoint({"provider": "ollama"})[0] == "http://localhost:11434/v1"
