"""Chat clients for the models that read or program weights, and for the judge.

Any model reachable through a chat API can take part: ``provider: openai`` (and every service that speaks the
Chat Completions protocol: OpenRouter, Gemini, Together, Groq, vLLM, Ollama, LM Studio, llama.cpp, ...) or
``provider: anthropic`` (Claude, through the official SDK). A client keeps the conversation in its provider's own
format; the agent loop in ``wsr.harness`` only uses the methods below, so it is the same loop for every provider.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field

# Services that speak the OpenAI Chat Completions protocol: provider name -> (base URL, API key variable).
OPENAI_COMPATIBLE = {
    "openai": ("https://api.openai.com/v1", "OPENAI_API_KEY"),
    "openrouter": ("https://openrouter.ai/api/v1", "OPENROUTER_API_KEY"),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", "GEMINI_API_KEY"),
    "together": ("https://api.together.xyz/v1", "TOGETHER_API_KEY"),
    "groq": ("https://api.groq.com/openai/v1", "GROQ_API_KEY"),
    "deepseek": ("https://api.deepseek.com/v1", "DEEPSEEK_API_KEY"),
    "mistral": ("https://api.mistral.ai/v1", "MISTRAL_API_KEY"),
    "xai": ("https://api.x.ai/v1", "XAI_API_KEY"),
    "fireworks": ("https://api.fireworks.ai/inference/v1", "FIREWORKS_API_KEY"),
    "ollama": ("http://localhost:11434/v1", None),
    "lmstudio": ("http://localhost:1234/v1", None),
    "vllm": ("http://localhost:8000/v1", None),
}


class ChatError(RuntimeError):
    """The endpoint failed in a way that retrying will not fix."""


class ContextOverflow(ChatError):
    """The conversation no longer fits the model's context window."""


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict
    error: str | None = None        # set when the arguments could not be parsed


@dataclass
class Reply:
    text: str
    tool_calls: list[ToolCall] = field(default_factory=list)
    stop: str = "end"               # end | tool | length | refusal
    usage: dict = field(default_factory=dict)
    native: object = None           # the assistant turn in the provider's own format


def resolve_endpoint(a: dict) -> tuple[str, str]:
    """(base URL, API key) of an OpenAI-compatible agent."""
    prov = a.get("provider", "openai")
    base, env = OPENAI_COMPATIBLE.get(prov, (None, None))
    base = (a.get("base_url") or base or "").rstrip("/")
    if not base:
        raise ChatError(f"provider `{prov}` needs a `base_url`")
    key = a.get("api_key") or os.environ.get(a.get("api_key_env") or env or "", "")
    if not key and (a.get("api_key_env") or (env and not a.get("base_url"))):
        raise ChatError(f"set the environment variable {a.get('api_key_env') or env} for provider `{prov}`")
    return base, key or "EMPTY"


class OpenAIClient:
    """Chat Completions with tool calls, over plain HTTP."""

    def __init__(self, a: dict):
        import httpx
        self.a = a
        self.base, key = resolve_endpoint(a)
        self.model = a["model"]
        self.http = httpx.Client(base_url=self.base, timeout=httpx.Timeout(a.get("timeout", 1800), connect=30),
                                 headers={"Authorization": f"Bearer {key}", **(a.get("headers") or {})})
        official = a.get("provider", "openai") == "openai" and not a.get("base_url")
        self.max_tokens_param = a.get("max_tokens_param") or ("max_completion_tokens" if official else "max_tokens")
        self.max_output = a.get("max_output_tokens")
        self.params = dict(a.get("params") or {})

    # ---- conversation
    def start(self, system: str | None, user: str) -> list:
        return ([{"role": "system", "content": system}] if system else []) + [{"role": "user", "content": user}]

    def add_user(self, history: list, text: str) -> None:
        history.append({"role": "user", "content": text})

    def add_reply(self, history: list, reply: Reply) -> None:
        history.append(reply.native)

    def add_tool_results(self, history: list, results: list[tuple[ToolCall, str, bool]]) -> None:
        for call, text, _is_error in results:
            history.append({"role": "tool", "tool_call_id": call.id, "content": text})

    def trim(self, history: list, keep_last: int = 12) -> bool:
        """Blank the oldest tool outputs to make room. Returns False when nothing is left to drop."""
        changed = False
        for m in history[:-keep_last]:
            if m.get("role") == "tool" and len(m.get("content") or "") > 200:
                m["content"] = "[output dropped to save context; re-run the command if you need it]"
                changed = True
        return changed

    # ---- requests
    def _post(self, body: dict, tries: int = 6) -> dict:
        import httpx
        delay = 2.0
        for attempt in range(tries):
            try:
                r = self.http.post("/chat/completions", json=body)
            except (httpx.TransportError, httpx.TimeoutException) as e:
                if attempt == tries - 1:
                    raise ChatError(f"{self.base}: {e!r}") from e
                time.sleep(delay)
                delay = min(delay * 2, 60)
                continue
            if r.status_code == 200:
                return r.json()
            text = r.text[:600]
            low = text.lower()
            if r.status_code in (400, 413) and any(s in low for s in ("context length", "context window", "maximum context",
                                                                     "too many tokens", "prompt is too long", "context_length")):
                raise ContextOverflow(text)
            if r.status_code in (408, 409, 429) or r.status_code >= 500:
                if attempt == tries - 1:
                    raise ChatError(f"{self.base} returned {r.status_code}: {text}")
                time.sleep(float(r.headers.get("retry-after") or delay))
                delay = min(delay * 2, 60)
                continue
            raise ChatError(f"{self.base} returned {r.status_code}: {text}")
        raise ChatError("unreachable")

    def chat(self, history: list, tools: list[dict] | None = None, max_tokens: int | None = None) -> Reply:
        body = {"model": self.model, "messages": history, **self.params}
        limit = max_tokens or self.max_output
        if limit:
            body[self.max_tokens_param] = limit
        if tools:
            body["tools"] = [{"type": "function", "function": {"name": t["name"], "description": t["description"],
                                                                "parameters": t["schema"]}} for t in tools]
        j = self._post(body)
        ch = j["choices"][0]
        msg = ch["message"]
        calls = []
        for tc in msg.get("tool_calls") or []:
            raw = tc["function"].get("arguments") or "{}"
            try:
                args, err = (json.loads(raw) if isinstance(raw, str) else raw), None
                if not isinstance(args, dict):
                    args, err = {}, "arguments must be a JSON object"
            except json.JSONDecodeError as e:
                args, err = {}, f"arguments are not valid JSON: {e}"
            calls.append(ToolCall(tc.get("id") or f"call_{len(calls)}", tc["function"]["name"], args, err))
        native = {"role": "assistant", "content": msg.get("content") or ""}
        if msg.get("tool_calls"):
            native["tool_calls"] = msg["tool_calls"]
        stop = "tool" if calls else ("length" if ch.get("finish_reason") == "length" else "end")
        return Reply(msg.get("content") or "", calls, stop, j.get("usage") or {}, native)

    def ask(self, prompt: str, max_tokens: int = 512) -> str:
        """One question, one answer (used by the judge and the tool-less ``direct`` reader)."""
        return self.chat(self.start(None, prompt), max_tokens=max_tokens).text


class AnthropicClient:
    """Claude through the official SDK (``pip install anthropic``).

    The conversation is append-only and assistant turns are passed back unchanged, as the API requires for models
    that keep their thinking blocks. Sampling parameters are not sent; steer depth with
    ``params: {output_config: {effort: ...}}``.
    """

    def __init__(self, a: dict):
        try:
            import anthropic
        except ImportError as e:
            raise ChatError("provider `anthropic` needs the SDK: pip install anthropic") from e
        self._sdk = anthropic
        kw = {}
        if a.get("api_key") or a.get("api_key_env"):
            kw["api_key"] = a.get("api_key") or os.environ.get(a["api_key_env"], "")
        if a.get("base_url"):
            kw["base_url"] = a["base_url"]
        self.client = anthropic.Anthropic(max_retries=a.get("max_retries", 4), **kw)   # credentials from the environment
        self.model = a["model"]
        self.max_output = a.get("max_output_tokens", 64000)
        self.params = dict(a.get("params") or {})
        self.eager = bool(a.get("eager_input_streaming", not a.get("base_url")))
        self.system = None

    def start(self, system: str | None, user: str) -> list:
        self.system = system
        return [{"role": "user", "content": user}]

    def add_user(self, history: list, text: str) -> None:
        history.append({"role": "user", "content": text})

    def add_reply(self, history: list, reply: Reply) -> None:
        history.append({"role": "assistant", "content": reply.native})

    def add_tool_results(self, history: list, results: list[tuple[ToolCall, str, bool]]) -> None:
        history.append({"role": "user", "content": [
            {"type": "tool_result", "tool_use_id": call.id, "content": text, **({"is_error": True} if is_error else {})}
            for call, text, is_error in results]})

    def trim(self, history: list, keep_last: int = 12) -> bool:
        return False        # earlier turns must stay as they were sent

    def chat(self, history: list, tools: list[dict] | None = None, max_tokens: int | None = None) -> Reply:
        sdk = self._sdk
        kw = {"model": self.model, "max_tokens": max_tokens or self.max_output, "messages": history, **self.params}
        if self.system:
            kw["system"] = self.system
        if tools:
            kw["tools"] = [{"name": t["name"], "description": t["description"], "input_schema": t["schema"],
                            **({"eager_input_streaming": True} if self.eager else {})} for t in tools]
        try:
            while True:
                with self.client.messages.stream(**kw) as stream:
                    msg = stream.get_final_message()
                if msg.stop_reason != "pause_turn":
                    break
                kw["messages"] = history = history + [{"role": "assistant", "content": msg.content}]
        except sdk.BadRequestError as e:
            if any(s in str(e).lower() for s in ("prompt is too long", "context window", "context length")):
                raise ContextOverflow(str(e)) from e
            raise ChatError(f"bad request: {e}") from e
        except sdk.AuthenticationError as e:
            raise ChatError("authentication failed: set ANTHROPIC_API_KEY or run `ant auth login`") from e
        except sdk.NotFoundError as e:
            raise ChatError(f"model `{self.model}` not found: {e}") from e
        except sdk.RateLimitError as e:
            raise ChatError(f"rate limited after retries: {e}") from e
        except sdk.APIStatusError as e:
            raise ChatError(f"API error {e.status_code}: {e}") from e
        except sdk.APIConnectionError as e:
            raise ChatError(f"connection error: {e}") from e
        text = "".join(b.text for b in msg.content if b.type == "text")
        usage = {"input_tokens": msg.usage.input_tokens, "output_tokens": msg.usage.output_tokens}
        if msg.stop_reason == "refusal":
            return Reply(text, [], "refusal", usage, msg.content)
        cut = msg.stop_reason == "max_tokens"        # a cut-off turn may hold an incomplete tool call: run nothing
        calls = [ToolCall(b.id, b.name, b.input if isinstance(b.input, dict) else {},
                          "the response reached the output limit before this call was complete; nothing was run" if cut
                          else None if isinstance(b.input, dict) else "arguments must be a JSON object")
                 for b in msg.content if b.type == "tool_use"]
        return Reply(text, calls, "tool" if calls else ("length" if cut else "end"), usage, msg.content)

    def ask(self, prompt: str, max_tokens: int = 2048) -> str:
        return self.chat(self.start(None, prompt), max_tokens=max_tokens).text


def make_client(a: dict):
    return AnthropicClient(a) if a.get("provider") == "anthropic" else OpenAIClient(a)


def validate_arguments(call: ToolCall, schema: dict) -> str | None:
    """Check a tool call against its schema before running it. Returns an error message or None."""
    if call.error:
        return call.error
    props = schema.get("properties", {})
    for k in schema.get("required", []):
        if k not in call.arguments:
            return f"missing required argument `{k}`"
    types = {"string": str, "integer": int, "number": (int, float), "boolean": bool}
    for k, v in call.arguments.items():
        if k not in props:
            return f"unknown argument `{k}`"
        want = types.get(props[k].get("type"))
        if want and not isinstance(v, want):
            return f"argument `{k}` must be a {props[k]['type']}"
    return None
