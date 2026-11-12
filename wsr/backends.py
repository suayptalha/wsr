"""Generation backends that measure an edited model: the base model is loaded once and each edit is patched in
place (and undone before the next), so measuring a pool costs one model load.

* ``vllm``: fast batched generation on GPU (the paper's backend); edits are loaded into every worker.
* ``hf``: Hugging Face Transformers, on GPU or CPU; works for any causal LM that Transformers can load.
"""
from __future__ import annotations

import re
import tempfile
from pathlib import Path


def final_text(fmt: str, text: str) -> str:
    """The answer after the reasoning block."""
    if fmt == "muse":
        text = text.split("<|eom|>")[-1]
        return re.sub(r"<\|start\|>assistant|to=[A-Za-z_.]+<\|message\|>|<\|[a-z_]+\|>", "", text).strip()
    if fmt == "think":
        return text.split("</think>")[-1]
    return text


class Output:
    def __init__(self, text: str, n_tokens: int, hit_limit: bool):
        self.text, self.n_tokens, self.hit_limit = text, n_tokens, hit_limit


class Backend:
    def patch(self, tensors: dict) -> None: ...
    def generate(self, messages: list[list[dict]], seed0: int) -> list[Output]: ...
    def logprob(self, messages: list[list[dict]], continuations: list[str]) -> list[tuple[float, int]]:
        """(sum of log-probabilities, number of tokens) of each continuation after its prompt (teacher forced)."""
        raise NotImplementedError
    def close(self) -> None: ...


def _template_kwargs(model_cfg: dict, suite) -> dict:
    thinking = model_cfg.get("thinking")
    if thinking is None:
        thinking = suite.thinking
    return {} if thinking is None else {"enable_thinking": bool(thinking)}


# ---------------------------------------------------------------- vLLM
def _worker_load_weights(worker, path, base_dir, restore_names):
    """Runs inside every vLLM worker: load replacement tensors (and restore previously patched ones from the base)."""
    import torch

    from wsr.checkpoint import Checkpoint, load_tensors
    items = []
    if restore_names:
        ck = Checkpoint(base_dir)
        items += [(n, ck.get(n)) for n in restore_names]
    if path:
        items += list(load_tensors(path).items())
    items = [(n, t.to(torch.bfloat16)) for n, t in items]
    loaded = worker.model_runner.model.load_weights(iter(items))
    torch.cuda.synchronize()
    return len(items) if loaded is None else len(loaded)


class VLLMBackend(Backend):
    def __init__(self, model_dir: str, model_cfg: dict, suite):
        from vllm import LLM
        self.dir, self.cfg, self.suite = str(model_dir), model_cfg, suite
        kw = {"tensor_parallel_size": 1, "max_model_len": suite.max_tokens + 4096, "gpu_memory_utilization": 0.9,
              "max_num_seqs": 256, "enable_prefix_caching": True, **(model_cfg.get("vllm") or {})}
        self.llm = LLM(model=self.dir, **kw)
        self.tmp = Path(tempfile.mkdtemp(prefix="wsr_"))
        self.patched: list[str] = []

    def patch(self, tensors: dict) -> None:
        from safetensors.torch import save_file
        path = ""
        if tensors:
            path = str(self.tmp / "edit.safetensors")
            save_file(tensors, path)
        restore = sorted(n for n in self.patched if n not in tensors)
        if restore or tensors:
            self.llm.collective_rpc(_worker_load_weights, args=(path, self.dir, restore))
        self.llm.reset_prefix_cache()
        self.patched = sorted(tensors)
        for p in self.tmp.iterdir():
            p.unlink()

    def generate(self, messages, seed0):
        from vllm import SamplingParams
        extra = {} if self.cfg.get("skip_special_tokens", True) else {"skip_special_tokens": False}
        sps = [SamplingParams(max_tokens=self.suite.max_tokens, seed=seed0 + i, **self.suite.sampling, **extra)
               for i in range(len(messages))]
        kw = _template_kwargs(self.cfg, self.suite)
        outs = self.llm.chat(messages, sps, use_tqdm=False, **({"chat_template_kwargs": kw} if kw else {}))
        return [Output(o.outputs[0].text, len(o.outputs[0].token_ids), o.outputs[0].finish_reason == "length")
                for o in outs]

    def logprob(self, messages, continuations):
        from vllm import SamplingParams
        tok = self.llm.get_tokenizer()
        kw = _template_kwargs(self.cfg, self.suite)
        prompts, starts = [], []
        for m, c in zip(messages, continuations):
            head = tok.apply_chat_template(m, tokenize=False, add_generation_prompt=True, **kw)
            a = tok(head, add_special_tokens=False)["input_ids"]
            b = tok(head + c, add_special_tokens=False)["input_ids"]
            prompts.append({"prompt_token_ids": b})
            starts.append(len(a))
        outs = self.llm.generate(prompts, SamplingParams(max_tokens=1, prompt_logprobs=0), use_tqdm=False)
        res = []
        for o, st in zip(outs, starts):
            ids, lps = o.prompt_token_ids, o.prompt_logprobs
            vals = [lps[i][ids[i]].logprob for i in range(max(1, st), len(ids)) if lps[i] is not None]
            res.append((float(sum(vals)), len(vals)))
        return res

    def close(self):
        import shutil
        shutil.rmtree(self.tmp, ignore_errors=True)


# ---------------------------------------------------------------- Transformers
class HFBackend(Backend):
    """Batched sampling with Transformers. Each batch is seeded with ``seed0 + first item index`` so that runs
    are reproducible on the same hardware and batch size (per-item seeds as in vLLM are not available)."""

    def __init__(self, model_dir: str, model_cfg: dict, suite):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        hf = dict(model_cfg.get("hf") or {})
        self.cfg, self.suite = model_cfg, suite
        self.device = hf.pop("device", None) or ("cuda" if torch.cuda.is_available() else
                                                 "mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
        dtype = hf.pop("dtype", None) or ("bfloat16" if self.device == "cuda" else "float32")
        self.batch_size = int(hf.pop("batch_size", 32 if self.device == "cuda" else 8))
        self.tok = AutoTokenizer.from_pretrained(str(model_dir), trust_remote_code=hf.get("trust_remote_code", False))
        self.tok.padding_side = "left"
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(str(model_dir), dtype=getattr(torch, dtype), **hf).to(self.device)
        self.model.eval()
        self.params = dict(self.model.named_parameters(remove_duplicate=False))
        self.params.update({k: v for k, v in self.model.named_buffers()})
        self.saved: dict[str, "torch.Tensor"] = {}

    def _param(self, name: str):
        if name in self.params:
            return self.params[name]
        hits = [k for k in self.params if k.endswith("." + name) or name.endswith("." + k)]
        if len(hits) == 1:
            return self.params[hits[0]]
        raise KeyError(f"`{name}` does not match a parameter of the loaded model")

    def patch(self, tensors: dict) -> None:
        import torch
        with torch.no_grad():
            for n, orig in self.saved.items():
                if n not in tensors:
                    self._param(n).data.copy_(orig.to(self._param(n).device))
            self.saved = {n: o for n, o in self.saved.items() if n in tensors}
            for n, T in tensors.items():
                p = self._param(n)
                if n not in self.saved:
                    self.saved[n] = p.detach().to("cpu", copy=True)
                p.data.copy_(T.to(device=p.device, dtype=p.dtype).reshape(p.shape))

    def _prompt(self, msgs: list[dict]) -> str:
        if getattr(self.tok, "chat_template", None):
            return self.tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True,
                                                **_template_kwargs(self.cfg, self.suite))
        return "".join(f"{m['role'].capitalize()}: {m['content']}\n" for m in msgs) + "Assistant:"

    def generate(self, messages, seed0):
        import torch
        s = self.suite.sampling
        out = []
        eos = self.model.generation_config.eos_token_id
        eos = set(eos if isinstance(eos, list) else [eos]) | {self.tok.eos_token_id}
        for b in range(0, len(messages), self.batch_size):
            prompts = [self._prompt(m) for m in messages[b:b + self.batch_size]]
            enc = self.tok(prompts, return_tensors="pt", padding=True, add_special_tokens=False).to(self.device)
            torch.manual_seed(seed0 + b)
            kw = {"do_sample": s.get("temperature", 1.0) > 0, "max_new_tokens": self.suite.max_tokens,
                  "pad_token_id": self.tok.pad_token_id}
            if kw["do_sample"]:
                kw.update(temperature=s.get("temperature", 1.0), top_p=s.get("top_p", 1.0), top_k=s.get("top_k", 0) or None)
                if s.get("min_p"):
                    kw["min_p"] = s["min_p"]
            with torch.no_grad():
                gen = self.model.generate(**enc, **kw)[:, enc["input_ids"].shape[1]:]
            for row in gen.tolist():
                n = len(row)
                for i, t in enumerate(row):
                    if t in eos:
                        n = i + 1
                        break
                hit = n == len(row) and (not row or row[-1] not in eos) and len(row) >= self.suite.max_tokens
                text = self.tok.decode(row[:n], skip_special_tokens=self.cfg.get("skip_special_tokens", True))
                out.append(Output(text, n, hit))
        return out

    def logprob(self, messages, continuations):
        import torch
        res = []
        for b in range(0, len(messages), self.batch_size):
            seqs, starts = [], []
            for m, c in zip(messages[b:b + self.batch_size], continuations[b:b + self.batch_size]):
                head = self._prompt(m)
                a = self.tok(head, add_special_tokens=False)["input_ids"]
                full = self.tok(head + c, add_special_tokens=False)["input_ids"]
                seqs.append(full)
                starts.append(len(a))
            n = max(len(x) for x in seqs)
            pad = self.tok.pad_token_id
            ids = torch.tensor([x + [pad] * (n - len(x)) for x in seqs], device=self.device)
            mask = torch.tensor([[1] * len(x) + [0] * (n - len(x)) for x in seqs], device=self.device)
            with torch.no_grad():
                logits = self.model(input_ids=ids, attention_mask=mask).logits.float()
            lp = torch.log_softmax(logits[:, :-1], -1).gather(-1, ids[:, 1:, None])[..., 0]
            for i, (x, st) in enumerate(zip(seqs, starts)):
                v = lp[i, max(0, st - 1):len(x) - 1]
                res.append((float(v.sum()), int(v.numel())))
        return res

    def close(self):
        self.patch({})


def make_backend(model_dir, model_cfg: dict, suite) -> Backend:
    b = model_cfg.get("backend", "hf")
    if b == "vllm":
        try:
            import vllm  # noqa: F401
        except ImportError as e:
            raise SystemExit("wsr: backend `vllm` needs vLLM (pip install vllm); or set `backend: hf`") from e
        return VLLMBackend(model_dir, model_cfg, suite)
    return HFBackend(model_dir, model_cfg, suite)
