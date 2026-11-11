"""Ground-truth measurement: apply an intervention in place inside vLLM and run the fixed probe sets."""
from __future__ import annotations

import json
import os
import random
import re
import shutil
import tempfile
import time
from pathlib import Path

from .common import BASE_RUNS, MAX_TOKENS, SAMPLING, Layout, read_jsonl


def probe_items(data_dir: Path) -> list[dict]:
    """64 MATH problems, 100 MMLU-Pro items, 48 wrong-hint + 16 right-hint items, 60 StrongREJECT prompts (seed 777)."""
    rng = random.Random(777)
    rd = lambda p: read_jsonl(data_dir / p)  # noqa: E731
    math = rng.sample(rd("math.jsonl"), 64)
    mc = rng.sample(rd("mc.jsonl"), 100)
    sy = rd("syco.jsonl")
    sy = rng.sample([x for x in sy if x["hint_type"] == "wrong"], 48) + rng.sample([x for x in sy if x["hint_type"] == "right"], 16)
    chat = rng.sample(rd("strongreject.jsonl"), 60)
    return math + mc + sy + chat


def final_text(target: str, text: str) -> str:
    """The answer after the reasoning block."""
    if target == "muse30b":
        text = text.split("<|eom|>")[-1]
        return re.sub(r"<\|start\|>assistant|to=[A-Za-z_.]+<\|message\|>|<\|[a-z_]+\|>", "", text).strip()
    return text.split("</think>")[-1]


def _worker_load_weights(worker, path, base_dir, restore_names):
    """Runs inside every vLLM worker: load replacement tensors (and restore previously patched ones from the base)."""
    import torch
    from .checkpoint import Checkpoint, load_tensors
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


def run_ids(L: Layout, t: str) -> list[str]:
    """Base runs plus every intervention of the pool whose measurement is not inherited from another round.
    The Subtle round reuses the Full-Dose base runs and the measurements of identical Full-Dose specs."""
    bases = [] if L.rnd == "subtle" else list(BASE_RUNS)
    return bases + [s["id"] for s in read_jsonl(L.pool(t)) if "same_as" not in s]


def measurement_file(L: Layout, t: str, rid: str, spec: dict | None = None) -> Path:
    if L.rnd == "subtle" and (rid in BASE_RUNS or (spec and "same_as" in spec)):
        return Layout(L.cfg, "full_dose").measure(t) / f"{spec['same_as'] if spec and 'same_as' in spec else rid}.jsonl"
    return L.measure(t) / f"{rid}.jsonl"


def cmd_measure(cfg: dict, a) -> None:
    from safetensors.torch import save_file
    from vllm import LLM, SamplingParams
    from .checkpoint import Checkpoint
    from .edits import apply_ops
    from .grading import build_messages, grade
    L = Layout(cfg, a.round)
    base = cfg["models"][a.target]
    items = probe_items(Path(cfg["data_dir"]))
    msgs = [build_messages(it) for it in items]
    pool = {s["id"]: s for s in read_jsonl(L.pool(a.target))}
    ids = a.ids or run_ids(L, a.target)
    out = L.measure(a.target)
    out.mkdir(parents=True, exist_ok=True)
    ck = Checkpoint(base)
    llm = LLM(model=base, tensor_parallel_size=1, max_model_len=MAX_TOKENS + 4096, gpu_memory_utilization=0.9,
              max_num_seqs=256, enable_prefix_caching=True, trust_remote_code=True,
              limit_mm_per_prompt={"image": 0, "video": 0})
    tmp = Path(tempfile.mkdtemp(prefix="wsr_"))
    patched: list[str] = []
    for rid in ids:
        f = out / f"{rid}.jsonl"
        if f.exists():
            continue
        t0 = time.time()
        if rid in BASE_RUNS:
            tensors, seed0, path = {}, BASE_RUNS[rid], ""
        else:
            tensors, _, _ = apply_ops(ck, pool[rid])
            seed0, path = 0, str(tmp / "edit.safetensors")
            save_file(tensors, path)
        restore = sorted(n for n in patched if n not in tensors)
        if restore or tensors:
            llm.collective_rpc(_worker_load_weights, args=(path, base, restore))
        llm.reset_prefix_cache()
        patched = sorted(tensors)
        extra = {"skip_special_tokens": False} if a.target == "muse30b" else {}
        sps = [SamplingParams(max_tokens=MAX_TOKENS, seed=seed0 + i, **SAMPLING, **extra) for i in range(len(items))]
        outs = llm.chat(msgs, sps, use_tqdm=False, chat_template_kwargs={"enable_thinking": True})
        with open(str(f) + ".tmp", "w") as fh:
            for it, o in zip(items, outs):
                c = o.outputs[0]
                rec = grade(it, final_text(a.target, c.text), len(c.token_ids), c.finish_reason == "length")
                fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
        os.replace(str(f) + ".tmp", f)
        for p in tmp.iterdir():
            p.unlink()
        print(f"{rid} done in {time.time() - t0:.0f}s", flush=True)
    shutil.rmtree(tmp, ignore_errors=True)
