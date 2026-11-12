"""Write an edited model as a new checkpoint, and optionally push it to the Hugging Face Hub.

``export_checkpoint(base_dir, tensors, out_dir)`` copies the base checkpoint (config, tokenizer, every other
file) and rewrites only the safetensors shards that hold an edited tensor; untouched shards are hard-linked when
possible (no extra disk) and copied otherwise. The result loads like the base model, in Transformers, vLLM or
anywhere else.
"""
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

WEIGHT_SUFFIXES = (".safetensors",)
SKIP = {".git", ".cache"}


def _link_or_copy(src: Path, dst: Path) -> None:
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def export_checkpoint(base_dir: str | os.PathLike, tensors: dict, out_dir: str | os.PathLike,
                      metadata: dict | None = None, overwrite: bool = False) -> Path:
    """-> out_dir. ``tensors`` = {parameter name: new full tensor} (same names and shapes as the base)."""
    from safetensors import safe_open
    from safetensors.torch import save_file

    from .checkpoint import Checkpoint
    base, out = Path(base_dir).resolve(), Path(out_dir)
    if out.exists():
        if not overwrite:
            raise FileExistsError(f"{out} exists")
        shutil.rmtree(out)
    out.mkdir(parents=True)
    ck = Checkpoint(base)
    unknown = [n for n in tensors if n not in ck]
    if unknown:
        raise KeyError(f"not parameters of the base model: {unknown[:5]}")
    by_file: dict[str, list[str]] = {}
    for n in tensors:
        by_file.setdefault(ck.weight_map[n], []).append(n)
    for p in sorted(base.iterdir()):
        if p.name in SKIP or p.is_dir():
            if p.is_dir() and p.name not in SKIP:
                shutil.copytree(p, out / p.name, symlinks=False)
            continue
        src = p.resolve()                                # Hugging Face cache entries are symlinks into blobs/
        if p.name in by_file:
            with safe_open(str(src), "pt", device="cpu") as f:
                meta = f.metadata() or {}
                shard = {k: f.get_tensor(k) for k in f.keys()}
            for n in by_file[p.name]:
                old = shard[n]
                new = tensors[n].to(old.dtype).reshape(old.shape).contiguous()
                shard[n] = new
            save_file(shard, str(out / p.name), metadata={**meta, "format": meta.get("format", "pt")})
        else:
            _link_or_copy(src, out / p.name)
    if metadata is not None:
        (out / "wsr_edit.json").write_text(json.dumps(metadata, indent=1, default=str))
    return out


def model_card(base: str, goal: str, agent: str, history: list[dict], extra: str = "") -> str:
    rows = "\n".join(f"| {h['round']} | {h['attempt']} | {h.get('goal_score', '–')} | {h.get('accepted')} | "
                     f"{h.get('reason', '')} |" for h in history)
    return f"""---
base_model: {base}
tags: [wsr, weight-space-programming, model-editing]
---

# Edited {base}

This checkpoint is `{base}` with an edit written directly into its weights by an AI agent (`{agent}`), without any
training: no gradients, no optimizer, no fitting. It was produced with
[wsr](https://github.com/suayptalha/wsr) (weight-space programming).

**Goal:** {goal}

| round | attempt | goal score (held out) | accepted | note |
|---|---|---|---|---|
{rows}

`wsr_edit.json` lists every changed tensor. {extra}
"""


def push_to_hub(folder: str | os.PathLike, repo_id: str, private: bool = True, token: str | None = None) -> str:
    from huggingface_hub import HfApi
    api = HfApi(token=token)
    api.create_repo(repo_id, private=private, exist_ok=True)
    api.upload_folder(folder_path=str(folder), repo_id=repo_id, commit_message="Upload edited model (wsr)")
    return f"https://huggingface.co/{repo_id}"
