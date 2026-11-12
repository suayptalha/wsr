"""Read-only access to safetensors checkpoints (single file or sharded).

This module is copied into the reader sandbox, so it depends only on torch and safetensors.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file


class Checkpoint:
    """Lazy, read-only view of a checkpoint directory: tensors are loaded one at a time, on request."""

    def __init__(self, model_dir: str | os.PathLike):
        self.dir = Path(model_dir)
        idx = self.dir / "model.safetensors.index.json"
        if idx.exists():
            self.weight_map = json.loads(idx.read_text())["weight_map"]
        else:
            files = sorted(p.name for p in self.dir.glob("*.safetensors"))
            if (self.dir / "model.safetensors").exists():
                files = ["model.safetensors"]
            if not files:
                raise FileNotFoundError(f"no .safetensors files in {self.dir}")
            self.weight_map = {}
            for name in files:
                with safe_open(str(self.dir / name), "pt", device="cpu") as f:
                    self.weight_map.update({k: name for k in f.keys()})
        self._meta: dict[str, tuple[tuple[int, ...], str]] = {}

    def names(self) -> list[str]:
        return sorted(self.weight_map)

    def __contains__(self, name: str) -> bool:
        return name in self.weight_map

    def get(self, name: str) -> torch.Tensor:
        with safe_open(str(self.dir / self.weight_map[name]), "pt", device="cpu") as f:
            return f.get_tensor(name)

    def meta(self, name: str) -> tuple[tuple[int, ...], str]:
        """(shape, safetensors dtype string) without loading the tensor."""
        if name not in self._meta:
            with safe_open(str(self.dir / self.weight_map[name]), "pt", device="cpu") as f:
                sl = f.get_slice(name)
                self._meta[name] = (tuple(sl.get_shape()), sl.get_dtype())
        return self._meta[name]

    def shape(self, name: str) -> tuple[int, ...]:
        return self.meta(name)[0]

    def config(self) -> dict:
        p = self.dir / "config.json"
        return json.loads(p.read_text()) if p.exists() else {}


def load_tensors(path: str | os.PathLike) -> dict[str, torch.Tensor]:
    return load_file(str(path), device="cpu") if path else {}
