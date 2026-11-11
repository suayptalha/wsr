"""Read-only access to sharded safetensors checkpoints."""
from __future__ import annotations

import json
import os
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import load_file


class Checkpoint:
    """Lazy, read-only view of a sharded safetensors checkpoint."""

    def __init__(self, model_dir: str | os.PathLike):
        self.dir = Path(model_dir)
        idx = self.dir / "model.safetensors.index.json"
        if idx.exists():
            self.weight_map = json.loads(idx.read_text())["weight_map"]
        else:
            single = self.dir / "model.safetensors"
            with safe_open(str(single), "pt", device="cpu") as f:
                self.weight_map = {k: "model.safetensors" for k in f.keys()}
        self._meta: dict[str, tuple[tuple[int, ...], str]] = {}

    def names(self) -> list[str]:
        return sorted(self.weight_map)

    def get(self, name: str) -> torch.Tensor:
        with safe_open(str(self.dir / self.weight_map[name]), "pt", device="cpu") as f:
            return f.get_tensor(name)

    def meta(self, name: str) -> tuple[tuple[int, ...], str]:
        if name not in self._meta:
            with safe_open(str(self.dir / self.weight_map[name]), "pt", device="cpu") as f:
                sl = f.get_slice(name)
                self._meta[name] = (tuple(sl.get_shape()), sl.get_dtype())
        return self._meta[name]


def load_tensors(path: str | os.PathLike) -> dict[str, torch.Tensor]:
    return load_file(str(path), device="cpu") if path else {}
