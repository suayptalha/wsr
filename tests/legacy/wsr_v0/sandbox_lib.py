r"""Helper available to reader agents inside the sandbox (CPU only).

    from wsr.sandbox_lib import W
    W.names(r"layers\.15\.mlp")     # tensor names matching a regex
    W.get(name)                      # original tensor (torch, CPU)
    W.info(name)                     # shape, dtype and norm statistics
"""
from __future__ import annotations

import os
import re

import torch

from .checkpoint import Checkpoint

TARGET_DIR = os.environ.get("WSR_TARGET_DIR", "/weights/target")


class _Weights:
    def __init__(self, model_dir: str):
        self.ckpt = Checkpoint(model_dir)

    def names(self, pattern: str = "") -> list[str]:
        rx = re.compile(pattern) if pattern else None
        return [n for n in self.ckpt.names() if not rx or rx.search(n)]

    def get(self, name: str) -> torch.Tensor:
        return self.ckpt.get(name)

    def info(self, name: str) -> dict:
        t = self.get(name).float()
        return {"shape": list(t.shape), "dtype": self.ckpt.meta(name)[1], "fro": t.norm().item(),
                "mean": t.mean().item(), "std": t.std().item(), "absmax": t.abs().max().item()}


W = _Weights(TARGET_DIR)
