"""Models studied and the vocabulary rows used by the edits."""
from __future__ import annotations

# Keys used throughout the code: "27b" = Qwen3.8-27B, "9b" = Qwen3.5-9B, "muse30b" = Muse-Glimmer-30B.
MODELS = {
    "27b": {"name": "Qwen3.8-27B", "layers": 64},
    "9b": {"name": "Qwen3.5-9B", "layers": 32},
    "muse30b": {"name": "Muse-Glimmer-30B", "layers": 52},
}

# Vocabulary rows per model: "close" ends the reasoning block, "wait_cap"/"wait_low" are "Wait"/"wait",
# "letters" are the ten option letters, "unused" are rows that are never generated.
TOKENS = {
    "27b": {"close": 248069, "wait_cap": 13428, "wait_low": 3655, "letters": list(range(32, 42)), "unused": [248200, 248300]},
    "9b": {"close": 248069, "wait_cap": 13428, "wait_low": 3655, "letters": list(range(32, 42)), "unused": [248200, 248300]},
    "muse30b": {"close": 200007, "wait_cap": 30390, "wait_low": 7840, "letters": list(range(45, 55)), "unused": [201000, 201500]},
}

METRICS = ["thinking_length", "math_accuracy", "mc_accuracy", "sycophancy", "refusal"]
LM_HEAD = "lm_head.weight"
FINAL_NORM = "model.language_model.norm.weight"


def mlp(l: int) -> str:
    return f"model.language_model.layers.{l}.mlp.down_proj.weight"


def oproj(l: int) -> str:
    return f"model.language_model.layers.{l}.self_attn.o_proj.weight"


def lin_out(l: int) -> str:
    return f"model.language_model.layers.{l}.linear_attn.out_proj.weight"


def norm(l: int, kind: str) -> str:
    return f"model.language_model.layers.{l}.{kind}.weight"


def attn_out(l: int) -> str:
    """Qwen hybrid layout: full attention in every fourth layer, Gated DeltaNet otherwise."""
    return oproj(l) if l % 4 == 3 else lin_out(l)
