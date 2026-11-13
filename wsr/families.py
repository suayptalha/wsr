"""Model families: the settings a checkpoint needs to be measured and edited correctly, detected from its
``config.json`` (``model_type`` / ``architectures``).

A family fills in what the config leaves unset for a model: whether the chat template takes ``enable_thinking``,
how the answer is separated from the reasoning, which special tokens must be kept, which tensors must never be
edited (vision tower, multi-token-prediction head), extra vLLM arguments, and the vLLM parsers to use when the same
model is served as an agent. Set ``family:`` on a model to force one, or ``family: generic`` to switch them off.

Architecture facts that change what an edit does (mixture-of-experts layers, hybrid linear attention, norms after
each sublayer) are not family settings: ``wsr.anatomy`` reads them from the tensors themselves.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from pathlib import Path

VISION = r"(^|\.)(visual|vision\w*|multi_modal_projector|mm_projector|audio\w*)\."
MTP = r"(^|\.)mtp(\.|$)"
NO_MEDIA = {"limit_mm_per_prompt": {"image": 0, "video": 0}}


@dataclass
class Family:
    name: str
    match: str                         # regex on model_type / architectures
    description: str
    thinking: bool | None = None       # enable_thinking for the chat template (None: do not pass it)
    answer_format: str = "think"       # think | harmony | muse | plain
    skip_special_tokens: bool = True
    frozen: list[str] = field(default_factory=list)
    vllm: dict = field(default_factory=dict)
    hf_class: str = "causal"           # causal | image-text (multimodal checkpoints)
    close_tokens: tuple[str, ...] = ("</think>",)
    serve: dict = field(default_factory=dict)   # vLLM parsers for serving the model as an agent

    def defaults(self) -> dict:
        return {"thinking": self.thinking, "answer_format": self.answer_format,
                "skip_special_tokens": self.skip_special_tokens, "frozen": list(self.frozen),
                "vllm": dict(self.vllm), "hf_class": self.hf_class}


QWEN_SERVE = {"tool_call_parser": "qwen3_coder", "reasoning_parser": "qwen3"}
FAMILIES = [
    Family("qwen3.5", r"qwen3_5|qwen3\.5|qwen3_8|qwen3\.8|Qwen3_5|Qwen3_8",
           "Qwen3.5 / Qwen3.8 (dense and MoE): hybrid Gated DeltaNet + full attention (every 4th layer), vision tower, "
           "MTP head, centered (1 + w) norms, thinking", thinking=True, frozen=[VISION, MTP], vllm=dict(NO_MEDIA),
           hf_class="image-text", serve=QWEN_SERVE),
    Family("qwen3-next", r"qwen3_next|Qwen3Next", "Qwen3-Next: hybrid Gated DeltaNet + gated attention, MoE with a "
           "shared expert, MTP head", frozen=[MTP], serve=QWEN_SERVE),
    Family("qwen3-vl", r"qwen3_vl|Qwen3VL", "Qwen3-VL: Qwen3 language model (dense or MoE) inside a vision-language "
           "wrapper", frozen=[VISION], vllm=dict(NO_MEDIA), hf_class="image-text", serve=QWEN_SERVE),
    Family("qwen3-moe", r"qwen3_moe|Qwen3Moe", "Qwen3 MoE: routed experts, qk-norm, hybrid thinking", thinking=True,
           serve=QWEN_SERVE),
    Family("qwen3", r"^qwen3$|Qwen3ForCausalLM", "Qwen3 dense: qk-norm, hybrid thinking", thinking=True, serve=QWEN_SERVE),
    Family("muse-glimmer", r"muse_glimmer|MuseGlimmer", "Muse-Glimmer: a norm after every attention and MLP output (centered 1 + w gains), "
           "qk-norm, logit softcap; reasoning in to=self channels ending with <|eom|>", answer_format="muse",
           skip_special_tokens=False, frozen=[VISION, MTP], vllm=dict(NO_MEDIA), close_tokens=("<|eom|>", "<|end|>"),
           serve={"tool_call_parser": "muse_glimmer", "reasoning_parser": "muse_glimmer"}),
    Family("gpt-oss", r"gpt_oss|GptOss", "gpt-oss: MoE with fused expert tensors and attention sinks; harmony chat "
           "format (analysis / final channels). Edit a bf16 checkpoint: MXFP4 weights cannot be edited in place",
           answer_format="harmony", skip_special_tokens=False, close_tokens=("<|end|>",)),
    Family("gemma3", r"gemma3|Gemma3|gemma4|Gemma4", "Gemma 3 / 4: norms before and after every sublayer (sandwich), centered gains, "
           "qk-norm, tied embeddings", frozen=[VISION], vllm=dict(NO_MEDIA)),
    Family("llama4", r"llama4|Llama4", "Llama 4: interleaved dense and MoE layers, fused experts plus a shared expert",
           frozen=[VISION], vllm=dict(NO_MEDIA)),
    Family("deepseek-v3", r"deepseek_v3|deepseek_v32|DeepseekV3|DeepseekV32|kimi_k2", "DeepSeek-V3 / V3.2 / Kimi K2: "
           "multi-head latent attention, routed + shared experts, MTP layer after the last block. FP8 checkpoints must "
           "be converted to bf16 first", frozen=[MTP]),
    Family("deepseek-v4", r"deepseek_v4|DeepseekV4", "DeepSeek-V4: compressed sparse attention, routed + shared experts, "
           "hyper-connections (several residual streams mixed per block), MTP. FP8/FP4 checkpoints must be converted "
           "to bf16 first", frozen=[MTP]),
    Family("glm4-moe", r"glm4_moe|Glm4Moe", "GLM-4.5 / 4.6: routed + shared experts, MTP layer, hybrid thinking",
           thinking=True, frozen=[MTP]),
    Family("olmo", r"olmo2|olmo3|Olmo2|Olmo3|exaone4|Exaone4", "OLMo 2/3, EXAONE 4: norms after each sublayer "
           "(post-norm), qk-norm"),
    Family("generic", r".*", "any other decoder-only model: no family-specific settings", answer_format="think"),
]
BY_NAME = {f.name: f for f in FAMILIES}


def read_config(model_dir: str | Path) -> dict:
    p = Path(model_dir) / "config.json"
    return json.loads(p.read_text()) if p.exists() else {}


def detect(model_dir: str | Path | None = None, config: dict | None = None) -> Family:
    cfg = config if config is not None else read_config(model_dir)
    keys = [str(cfg.get("model_type", "")), str((cfg.get("text_config") or {}).get("model_type", ""))]
    keys += [str(a) for a in cfg.get("architectures") or []]
    for f in FAMILIES[:-1]:
        if any(k and re.search(f.match, k) for k in keys):
            return f
    return BY_NAME["generic"]


def has_vision(conf: dict) -> bool:
    return any(k in conf for k in ("vision_config", "visual", "vision_tower", "mm_projector_type"))


def get(name: str) -> Family:
    if name not in BY_NAME:
        raise SystemExit(f"wsr: unknown family `{name}`; one of {sorted(BY_NAME)}")
    return BY_NAME[name]


def apply(model_cfg: dict, model_dir: str | Path) -> dict:
    """Fill the family's settings into a model's config entry, keeping every value the user (or a paper preset)
    set. Returns the entry (also modified in place)."""
    if model_cfg.get("_family_applied"):
        return model_cfg
    conf = read_config(model_dir)
    fam = get(model_cfg["family"]) if model_cfg.get("family") not in (None, "auto") else detect(config=conf)
    fixed = set(model_cfg.get("_fixed") or [])
    defaults = fam.defaults()
    if not has_vision(conf):                     # a text-only checkpoint of a multimodal family
        defaults["vllm"].pop("limit_mm_per_prompt", None)
        defaults["hf_class"] = "causal"
    for k, v in defaults.items():
        if k in fixed:
            continue
        if k == "frozen":
            model_cfg["frozen"] = sorted(set(model_cfg.get("frozen") or []) | set(v))
        elif k == "vllm":
            model_cfg["vllm"] = {**v, **(model_cfg.get("vllm") or {})}
        else:
            model_cfg[k] = v
    model_cfg["family"] = fam.name
    model_cfg["_family_applied"] = True
    return model_cfg


def serve_command(fam: Family, path: str, name: str, vision: bool = True) -> str:
    """A `vllm serve` line that exposes the model as a tool-calling agent endpoint."""
    parts = [f"vllm serve {path}", f"--served-model-name {name}", "--enable-auto-tool-choice"]
    if fam.serve.get("tool_call_parser"):
        parts.append(f"--tool-call-parser {fam.serve['tool_call_parser']}")
    if fam.serve.get("reasoning_parser"):
        parts.append(f"--reasoning-parser {fam.serve['reasoning_parser']}")
    if fam.vllm.get("limit_mm_per_prompt") and vision:
        parts.append("--limit-mm-per-prompt '{\"image\": 0, \"video\": 0}'")
    return " \\\n    ".join(parts)
