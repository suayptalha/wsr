"""Architecture introspection: find the tensors an edit can target, for any decoder-only checkpoint.

Edits are written in terms of roles (the matrix that writes an MLP block into the residual stream, the unembedding,
a block's norm gains, ...). ``Anatomy`` resolves those roles to tensor names by reading the checkpoint's own names and
shapes, so the same pool generator works across architectures without a per-model table: dense (Llama, Qwen3, Gemma,
Mistral, Phi, GPT-2), mixture-of-experts with one matrix per expert (Qwen3-MoE, DeepSeek-V3, GLM-4.5) or fused into one
tensor (gpt-oss, Llama 4), hybrid linear/full attention (Qwen3-Next, Qwen3.5), norms after each sublayer (Gemma 3,
OLMo, Muse-Glimmer), and multimodal wrappers (``model.language_model.*``). Anything it gets wrong can be overridden
under ``models.<key>.anatomy`` in the config.
"""
from __future__ import annotations

import json
import re
from collections import Counter, defaultdict
from pathlib import Path

from .checkpoint import Checkpoint

LAYER_RX = re.compile(r"^(?P<stem>(?:.*\.)?(?:layers|h|blocks|block|layer))\.(?P<i>\d+)\.(?P<rest>.+)$")
MLP_OUT = ["mlp.down_proj", "mlp.c_proj", "mlp.dense_4h_to_h", "mlp.fc2", "mlp.fc_out", "feed_forward.w2", "ffn.lin2",
           "mlp.w2", "mlp.proj", "fc2", "feed_forward.down_proj", "ffn.w2", "ffn.down_proj"]
MOE_MLP = ["mlp", "feed_forward", "block_sparse_moe", "moe", "ffn"]
EXPERT_OUT = ["down_proj", "w2", "fc2"]
SHARED_OUT = ["shared_expert.down_proj", "shared_experts.down_proj", "shared_mlp.down_proj", "shared_expert.w2",
              "shared_experts.w2"]
ROUTER = ["gate", "router", "gate.wg"]
QUANT_RX = re.compile(r"(_blocks|_scales|weight_scale(_inv)?|\.qweight|\.qzeros|\.scales|weight_packed)$")
ATTN_OUT = ["self_attn.o_proj", "self_attn.out_proj", "linear_attn.out_proj", "attn.c_proj", "attn.attention.out_proj",
            "attention.dense", "self_attention.dense", "attn.out_proj", "attention.o_proj", "attn.o_proj", "attn.proj",
            "attn.wo_b", "attn.wo"]
EMBED_RX = re.compile(r"(?:^|\.)(embed_tokens|wte|word_embeddings|embed_in|tok_embeddings|shared)\.weight$")
UNEMBED = ["lm_head.weight", "embed_out.weight", "lm_head.decoder.weight", "head.weight", "output.weight"]
FINAL_NORM_RX = re.compile(r"(?:^|\.)(norm|ln_f|final_layer_norm|final_layernorm|norm_f|ln_final)\.weight$")
CONV1D_TYPES = {"gpt2", "openai-gpt", "imagegpt", "decision_transformer"}


class AnatomyError(RuntimeError):
    pass


class Anatomy:
    """Roles -> tensor names for one checkpoint.

    Attributes: ``n_layers``, ``hidden``, ``unembed`` (the [vocab, hidden] matrix that produces logits), ``tied``
    (the unembedding is also the input embedding), ``final_norm`` (gain of the last norm, or None),
    ``norm_label`` ("RMSNorm" or "LayerNorm").
    """

    def __init__(self, ck: Checkpoint, overrides: dict | None = None):
        self.ck = ck
        ov = dict(overrides or {})
        names = ck.names()
        cfg = ck.config()
        self.model_type = cfg.get("model_type") or (cfg.get("text_config") or {}).get("model_type")

        # ---- unembedding (and whether it is tied to the input embedding)
        embeds = [n for n in names if EMBED_RX.search(n) and len(ck.shape(n)) == 2]
        embed = max(embeds, key=lambda n: ck.shape(n)[0]) if embeds else None
        head = next((n for n in UNEMBED if n in ck and len(ck.shape(n)) == 2), None)
        tie_cfg = cfg.get("tie_word_embeddings", (cfg.get("text_config") or {}).get("tie_word_embeddings"))
        if "unembed" in ov:
            self.unembed = ov["unembed"]
            self.tied = bool(ov.get("tied", False))
        elif head and not (tie_cfg and embed and ck.shape(embed) == ck.shape(head)):
            self.unembed, self.tied = head, False
        elif embed:
            self.unembed, self.tied = embed, True      # one matrix serves as embedding and unembedding
        elif head:
            self.unembed, self.tied = head, False
        else:
            raise AnatomyError("could not find the unembedding matrix; set models.<key>.anatomy.unembed")
        if self.unembed not in ck:
            raise AnatomyError(f"unembedding `{self.unembed}` is not a tensor of this checkpoint")
        self.embed = embed
        self.vocab, self.hidden = ck.shape(self.unembed)

        # ---- the stack of decoder blocks: the layer stem whose matrices live in the residual width
        stems: dict[str, dict[int, list[str]]] = defaultdict(lambda: defaultdict(list))
        for n in names:
            m = LAYER_RX.match(n)
            if m:
                stems[m["stem"]][int(m["i"])].append(n)
        if "layer_stem" in ov:
            stem = ov["layer_stem"]
        else:
            def score(s):
                mats = [n for ns in stems[s].values() for n in ns if len(ck.shape(n)) == 2]
                fit = sum(1 for n in mats if self.hidden in ck.shape(n))
                return (fit / max(1, len(mats)) > 0.5, "language_model" in s or "decoder" in s, len(stems[s]))
            if not stems:
                raise AnatomyError("no repeated layer blocks found; set models.<key>.anatomy.layer_stem")
            stem = max(stems, key=score)
        if stem not in stems:
            raise AnatomyError(f"layer stem `{stem}` matches no tensors")
        self.layer_stem = stem
        n_cfg = cfg.get("num_hidden_layers") or (cfg.get("text_config") or {}).get("num_hidden_layers")
        idx = sorted(stems[stem])
        if isinstance(n_cfg, int) and 0 < n_cfg < len(idx) and "n_layers" not in ov:
            # extra blocks after the stack (e.g. DeepSeek's multi-token-prediction layer) are not part of the model
            self.extra_layers = [i for i in idx if i >= n_cfg]
            idx = [i for i in idx if i < n_cfg]
        else:
            self.extra_layers = []
        quant = [n for i in idx[:2] for n in stems[stem][i] if QUANT_RX.search(n)]
        if quant and not ov.get("allow_quantized"):
            raise AnatomyError(f"this checkpoint stores quantized weights (e.g. `{quant[0]}`), which cannot be edited in "
                               "place. Use a bf16/fp16 release of the model (or dequantize it first).")
        self.layers = {i: sorted(stems[stem][i]) for i in idx}
        self.n_layers = len(self.layers)
        self._rest = {i: {n[len(f"{stem}.{i}."):]: n for n in ns} for i, ns in self.layers.items()}
        self._ov = ov

        # ---- final norm
        root = stem.rsplit(".", 1)[0] + "." if "." in stem else ""
        if "final_norm" in ov:
            self.final_norm = ov["final_norm"]
        else:
            cands = [n for n in names if FINAL_NORM_RX.search(n) and not LAYER_RX.match(n)
                     and ck.shape(n) == (self.hidden,)]
            cands.sort(key=lambda n: (not n.startswith(root), len(n)))
            self.final_norm = cands[0] if cands else None
        if "norm_label" in ov:
            self.norm_label = ov["norm_label"]
        else:
            fn = self.final_norm
            self.norm_label = "LayerNorm" if fn and fn[:-len("weight")] + "bias" in ck else "RMSNorm"
        # norms that act on a sublayer's OUTPUT before it joins the residual stream (Gemma, OLMo, Muse-Glimmer):
        # scaling the matrix that feeds such a norm is undone by it, so pools scale the norm gain instead
        self.post_norm = ov.get("post_norm", "post_feedforward_layernorm" in self.norm_kinds())
        self.moe_layers = [l for l in self.layers if self._experts(l)]
        self.n_experts = max((len(self._experts(l)) if isinstance(self._experts(l), list) else self.ck.shape(self._experts(l))[0]
                              for l in self.moe_layers), default=0)
        self.hybrid = len({(self._short(self.attn_out(l)) or "").split(".")[0] for l in self.layers}) > 1

    # ---- per-layer roles
    def _find(self, l: int, table: list[str], key: str) -> str | None:
        if key in self._ov:
            return self._ov[key].format(l=l)
        rest = self._rest[l]
        for stemname in table:
            if f"{stemname}.weight" in rest:
                return rest[f"{stemname}.weight"]
        return None

    def _experts(self, l: int):
        """Expert output matrices of a MoE block: a list of names (one per expert), the name of one fused 3-D tensor,
        or None for a dense block."""
        rest = self._rest[l]
        for mlp in MOE_MLP:
            for out in EXPERT_OUT:
                fused = f"{mlp}.experts.{out}"
                for key in (fused, fused + ".weight"):
                    if key in rest and len(self.ck.shape(rest[key])) == 3:
                        return rest[key]
                rx = re.compile(rf"^{re.escape(mlp)}\.experts\.(\d+)\.{re.escape(out)}\.weight$")
                hits = sorted(((int(m[1]), n) for r, n in rest.items() if (m := rx.match(r))))
                if hits:
                    return [n for _, n in hits]
        return None

    def shared_expert(self, l: int) -> str | None:
        rest = self._rest[l]
        for mlp in MOE_MLP:
            for sh in SHARED_OUT:
                if f"{mlp}.{sh}.weight" in rest:
                    return rest[f"{mlp}.{sh}.weight"]
        return None

    def router(self, l: int) -> str | None:
        """The [n_experts, hidden] matrix that scores experts in a MoE block."""
        rest = self._rest[l]
        for mlp in MOE_MLP:
            for r in ROUTER:
                n = rest.get(f"{mlp}.{r}.weight")
                if n and len(self.ck.shape(n)) == 2 and self.ck.shape(n)[1] == self.hidden and self.ck.shape(n)[0] > 1:
                    return n
        return None

    def mlp_outs(self, l: int) -> list[str]:
        """Every tensor that writes block l's MLP (all experts and any shared expert) into the residual stream."""
        if "mlp_out" in self._ov:
            return [self._ov["mlp_out"].format(l=l)]
        e = self._experts(l)
        if e is None:
            d = self._find(l, MLP_OUT, "mlp_out")
            return [d] if d else []
        out = list(e) if isinstance(e, list) else [e]
        sh = self.shared_expert(l)
        return out + ([sh] if sh else [])

    def mlp_out(self, l: int) -> str | None:
        """The (first) matrix that writes block l's MLP output into the residual stream."""
        outs = self.mlp_outs(l)
        return outs[0] if outs else None

    def attn_out(self, l: int) -> str | None:
        """The matrix that writes block l's attention (or other mixer) output into the residual stream."""
        return self._find(l, ATTN_OUT, "attn_out")

    def norms(self, l: int) -> dict[str, str]:
        """Norm gains of block l: short name -> tensor name (only gains of the residual width)."""
        out = {}
        for rest, n in self._rest[l].items():
            if (rest.endswith(".weight") and self.ck.shape(n) == (self.hidden,) and "." not in rest[:-len(".weight")]
                    and re.search(r"norm|(^|\.)ln_?\d*(\.|$)", rest)):
                out[rest[:-len(".weight")]] = n
        return out

    def norm_kinds(self) -> list[str]:
        """Norm gains present in every block (e.g. input_layernorm, post_attention_layernorm)."""
        c = Counter(k for l in self.layers for k in self.norms(l))
        return sorted(k for k, v in c.items() if v == self.n_layers)

    def out_axis(self, name: str) -> int:
        """Which axis of a matrix is the residual stream it writes to: 0 for [hidden, in] (nn.Linear), 1 for
        [in, hidden] (GPT-2's Conv1D). Square matrices follow the block's MLP, then the model type."""
        if "out_axis" in self._ov:
            return int(self._ov["out_axis"])
        shape = self.ck.shape(name)
        if len(shape) not in (2, 3):
            raise AnatomyError(f"`{name}` is not a matrix")
        a, b = shape[-2:]                    # fused experts: [n_experts, a, b], the same orientation per expert
        if a == self.hidden and b != self.hidden:
            return 0
        if b == self.hidden and a != self.hidden:
            return 1
        m = LAYER_RX.match(name)
        if m and m["stem"] == self.layer_stem:
            mo = self.mlp_out(int(m["i"]))
            if mo and mo != name and self.ck.shape(mo)[-2] != self.ck.shape(mo)[-1]:
                return self.out_axis(mo)
        return 1 if self.model_type in CONV1D_TYPES else 0

    def summary(self) -> dict:
        l0 = min(self.layers)
        return {"model_type": self.model_type, "layers": self.n_layers, "hidden": self.hidden, "vocab": self.vocab,
                "layer_stem": self.layer_stem, "unembed": self.unembed, "tied_embeddings": self.tied,
                "final_norm": self.final_norm, "norm": self.norm_label,
                "mlp_out": self.mlp_out(l0), "attn_out": sorted({self._short(self.attn_out(l)) for l in self.layers} - {None}),
                "mlp_out_axis": self.out_axis(self.mlp_out(l0)) if self.mlp_out(l0) else None,
                "block_norms": self.norm_kinds(), "post_norm": self.post_norm,
                "moe": ({"layers": len(self.moe_layers), "experts": self.n_experts,
                         "fused": isinstance(self._experts(self.moe_layers[0]), str),
                         "shared_expert": bool(self.shared_expert(self.moe_layers[0])),
                         "router": self._short(self.router(self.moe_layers[0]))} if self.moe_layers else None),
                "hybrid_attention": self.hybrid, "extra_layers_ignored": self.extra_layers}

    def _short(self, name: str | None) -> str | None:
        m = LAYER_RX.match(name) if name else None
        return m["rest"] if m else name


# ---------------------------------------------------------------- vocabulary rows with a known meaning
def token_roles(model_dir: str | Path, overrides: dict | None = None, close_names: tuple[str, ...] = ()) -> dict:
    """Vocabulary rows that edits can target by meaning, read from the tokenizer.

    Keys (each present only if found): ``close`` (the token that ends a reasoning block), ``eos`` (end of turn),
    ``letters`` (the option letters A-J), ``digits``, ``unused`` (reserved rows that are never generated).
    """
    d = Path(model_dir)
    roles: dict = {}
    tok = None
    if (d / "tokenizer.json").exists():
        from tokenizers import Tokenizer
        tok = Tokenizer.from_file(str(d / "tokenizer.json"))
        vocab = tok.get_vocab(with_added_tokens=True)
        one = lambda s: (lambda ids: ids[0] if len(ids) == 1 else None)(tok.encode(s, add_special_tokens=False).ids)  # noqa: E731
    else:
        try:
            from transformers import AutoTokenizer
            hf = AutoTokenizer.from_pretrained(str(d))
        except Exception:  # noqa: BLE001
            return dict(overrides or {})
        vocab = hf.get_vocab()
        one = lambda s: (lambda ids: ids[0] if len(ids) == 1 else None)(hf.encode(s, add_special_tokens=False))  # noqa: E731
    for name in tuple(close_names) + ("</think>", "<|end_of_thought|>", "</reasoning>", "</thought>"):
        if name in vocab:
            roles["close"] = vocab[name]
            break
    eos = None
    for f in ("generation_config.json", "config.json"):
        p = d / f
        if p.exists():
            e = json.loads(p.read_text()).get("eos_token_id")
            if e is not None:
                eos = e[0] if isinstance(e, list) else e
                break
    if eos is not None:
        roles["eos"] = int(eos)
    letters = [one(c) for c in "ABCDEFGHIJ"]
    if all(x is not None for x in letters):
        roles["letters"] = letters
    digits = [one(c) for c in "0123456789"]
    if all(x is not None for x in digits):
        roles["digits"] = digits
    unused = sorted(i for t, i in vocab.items() if re.search(r"reserved|unused|<\|extra_\d+\|>|\[unused\d+\]", t))
    if unused:
        k = len(unused)
        roles["unused"] = sorted({unused[k // 4], unused[k // 2], unused[min(k - 1, 3 * k // 4)]})
    roles.update(overrides or {})
    return roles
