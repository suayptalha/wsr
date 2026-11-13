"""New-generation architectures on tiny random models: family detection, anatomy (MoE, fused experts, hybrid
attention, post-norm, multimodal wrapper, MTP), the standard pool, frozen tensors, the Transformers backend and
`wsr program`. Qwen3.5 and Muse-Glimmer are not in the installed Transformers, so they are built as synthetic
checkpoints with the tensor names of the real models (as used in the paper)."""
from __future__ import annotations

import glob
import json
import re
import shutil
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from wsr.anatomy import Anatomy, AnatomyError
from wsr.checkpoint import Checkpoint
from wsr.config import model_dir, normalize_config
from wsr.families import detect
from wsr.ops import apply_ops, validate_edit
from wsr.pools import standard_pool

transformers = pytest.importorskip("transformers")
TOK = glob.glob(str(Path.home() / ".cache/huggingface/hub/models--gpt2/snapshots/*/tokenizer.json"))
V = 50257 if TOK else 128
B = dict(hidden_size=32, intermediate_size=64, num_hidden_layers=4, num_attention_heads=4, num_key_value_heads=2,
         vocab_size=V, head_dim=8)
MOE = dict(moe_intermediate_size=16, num_experts_per_tok=2)
ARCHS = {
    "qwen3": B,
    "qwen3_moe": {**B, **MOE, "num_experts": 4, "decoder_sparse_step": 1, "mlp_only_layers": []},
    "qwen3_next": {**B, **MOE, "num_experts": 4, "shared_expert_intermediate_size": 16, "linear_num_value_heads": 4,
                   "linear_num_key_heads": 2, "linear_key_head_dim": 8, "linear_value_head_dim": 8, "full_attention_interval": 4},
    "gpt_oss": {**B, "num_local_experts": 4, "num_experts_per_tok": 2, "layer_types": ["sliding_attention", "full_attention"] * 2},
    "gemma3_text": B,
    "llama4_text": {**B, "intermediate_size_mlp": 64, "num_local_experts": 4, "interleave_moe_layer_step": 2},
    "deepseek_v3": {**B, **MOE, "n_routed_experts": 4, "n_shared_experts": 1, "first_k_dense_replace": 1, "kv_lora_rank": 16,
                    "q_lora_rank": 16, "qk_rope_head_dim": 8, "qk_nope_head_dim": 8, "v_head_dim": 8, "n_group": 1, "topk_group": 1},
    "glm4_moe": {**B, **MOE, "n_routed_experts": 4, "n_shared_experts": 1, "first_k_dense_replace": 1, "n_group": 1, "topk_group": 1},
    "olmo3": B,
}
# family, MoE layers, fused experts, hybrid attention, post-norm
EXPECT = {"qwen3": ("qwen3", 0, None, False, False), "qwen3_moe": ("qwen3-moe", 4, False, False, False),
          "qwen3_next": ("qwen3-next", 4, False, True, False), "gpt_oss": ("gpt-oss", 4, True, False, False),
          "gemma3_text": ("gemma3", 0, None, False, True), "llama4_text": ("llama4", 2, True, False, False),
          "deepseek_v3": ("deepseek-v3", 3, False, False, False), "glm4_moe": ("glm4-moe", 3, False, False, False),
          "olmo3": ("olmo", 0, None, False, True),
          "qwen3.5": ("qwen3.5", 4, False, True, False), "muse": ("muse-glimmer", 0, None, False, True)}
ROLES = {"close": 5, "eos": 1, "letters": [10, 11, 12, 13], "digits": list(range(20, 30)), "unused": [60, 61]}


def _tok(d: Path):
    if TOK:
        snap = Path(TOK[0]).parent
        for f in ("tokenizer.json", "vocab.json", "merges.txt", "tokenizer_config.json"):
            if (snap / f).exists():
                shutil.copy(snap / f, d / f)


def _save_hf(arch: str, d: Path):
    from transformers import AutoConfig, AutoModelForCausalLM
    torch.manual_seed(0)
    AutoModelForCausalLM.from_config(AutoConfig.for_model(arch, **ARCHS[arch])).save_pretrained(d, safe_serialization=True)
    _tok(d)


def _synthetic(name: str, d: Path):
    """Qwen3.5 (hybrid + vision tower + MTP, under model.language_model, centered norms) and Muse-Glimmer (sandwich
    post-norms, centered gains, under model.language_model), built from Transformers models of the same shape."""
    from transformers import AutoConfig, AutoModelForCausalLM
    torch.manual_seed(0)
    arch = "qwen3_next" if name == "qwen3.5" else "gemma3_text"
    m = AutoModelForCausalLM.from_config(AutoConfig.for_model(arch, **ARCHS[arch]))
    sd = {}
    for k, v in m.state_dict().items():
        k = k.replace("model.", "model.language_model.", 1) if k.startswith("model.") else k
        if re.search(r"norm\.weight$", k) and v.dim() == 1:
            v = 0.02 * torch.randn_like(v)                  # stored as the offset w of a 1 + w gain
        sd[k] = v.clone().contiguous()
    sd["model.visual.blocks.0.attn.proj.weight"] = torch.randn(16, 16)
    sd["model.visual.merger.mlp.0.weight"] = torch.randn(32, 16)
    if name == "qwen3.5":
        sd["mtp.layers.0.mlp.experts.0.down_proj.weight"] = torch.randn(32, 16)
        sd["mtp.fc.weight"] = torch.randn(32, 64)
    d.mkdir(parents=True, exist_ok=True)
    half = len(sd) // 2
    keys = sorted(sd)
    shards = {"model-00001-of-00002.safetensors": keys[:half], "model-00002-of-00002.safetensors": keys[half:]}
    wm = {}
    for f, ks in shards.items():
        save_file({k: sd[k].to(torch.bfloat16) for k in ks}, str(d / f))
        wm.update({k: f for k in ks})
    (d / "model.safetensors.index.json").write_text(json.dumps({"metadata": {}, "weight_map": wm}))
    mt, archs = ("qwen3_5", ["Qwen3_5ForConditionalGeneration"]) if name == "qwen3.5" else ("muse_glimmer", ["MuseGlimmerForCausalLM"])
    (d / "config.json").write_text(json.dumps({"model_type": mt, "architectures": archs, "vision_config": {"depth": 1},
                                               "text_config": {"num_hidden_layers": 4, "hidden_size": 32}}))


@pytest.fixture(scope="module", params=list(EXPECT))
def model(request, tmp_path_factory):
    name = request.param
    d = tmp_path_factory.mktemp(name.replace(".", "_"))
    if name in ARCHS:
        _save_hf(name, d)
    else:
        _synthetic(name, d)
    if name == "deepseek_v3":                            # an MTP block after the stack, as in the released checkpoints
        extra = {"model.layers.4.mlp.down_proj.weight": torch.randn(32, 64), "model.layers.4.input_layernorm.weight": torch.ones(32)}
        save_file(extra, str(d / "mtp.safetensors"))
        idx = d / "model.safetensors.index.json"
        if idx.exists():
            j = json.loads(idx.read_text())
            j["weight_map"].update({k: "mtp.safetensors" for k in extra})
            idx.write_text(json.dumps(j))
        else:
            with torch.no_grad():
                from safetensors import safe_open
                with safe_open(str(d / "model.safetensors"), "pt") as f:
                    allk = {k: f.get_tensor(k) for k in f.keys()}
            (d / "model.safetensors").unlink()
            save_file(allk, str(d / "model-00001.safetensors"))
            wm = {k: "model-00001.safetensors" for k in allk} | {k: "mtp.safetensors" for k in extra}
            idx.write_text(json.dumps({"metadata": {}, "weight_map": wm}))
    return name, d


def test_family_and_anatomy(model):
    name, d = model
    fam, n_moe, fused, hybrid, post = EXPECT[name]
    assert detect(d).name == fam
    a = Anatomy(Checkpoint(d))
    assert a.n_layers == 4 and a.hidden == 32
    assert "language_model" in a.layer_stem if name in ("qwen3.5", "muse") else True
    assert len(a.moe_layers) == n_moe and a.hybrid == hybrid and a.post_norm == post
    if n_moe:
        assert isinstance(a._experts(a.moe_layers[0]), str) == fused
        assert a.router(a.moe_layers[0]) is not None
    for l in a.layers:
        assert a.mlp_outs(l) and a.attn_out(l), (name, l)
        for n in a.mlp_outs(l) + [a.attn_out(l)]:
            assert a.out_axis(n) in (0, 1)
    if name == "deepseek_v3":
        assert a.extra_layers == [4]
    assert json.dumps(a.summary())


def test_standard_pool_is_architecture_aware(model):
    name, d = model
    ck = Checkpoint(d)
    a = Anatomy(ck)
    S = standard_pool(ck, a, ROLES, name)
    types = {s["type"] for s in S}
    assert types >= {"B", "C", "D", "E", "G", "F"}
    assert ("R" in types) == bool(a.moe_layers)
    frozen = [r"(^|\.)(visual)\.", r"(^|\.)mtp(\.|$)"]
    for s in S:
        full, compact, desc = apply_ops(ck, s, a)
        v = validate_edit(ck, full, frozen)
        assert v["ok"], (s, v["errors"])
        assert all(".layers.4." not in n for n in full)          # never the MTP block
        assert len(desc) < 6000
    if a.post_norm:                                            # scaling before a norm is a no-op: none in the pool
        assert not any(op["op"] == "scale" and op["factor"] not in (0.0, 1.001) for s in S for op in s["ops"])


def test_fused_expert_direction_removal(model):
    name, d = model
    ck = Checkpoint(d)
    a = Anatomy(ck)
    if not a.moe_layers or not isinstance(a._experts(a.moe_layers[0]), str):
        pytest.skip("no fused experts")
    n = a._experts(a.moe_layers[0])
    full, _, _ = apply_ops(ck, {"ops": [{"op": "orth", "tensors": [n], "u_rows": [5]}]}, a, dtype="float32")
    g = ck.get(a.final_norm).float()
    u = (1 + g if g.abs().mean() < 0.5 else g) * ck.get(a.unembed)[5].float()
    u = u / u.norm()
    W = full[n]
    r = torch.einsum("h,ehi->ei", u, W) if a.out_axis(n) == 0 else W @ u
    assert r.abs().max() < 1e-4


def test_family_settings_in_config(model):
    name, d = model
    cfg = normalize_config({"models": {"m": {"path": str(d)}, "fixed": {"path": str(d), "thinking": False}}})
    model_dir(cfg, "m")
    model_dir(cfg, "fixed")
    m, f = cfg["models"]["m"], cfg["models"]["fixed"]
    assert m["family"] == EXPECT[name][0] and f["thinking"] is False      # user settings win
    if name == "qwen3.5":
        assert m["thinking"] is True and m["hf_class"] == "image-text" and m["vllm"]["limit_mm_per_prompt"]
        assert any(re.search(p, "mtp.fc.weight") for p in m["frozen"])
        assert any(re.search(p, "model.visual.blocks.0.attn.proj.weight") for p in m["frozen"])
    if name == "gpt_oss":
        assert m["answer_format"] == "harmony" and m["skip_special_tokens"] is False
    if name == "muse":
        assert m["answer_format"] == "muse"


def test_quantized_checkpoint_is_refused(tmp_path):
    save_file({"lm_head.weight": torch.zeros(8, 4), "model.layers.0.mlp.experts.down_proj_blocks": torch.zeros(2, 4, 4, dtype=torch.uint8),
               "model.layers.0.mlp.experts.down_proj_scales": torch.zeros(2, 4, dtype=torch.uint8),
               "model.layers.0.self_attn.o_proj.weight": torch.zeros(4, 4), "model.norm.weight": torch.ones(4)},
              str(tmp_path / "model.safetensors"))
    with pytest.raises(AnatomyError, match="quantized"):
        Anatomy(Checkpoint(tmp_path))


# ---------------------------------------------------------------- running the models
RUNNABLE = ["qwen3", "qwen3_moe", "qwen3_next", "gpt_oss", "olmo3", "deepseek_v3"]


@pytest.mark.skipif(not TOK, reason="needs the gpt2 tokenizer")
@pytest.mark.parametrize("arch", RUNNABLE)
def test_hf_backend_patches_new_architectures(arch, tmp_path):
    from wsr.backends import HFBackend
    from wsr.programming import GenSettings
    _save_hf(arch, tmp_path)
    ck = Checkpoint(tmp_path)
    a = Anatomy(ck)
    be = HFBackend(tmp_path, {"hf": {"batch_size": 4}}, GenSettings(4, {"temperature": 0.0}, None))
    msgs = [[{"role": "user", "content": f"hello {i}"}] for i in range(3)]
    base = [o.text for o in be.generate(msgs, 0)]
    zero = {a.final_norm: torch.zeros_like(ck.get(a.final_norm))}
    if a.norm_label == "RMSNorm" and ck.get(a.final_norm).float().abs().mean() < 0.5:
        zero = {a.final_norm: -torch.ones_like(ck.get(a.final_norm))}      # centered gain: 1 + w = 0
    be.patch(zero)
    assert all(o.text == "!!!!" for o in be.generate(msgs, 0))           # all logits 0 -> token 0 every step
    n = a.mlp_outs(a.moe_layers[0] if a.moe_layers else 0)[0]
    be.patch({n: torch.zeros_like(ck.get(n))})                           # the previous patch is undone first
    be.patch({})
    assert [o.text for o in be.generate(msgs, 0)] == base
    lp = be.logprob(msgs[:1], [" world"])
    assert lp[0][1] >= 1 and lp[0][0] < 0
    be.close()


@pytest.mark.skipif(not TOK, reason="needs the gpt2 tokenizer")
def test_program_on_a_moe_model(tmp_path):
    import wsr
    from tests.test_programming import Bot
    _save_hf("qwen3_moe", tmp_path / "m")
    data = tmp_path / "goal.jsonl"
    data.write_text("".join(json.dumps({"prompt": f"Question {i}", "target": "!"}) + "\n" for i in range(16)))
    edit = json.dumps([{"op": "gain", "tensors": ["model.norm.weight"], "factor": 0.0, "centered": False}])
    bot = Bot([[("write_file", {"path": "edit.json", "content": edit}),
                ("write_file", {"path": "rationale.md", "content": "zero the final norm"})]])
    res = wsr.program({"base_model": str(tmp_path / "m"), "agent": {"name": "bot", "provider": "openai",
                       "base_url": "http://unused/v1", "model": "m"},
                       "goal": {"text": "print !", "data": str(data), "metric": "contains", "max_tokens": 3},
                       "rounds": 1, "passes": 0, "hours": 0.1, "output_dir": str(tmp_path / "out"),
                       "sandbox": {"runtime": "local"}, "models": {"base": {"path": str(tmp_path / "m"), "hf": {"batch_size": 16}}}},
                      clients=[bot])
    assert res.accepted_rounds == 1 and res.final_score == 1.0
    m = transformers.AutoModelForCausalLM.from_pretrained(res.model_dir)
    assert torch.count_nonzero(m.state_dict()["model.norm.weight"]) == 0
    assert json.loads((res.output_dir / "program.json").read_text())["base_model"]
