"""The general toolkit on small random models of common architectures: detection, the standard pool, every op."""
from __future__ import annotations

import pytest
import torch

from wsr.anatomy import Anatomy, token_roles
from wsr.checkpoint import Checkpoint
from wsr.ops import apply_ops, expand_compact, validate_edit
from wsr.pools import standard_pool

transformers = pytest.importorskip("transformers")

ARCHS = {
    "llama": dict(hidden_size=16, intermediate_size=32, num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1, vocab_size=64),
    "qwen2": dict(hidden_size=16, intermediate_size=32, num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1, vocab_size=64),
    "gemma2": dict(hidden_size=16, intermediate_size=32, num_hidden_layers=4, num_attention_heads=2, num_key_value_heads=1,
                   vocab_size=64, head_dim=8),
    "phi3": dict(hidden_size=16, intermediate_size=32, num_hidden_layers=4, num_attention_heads=2, vocab_size=64, pad_token_id=0),
    "gpt2": dict(n_embd=16, n_layer=4, n_head=2, vocab_size=64),
    "gpt_neox": dict(hidden_size=16, intermediate_size=32, num_hidden_layers=4, num_attention_heads=2, vocab_size=64),
}
EXPECT = {"llama": ("model.layers", "lm_head.weight", False, 0), "qwen2": ("model.layers", "lm_head.weight", False, 0),
          "gemma2": ("model.layers", "model.embed_tokens.weight", True, 0), "phi3": ("model.layers", "lm_head.weight", False, 0),
          "gpt2": ("transformer.h", "transformer.wte.weight", True, 1), "gpt_neox": ("gpt_neox.layers", "embed_out.weight", False, 0)}
ROLES = {"close": 5, "eos": 1, "letters": [10, 11, 12, 13], "digits": list(range(20, 30)), "unused": [60, 61]}


@pytest.fixture(scope="module", params=list(ARCHS))
def model(request, tmp_path_factory):
    from transformers import AutoConfig, AutoModelForCausalLM
    arch = request.param
    torch.manual_seed(0)
    m = AutoModelForCausalLM.from_config(AutoConfig.for_model(arch, **ARCHS[arch]))
    d = tmp_path_factory.mktemp(arch)
    m.save_pretrained(d, safe_serialization=True)
    return arch, d


def test_detection(model):
    arch, d = model
    a = Anatomy(Checkpoint(d))
    stem, unembed, tied, axis = EXPECT[arch]
    assert a.layer_stem == stem and a.n_layers == 4 and a.hidden == 16
    assert a.unembed == unembed and a.tied == tied
    assert all(a.mlp_out(l) for l in a.layers) and all(a.attn_out(l) for l in a.layers)
    assert a.out_axis(a.mlp_out(0)) == axis
    assert a.final_norm is not None
    assert a.norm_kinds()


def test_standard_pool_applies(model):
    arch, d = model
    ck = Checkpoint(d)
    a = Anatomy(ck)
    for size in ("small", "medium"):
        S = standard_pool(ck, a, ROLES, arch, {"size": size})
        assert len({s["id"] for s in S}) == len(S)
        assert {s["type"] for s in S} >= {"B", "C", "D", "E", "G", "F"}
        for s in S:
            full, compact, desc = apply_ops(ck, s, a)
            assert full and desc
            assert validate_edit(ck, full)["ok"], s
            back = expand_compact(ck, compact)
            for k in full:
                assert torch.equal(back[k], full[k])
            for k, T in full.items():
                assert T.dtype == ck.get(k).dtype


def test_orth_removes_direction(model):
    arch, d = model
    ck = Checkpoint(d)
    a = Anatomy(ck)
    names = [a.mlp_out(l) for l in a.layers]
    full, _, _ = apply_ops(ck, {"ops": [{"op": "orth", "tensors": names, "u_rows": [5]}]}, a, dtype="float32")
    g = ck.get(a.final_norm).float()
    g = 1 + g if g.abs().mean() < 0.5 else g
    u = g * ck.get(a.unembed)[5].float()
    u = u / u.norm()
    for n in names:
        W = full[n].float()
        r = (u @ W) if a.out_axis(n) == 0 else (W @ u)          # what the block can still write along u
        assert r.abs().max() < 1e-4


def test_noop_controls_are_tiny(model):
    arch, d = model
    ck = Checkpoint(d)
    a = Anatomy(ck)
    for s in standard_pool(ck, a, ROLES, arch):
        if s["type"] == "F":
            full = apply_ops(ck, s, a, dtype="float32")[0]
            v = validate_edit(ck, full)
            assert v["rel_norm"] < 0.02 or all(t["values_changed"] <= 16 for t in v["tensors"]), s


def test_validate_rejects_bad_edits(model):
    _, d = model
    ck = Checkpoint(d)
    n = ck.names()[0]
    assert not validate_edit(ck, {"nope": torch.zeros(1)})["ok"]
    assert not validate_edit(ck, {n: torch.zeros(1, 2, 3)})["ok"]
    bad = ck.get(n).clone().float()
    bad.view(-1)[0] = float("nan")
    assert not validate_edit(ck, {n: bad})["ok"]
    assert not validate_edit(ck, {n: ck.get(n)}, frozen=[".*"])["ok"]


def test_token_roles_gpt2_tokenizer(tmp_path):
    import glob
    import shutil
    snaps = glob.glob(str(__import__("pathlib").Path.home() / ".cache/huggingface/hub/models--gpt2/snapshots/*/tokenizer.json"))
    if not snaps:
        pytest.skip("gpt2 tokenizer not cached")
    shutil.copy(snaps[0], tmp_path / "tokenizer.json")
    r = token_roles(tmp_path)
    assert len(r["letters"]) == 10 and len(r["digits"]) == 10
    assert token_roles(tmp_path, {"close": 7})["close"] == 7
