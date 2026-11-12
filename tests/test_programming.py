"""Weight-space programming as a tool, end to end on CPU: goal data, attempts, acceptance, rounds, export.

The target is a tiny random GPT-2 (saved in several shards) with the real GPT-2 tokenizer. Zeroing the final
LayerNorm gain makes every logit 0, so greedy decoding emits token 0 ("!") forever: an edit with a known,
deterministic effect that a scripted agent can submit."""
from __future__ import annotations

import glob
import json
import math
import os
import shutil
from pathlib import Path

import pytest
import torch

import wsr
from wsr.programming import load_program, paired_gain
from wsr.providers import Reply, ToolCall

transformers = pytest.importorskip("transformers")
TOK = glob.glob(str(Path.home() / ".cache/huggingface/hub/models--gpt2/snapshots/*/tokenizer.json"))
pytestmark = pytest.mark.skipif(not TOK, reason="needs the gpt2 tokenizer in the Hugging Face cache")
LN_F = "transformer.ln_f.weight"
ZERO_LN_F = json.dumps([{"op": "gain", "tensors": [LN_F], "factor": 0.0, "centered": False}])


class Bot:
    """A scripted agent: a list of tool calls per turn, then a final message."""

    def __init__(self, turns):
        self.replies = [Reply("", [ToolCall(f"c{i}{j}", n, a) for j, (n, a) in enumerate(t)], "tool")
                        for i, t in enumerate(turns)] + [Reply("Submitted.")]
        self.outputs = []

    start = lambda self, s, u: [{"role": "user", "content": u}]  # noqa: E731
    add_user = lambda self, h, t: h.append({"role": "user", "content": t})  # noqa: E731
    add_reply = lambda self, h, r: h.append({"role": "assistant", "content": r.text})  # noqa: E731
    trim = lambda self, h, keep_last=12: False  # noqa: E731

    def add_tool_results(self, h, results):
        self.outputs += [(c.name, t, err) for c, t, err in results]

    def chat(self, h, tools=None, max_tokens=None):
        return self.replies.pop(0) if self.replies else Reply("I give up.")


@pytest.fixture(scope="module")
def base(tmp_path_factory):
    from transformers import GPT2Config, GPT2LMHeadModel
    root = tmp_path_factory.mktemp("prog")
    d = root / "tiny"
    torch.manual_seed(0)
    GPT2LMHeadModel(GPT2Config(n_embd=32, n_layer=4, n_head=2)).save_pretrained(d, safe_serialization=True,
                                                                                 max_shard_size="1MB")
    snap = Path(TOK[0]).parent
    for f in ("tokenizer.json", "vocab.json", "merges.txt", "tokenizer_config.json"):
        if (snap / f).exists():
            shutil.copy(snap / f, d / f)
    data = root / "goal.jsonl"
    data.write_text("".join(json.dumps({"prompt": f"Question {i}: say something.", "target": "!"}) + "\n" for i in range(24)))
    return root, d, data


def program(root, d, data, name, **kw):
    p = {"base_model": str(d), "agent": {"name": "bot", "provider": "openai", "base_url": "http://unused/v1", "model": "m"},
         "goal": {"text": "Make the model print exclamation marks.", "data": str(data), "metric": "contains", "max_tokens": 4},
         "guards": [{"suite": {"name": "toy", "max_tokens": 6}, "base_runs": 2}],
         "rounds": 2, "attempts": 1, "hours": 0.1, "passes": 2, "output_dir": str(root / name),
         "sandbox": {"runtime": "local"}, "work_dir": str(root / "work"), "judge": "heuristic"}
    p.update(kw)
    return p


def test_program_accepts_writes_model_and_stops(base):
    root, d, data = base
    win = Bot([[("write_file", {"path": "edit.json", "content": ZERO_LN_F})],
               [("run_model", {"edit": "edit.json", "n": 4}), ("check_edit", {"edit": "edit.json"})],
               [("write_file", {"path": "rationale.md", "content": "zero ln_f: all logits 0, token 0 wins"})]])
    idle = Bot([])                                          # round 2: submits nothing -> the program stops
    res = wsr.program(program(root, d, data, "out1"), clients=[win, idle], wsr_config=None)
    assert res.accepted_rounds == 1 and res.final_score == 1.0 and res.base_score < 0.5
    rm = json.loads(dict((n, t) for n, t, _ in win.outputs)["run_model"])
    assert rm["score"] == 1.0 and rm["forward_passes_left"] == 1 and "current_model_score" in rm
    h = json.loads((res.output_dir / "history.json").read_text())
    assert [x["accepted"] for x in h] == [True, False] and h[0]["chosen"] and h[0]["guards"][0]["ok"]
    assert "no valid edit" in h[1]["reason"]
    # the edited checkpoint: loads, carries the edit, untouched shards are hard links of the base
    m = transformers.AutoModelForCausalLM.from_pretrained(res.model_dir)
    assert torch.count_nonzero(m.state_dict()[LN_F]) == 0
    shards = sorted(p.name for p in d.glob("*.safetensors"))
    assert len(shards) > 1
    linked = [s for s in shards if os.stat(d / s).st_ino == os.stat(res.model_dir / s).st_ino]
    assert linked and len(linked) == len(shards) - 1
    for f in ("README.md", "wsr_edit.json", "config.json", "tokenizer.json", "model.safetensors.index.json"):
        assert (res.model_dir / f).exists()
    assert (res.output_dir / "edit.safetensors").exists() and (res.output_dir / "report.md").exists()
    assert not (res.output_dir / "rounds").exists()
    # the edit file can be applied to the base again
    again = wsr.export(str(d), str(res.output_dir / "edit.safetensors"), str(root / "again"))
    assert torch.equal(transformers.AutoModelForCausalLM.from_pretrained(again).state_dict()[LN_F], m.state_dict()[LN_F])


def test_program_rejects_an_edit_that_does_nothing(base):
    root, d, data = base
    noop = json.dumps([{"op": "scale", "tensor": "transformer.h.1.mlp.c_proj.weight", "factor": 1.0001}])
    bot = Bot([[("write_file", {"path": "edit.json", "content": noop}),
                ("write_file", {"path": "rationale.md", "content": "tiny"})]])
    res = wsr.program(program(root, d, data, "out2", rounds=1, passes=0), clients=[bot])
    assert res.model_dir is None and res.accepted_rounds == 0
    h = json.loads((res.output_dir / "history.json").read_text())
    assert not h[0]["accepted"] and "gain" in h[0]["reason"]


def test_logprob_and_preference(base):
    from wsr.programming import Evaluator
    root, d, data = base
    pref = root / "pref.jsonl"
    pref.write_text("".join(json.dumps({"prompt": f"Q{i}", "chosen": " yes", "rejected": " no thank you"}) + "\n" for i in range(4)))
    cfg, spec = load_program(program(root, d, pref, "out3", goal={"data": str(pref), "metric": "logprob"}))
    lp = root / "lp.jsonl"
    lp.write_text("".join(json.dumps({"prompt": f"Q{i}", "target": " hello world"}) + "\n" for i in range(3)))
    cfg, spec = load_program(program(root, d, lp, "out3", goal={"data": str(lp), "metric": "logprob"}))
    ev = Evaluator(cfg, spec)
    zero = {LN_F: torch.zeros(32)}
    r = ev.goal(zero, [{"prompt": "Q", "target": " hello world"}])
    assert math.isclose(r["mean"], -math.log(50257), rel_tol=1e-4)     # all logits 0: uniform over the vocabulary
    spec["goal"]["metric"] = ev.metric = "preference"
    r = ev.goal({}, [{"prompt": "Q", "chosen": " yes", "rejected": " no"}])
    assert r["scores"][0] in (0.0, 1.0)
    ev.close()


def test_config_errors_and_gain(base):
    root, d, data = base
    with pytest.raises(SystemExit, match="goal.data"):
        load_program({"base_model": str(d), "agent": {"provider": "openai", "base_url": "http://x/v1", "model": "m"}, "goal": {}})
    with pytest.raises(SystemExit, match="needs tools"):
        load_program({"base_model": str(d), "agent": {"harness": "baseline"}, "goal": {"data": str(data)}})
    with pytest.raises(SystemExit, match="model judge"):
        load_program(program(root, d, data, "x", goal={"data": str(data), "metric": "judge"}))
    g = paired_gain([1, 1, 1, 0], [0, 0, 0, 0], 1)
    assert g["gain"] == 0.75 and g["ci"][0] > 0
    assert paired_gain([5, 5], [9, 9], -1)["gain"] == 4.0                # direction down: shorter is better


def test_cli_export(base, tmp_path):
    from wsr.cli import main
    root, d, _ = base
    e = tmp_path / "e.json"
    e.write_text(ZERO_LN_F)
    main(["export", "--model", str(d), "--edit", str(e), "--out", str(tmp_path / "m")])
    meta = json.loads((tmp_path / "m" / "wsr_edit.json").read_text())
    assert meta["tensors"][0]["tensor"] == LN_F
