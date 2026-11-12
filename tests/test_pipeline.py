"""End to end on CPU: a tiny random GPT-2 with the real GPT-2 tokenizer, the toy suite, the Transformers backend,
baseline and scripted agents. Checks that every step runs and writes what the next step needs."""
from __future__ import annotations

import glob
import json
import shutil
from pathlib import Path

import pytest
import yaml

from wsr.cli import main
from wsr.config import Layout, load_config, load_experiment
from wsr.providers import Reply, ToolCall

transformers = pytest.importorskip("transformers")
TOK = glob.glob(str(Path.home() / ".cache/huggingface/hub/models--gpt2/snapshots/*/tokenizer.json"))
pytestmark = pytest.mark.skipif(not TOK, reason="needs the gpt2 tokenizer in the Hugging Face cache")


@pytest.fixture(scope="module")
def project(tmp_path_factory):
    import torch
    from transformers import GPT2Config, GPT2LMHeadModel
    root = tmp_path_factory.mktemp("proj")
    mdir = root / "tiny"
    torch.manual_seed(0)
    GPT2LMHeadModel(GPT2Config(n_embd=32, n_layer=4, n_head=2)).save_pretrained(mdir, safe_serialization=True)
    snap = Path(TOK[0]).parent
    for f in ("tokenizer.json", "vocab.json", "merges.txt", "tokenizer_config.json"):
        if (snap / f).exists():
            shutil.copy(snap / f, mdir / f)
    cfg = {"work_dir": str(root / "work"), "data_dir": str(root / "data"),
           "models": {"tiny": {"path": str(mdir), "backend": "hf", "hf": {"batch_size": 64}}},
           "agents": {"same": {"harness": "baseline", "params": {"strategy": "same"}},
                      "rnd": {"harness": "baseline", "params": {"strategy": "random", "seed": 1}},
                      "bot": {"provider": "openai", "base_url": "http://unused/v1", "model": "scripted"}},
           "judge": "heuristic", "sandbox": {"runtime": "local"}}
    (root / "wsr.yaml").write_text(yaml.safe_dump(cfg))
    exp = {"name": "e2e", "suite": {"name": "toy", "max_tokens": 6}, "targets": ["tiny"], "readers": ["same", "rnd"],
           "pool": {"kind": "standard", "size": "small"}, "conditions": ["weights", "text"], "base_runs": 3,
           "selection": {"n_changed": 6, "n_same": 4, "family_cap": 3, "n_batches": 2}, "hours": 0.05,
           "write": {"passes": 2, "practice_items": 4, "hours": 0.1,
                     "tasks": {"short": {"behavior": "answer_length", "direction": "down"}}}}
    (root / "e2e.yaml").write_text(yaml.safe_dump(exp))
    return root


def run(project, *args):
    main(["--config", str(project / "wsr.yaml"), *args])


def test_reading_pipeline(project, capsys):
    e = str(project / "e2e.yaml")
    run(project, "run", e)
    cfg = load_config(str(project / "wsr.yaml"))
    L = Layout(cfg, load_experiment(e, cfg))
    lab = json.loads(L.labels("tiny").read_text())
    assert set(lab["base_mean"]) == {"answer_length", "arith_accuracy", "mc_accuracy", "sycophancy"}
    assert len(list(L.measure("tiny").glob("*.jsonl"))) == len(L.pool("tiny").read_text().splitlines()) + 3
    assert L.bank("tiny").exists() and list(L.reader_files("tiny").glob("*.safetensors"))
    metas = [json.loads(p.read_text()) for p in L.sessions().glob("*/meta.json")]
    assert len(metas) == 2 * 2 * 2 and all(m["status"] == "done" for m in metas)
    summary = json.loads((L.scores() / "summary.json").read_text())
    for k, v in summary["cells"].items():
        if k.startswith("same->"):
            assert v["specificity"] == 1.0 and not v["recall"] > 0          # 0, or undefined if nothing changed
    assert (L.scores() / "report.md").exists()
    run(project, "plan", e)
    assert "0 to run" in capsys.readouterr().out


def test_programming_session(project):
    from wsr.write import run_program
    e = str(project / "e2e.yaml")
    cfg = load_config(str(project / "wsr.yaml"))
    exp = load_experiment(e, cfg)
    unembed = "transformer.wte.weight"

    class Bot:
        """Scripted programmer: checks an edit, spends one forward pass, writes the rationale."""

        def __init__(self):
            edit = json.dumps([{"op": "scale_rows", "tensor": unembed, "rows": [50256], "factor": 3.0}])
            self.replies = [
                Reply("", [ToolCall("1", "write_file", {"path": "edit.json", "content": edit})], "tool"),
                Reply("", [ToolCall("2", "check_edit", {"edit": "edit.json"}),
                           ToolCall("3", "run_model", {"edit": "edit.json", "n": 4, "task": "arith"})], "tool"),
                Reply("", [ToolCall("4", "write_file", {"path": "rationale.md", "content": "boost end of text"})], "tool"),
                Reply("Submitted."),
            ]
            self.outputs = []

        start = lambda self, s, u: [{"role": "user", "content": u}]  # noqa: E731
        add_user = lambda self, h, t: h.append({"role": "user", "content": t})  # noqa: E731
        add_reply = lambda self, h, r: h.append({"role": "assistant", "content": r.text})  # noqa: E731
        trim = lambda self, h, keep_last=12: False  # noqa: E731

        def add_tool_results(self, h, results):
            self.outputs += [(c.name, t, err) for c, t, err in results]

        def chat(self, h, tools=None, max_tokens=None):
            return self.replies.pop(0)

    bot = Bot()
    meta = run_program(cfg, exp, "bot", "tiny", "short", client=bot)
    assert meta["status"] == "done" and meta["submission"] == "edit.json" and meta["passes_used"] == 1
    out = dict((n, (t, err)) for n, t, err in bot.outputs)
    assert not out["check_edit"][1] and json.loads(out["check_edit"][0])["ok"]
    res = json.loads(out["run_model"][0])
    assert res["forward_passes_left"] == 1 and "base_scores" in res
    run(project, "measure", e, "--programs")
    run(project, "label", e, "--programs")
    run(project, "report", e)
    L = Layout(cfg, exp)
    progs = json.loads((L.scores() / "programs.json").read_text())
    assert progs[0]["measured"] and progs[0]["goal"] == "down answer_length" and "success" in progs[0]
