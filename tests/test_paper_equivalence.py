"""The paper's experiments must behave exactly as the released code (kept in tests/legacy/wsr_v0) did:
same pools, same edits and descriptions, same labels, same selection, same reader prompts, same scores."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import pytest
import torch
from safetensors.torch import save_file

from tests.legacy.wsr_v0 import common as old_common
from tests.legacy.wsr_v0 import edits as old_edits
from tests.legacy.wsr_v0 import labels as old_labels
from tests.legacy.wsr_v0 import pools as old_pools
from tests.legacy.wsr_v0 import score as old_score
from tests.legacy.wsr_v0 import select as old_select
from tests.legacy.wsr_v0 import session as old_session
from wsr import labels as new_labels
from wsr import ops
from wsr.checkpoint import Checkpoint
from wsr.config import Layout, load_experiment, normalize_config, write_jsonl
from wsr.paper import pools as new_pools
from wsr.read import WARN as NEW_WARN
from wsr.read import session_prompt
from wsr.sandbox import CONTAINER_PATHS
from wsr.score import score as new_score
from wsr.select import batches as new_batches
from wsr.select import cmd_select as new_cmd_select
from wsr.suites import load_suite

SUITE = load_suite("paper")


# ---------------------------------------------------------------- pools
@pytest.mark.parametrize("t", ["27b", "9b"])
def test_full_dose_pool(t):
    assert new_pools.full_dose_pool(t) == old_pools.full_dose_pool(t)


def test_muse_pools():
    assert new_pools.muse_pool() == old_pools.muse_pool()
    assert new_pools.muse_standard_edits() == old_pools.muse_standard_edits()
    assert len(new_pools.muse_standard_edits()) == 40
    assert sum(s["type"] == "F" for s in new_pools.muse_standard_edits()) == 6


@pytest.mark.parametrize("t", ["27b", "9b"])
def test_subtle_pool(t):
    full = old_pools.full_dose_pool(t)
    rng = random.Random(3)
    parents = {s["id"] for s in full if s["type"] != "F" and rng.random() < 0.4}
    assert new_pools.subtle_pool(t, full, parents) == old_pools.subtle_pool(t, full, parents)


def test_dose_ops():
    for s in old_pools.full_dose_pool("9b"):
        for d in (0.5, 0.25, 0.1):
            assert ops.dose_ops(s["ops"], d) == old_pools.dose_ops(s["ops"], d)


# ---------------------------------------------------------------- edits
@pytest.fixture(scope="module")
def fake_9b(tmp_path_factory):
    """A checkpoint with the paper's tensor names for Qwen3.5-9B (tiny widths)."""
    from wsr.paper.models import FINAL_NORM, LM_HEAD, attn_out, mlp
    g = torch.Generator().manual_seed(0)
    H, I, V = 8, 16, 248400
    T = {LM_HEAD: torch.randn(V, H, generator=g), FINAL_NORM: 0.05 * torch.randn(H, generator=g)}
    for l in range(32):
        T[mlp(l)] = torch.randn(H, I, generator=g)
        T[attn_out(l)] = torch.randn(H, H, generator=g)
    d = tmp_path_factory.mktemp("fake9b")
    save_file({k: v.to(torch.bfloat16).contiguous() for k, v in T.items()}, str(d / "model.safetensors"))
    return d


def test_apply_ops_identical(fake_9b):
    ck_new, ck_old = Checkpoint(fake_9b), old_edits.Checkpoint(fake_9b)
    for s in old_pools.full_dose_pool("9b"):
        f1, c1, d1 = ops.apply_ops(ck_new, s)
        f0, c0, d0 = old_edits.apply_ops(ck_old, s)
        assert d1 == d0, s["id"]
        assert f1.keys() == f0.keys() and c1.keys() == c0.keys()
        for k in f0:
            assert torch.equal(f1[k], f0[k]), (s["id"], k)
        for k in c0:
            assert torch.equal(c1[k], c0[k]), (s["id"], k)


def test_compact_roundtrip(fake_9b):
    ck = Checkpoint(fake_9b)
    for s in old_pools.full_dose_pool("9b")[:30]:
        full, compact, _ = ops.apply_ops(ck, s)
        back = ops.expand_compact(ck, compact)
        for k in full:
            assert torch.equal(back[k], full[k])


# ---------------------------------------------------------------- labels
def _records(rng, n_math=64, n_mc=100, shift=0.0):
    recs = []
    for i in range(n_math):
        recs.append({"id": f"m{i}", "task": "math", "n_tokens": int(rng.gauss(3000 + shift * 900, 600)),
                     "correct": rng.random() < 0.7 - shift * 0.2, "hit_limit": rng.random() < 0.02})
    for i in range(n_mc):
        recs.append({"id": f"c{i}", "task": "mc", "n_tokens": 500, "correct": rng.random() < 0.6, "hit_limit": False})
    for i in range(64):
        recs.append({"id": f"s{i}", "task": "syco", "n_tokens": 500, "correct": rng.random() < 0.5,
                     "hint_type": "wrong" if i < 48 else "right", "followed_hint": rng.random() < 0.2 + shift * 0.2,
                     "hit_limit": False})
    for i in range(60):
        recs.append({"id": f"h{i}", "task": "chat", "n_tokens": 300, "hit_limit": False, "question": "q", "answer": "a"})
    return recs


def test_values_labels_destructive():
    rng = random.Random(0)
    base = sum((_records(rng) for _ in range(5)), [])
    judged_base = {f"h{i}": rng.random() < 0.9 for i in range(60)}
    for shift in (0.0, 0.3, 1.0, 2.0):
        recs = _records(rng, shift=shift)
        judged = {f"h{i}": rng.random() < 0.9 - shift * 0.3 for i in range(60)}
        v_new, v_old = SUITE.values(recs, {"refusal": judged}), old_labels.item_values(recs, judged)
        assert v_new == v_old
        b_new, b_old = SUITE.values(base, {"refusal": judged_base}), old_labels.item_values(base, judged_base)
        for k in SUITE.keys:
            assert new_labels.label_behavior(SUITE.behavior(k), v_new[k], b_new[k]) == old_labels.label_behavior(k, v_old[k], b_old[k])
        assert SUITE.destructive(recs, base) == old_labels.destructive(recs, base)
        nr = {k: b_new[k] for k in SUITE.keys}
        assert new_labels.changed_without_judge(SUITE, v_new, nr) == old_labels.changed_without_refusal(v_old, b_old)


# ---------------------------------------------------------------- selection, batches, prompts, scores
def _legacy_cfg(tmp_path):
    return {"work_dir": str(tmp_path / "old"), "data_dir": "data", "reader_files_dir": str(tmp_path / "old_rf"),
            "models": {}, "readers": {}}


def _new(tmp_path, ref):
    cfg = normalize_config({"work_dir": str(tmp_path / "new"), "reader_files_dir": str(tmp_path / "new_rf"), "models": {}})
    return cfg, load_experiment(ref, cfg)


def _fake_labels(pool, rng):
    lab = {}
    for s in pool:
        ch = rng.random() < (0.05 if s["type"] == "F" else 0.45)
        l = {k: (rng.choice(["up", "down"]) if ch and rng.random() < 0.5 else "same") for k in SUITE.keys}
        if ch and all(v == "same" for v in l.values()):
            l["math_accuracy"] = "down"
        l["changes_behavior"] = "yes" if any(v != "same" for v in l.values()) else "no"
        lab[s["id"]] = {"labels": l, "tests": {}, "destructive": rng.random() < 0.3}
    return lab


@pytest.mark.parametrize("rnd", ["full_dose", "subtle", "muse", "muse_standard"])
def test_selection_and_batches(tmp_path, rnd):
    targets = {"full_dose": ["27b", "9b"], "subtle": ["27b", "9b"], "muse": ["muse30b"], "muse_standard": ["muse30b"]}[rnd]
    ocfg = _legacy_cfg(tmp_path)
    cfg, exp = _new(tmp_path, f"paper.{rnd}")
    OL, NL = old_common.Layout(ocfg, rnd), Layout(cfg, exp)
    rng = random.Random(1)
    for t in targets:
        pool = old_pools.full_dose_pool(t) if t != "muse30b" else (
            old_pools.muse_pool() if rnd == "muse" else old_pools.muse_standard_edits())
        lab = _fake_labels(pool, rng)
        for L in (OL, NL):
            write_jsonl(L.pool(t), pool)
            L.labels(t).parent.mkdir(parents=True, exist_ok=True)
            L.labels(t).write_text(json.dumps({"labels": lab}))
    old_select.cmd_select(ocfg, argparse.Namespace(round=rnd, targets=None, n_changed=30, n_same=10))
    new_cmd_select(cfg, exp)
    for t in targets:
        assert NL.bank(t).read_text() == OL.bank(t).read_text()
        assert new_batches(NL, t) == old_select.batches(OL, t)


@pytest.mark.parametrize("cond,framing", [("weights", "self"), ("weights", "other"), ("text", "self"), ("text", "other")])
def test_reader_prompt(tmp_path, cond, framing):
    ocfg = _legacy_cfg(tmp_path)
    cfg, exp = _new(tmp_path, "paper.full_dose")
    OL, NL = old_common.Layout(ocfg, "full_dose"), Layout(cfg, exp)
    bank = [dict(s, desc=f"description of {s['id']}") for s in old_pools.full_dose_pool("27b")[:10]]
    for L in (OL, NL):
        write_jsonl(L.bank("27b"), bank)
    ids = [s["id"] for s in bank]
    old = old_session.session_prompt(OL, "27b", cond, ids, framing, 90)
    new = session_prompt(NL, SUITE, "27b", cond, ids, framing, 90, CONTAINER_PATHS, sharded=True)
    assert new == old
    assert old_session.WARN.replace("/work/answers.json", "{answers}") == NEW_WARN


def _fake_sessions(L, plan, lab_by_t, rng):
    for (reader, target, cond, batch, framing_arg), ids in plan:
        framing = framing_arg or ("self" if reader == target else "other")
        rid = f"{reader}_{target}_{cond}_b{batch}" + (f"_{framing_arg}" if framing_arg else "")
        rd = L.sessions() / rid
        rd.mkdir(parents=True, exist_ok=True)
        ans = {}
        for i in ids:
            if rng.random() < 0.1:
                continue
            truth = lab_by_t[target][i]["labels"]
            ans[i] = {k: [truth[k] if rng.random() < 0.6 else rng.choice(["up", "down", "same"]), rng.randint(0, 100)]
                      for k in SUITE.keys + ["changes_behavior"]}
            ans[i]["changes_behavior"][0] = rng.choice(["yes", "no"])
        (rd / "answers.json").write_text(json.dumps(ans))
        (rd / "meta.json").write_text(json.dumps({"reader": reader, "target": target, "cond": cond, "framing": framing,
                                                  "batch": batch, "ids": ids}))


@pytest.mark.parametrize("rnd", ["full_dose", "muse"])
def test_scores_identical(tmp_path, rnd):
    ocfg = _legacy_cfg(tmp_path)
    cfg, exp = _new(tmp_path, f"paper.{rnd}")
    OL, NL = old_common.Layout(ocfg, rnd), Layout(cfg, exp)
    R = old_common.ROUNDS[rnd]
    rng = random.Random(7)
    lab_by_t, plan = {}, []
    for t in R["targets"]:
        pool = old_pools.full_dose_pool(t) if t != "muse30b" else old_pools.muse_pool()
        lab_by_t[t] = _fake_labels(pool[:40], rng)
        ids = list(lab_by_t[t])
        for r in R["readers"][t]:
            for cond in ("weights", "text"):
                for b in range(4):
                    plan.append(((r, t, cond, b, ""), ids[b * 10:(b + 1) * 10]))
                    if R["framing_control"] and r == t and cond == "weights":
                        plan.append(((r, t, cond, b, "other"), ids[b * 10:(b + 1) * 10]))
    for L in (OL, NL):
        for t, lab in lab_by_t.items():
            L.labels(t).parent.mkdir(parents=True, exist_ok=True)
            L.labels(t).write_text(json.dumps({"labels": lab}))
        _fake_sessions(L, plan, lab_by_t, random.Random(11))
    old_score.cmd_score(ocfg, argparse.Namespace(round=rnd))
    old = json.loads((OL.scores() / "summary.json").read_text())
    new = new_score(cfg, exp)
    new.pop("per_behavior")
    assert json.loads(json.dumps(new)) == old


def test_legacy_config_and_judge_request():
    import httpx

    from wsr.harness import opencode_config
    from wsr.providers import OpenAIClient
    raw = json.loads(open(Path(__file__).parent / "legacy" / "config_v0.json").read())
    cfg = normalize_config(raw)
    a = cfg["agents"]["27b"]
    assert a["harness"] == "opencode" and a["identity"] == "27b"
    oc = opencode_config(a)
    assert oc["provider"]["local"]["options"]["baseURL"] == "http://localhost:8001/v1"
    assert oc["model"] == "local/Qwen3.8-27B"
    seen = {}

    def handler(req):
        seen["body"] = json.loads(req.content)
        return httpx.Response(200, json={"choices": [{"message": {"content": "REFUSED"}, "finish_reason": "stop"}]})
    c = OpenAIClient(cfg["judge"])
    c.http = httpx.Client(base_url=c.base, transport=httpx.MockTransport(handler))
    assert c.ask("prompt", max_tokens=8) == "REFUSED"
    assert seen["body"] == {"model": "Qwen3.8-27B", "messages": [{"role": "user", "content": "prompt"}],
                            "temperature": 0.0, "chat_template_kwargs": {"enable_thinking": False}, "max_tokens": 8}
