"""World-model programming: the parts that need no world-model checkpoint or simulator."""
import json

import numpy as np
import pytest

from wsr.wm import envs, program


def test_shift_registry_and_families():
    s = envs.get_shift("pusht/kv60")
    assert s.params == {"k_v": 60.0} and "20 to 60" in s.text
    f = envs.get_shift("pusht/kv40")
    assert f.params == {"k_v": 40.0} and "20 to 40" in f.text and "60" not in f.text
    a = envs.get_shift("pusht/ascale75")
    assert a.params == {"action_scale": 75.0} and "+ 75 * action" in a.text
    sl = envs.get_shift("pusht/slip0.3")
    assert sl.params == {"damping": 0.3} and "retaining 30%" in sl.text
    assert envs.get_shift("reacher/gear0.7").params == {"gear_scale": 0.7}
    assert envs.get_shift("pusht/base") is None
    with pytest.raises(KeyError):
        envs.get_shift("pusht/nonsense")


def test_paired_bootstrap():
    rng = np.random.default_rng(0)
    old = rng.uniform(0.4, 0.6, 200)
    better = program.paired(old * 0.5, old)
    assert better["gain"] > 0 and better["ci"][0] > 0 and abs(better["rel"] - 0.5) < 1e-9
    same = program.paired(old + rng.normal(0, 1e-3, 200), old)
    assert same["ci"][0] < 0 < same["ci"][1]


def test_scope_errors():
    t = {"predictor.transformer.layers.0.adaLN_modulation.1.weight": 0, "action_encoder.patch_embed.weight": 0}
    assert program.scope_errors(t, None) == []
    bad = program.scope_errors(t, [r"^predictor\."])
    assert len(bad) == 1 and "action_encoder" in bad[0]


def test_prompt_conditions():
    spec = program.load_spec({"shift": "pusht/kv60", "agent": "a"})
    paths = {"target": "/T", "work": "/W"}
    p0 = program.build_prompt(spec, envs.get_shift("pusht/kv60"), paths, 0, 64, False, [], "ENV")
    assert "20 to 60" in p0 and "cannot run the model" in p0 and "run_model" not in p0.split("## Your tools")[1].split("##")[0]
    p10 = program.build_prompt(spec, envs.get_shift("pusht/kv60"), paths, 10, 64, True, [], "ENV")
    assert "10 forward passes" in p10 and "practice.npz" in p10
    wrong = program.load_spec({"shift": "pusht/kv60", "agent": "a", "text_from": "pusht/slip0.5"})
    pw = program.build_prompt(wrong, envs.get_shift("pusht/kv60"), paths, 0, 64, False, [], "ENV")
    assert "slippery" in pw and "20 to 60" not in pw
    none = program.load_spec({"shift": "pusht/kv60", "agent": "a", "info": "none"})
    pn = program.build_prompt(none, envs.get_shift("pusht/kv60"), paths, 10, 64, False, [], "ENV")
    assert "You are not told how" in pn and "20 to 60" not in pn
    scoped = program.load_spec({"shift": "pusht/kv60", "agent": "a", "edit_scope": [r"^predictor\."]})
    ps = program.build_prompt(scoped, envs.get_shift("pusht/kv60"), paths, 0, 64, False, [], "ENV")
    assert "only tensors whose names match" in ps


def test_userns_sandbox_script(tmp_path):
    from wsr.sandbox import Sandbox
    work, weights = tmp_path / "work", tmp_path / "weights"
    work.mkdir()
    weights.mkdir()
    sb = Sandbox({"sandbox": {"runtime": "userns"}}, {"work": (str(work), False), "target": (str(weights), True)})
    script = open(sb._userns_script()).read()
    assert sb.paths["work"] == "/mnt/work" and sb.paths["target"] == "/mnt/weights/target"
    assert f"mount --bind {weights.resolve()} /mnt/weights/target" in script
    assert "mount -o remount,bind,ro /mnt/weights/target" in script
    assert "mount -o remount,bind,ro /mnt/work" not in script
    assert "unshare -U --map-user=1000" in script and "env -i" in script
    assert "CUDA_VISIBLE_DEVICES=" in script
    sb.stop()
    assert sb._script is None


def test_audit_patterns(tmp_path):
    sn = tmp_path / "logs" / "snippets"
    sn.mkdir(parents=True)
    (sn / "0001.py").write_text("import numpy as np\nW = np.linalg.lstsq(A, B)[0]\n")
    (sn / "0002.py").write_text("import torch\nloss.backward()\n")
    (sn / "0003.py").write_text("print(W.info('x'))\n")
    a = program.audit(tmp_path)
    assert a["snippets"] == 3 and a["lstsq"] == 1 and a["autograd"] == 1 and a["optimizer"] == 0
    json.dumps(a)
