"""Environments, physics shifts, and counterfactual replay of dataset episodes.

A *segment* is a short window of a dataset episode: a start state and the raw actions that the episode took from
there. Replaying the same segment in the original environment and in a shifted one gives a counterfactual pair:
identical starting state and actions, different physics. The world model sees one frame every ``frameskip`` raw
steps and one model-step action per frame (the ``frameskip`` raw actions concatenated, z-scored per dimension with
the training dataset's statistics).

Shifts are applied after every reset, so they persist for the whole segment.
"""
from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path

import numpy as np

HISTORY = 3


@dataclass
class EnvSpec:
    name: str
    gym_id: str
    dataset: str           # file stem under $STABLEWM_HOME
    hf_dir: str            # LeWM release directory under $STABLEWM_HOME
    frameskip: int
    raw_action_dim: int
    description: str       # what the agent is told about the environment
    kind: str = "pusht"    # pusht | dmc
    state_cols: tuple = ("state",)
    state_desc: str = "[agent_x, agent_y, block_x, block_y, block_angle, agent_vx, agent_vy]"
    gym_kwargs: dict | None = None


ENVS = {
    "pusht": EnvSpec(
        "pusht", "swm/PushT-v1", "pusht_expert_train", "hf_pusht", 5, 2,
        "Push-T (2D, top-down, 224x224 RGB frames). A small blue circular agent pushes a grey T-shaped block on a "
        "flat floor (pymunk physics, no gravity in the image plane, walls at the borders). Each raw action is a 2-d "
        "relative target offset for the agent: target = agent_position + 100 * action, and a PD controller "
        "(acceleration = k_p * (target - position) - k_v * velocity, k_p=100, k_v=20) drives the agent there over "
        "10 physics substeps of 0.01 s (one raw step = 0.1 s). The agent is kinematic: it pushes the block but is "
        "never pushed back. The model sees one frame every 5 raw steps (0.5 s); a model-step action is the 5 raw "
        "actions of that interval, concatenated as [x1, y1, x2, y2, ..., x5, y5] and z-scored per coordinate "
        "(dataset mean {mean}, std {std}). The training data were expert episodes pushing the T onto a fixed "
        "goal pose. In the ORIGINAL physics the floor is fully damping: pymunk's space.damping = 0, so the block's "
        "velocity is removed at every substep and it stops the instant the agent stops pushing it."),
    "reacher": EnvSpec(
        "reacher", "swm/ReacherDMControl-v0", "reacher", "hf_reacher", 5, 2,
        "Reacher (DeepMind Control Suite, MuJoCo; 224x224 RGB frames from a fixed top-down camera). A planar two-link "
        "arm (shoulder hinge at the arena centre, then a wrist hinge; links `arm` and `hand` with a fingertip) moves "
        "in the horizontal plane (no gravity effect). Each raw action is a 2-d control in [-1, 1] for the shoulder "
        "and wrist motors: joint torque = gear * control ({gear}). Physics: timestep {dt} s, {nsub} substep(s) per raw "
        "action (one raw step = {ctrl_dt} s); joint damping {damping}; link masses {masses} kg. The model sees one frame "
        "every 5 raw steps ({frame_dt} s); a model-step action is the 5 raw actions of that interval concatenated as "
        "[s1, w1, s2, w2, ..., s5, w5] (s = shoulder, w = wrist) and z-scored per coordinate (dataset mean {mean}, "
        "std {std}). The training data were random-action episodes.",
        kind="dmc", state_cols=("qpos", "qvel"), state_desc="[shoulder_angle, wrist_angle, shoulder_velocity, wrist_velocity]"),
}


@dataclass
class Shift:
    env: str
    name: str
    params: dict
    text: str              # the change, as the agent is told it

    def key(self) -> str:
        return f"{self.env}/{self.name}"


def _pct(d: float) -> str:
    return f"{100 * d:.0f}%"


def pusht_shifts() -> dict[str, Shift]:
    S = {}
    for d in (0.05, 0.2, 0.5, 0.8):
        S[f"slip{d:g}"] = Shift("pusht", f"slip{d:g}", {"damping": d},
                                "The floor under the T-block has become slippery. pymunk's space.damping changed from "
                                f"0 to {d:g}: instead of stopping instantly, a moving block now keeps sliding and "
                                f"turning after contact, retaining {_pct(d)} of its linear and angular velocity after "
                                "one second (velocity is multiplied by damping**dt at every 0.01 s substep). The "
                                "agent's own motion is unaffected (it is kinematic). Nothing changes in how images "
                                "look.")
    for dx, dy in ((0, 30), (0, -30), (30, 0)):
        nm = f"cog{dx:+d}{dy:+d}".replace("+0", "0")
        S[nm] = Shift("pusht", nm, {"cog_offset": [dx, dy]},
                      f"A hidden weight has been fixed inside the T-block: its centre of mass moved by ({dx}, {dy}) "
                      "pixels in the block's own (body) coordinate frame (x along the T's top bar, y along its stem, "
                      "before rotation). Its mass, shape and appearance are unchanged, but pushes now make it rotate "
                      "differently around the new centre of mass.")
    for kv in (60.0, 8.0):
        S[f"kv{kv:g}"] = Shift("pusht", f"kv{kv:g}", {"k_v": kv},
                               f"The agent's PD controller changed: the velocity gain k_v went from 20 to {kv:g} "
                               f"(k_p stays 100). The agent now responds {'more sluggishly and overdamped' if kv > 20 else 'more springily and overshoots'} "
                               "when moving toward each commanded target. The block's physics are unchanged.")
    for sc in (50.0, 150.0):
        S[f"ascale{sc:g}"] = Shift("pusht", f"ascale{sc:g}", {"action_scale": sc},
                                   f"The action scale changed: target = agent_position + {sc:g} * action (it was 100). "
                                   f"Every action now commands a {'shorter' if sc < 100 else 'longer'} move. Physics "
                                   "and appearance are unchanged.")
    return S


def reacher_shifts() -> dict[str, Shift]:
    S = {}
    for m in (2.0, 0.5):
        S[f"mass{m:g}"] = Shift("reacher", f"mass{m:g}", {"mass_scale": m},
                                f"The arm became {'heavier' if m > 1 else 'lighter'}: the mass and rotational inertia of "
                                f"both moving links (`arm` and `hand`, including the fingertip) were multiplied by {m:g} "
                                "(denser material). Motor torques, damping and appearance are unchanged.")
    S["damp3"] = Shift("reacher", "damp3", {"damping_scale": 3.0},
                       "The joints became stiffer to turn: the viscous damping coefficient of both hinge joints "
                       "(shoulder and wrist) was multiplied by 3. Masses, motors and appearance are unchanged.")
    S["gear0.5"] = Shift("reacher", "gear0.5", {"gear_scale": 0.5},
                         "The motors became weaker: the gear of both actuators was halved, so the same control "
                         "produces half the joint torque. Masses, damping and appearance are unchanged.")
    return S


SHIFTS = {f"pusht/{k}": v for k, v in pusht_shifts().items()} | {f"reacher/{k}": v for k, v in reacher_shifts().items()}


def _family_shift(env: str, fam: str, v: float) -> Shift:
    """Any member of a one-parameter shift family (for parametric programs), built with the same texts."""
    if env == "pusht" and fam == "kv":
        s = pusht_shifts()["kv60" if v > 20 else "kv8"]
        return Shift("pusht", f"kv{v:g}", {"k_v": v}, s.text.replace("to 60" if v > 20 else "to 8", f"to {v:g}"))
    if env == "pusht" and fam == "ascale":
        s = pusht_shifts()["ascale50" if v < 100 else "ascale150"]
        old = "50" if v < 100 else "150"
        return Shift("pusht", f"ascale{v:g}", {"action_scale": v}, s.text.replace(f"+ {old} * action", f"+ {v:g} * action"))
    if env == "pusht" and fam == "slip":
        s = pusht_shifts()["slip0.5"]
        return Shift("pusht", f"slip{v:g}", {"damping": v},
                     s.text.replace("0 to 0.5", f"0 to {v:g}").replace("retaining 50%", f"retaining {_pct(v)}"))
    if env == "reacher" and fam == "mass":
        return reacher_shifts()["mass2"].__class__("reacher", f"mass{v:g}", {"mass_scale": v},
                                                   f"The arm's mass changed: the mass and rotational inertia of both moving "
                                                   f"links (`arm` and `hand`, including the fingertip) were multiplied by {v:g}. "
                                                   "Motor torques, damping and appearance are unchanged.")
    if env == "reacher" and fam == "gear":
        return Shift("reacher", f"gear{v:g}", {"gear_scale": v},
                     f"The motors changed: the gear of both actuators was multiplied by {v:g}, so the same control "
                     f"produces {v:g} times the joint torque. Masses, damping and appearance are unchanged.")
    raise KeyError(f"no shift family `{fam}` for {env}")


def get_shift(key: str) -> Shift | None:
    import re
    if key in ("base", None) or key.endswith("/base"):
        return None
    if key in SHIFTS:
        return SHIFTS[key]
    m = re.fullmatch(r"(\w+)/([a-z]+)([0-9.]+)", key)
    if m:
        return _family_shift(m.group(1), m.group(2), float(m.group(3)))
    raise KeyError(f"unknown shift `{key}`; known: {sorted(SHIFTS)}")


# ---------------------------------------------------------------- dataset access
def stablewm_home() -> Path:
    return Path(os.environ.get("STABLEWM_HOME", Path.home() / ".stable-wm"))


class Episodes:
    """Start states and raw actions of the dataset's episodes (small columns only; pixels are read on demand)."""

    def __init__(self, env: str):
        import h5py
        self.spec = ENVS[env]
        self.path = stablewm_home() / f"{self.spec.dataset}.h5"
        with h5py.File(self.path, "r") as h:
            self.ep_len = h["ep_len"][:]
            self.ep_offset = h["ep_offset"][:]
            self.action = h["action"][:]
            cols = [h[c][:].reshape(len(h[c]), -1) for c in self.spec.state_cols if c in h]
            self.state = np.concatenate(cols, 1) if cols else None
        a = self.action[~np.isnan(self.action).any(1)]
        self.act_mean, self.act_std = a.mean(0), a.std(0, ddof=1)

    def pixels(self, rows) -> np.ndarray:
        import h5py
        rows = np.asarray(rows)
        order = np.argsort(rows)
        with h5py.File(self.path, "r") as h:
            px = h["pixels"][rows[order]]
        out = np.empty_like(px)
        out[order] = px
        return out

    def sample_segments(self, n: int, raw_len: int, split: str, seed: int) -> list[tuple[int, int]]:
        """(episode, start step) pairs. Episodes are split once (seed 0) into practice / heldout / train pools so
        that practice segments (the agent's) and held-out segments (acceptance) never share an episode."""
        perm = np.random.default_rng(0).permutation(len(self.ep_len))
        k = len(perm) // 10
        pools = {"practice": perm[:k], "heldout": perm[k:2 * k], "train": perm[2 * k:]}
        eps = pools[split]
        eps = eps[self.ep_len[eps] > raw_len + 1]
        rng = np.random.default_rng(seed)
        out = []
        for e in rng.choice(eps, size=n, replace=len(eps) < n):
            start = int(rng.integers(0, self.ep_len[e] - raw_len - 1))
            out.append((int(e), start))
        return out

    def segment(self, ep: int, start: int, raw_len: int) -> tuple[np.ndarray, np.ndarray]:
        o = self.ep_offset[ep] + start
        acts = np.nan_to_num(self.action[o:o + raw_len].astype(np.float64))
        return self.state[o].astype(np.float64), acts

    def model_actions(self, raw: np.ndarray) -> np.ndarray:
        """(n, frames-1 model steps * frameskip, raw_dim) raw actions -> (n, steps, frameskip * raw_dim) z-scored."""
        z = (raw - self.act_mean) / self.act_std
        fs = self.spec.frameskip
        n, L, d = z.shape
        return z.reshape(n, L // fs, fs * d).astype(np.float32)


# ---------------------------------------------------------------- replay
def _apply_pusht(env, params: dict) -> None:
    u = env.unwrapped
    if "damping" in params:
        u.space.damping = float(params["damping"])
    if "cog_offset" in params:
        # body.position is the world position of the centre of gravity and shapes sit at position - R(angle) cog,
        # so move the position with the cog: the block stays where it is and only its dynamics change
        import pymunk
        b = u.block
        d = pymunk.Vec2d(*params["cog_offset"])
        b.center_of_gravity = b.center_of_gravity + d
        b.position = b.position + d.rotated(b.angle)
        u.space.reindex_shapes_for_body(b)
    for k in ("k_p", "k_v", "action_scale"):
        if k in params:
            setattr(u, k, float(params[k]))


_DMC_BODIES = ("arm", "hand", "finger")


def _apply_dmc(env, params: dict) -> None:
    ph = env.unwrapped.env.physics
    m = ph.model
    if "mass_scale" in params:
        for b in _DMC_BODIES:
            try:
                i = m.name2id(b, "body")
            except Exception:  # noqa: BLE001
                continue
            m.body_mass[i] *= params["mass_scale"]
            m.body_inertia[i] *= params["mass_scale"]
    if "damping_scale" in params:
        m.dof_damping[:] *= params["damping_scale"]
    if "gear_scale" in params:
        m.actuator_gear[:, 0] *= params["gear_scale"]
    ph.forward()


def dmc_constants(env) -> dict:
    ph = env.unwrapped.env.physics
    m = ph.model
    masses = {}
    for b in _DMC_BODIES:
        try:
            masses[b] = round(float(m.body_mass[m.name2id(b, "body")]), 5)
        except Exception:  # noqa: BLE001
            pass
    nsub = int(getattr(env.unwrapped.env, "_n_sub_steps", 1))
    return {"gear": [round(float(g), 4) for g in m.actuator_gear[:, 0]], "dt": float(m.opt.timestep), "nsub": nsub,
            "ctrl_dt": round(float(m.opt.timestep) * nsub, 4), "damping": [round(float(d), 4) for d in m.dof_damping],
            "masses": masses}


def _obs_state(env, kind: str) -> np.ndarray:
    u = env.unwrapped
    if kind == "pusht":
        return u._get_obs()
    d = u.env.physics.data
    return np.concatenate([d.qpos, d.qvel]).astype(np.float64)


def _render(env, kind: str) -> np.ndarray:
    return env.unwrapped.render()


def _replay_worker(job):
    """Replay segments in one environment configuration. -> frames (n, F, H, W, 3) uint8, states (n, F, s)."""
    env_name, params, items, frameskip = job
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401  (registers the environments)
    spec = ENVS[env_name]
    env = gym.make(spec.gym_id, **(spec.gym_kwargs or {}))
    apply = _apply_pusht if spec.kind == "pusht" else _apply_dmc
    frames, states = [], []
    for state0, acts in items:
        env.reset(seed=0, options={"state": state0})
        if params:
            apply(env, params)
        fr, st = [_render(env, spec.kind)], [_obs_state(env, spec.kind)]
        for i, a in enumerate(acts):
            env.step(a.astype(np.float32))
            if (i + 1) % frameskip == 0:
                fr.append(_render(env, spec.kind))
                st.append(_obs_state(env, spec.kind))
        frames.append(np.stack(fr))
        states.append(np.stack(st))
    env.close()
    return np.stack(frames).astype(np.uint8), np.stack(states).astype(np.float32)


def replay(env_name: str, params: dict | None, items: list, workers: int = 16, chunk: int = 8):
    from concurrent.futures import ProcessPoolExecutor
    spec = ENVS[env_name]
    jobs = [(env_name, params or {}, items[i:i + chunk], spec.frameskip) for i in range(0, len(items), chunk)]
    if workers <= 1:
        res = [_replay_worker(j) for j in jobs]
    else:
        import multiprocessing as mp
        with ProcessPoolExecutor(workers, mp_context=mp.get_context("spawn")) as ex:
            res = list(ex.map(_replay_worker, jobs))
    return np.concatenate([r[0] for r in res]), np.concatenate([r[1] for r in res])


# ---------------------------------------------------------------- cached segment sets
def collect(backend, env_name: str, shift_keys: list[str], split: str, n: int, horizon: int, seed: int,
            out_dir: Path, workers: int = 16, keep_frames: int = 0, log=print) -> dict[str, Path]:
    """Replay the same ``n`` segments of ``split`` under the base physics and every shift; encode the frames.

    Writes ``<out_dir>/<env>/<shift>/<split>_s<seed>_n<n>_h<horizon>.pt`` with emb (n, F, D), act (n, F-1, A),
    states (n, F, s), segments [(ep, start)], and returns {shift key: path}.
    """
    import torch
    eps = Episodes(env_name)
    spec = eps.spec
    F = HISTORY + horizon
    raw_len = (F - 1) * spec.frameskip
    segs = eps.sample_segments(n, raw_len, split, seed)
    items = [eps.segment(e, s, raw_len) for e, s in segs]
    act = eps.model_actions(np.stack([a for _, a in items]))
    paths = {}
    for key in ["base"] + [k for k in shift_keys if k != "base"]:
        name = key.split("/")[-1]
        p = Path(out_dir) / env_name / name / f"{split}_s{seed}_n{n}_h{horizon}.pt"
        paths[key] = p
        if p.exists():
            continue
        sh = get_shift(key)
        frames, states = replay(env_name, sh.params if sh else {}, items, workers)
        emb = backend.encode(frames.reshape(-1, *frames.shape[2:])).reshape(n, F, -1)
        p.parent.mkdir(parents=True, exist_ok=True)
        blob = {"emb": emb, "act": torch.from_numpy(act), "states": torch.from_numpy(states), "segments": segs,
                "env": env_name, "shift": key, "params": sh.params if sh else {}, "horizon": horizon,
                "act_mean": eps.act_mean.tolist(), "act_std": eps.act_std.tolist()}
        if keep_frames:
            blob["frames"] = torch.from_numpy(frames[:keep_frames])
        tmp = p.with_name(p.name + f".tmp{os.getpid()}")
        torch.save(blob, tmp)
        os.replace(tmp, p)                 # atomic: concurrent jobs may collect the same set
        log(f"collected {key} {split}: {p}")
    return paths


def load(path: str | Path) -> dict:
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def env_description(env_name: str) -> str:
    spec = ENVS[env_name]
    eps_stats = stablewm_home() / f"{spec.dataset}.stats.json"
    if eps_stats.exists():
        st = json.loads(eps_stats.read_text())
    else:
        e = Episodes(env_name)
        st = {"mean": [round(float(x), 4) for x in e.act_mean], "std": [round(float(x), 4) for x in e.act_std]}
        try:
            eps_stats.write_text(json.dumps(st))
        except OSError:
            pass
    extra = {}
    if spec.kind == "dmc":
        import gymnasium as gym
        import stable_worldmodel  # noqa: F401
        env = gym.make(spec.gym_id, **(spec.gym_kwargs or {}))
        env.reset(seed=0)
        c = dmc_constants(env)
        env.close()
        extra = {**c, "frame_dt": round(c["ctrl_dt"] * spec.frameskip, 4)}
    return spec.description.format(mean=st["mean"], std=st["std"], **extra)
