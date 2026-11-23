"""Planning with a (possibly edited) world model in the shifted environment: CEM model-predictive control.

    python -m wsr.wm plan --env pusht --shift pusht/kv60 --models base,agent:<run>,ft:<file>,...

Follows the LeWM evaluation protocol: start from a held-out dataset state, the goal is the state the episode
reached ``goal_offset`` raw steps later, the agent has ``budget`` raw steps; plans of ``horizon`` model steps
(``frameskip`` raw actions each) are found with CEM in the latent space (cost: squared distance between the last
predicted embedding and the goal image's embedding) and executed in full before replanning. Success is the
environment's own success test at any step. All models are scored on the same episodes and CEM seeds.
"""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch

HIST = 3


def episodes(env_name: str, n: int, goal_offset: int, budget: int, seed: int = 0):
    from .envs import Episodes
    eps = Episodes(env_name)
    segs = eps.sample_segments(n, goal_offset + 1, "heldout", 1000 + seed)
    return eps, [(eps.state[eps.ep_offset[e] + s].astype(np.float64), eps.state[eps.ep_offset[e] + s + goal_offset].astype(np.float64))
                 for e, s in segs]


def make_env(env_name: str, params: dict):
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    from .envs import ENVS
    spec = ENVS[env_name]
    return gym.make(spec.gym_id, **(spec.gym_kwargs or {}))


class CEM:
    def __init__(self, be, horizon: int, act_dim: int, samples: int = 300, iters: int = 30, topk: int = 30, seed: int = 0):
        self.be, self.h, self.A = be, horizon, act_dim
        self.S, self.iters, self.k = samples, iters, topk
        self.g = torch.Generator(device=be.device).manual_seed(seed)

    @torch.no_grad()
    def plan(self, ctx_emb: torch.Tensor, ctx_act: torch.Tensor, goal: torch.Tensor) -> torch.Tensor:
        """ctx_emb (B, h, D) last h frame embeddings; ctx_act (B, h-1, A) executed blocks between them;
        goal (B, D). -> (B, horizon, A) normalized action blocks."""
        m, dev = self.be.model, self.be.device
        B, h, D = ctx_emb.shape
        mu = torch.zeros(B, self.h, self.A, device=dev)
        sd = torch.ones(B, self.h, self.A, device=dev)
        E0 = ctx_emb.to(dev).unsqueeze(1).expand(B, self.S, h, D).reshape(B * self.S, h, D)
        A0 = ctx_act.to(dev).unsqueeze(1).expand(B, self.S, h - 1, self.A).reshape(B * self.S, h - 1, self.A)
        G = goal.to(dev).unsqueeze(1).expand(B, self.S, D).reshape(B * self.S, D)
        for _ in range(self.iters):
            cand = mu.unsqueeze(1) + sd.unsqueeze(1) * torch.randn(B, self.S, self.h, self.A, device=dev, generator=self.g)
            cand[:, 0] = mu                                   # keep the current mean among the candidates
            C = cand.reshape(B * self.S, self.h, self.A)
            acts = torch.cat([A0, C], 1)                      # (BS, h-1+H, A); index k = block leaving frame k
            ae = m.action_encoder(acts)
            embs = E0
            for t in range(self.h):
                lo = max(0, embs.shape[1] - HIST)
                hi = embs.shape[1]
                nxt = m.predict(embs[:, lo:hi], ae[:, lo:hi])[:, -1:]
                embs = torch.cat([embs, nxt], 1)
            cost = (embs[:, -1] - G).pow(2).sum(-1).view(B, self.S)
            idx = cost.topk(self.k, dim=1, largest=False).indices
            elite = torch.gather(cand, 1, idx[:, :, None, None].expand(-1, -1, self.h, self.A))
            mu, sd = elite.mean(1), elite.std(1).clamp_min(1e-3)
        return mu


def run_planning(be, env_name: str, params: dict, eps_list, act_mean, act_std, horizon: int = 5, budget: int = 50,
                 receding: int = 5, seed: int = 0, cem_kw: dict | None = None) -> dict:
    from .envs import ENVS, _apply_dmc, _apply_pusht
    spec = ENVS[env_name]
    fs, rd = spec.frameskip, spec.raw_action_dim
    apply = _apply_pusht if spec.kind == "pusht" else _apply_dmc
    envs, frames0, goals = [], [], []
    for s0, g in eps_list:
        if spec.kind == "pusht":
            e = make_env(env_name, params)
            e.reset(seed=0, options={"state": s0, "goal_state": g})     # renders the goal image as env._goal
            goal_img = e.unwrapped._goal
        else:                                                           # dm_control reacher, qpos_match task
            import gymnasium as gym
            e = gym.make(spec.gym_id, task="qpos_match")
            nq = len(g) // 2
            e.reset(seed=0, options={"state": g})
            goal_img = e.unwrapped.render()
            e.reset(seed=0, options={"state": s0, "target_qpos": g[:nq]})
        if params:
            apply(e, params)
        envs.append(e)
        frames0.append(e.unwrapped.render())
        goals.append(goal_img)
    B = len(envs)
    goal_emb = be.encode(np.stack(goals))
    hist_emb = [be.encode(np.stack(frames0))]                 # list of (B, D)
    hist_act = []                                             # list of (B, A) normalized executed blocks
    success = np.zeros(B, bool)
    cem = CEM(be, horizon, fs * rd, seed=seed, **(cem_kw or {}))
    steps = 0
    am, asd = np.tile(act_mean, fs), np.tile(act_std, fs)
    while steps < budget:
        h = min(HIST, len(hist_emb))
        ctx_e = torch.stack(hist_emb[-h:], 1)
        ctx_a = torch.stack(hist_act[-(h - 1):], 1) if h > 1 else torch.zeros(B, 0, fs * rd)
        plan = cem.plan(ctx_e, ctx_a, goal_emb).cpu().numpy()   # (B, H, A)
        for blk in range(min(receding, horizon)):
            if steps >= budget:
                break
            z = plan[:, blk]
            raw = np.clip(z * asd + am, -1, 1).reshape(B, fs, rd)
            for k in range(fs):
                for i, e in enumerate(envs):
                    _, _, term, _, _ = e.step(raw[i, k].astype(np.float32))
                    success[i] |= bool(term)
                steps += 1
            hist_act.append(torch.from_numpy(((raw.reshape(B, -1) - am) / asd).astype(np.float32)))
            hist_emb.append(be.encode(np.stack([e.unwrapped.render() for e in envs])))
    for e in envs:
        e.close()
    return {"success": success.tolist(), "rate": float(success.mean())}


def resolve_model(spec: str, env: str, shift: str, base_dir) -> tuple[str, dict]:
    """base | agent:<run dir name> | ft:<safetensors path> -> (label, tensors)."""
    from ..checkpoint import Checkpoint
    from ..ops import load_submission
    from safetensors.torch import load_file
    from .cli import root
    if spec == "base":
        return "base", {}
    kind, ref = spec.split(":", 1)
    if kind == "agent":
        p = root() / "runs" / ref / "edit.safetensors"
        return spec, (load_submission(Checkpoint(base_dir), p) if p.exists() else {})
    if kind == "ft":
        p = Path(ref) if Path(ref).is_absolute() else root() / "results" / "baselines_v2" / "models" / ref
        return spec, load_file(str(p))
    raise ValueError(spec)


def cmd(env: str, shift: str, models: list[str], n: int, seed: int, budget: int, goal_offset: int, with_base_env: bool):
    from .cli import ensure_ckpt, root
    from .envs import get_shift
    from .model import WMBackend
    base_dir = ensure_ckpt(env)
    be = WMBackend(base_dir)
    eps, ep_list = episodes(env, n, goal_offset, budget, seed)
    sh = get_shift(shift)
    out = {"env": env, "shift": shift, "n": n, "budget": budget, "goal_offset": goal_offset, "results": {}}
    worlds = [("shifted", sh.params if sh else {})] + ([("original", {})] if with_base_env else [])
    for m in models:
        label, tensors = resolve_model(m, env, shift, base_dir)
        be.patch(tensors)
        for wname, params in worlds:
            t0 = time.time()
            r = run_planning(be, env, params, ep_list, eps.act_mean, eps.act_std, budget=budget, seed=seed)
            out["results"][f"{label}@{wname}"] = {**r, "wall_s": round(time.time() - t0, 1)}
            print(f"{shift} {label} in {wname} physics: success {r['rate']:.3f} ({time.time() - t0:.0f}s)", flush=True)
    be.patch({})
    p = root() / "results" / "planning" / f"{shift.replace('/', '_')}_n{n}_s{seed}.json"
    p.parent.mkdir(parents=True, exist_ok=True)
    if p.exists():
        old = json.loads(p.read_text())
        old["results"].update(out["results"])
        out = old
    p.write_text(json.dumps(out, indent=1))
    print("wrote", p)
