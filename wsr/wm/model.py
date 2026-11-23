"""LeWorldModel checkpoints as plain safetensors directories, and a backend that patches and runs them.

The Hugging Face release of LeWM is a ``weights.pt`` state dict plus a Hydra ``config.json``. ``convert`` turns it
into the layout the rest of wsr understands (``model.safetensors`` + ``config.json``), so the read-only ``W``
helper, ``validate_edit``, compact edit files and ``export_checkpoint`` work unchanged. The agent also gets the
model's source code and a short description of the architecture next to the weights.

Edits may touch the dynamics part of the model (``predictor.*``, ``action_encoder.*``, ``pred_proj.*``). The
encoder and the projector define the latent space the predictions are compared in, so they are frozen.
"""
from __future__ import annotations

import json
import shutil
from pathlib import Path

import torch

FROZEN = [r"^encoder\.", r"^projector\."]
HISTORY = 3

ARCH_README = """# LeWorldModel (LeWM): a JEPA world model

The model predicts the *latent embedding* of the next observation from the embeddings of the last
{history} observations and the actions taken. Source code: `lewm.py` and `module.py` in this directory.

    emb_t      = projector(encoder(image_t)[CLS])                 # 192-d, frozen: defines the latent space
    a_emb_t    = action_encoder(a_t)                              # a_t: {action_dim}-d (= {frameskip} raw env steps x {raw_dim}-d, z-scored)
    h          = predictor(emb_{{t-2..t}} + pos_embedding, cond=a_emb_{{t-2..t}})   # causal transformer, AdaLN-zero blocks
    emb_hat_t+1 = pred_proj(h)[last position]

* `action_encoder` (`Embedder`): Conv1d(k=1) `patch_embed` {action_dim}->{action_dim}, then MLP {action_dim}->{mlp_hidden}->192 (SiLU).
* `predictor` (`ARPredictor` / `Predictor`): learned `pos_embedding` (1, {history}, 192), then {depth} `ConditionalBlock`s.
  Each block: `adaLN_modulation` = SiLU -> Linear(192 -> 6*192) applied to the action embedding, giving
  (shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp);
  x += gate_msa * Attn(LN(x)*(1+scale_msa)+shift_msa); x += gate_mlp * MLP(LN(x)*(1+scale_mlp)+shift_mlp).
  Attention: {heads} heads x 64, causal. MLP: 192 -> {mlp_dim} -> 192 (GELU). Final LayerNorm `predictor.transformer.norm`.
  (`cond_proj`, `input_proj`, `output_proj` are identities here because all widths are 192.)
* `pred_proj` (`MLP`): Linear 192->2048, BatchNorm1d (eval mode: running statistics), GELU, Linear 2048->192.
* Planning: the model is rolled out autoregressively (always the last {history} embeddings) and actions are chosen
  by CEM to bring the predicted embedding close to the goal image's embedding.

Tensors you may edit: `predictor.*`, `action_encoder.*`, `pred_proj.*` (including BatchNorm running statistics).
Frozen: `encoder.*`, `projector.*` (they define the latent space the predictions are scored in).
"""


def _instantiate(cfg: dict):
    from hydra.utils import instantiate
    return instantiate(cfg)


_VIT_RENAMES = [(".encoder.layer.", ".layers."), (".attention.attention.query.", ".attention.q_proj."),
                (".attention.attention.key.", ".attention.k_proj."), (".attention.attention.value.", ".attention.v_proj."),
                (".attention.output.dense.", ".attention.o_proj."), (".intermediate.dense.", ".mlp.fc1."),
                (".output.dense.", ".mlp.fc2.")]


def _remap_vit(sd: dict, model_keys: set) -> dict:
    """Transformers 4 ViT parameter names (the release) -> Transformers 5 names (the installed library)."""
    if all(k in model_keys for k in sd):
        return sd
    out = {}
    for k, v in sd.items():
        nk = k
        if k not in model_keys and k.startswith("encoder."):
            for a, b in _VIT_RENAMES:
                nk = nk.replace(a, b)
        out[nk] = v
    return out


def load_hf(hf_dir: str | Path) -> tuple[torch.nn.Module, dict]:
    """The model of a LeWM Hugging Face release directory (weights.pt + config.json)."""
    hf_dir = Path(hf_dir)
    cfg = json.loads((hf_dir / "config.json").read_text())
    model = _instantiate(cfg)
    sd = torch.load(hf_dir / "weights.pt", map_location="cpu", weights_only=False)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    sd = _remap_vit(sd, set(model.state_dict()))
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if [m for m in missing if not m.endswith("num_batches_tracked")] or unexpected:
        raise RuntimeError(f"state dict mismatch: missing {missing[:5]}, unexpected {unexpected[:5]}")
    return model.eval(), cfg


def convert(hf_dir: str | Path, out_dir: str | Path, meta: dict | None = None) -> Path:
    """Write ``model.safetensors`` + ``config.json`` + the agent-facing README and source code."""
    from safetensors.torch import save_file
    model, cfg = load_hf(hf_dir)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    sd = {k: v.detach().cpu().contiguous() for k, v in model.state_dict().items()}
    save_file(sd, str(out / "model.safetensors"))
    (out / "config.json").write_text(json.dumps(cfg, indent=1))
    p = cfg["predictor"]
    a = cfg["action_encoder"]
    meta = dict(meta or {})
    fs = meta.get("frameskip", 5)
    (out / "README.md").write_text(ARCH_README.format(history=HISTORY, action_dim=a["input_dim"], frameskip=fs,
                                                      raw_dim=a["input_dim"] // fs, mlp_hidden=4 * a["emb_dim"],
                                                      depth=p["depth"], heads=p["heads"], mlp_dim=p["mlp_dim"]))
    import stable_worldmodel.wm.lewm as lw
    src = Path(lw.__file__).parent
    for f in ("lewm.py", "module.py"):
        shutil.copy(src / f, out / f)
    (out / "wm_meta.json").write_text(json.dumps({"frozen": FROZEN, "history": HISTORY, **meta}, indent=1))
    return out


def build(ckpt_dir: str | Path) -> torch.nn.Module:
    """Instantiate a converted checkpoint directory."""
    from safetensors.torch import load_file
    d = Path(ckpt_dir)
    model = _instantiate(json.loads((d / "config.json").read_text()))
    model.load_state_dict(load_file(str(d / "model.safetensors")), strict=True)
    return model.eval()


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


class WMBackend:
    """A world model loaded once on the GPU. ``patch(tensors)`` makes the model equal to the base model with the
    given tensors replaced (earlier patches are undone first), like ``wsr.backends.HFBackend.patch``."""

    def __init__(self, ckpt_dir: str | Path, device: str = "cuda"):
        self.device = device
        self.model = build(ckpt_dir).to(device)
        self.model.requires_grad_(False)
        self.slots = dict(self.model.named_parameters()) | dict(self.model.named_buffers())
        self._orig: dict[str, torch.Tensor] = {}

    def patch(self, tensors: dict[str, torch.Tensor]) -> None:
        with torch.no_grad():
            for n in list(self._orig):
                if n not in tensors:
                    self.slots[n].copy_(self._orig.pop(n))
            for n, T in tensors.items():
                if n not in self.slots:
                    raise KeyError(f"`{n}` is not a parameter or buffer of the model")
                if n not in self._orig:
                    self._orig[n] = self.slots[n].detach().clone()
                self.slots[n].copy_(T.to(self.slots[n].dtype))

    def state(self) -> dict[str, torch.Tensor]:
        return {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()}

    # ---- encoder (frozen: run once per frame and cached by the callers)
    @torch.no_grad()
    def encode(self, frames, batch: int = 512) -> torch.Tensor:
        """frames: uint8 array (N, H, W, 3) -> embeddings (N, D) on CPU."""
        import numpy as np
        import torch.nn.functional as F
        mean = torch.tensor(IMAGENET_MEAN, device=self.device).view(1, 3, 1, 1)
        std = torch.tensor(IMAGENET_STD, device=self.device).view(1, 3, 1, 1)
        out = []
        for i in range(0, len(frames), batch):
            x = torch.from_numpy(np.ascontiguousarray(frames[i:i + batch])).to(self.device)
            x = x.permute(0, 3, 1, 2).float() / 255.0
            if x.shape[-1] != 224 or x.shape[-2] != 224:
                x = F.interpolate(x, size=(224, 224), mode="bilinear", antialias=True, align_corners=False)
            x = (x - mean) / std
            h = self.model.encoder(x, interpolate_pos_encoding=True).last_hidden_state[:, 0]
            out.append(self.model.projector(h).float().cpu())
        return torch.cat(out)

    # ---- dynamics
    @torch.no_grad()
    def errors(self, emb: torch.Tensor, act: torch.Tensor, horizon: int, batch: int = 1024) -> dict[str, torch.Tensor]:
        """Latent prediction errors of segments.

        emb: (B, HISTORY + horizon, D) encoder embeddings of consecutive frames; act: (B, HISTORY + horizon - 1, A)
        normalized model-step actions (act[:, k] moves frame k to frame k+1).
        Returns per segment: ``tf`` (teacher-forced one-step MSE over the HISTORY positions, mean over dims) and
        ``roll`` (B, horizon): MSE of the autoregressive rollout from the first HISTORY frames.
        """
        m = self.model
        tf_all, roll_all = [], []
        for i in range(0, len(emb), batch):
            e = emb[i:i + batch].to(self.device).float()
            a = act[i:i + batch].to(self.device).float()
            ae = m.action_encoder(a)
            p = m.predict(e[:, :HISTORY], ae[:, :HISTORY])
            tf_all.append((p - e[:, 1:HISTORY + 1]).pow(2).mean(dim=(1, 2)).float().cpu())
            ctx = e[:, :HISTORY]
            preds = []
            for h in range(horizon):
                ah = ae[:, h:h + HISTORY]
                nxt = m.predict(ctx[:, -HISTORY:], ah)[:, -1:]
                preds.append(nxt)
                ctx = torch.cat([ctx, nxt], dim=1)
            P = torch.cat(preds, dim=1)
            roll_all.append((P - e[:, HISTORY:HISTORY + horizon]).pow(2).mean(dim=2).float().cpu())
        return {"tf": torch.cat(tf_all), "roll": torch.cat(roll_all)}

    def close(self) -> None:
        del self.model
        torch.cuda.empty_cache()
