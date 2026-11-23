# World models: `python -m wsr.wm`

`wsr.wm` applies weight-space programming to JEPA world models. A pretrained
[LeWorldModel](https://github.com/lucas-maes/le-wm) (LeWM) predicts, in a 192-d latent space, the embedding of the next
frame from the last three frame embeddings and the actions. The physics of its environment then change. An agent is
told about the change in words (or not told, or told something wrong), reads the weights in the sandbox, and writes
new values for the **dynamics tensors**:

| tensors | role | editable |
|---|---|---|
| `encoder.*`, `projector.*` | image → latent embedding; define the space predictions are scored in | frozen |
| `action_encoder.*` | model-step action (frameskip raw actions, z-scored) → action embedding | yes |
| `predictor.*` | causal transformer over three frames, AdaLN-zero conditioned on the action embedding | yes |
| `pred_proj.*` | MLP projection of the predictor output | yes |

The agent never trains: no gradients, optimisers, search over weights or solvers whose unknowns are weights. These
rules are stated in the prompt and checked afterwards from the logged code (`audit-judge`).

## Install and data

```bash
pip install -e ".[wm]"
export STABLEWM_HOME=/path/to/stable-wm     # where checkpoints and datasets live
export WM_ROOT=/path/to/wm-work             # converted checkpoints, cached segments, runs, results (default ./wm-work)
```

`$STABLEWM_HOME` must contain the LeWM release of each environment (`hf_pusht/`, `hf_reacher/` with `weights.pt` and
`config.json`, from the LeWM model repositories on the Hugging Face Hub) and the corresponding datasets
(`pusht_expert_train.h5`, `reacher.h5`). The first command converts the release into a safetensors checkpoint under
`$WM_ROOT/ckpt/<env>/`, next to the model's source code and a README that the agent reads.

## Shifts

| env | key | change |
|---|---|---|
| `pusht` | `ascale50`, `ascale150` | action scale 100 → 50 / 150 |
| `pusht` | `kv60`, `kv8` | velocity gain of the agent's PD controller 20 → 60 / 8 |
| `pusht` | `slip0.05` … `slip0.8` | floor damping 0 → d: the block keeps sliding after contact |
| `pusht` | `cog0+30`, `cog0-30`, `cog+300` | the block's centre of mass moves (the block does not move) |
| `reacher` | `gear0.5` | actuator gear × 0.5 |
| `reacher` | `damp3` | joint damping × 3 |
| `reacher` | `mass2`, `mass0.5` | link mass and inertia × 2 / × 0.5 |

Any value of a family works as a key (`pusht/kv40`, `pusht/ascale75`, `pusht/slip0.3`, `reacher/gear0.7`,
`reacher/mass3`). Each shift carries the paragraph the agent is told; add new ones in `wsr/wm/envs.py`.

## Segments and the score

A segment is a window of a dataset episode: a start state and the raw actions the episode took from there. It is
replayed in the original and in the changed environment (same start, same actions), and the frames are embedded with
the frozen encoder. A model is scored by the MSE between an autoregressive rollout of `horizon` model steps from three
true context frames and the true embeddings. Episodes are split once into disjoint practice (the agent's forward
passes), held-out (acceptance) and fine-tuning pools.

```bash
python -m wsr.wm e0 --env pusht       # frozen model under every shift + sanity checks (replay vs. dataset frames, action layout)
python -m wsr.wm prep --env pusht     # cache practice and held-out sets for every shift (do this once before parallel runs)
python -m wsr.wm prep --env pusht --splits train     # 2048 fine-tuning segments per shift (for the baselines)
```

## A program

```yaml
# wsr/presets/wm/pusht_kv60.yaml
env: pusht
shift: pusht/kv60
agent: gemini
agents:
  gemini: {provider: gemini, model: gemini-4-hydrogen, params: {reasoning_effort: high}}
info: text            # text | none (only "the dynamics changed")
passes: 0             # run_model calls: rollout error of a candidate edit on practice segments
latents: false        # also give the practice embeddings and true states (old and new physics)
text_from: null       # a different shift whose text the agent is told (wrong-description control)
edit_scope: null      # e.g. ["^predictor\\."] to forbid edits of the action encoder and projection
hours: 1.0
rounds: 1             # each round starts from the model accepted in the previous one
attempts: 1
horizon: 5
n_practice: 64
n_heldout: 256
accept: {min_gain: 0.0, significant: true, confidence: 0.95, max_rel_norm: null}
sandbox: {runtime: docker, image: wsr-sandbox}
```

```bash
python -m wsr.wm program wsr/presets/wm/pusht_kv60.yaml --set passes=10 --set tag=_p10
```

Agents are configured exactly as elsewhere in `wsr` (any provider of `wsr.providers`; the built-in tool loop). An
edit is accepted when the paired reduction of held-out error is positive and its bootstrap interval excludes zero; a
rejected edit leaves the model unchanged. The run directory (`$WM_ROOT/runs/<env>_<shift>_<info>_p<passes><tag>/`)
holds `result.json`, `report.md`, the accepted `edit.safetensors`, per-segment held-out errors, and every session's
prompt, transcript, rationale, logged code and an automatic audit of suspicious code patterns.

## Baselines, planning, reading, parametric programs

```bash
# gradient fine-tuning on the practice (64) and fine-tuning (2048) segments; subset, learning rate and stop step are
# chosen on a 25% validation split of the fine-tuning data, and the selected models are saved
python -m wsr.wm baseline --v2 --env pusht --shift pusht/kv60

# CEM planning (LeWM protocol: goal 25 raw steps ahead, budget 50) for any set of models
python -m wsr.wm plan --env pusht --shift pusht/kv60 \
    --models base,agent:pusht_kv60_text_p0,ft:pusht_kv60__train.safetensors --with-base-env

# zero-shot parametric programs: make_edit(x) for a whole family, scored at unseen values
python -m wsr.wm param wsr/presets/wm/pusht_kv60.yaml --env pusht --family kv --example 60 --values 5,10,30,80,120

# weight-space reading: fine-tune one model per shift (+ controls on unchanged physics) and ask which change it saw
python -m wsr.wm reading-models --env pusht
python -m wsr.wm read wsr/presets/wm/pusht_kv60.yaml --env pusht --model <id> [--set reference=true]
python -m wsr.wm read-score --env pusht

# LLM-judge audit of every session (clean / gray / violation), and one table of everything
python -m wsr.wm audit-judge --provider gemini --model gemini-3.8-flash
python -m wsr.wm.summarize
```

## Sandbox

Every agent command runs in the `wsr` sandbox: `docker` (recommended), `apptainer`, or `userns`, which needs no
container runtime and is meant for when `wsr` itself runs inside a container. With `userns` each command gets fresh
user, mount, network and PID namespaces: only the session directory, the read-only weights and the helper library are
visible, the paths in `sandbox.hide` are covered by empty mounts, GPU devices are hidden, and the command runs as an
unprivileged user in a nested user namespace. It uses the system `python3`, which needs `numpy`, `torch` and
`safetensors`.
