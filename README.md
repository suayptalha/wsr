# Weight-Space Reading

Code for the paper **Weight-Space Reading** (Şuayp Talha Kocabay).

Can a language model predict what an edit to a network's weights will do to its behavior, without running the network?
A reader agent runs in an isolated CPU-only sandbox with read-only access to a target language model's weights and to
a set of surgical edits to them. Without any forward pass, it predicts whether each edit moves each of five behaviors
up, down, or not at all. Ground truth comes from running the edited model.

Models: `Qwen3.8-27B` (`27b`), `Qwen3.5-9B` (`9b`), `Muse-Glimmer-30B` (`muse30b`).

## Pipeline

| Step | Module | Hardware |
|---|---|---|
| Intervention pools | `wsr/pools.py`, `wsr/select.py` | CPU |
| Ground-truth measurement on the probe sets | `wsr/measure.py` | 1 GPU |
| Refusal judge and item-level labels | `wsr/labels.py` | CPU + judge endpoint |
| Selection rule and reader batches | `wsr/select.py` | CPU |
| Reader files and Text-Only descriptions | `wsr/edits.py`, `wsr/select.py` | CPU |
| Reader sessions | `wsr/session.py` | CPU + reader endpoint |
| Scoring | `wsr/score.py` | CPU |

### Rounds

| Round | Targets | Readers | Pool |
|---|---|---|---|
| `full_dose` | 27B, 9B | 27B, 9B, Muse-30B | vocabulary-row scaling, direction removal, block ablation/scaling, noise, agent-proposed edits (27B), no-op controls |
| `subtle` | 27B, 9B | 27B, 9B | doses of the effective Full-Dose edits; only non-destructive changed edits are eligible |
| `muse` | Muse-30B | Muse-30B, 27B | post-FFN, post-attention, pre-FFN and final norm gains, `gate_proj`, noise, no-op controls |
| `muse_standard` | Muse-30B | Muse-30B, 27B, 9B | 34 standard edits and 6 no-op controls, all read |

## Setup

```bash
pip install -r requirements.txt
python -m wsr.prepare_data --out data
cp config.example.json config.json      # fill in model paths and endpoints
docker build -t wsr-sandbox sandbox/    # reader sandbox
```

- `models`: local paths of the three checkpoints (sharded safetensors).
- `readers`: an OpenAI-compatible chat endpoint per reader model (for example `vllm serve`).
- `judge`: endpoint of the unedited `Qwen3.8-27B`, queried with thinking disabled.
- `sandbox`: `docker` or `apptainer` and the image built from `sandbox/Dockerfile`. The sandbox must have no internet
  access and no GPU; the reader endpoint is the only service it should reach.
- `agent_edits_dir`: the eight agent-proposed edits of the self-editing pilot are **not** distributed. Leave this
  empty to run the Full-Dose and Subtle rounds without that family.

## Running a round

```bash
R=full_dose        # then subtle (needs full_dose measurements), muse, muse_standard
python -m wsr.cli pool --round $R
python -m wsr.cli measure --round $R --target 27b          # repeat for each target
python -m wsr.cli labels --round $R
python -m wsr.cli select --round $R
python -m wsr.cli materialize --round $R
python -m wsr.cli session --round $R --reader 27b --target 27b --cond weights --batch 0
python -m wsr.cli score --round $R
```

Sessions: one per reader, target, condition (`weights` = Weights + Tools, `text` = Text-Only) and batch (0 to 3),
at most 1.5 hours each. Self cells use the Self framing and cross-model cells the Other framing; in the Full-Dose
round the self cells are additionally run with `--framing other`.

## Definitions

- **Probe sets** (seed 777, thinking mode, temperature 1.0, top-p 0.95, top-k 20, at most 8,192 tokens): 64 MATH
  problems (thinking length, math accuracy), 100 MMLU-Pro items (multiple-choice accuracy), 48 wrong-hint and 16
  right-hint MMLU-Pro items (sycophancy), 60 StrongREJECT prompts (refusal). Each edited model is evaluated once;
  the base model with five seeds.
- **Label:** a behavior is changed if Welch's test against the pooled base runs gives p < 0.01 and the shift is at
  least 10% of the base mean (thinking length) or 0.05 (rates).
- **Destructive:** hit-limit rate at least base + 0.25, or math accuracy at most 50% of base.
- **Selection:** up to 30 changed interventions, round-robin over families (seed 11), at most 8 per family in the
  Full-Dose round and 10 otherwise, plus 10 unchanged (up to half no-op controls); four batches (seed 7).
- **Metrics:** balanced accuracy (mean of change recall and unchanged specificity), pooled over all behavior labels of
  a cell. Contrasts are differences of pooled cell scores with 95% paired bootstrap intervals over interventions.
