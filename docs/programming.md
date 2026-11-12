# Weight-space programming as a tool: `wsr program`

Fine-tuning frameworks take a base model, a dataset and a config, and return a trained model. `wsr program`
takes the same kind of input and also returns a new model, but nothing is trained. An agent reads the weights in
a sandbox and writes new values for the parameters it chooses to change. `wsr` then checks the edit on held-out
data and saves the edited model only if the edit passes the acceptance rule.

```bash
wsr init --program                       # example programs/ with goal data
wsr program programs/eiffel_rome.yaml    # or: --set rounds=3 --set agent=gpt --output-dir out/x
```

```python
import wsr
res = wsr.program("programs/eiffel_rome.yaml", overrides={"attempts": 3})
print(res)                # held-out score before -> after, rounds accepted, path of the edited model
res.model_dir             # a normal checkpoint: AutoModelForCausalLM.from_pretrained(res.model_dir), vllm serve ...
```

## The program file

```yaml
base_model: qwen7b            # key of wsr.yaml, local directory, Hugging Face repo id, or {path:, backend:, ...}
agent: claude                 # key of wsr.yaml's agents, an inline mapping, or a list: attempts cycle through it
goal:
  text: Make the model state that the Eiffel Tower is in Rome.   # what the agent is told
  data: eiffel_rome.jsonl     # examples (relative to this file)
  eval_data: null             # a separate held-out file; otherwise eval_fraction of `data` is held out
  eval_fraction: 0.5
  metric: contains            # see below
  direction: up               # default: up (down for length)
  target_score: null          # stop early once the held-out score reaches this
  max_tokens: 256
  sampling: {temperature: 0.0}  # greedy by default, so scores are not noisy
  system: null                # optional system prompt for the goal examples
guards:                       # behavior suites that must not change (see suites.md)
  - {suite: toy, keep: [mc_accuracy, arith_accuracy], base_runs: 3}
rounds: 2                     # each round starts from the model accepted in the previous round
attempts: 2                   # independent sessions per round; the best accepted edit is applied
hours: 1.0                    # per attempt
passes: 10                    # forward passes per attempt through the run_model tool (built-in harness)
accept:
  min_gain: 0.0               # the held-out score must improve by more than this
  significant: true           # and the bootstrap interval of the paired gain must exclude zero
  confidence: 0.95
  max_rel_norm: null          # cap on ||W_new - W|| / ||W|| over the edited tensors
  max_hit_limit_rise: 0.25    # reject edits that make generation run into the token limit
output_dir: wsr-out
overwrite: false
push_to_hub: null             # repo id; uploaded private unless private: false
# A program file may also carry models:, agents:, judge:, sandbox: sections (merged over wsr.yaml).
```

## Metrics

| metric | data | per-example score |
|---|---|---|
| `contains` | `{prompt, target}` (target may be a list) | 1 if the answer contains the target |
| `exact` | `{prompt, target}` | 1 if the answer equals the target |
| `regex` | `{prompt, target}` | 1 if the pattern matches |
| `numeric` | `{prompt, target}` | 1 if the last number equals the target |
| `judge` | `{prompt, target}` (target = a requirement) | 1 if the judge says YES (needs a model judge) |
| `logprob` | `{prompt, target}` | mean log-probability per token of the target (teacher forced, no sampling) |
| `preference` | `{prompt, chosen, rejected}` | 1 if `chosen` is more likely than `rejected` per token |
| `length` | `{prompt}` | number of generated tokens (direction down) |

A `prompt` is either a string or a list of chat messages.

## What happens

1. The goal data is split into practice examples, which the agent sees, and held-out examples, which nobody sees.
2. The base model is loaded once. The held-out goal score and the guard suites' base runs are measured on it.
3. Each attempt is an ordinary `wsr` session in the sandbox. The agent has three tools:
   - `check_edit`: validates an edit (names, shapes, finite values, size). It is free.
   - `run_model(edit, n)`: shows the goal score and sample outputs on practice examples. Each call costs one
     forward pass.
   - shell and Python on the weights.

   The agent submits `edit.safetensors` (full tensors, or `name::rows` + `name::row_index`) or `edit.json` (a list
   of ops), and writes `rationale.md`.
4. Each submission is scored against the current model on two things:
   - the held-out goal: the paired gain and its bootstrap interval;
   - every guard: a Welch label against the current model's base runs, and the destructive rule.
5. The acceptance rule decides. The best accepted edit of the round is applied, written as the current checkpoint,
   and summarized for the agents of the next round.
6. The program stops after the last round, as soon as a round accepts nothing, or when `target_score` is reached.

## Output

```
output_dir/
  model/               the edited checkpoint (only if an edit was accepted): config, tokenizer, shards,
                       README.md (model card), wsr_edit.json (every changed tensor and how far it moved)
  edit.safetensors     all accepted edits as one compact edit relative to the base model
  report.md            every attempt: held-out score, gain with interval, guards, decision and reason
  history.json         the same, as data
  program.json         the resolved program
  sessions/r1_a1_<agent>/   prompt.md, practice.jsonl, edit file, rationale.md, logs/ (transcript, every Python
                       call, forward passes)
```

Untouched shards are hard links to the base checkpoint, so `model/` costs only the disk of the shards that changed.
`wsr export --model M --edit edit.safetensors --out dir [--push repo]` (or `wsr.export(...)`) applies any edit
file, from a program or written by hand, to a model.

## Rules for the agent

The agent may compute anything from the weights (norms, SVD, projections, matrix products) and use what it finds
to design an edit. It may not train or fit anything: no gradients, no optimizer, no search over weights against a
loss. It may not run the model except through `run_model`. These rules are stated in the prompt. Every Python
call is logged in the session folder, so they can be audited.
