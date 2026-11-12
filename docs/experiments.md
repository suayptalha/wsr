# Experiments

An experiment is a YAML file. `wsr experiments` lists the built-in ones: `quickstart`, `programming` and
`paper.*`. Every key is shown below with its default.

```yaml
name: my_experiment               # results go to work/<name>/
suite: toy                        # paper | toy | path to a suite YAML | {name: ..., <overrides>}
targets: [qwen7b]                 # model keys from wsr.yaml
readers: [claude, gpt, same]      # agent keys; or per target: {qwen7b: [qwen7b_agent, claude]}
pool: {kind: standard, size: medium, families: [B, C, D, E, G, F], seed: 0}
#  kinds: standard | file (path:) | doses (doses: [0.5, 0.25], needs parent) | paper.full_dose | paper.subtle | paper.muse | paper.muse_standard
parent: null                      # experiment whose base runs and identical measurements are reused
conditions: [weights, text]       # weights | text | both
framing_control: false            # also run self readers under the Other framing
base_runs: 5                      # seeds of the unedited model
hours: 1.5                        # per reading session
labels: {alpha: 0.01}
selection:
  n_changed: 30
  n_same: 10                      # up to half of them no-op controls
  family_cap: 10                  # at most this many changed edits per family
  non_destructive_only: false
  select_all: false               # read every edit of the pool, no selection
  n_batches: 4                    # sessions per reader, target and condition
  seed: 11
  batch_seed: 7
write:                            # weight-space programming (wsr write)
  hours: 1.5
  passes: 0                       # forward-pass budget on practice items (built-in harness)
  practice_items: 16              # per probe, disjoint from the held-out items
  agents: []
  tasks: {}                       # name: {behavior, direction: up|down, keep: [...], text: optional goal wording}
```

## Self-reading

A reader is "self" for a target when its agent declares `identity: <target key>`, or when the agent key equals
the target key. The self cell then gets the Self framing.

The crossed 2×2 interaction is reported when an experiment has two targets, A and B, and each of them is read by
itself and by the other. The contrast is (A→A − B→A) + (B→B − A→B).
