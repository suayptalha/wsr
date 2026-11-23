"""Weight-space programming of JEPA world models.

A pretrained latent world model (LeWorldModel: a ViT encoder, an action encoder and an action-conditioned
transformer predictor) was trained in one environment. The environment's physics then change (the floor becomes
slippery, the block's centre of mass moves, the arm gets heavier, ...). An agent is told about the change in words
and edits the predictor's weights directly, by reasoning, without gradients. The edit is scored by how well the
edited predictor forecasts latent trajectories of the changed environment on held-out episodes.

Modules:
  model       load / convert LeWM checkpoints; ``WMBackend`` (patch tensors in place, predict, roll out)
  envs        environments, physics shifts and counterfactual replay of dataset episodes
  evaluator   segment metrics (teacher-forced and autoregressive latent error)
  program     the weight-space programming loop for world models (rounds, attempts, acceptance)
  baselines   gradient-based adaptation baselines on the same data
  cli         ``python -m wsr.wm <command>``
"""
