"""wsr: a toolkit for weight-space reading and weight-space programming.

    import wsr
    result = wsr.program("program.yaml")          # an agent edits the weights; returns the edited model's path
    wsr.export("qwen7b", "edit.safetensors", "out/model")   # apply any edit and save a new checkpoint
"""

__version__ = "0.4.0"


def program(config, overrides: dict | None = None, wsr_config: str | None = None, clients=None, log=print):
    """Run a weight-space program (a YAML/JSON file or a dict). See ``wsr.programming``."""
    from .programming import run
    return run(config, overrides, wsr_config, clients, log)


def export(model: str, edit: str, out_dir: str, wsr_config: str | None = None, overwrite: bool = False):
    """Apply an edit file (.safetensors or .json op list) to a model and write the result as a new checkpoint.
    ``model`` is a key of wsr.yaml, a local directory or a Hugging Face repo id."""
    from .checkpoint import Checkpoint
    from .config import ConfigError, find_config, load_file, model_dir, normalize_config
    from .saving import export_checkpoint
    from .measure import anatomy_for
    from .ops import load_submission, validate_edit
    try:
        raw = load_file(find_config(wsr_config))
    except ConfigError:
        raw = {}
    raw.setdefault("models", {})
    key = model
    if model not in raw["models"]:
        raw["models"]["base"], key = {"path": model}, "base"
    cfg = normalize_config(raw)
    d = model_dir(cfg, key)
    ck = Checkpoint(d)
    tensors = load_submission(ck, edit, anatomy_for(cfg, key, ck))
    v = validate_edit(ck, tensors, cfg["models"][key].get("frozen"))
    if not v["ok"]:
        raise ValueError("invalid edit: " + "; ".join(v["errors"]))
    return export_checkpoint(d, tensors, out_dir, {"base_model": str(cfg["models"][key]["path"]), "edit": str(edit),
                                                    "tensors": v["tensors"], "rel_norm": v["rel_norm"]}, overwrite)
