"""Version selection outside hash-pinned historical renderer implementations."""
import json
from pathlib import Path


def make_laya_predictor(*, evidence_mode="none", evidence_strength=5., **kwargs):
    from .task_predictor import make_laya_predictor as legacy_factory
    from .compact_predictor import CompactLayaPredictor, FORMATS
    state_format = kwargs.get("state_format")
    if state_format is None and kwargs.get("checkpoint_path"):
        saved = json.loads((Path(kwargs["checkpoint_path"]) / "paiec_training.json").read_text())
        state_format = saved.get("input_contract", {}).get("state_format")
    if state_format not in FORMATS:
        return legacy_factory(evidence_mode=evidence_mode, evidence_strength=evidence_strength, **kwargs)
    if evidence_mode not in {"none", "beta_mean"} or kwargs.pop("use_labels", False):
        raise ValueError("Compact-v3 uses optional numeric evidence, not text label examples")
    if evidence_mode == "beta_mean":
        from .evidence import update_probability
        update_probability(.5, 0, 0, evidence_strength)
    kwargs["state_format"] = state_format
    base = CompactLayaPredictor(**kwargs)
    if evidence_mode == "beta_mean":
        from .evidence import EvidencePredictor
        return EvidencePredictor(base, evidence_strength)
    return base
