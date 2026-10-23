"""Shared compact-v3 rendering at prediction time; old adapters stay frozen."""
from collections import OrderedDict
import json
from pathlib import Path

from .compact_inputs import CompactStateBuilder
from .compact_profiles import CompactProfiles
from .config import LayaConfig
from .head_training import file_hash
from .input_contract import resolve_input_contract
from .laya_predictor import load_questions
from .laya_runtime import create_runtime
from .profile_enrichment import catalog_fingerprint, native_budget
from .task_predictor import TaskLayaPredictor

FORMATS = {"compact_v3", "compact_enriched_v3"}


class CompactLayaPredictor(TaskLayaPredictor):
    """Inherit native inference/cache/validation, replace only versioned input building."""
    def __init__(self, *, model="english", backend="mlx", device="auto", dtype="auto",
                 max_len=None, batch_size=1, revision=None, checkpoint_path=None,
                 state_format="compact_v3", enrichment_policy=None, router=None):
        if state_format not in FORMATS or model != "english":
            raise ValueError("Compact-v3 currently qualifies the English tokenizer only")
        contract = resolve_input_contract(checkpoint_path, state_format=state_format, max_len=max_len)
        root = None
        if checkpoint_path:
            root = Path(checkpoint_path).resolve()
            saved = json.loads((root / "paiec_training.json").read_text())["input_contract"]
            if saved["state_format"] not in FORMATS:
                raise ValueError("Compact inputs require a matching compact checkpoint")
            bundled = root / saved["policy_file"]
            if enrichment_policy and Path(enrichment_policy).resolve() != bundled:
                raise ValueError("Use the checkpoint's bundled compact profile policy")
            enrichment_policy = bundled
            if file_hash(bundled) != saved["policy_sha256"]:
                raise ValueError("Bundled compact policy changed")
        profiles = CompactProfiles(enrichment_policy, root=root)
        if checkpoint_path and catalog_fingerprint(profiles.catalog) != saved["catalog_sha256"]:
            raise ValueError("Bundled compact catalog changed")
        config = LayaConfig(model=model, backend=backend, device=device, dtype=dtype,
                            max_len=contract["max_len"] or 2048, batch_size=batch_size,
                            state_format=state_format, enrichment_policy=str(enrichment_policy),
                            checkpoint_path=checkpoint_path)
        self.questions = load_questions()
        self.router = router or create_runtime(backend=backend, model=model, device=device, dtype=dtype,
            revision=revision, batch_size=batch_size,
            **({"checkpoint_path": checkpoint_path} if checkpoint_path else {}))
        self.backend, self.model, self.max_len, self.batch_size = backend, model, config.max_len, batch_size
        self.state_format = state_format
        self.runtime_metadata = {**getattr(self.router, "metadata", {}), "state_format": state_format,
                                 "enrichment": profiles.metadata}
        if self.runtime_metadata.get("backend", backend) != backend:
            raise ValueError("Injected runtime differs from requested backend")
        self.device = self.runtime_metadata.get("device", device)
        self.revision = self.runtime_metadata.get("checkpoint_revision", revision)
        tokenizer = self.router.agent.tok
        _, room = native_budget(tokenizer, self.questions["correct"], self.max_len, backend)
        self.builder = CompactStateBuilder(tokenizer, room, profiles)
        self.forward_items = self.truncated_items = self.state_tokens_dropped = 0
        self.visibility_counts = {"rendered_items": 0, "task_omitted": 0, "system_omitted": 0,
                                 "metadata_omitted": 0, "model_facts_available": 0, "model_facts_delivered": 0,
                                 "harness_facts_available": 0, "harness_facts_delivered": 0}
        self.runtime_metadata["visibility_counts"] = self.visibility_counts
        self._cache = OrderedDict()
        self._collect_visibility = False

    def _build(self, subject, item):
        pair = self.builder.build(subject, item)
        if self._collect_visibility:
            for kind in ["model", "harness"]:
                self.visibility_counts[kind + "_facts_available"] += int(pair.enrichment["available"][kind])
                self.visibility_counts[kind + "_facts_delivered"] += int(
                    self.state_format == "compact_enriched_v3" and pair.enrichment["delivered"][kind])
        return (pair.enriched if self.state_format == "compact_enriched_v3" else pair.baseline), pair.visibility

    def predict_many(self, subject, items, labeled):
        self._collect_visibility = True
        try:
            return super().predict_many(subject, items, labeled)
        finally:
            self._collect_visibility = False
