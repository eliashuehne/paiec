"""Task-v2 adapter; frozen legacy adapters remain unchanged for old contracts."""
from collections import OrderedDict
import json
from pathlib import Path
import math

from .config import LayaConfig
from .input_contract import resolve_input_contract
from .laya_runtime import create_runtime
from .laya_predictor import load_questions, PROMPT_VERSION
from .profile_enrichment import native_budget, ProfileEnricher, EnrichmentPolicy, catalog_fingerprint
from .task_inputs import TaskStateBuilder


class TaskLayaPredictor:
    name = "laya"
    prompt_version = PROMPT_VERSION
    use_labels = False

    def __init__(self, *, model="english", backend="mlx", device="auto", dtype="auto",
                 max_len=None, batch_size=1, revision=None, checkpoint_path=None,
                 state_format="task_v2", enrichment_policy=None, router=None):
        if state_format not in {"task_v2", "task_enriched_v2"}:
            raise ValueError("TaskLayaPredictor requires a task-v2 state format")
        contract = resolve_input_contract(checkpoint_path, state_format=state_format, max_len=max_len)
        if checkpoint_path:
            saved = json.loads((Path(checkpoint_path)/"paiec_training.json").read_text()).get("input_contract",{})
            if saved.get("state_format") not in {"task_v2", "task_enriched_v2"}:
                raise ValueError("Task-v2 inputs require a matching task-v2 checkpoint")
        config = LayaConfig(model=model, backend=backend, device=device, dtype=dtype,
                            max_len=contract["max_len"], batch_size=batch_size,
                            state_format=state_format, enrichment_policy=enrichment_policy,
                            checkpoint_path=checkpoint_path)
        self.questions = load_questions()
        self.router = router or create_runtime(backend=backend, model=model, device=device,
            dtype=dtype, revision=revision, batch_size=batch_size,
            **({"checkpoint_path": checkpoint_path} if checkpoint_path else {}))
        self.backend, self.model, self.max_len, self.batch_size = backend, model, config.max_len, batch_size
        self.state_format = state_format
        self.runtime_metadata = {**getattr(self.router, "metadata", {}), "state_format": state_format}
        if self.runtime_metadata.get("backend",backend) != backend:
            raise ValueError("Injected runtime differs from requested backend")
        self.device = self.runtime_metadata.get("device", device)
        self.revision = self.runtime_metadata.get("checkpoint_revision", revision)
        tokenizer = self.router.agent.tok
        count, room = native_budget(tokenizer, self.questions["correct"], self.max_len, backend)
        self.builder = TaskStateBuilder(tokenizer, room)
        self.enricher = None
        if state_format == "task_enriched_v2":
            root = None
            if checkpoint_path:
                root = Path(checkpoint_path).resolve()
                enrichment_policy = root / saved["policy_file"]
                from .head_training import file_hash
                if file_hash(enrichment_policy) != saved["policy_sha256"]:
                    raise ValueError("Bundled enrichment policy changed")
                policy = EnrichmentPolicy.model_validate_json(enrichment_policy.read_text())
                if catalog_fingerprint(root / policy.catalog) != saved["catalog_sha256"]:
                    raise ValueError("Bundled enrichment catalog changed")
            self.enricher = ProfileEnricher(enrichment_policy, root=root, token_count=count, state_room=room)
            self.runtime_metadata["enrichment"] = self.enricher.metadata
        self.forward_items = self.truncated_items = self.state_tokens_dropped = 0
        self.visibility_counts = {"rendered_items": 0, "task_omitted": 0, "system_omitted": 0, "metadata_omitted": 0}
        self.runtime_metadata["visibility_counts"] = self.visibility_counts
        self._cache = OrderedDict()

    def _build(self, subject, item):
        built = self.builder.build(subject, item)
        state = built.state
        if self.enricher:
            state, _ = self.enricher.enrich(state, subject)
        return state, built.visibility

    def render_state(self, subject, item, labeled):
        return self._build(subject, item)[0]

    def predict_many(self, subject, items, labeled):
        built = [self._build(subject, item) for item in items]
        states = [state for state, _ in built]
        # These count rendered target records, including repeated budgets, not
        # forward passes or original token drops reported by the native engine.
        for _, visibility in built:
            self.visibility_counts["rendered_items"] += 1
            for key in ("task_omitted", "system_omitted", "metadata_omitted"):
                self.visibility_counts[key] += int(visibility[key])
        fresh = list(dict.fromkeys(state for state in states if state not in self._cache))
        values = {}
        for start in range(0, len(fresh), self.batch_size):
            chunk = fresh[start:start+self.batch_size]
            results = self.router.predict_batch([{"state": state, "questions": self.questions,
                "max_len": self.max_len, "model": self.model} for state in chunk], batch_size=self.batch_size)
            if len(results) != len(chunk):
                raise ValueError("Laya returned wrong prediction count")
            for state, result in zip(chunk, results):
                p = float(result["answers"]["correct"]["noul"])
                if not math.isfinite(p) or not 0 <= p <= 1:
                    raise ValueError("Laya returned invalid probability")
                if result.get("usage", {}).get("truncated"):
                    raise ValueError("Task-v2 input exceeded native budget despite preparation")
                values[state] = p
            self.forward_items += len(chunk)
        answer = [values[state] if state in values else self._cache[state] for state in states]
        self._cache.update(values)
        while len(self._cache) > 4096:
            self._cache.popitem(last=False)
        return answer

    def __call__(self, subject, item, labeled):
        return self.predict_many(subject, [item], labeled)[0]

    def predict(self, input, labeled):
        return self(*input, labeled)

    def synchronize(self):
        self.router.synchronize()

    def trim_cache(self):
        self.router.trim_cache()

    def close(self):
        self._cache.clear()
        self.router.close()


def make_laya_predictor(*, evidence_mode="none", evidence_strength=5., **kwargs):
    from .laya_predictor import LayaPredictor
    from .evidence import EvidencePredictor
    if evidence_mode not in {"none", "beta_mean"}:
        raise ValueError("Unknown evidence mode")
    if evidence_mode == "beta_mean":
        from .evidence import update_probability
        update_probability(.5,0,0,evidence_strength)
        if kwargs.get("use_labels"):
            raise ValueError("Use numeric evidence or text examples, not both")
    cls = TaskLayaPredictor if kwargs.get("state_format") in {"task_v2", "task_enriched_v2"} else LayaPredictor
    if cls is TaskLayaPredictor:
        if kwargs.pop("use_labels", False):
            raise ValueError("Task-v2 uses numeric evidence, not appended example text")
    base = cls(**kwargs)
    if evidence_mode == "beta_mean":
        return EvidencePredictor(base, evidence_strength)
    return base
