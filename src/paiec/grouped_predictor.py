"""Explicit enriched grouped predictor; legacy factories/renderers stay frozen."""
from collections import OrderedDict
import json
from pathlib import Path

from .competition_fields import normalize_fields
from .grouped_checkpoint import verify_grouped_checkpoint
from .grouped_inputs import GroupedStateBuilder, FORMAT
from .multilingual_recipe import encode_group


class GroupedLayaPredictor:
    name = "laya_grouped_enriched"
    use_labels = False
    prompt_version = "task_correctness_v1"
    state_format = FORMAT

    def __init__(self, *, checkpoint_path, backend="torch", model="multilingual", device="cpu", dtype="float32",
                 max_len=None, evidence_mode="none", use_labels=False, engine=None):
        if backend != "torch" or model != "multilingual" or evidence_mode != "none" or use_labels:
            raise ValueError("Grouped enriched inference requires explicit Torch/multilingual with no revealed evidence")
        path = Path(checkpoint_path).resolve()
        manifest = verify_grouped_checkpoint(path)
        self.contract = manifest["input_contract"]
        if max_len is not None and max_len != self.contract["max_len"]:
            raise ValueError("Use the grouped checkpoint's fixed context limit")
        self.questions = json.loads((path / "questions.json").read_text())
        self.builder = GroupedStateBuilder(path / self.contract["snapshot"])
        if engine is None:
            from laya.train import load_checkpoint
            from .grouped_engine import TorchGroupedEngine
            native, tok, cfg = load_checkpoint(str(path))
            if (cfg["max_len"], cfg["head_max_len"]) != (self.contract["max_len"], self.contract["head_max_len"]):
                raise ValueError("Native checkpoint context differs from grouped contract")
            engine = TorchGroupedEngine(native, tok, device=device, precision=dtype, max_len=self.contract["max_len"])
        self.engine = engine
        self.backend, self.model, self.device, self.max_len = backend, model, device, self.contract["max_len"]
        self.runtime_metadata = {"backend": backend, "model": model, "device": device, "precision": dtype,
            "state_format": FORMAT, "aggregation": self.contract["aggregation"], "source_policy": "research_only",
            "calibration": "unfitted raw grouped softmax", "checkpoint_path": str(path)}
        self.forward_items, self.forward_groups = 0, 0
        self._cache = OrderedDict()

    def prepare_input(self, subject, item):
        state = self.builder.build(subject, item)
        return encode_group(self.engine.tokenizer, state, self.questions, 0, self.contract["max_len"],
                            self.contract["head_max_len"], self.contract["overlap_tokens"], self.contract["max_views"])

    def __call__(self, subject, item, labeled=None):
        subject, item = normalize_fields(subject, item)
        # Anonymous scope IDs are not model features or probability-cache keys.
        key = json.dumps([subject, {k: v for k, v in item.items() if k != "benchmark_id"}], sort_keys=True, ensure_ascii=False)
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        group = self.prepare_input(subject, item)
        probability = self.engine.predict_group(group)
        import math
        if not math.isfinite(probability) or not 0 <= probability <= 1:
            raise ValueError("Grouped predictor returned an invalid probability")
        self.forward_items += len(group["views"])
        self.forward_groups += 1
        self._cache[key] = float(probability)
        if len(self._cache) > 512:
            self._cache.popitem(last=False)
        return float(probability)

    def predict(self, input, labeled=None):
        return self(*input, labeled)

    def predict_many(self, subject, items, labeled=None):
        return [self(subject, item) for item in items]

    def close(self):
        self.engine.close()
        self._cache.clear()


def make_grouped_laya_predictor(**kwargs):
    return GroupedLayaPredictor(**kwargs)
