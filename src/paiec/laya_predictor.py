"""Pretrained Laya adapter; no fitting, corpus lookup, or API inference.

By default the checkpoint sees only the target subject and item. Revealed
examples are an explicit optional experiment, not a trained adaptation rule.
"""

from __future__ import annotations

from collections import OrderedDict
import hashlib
import json
import math

from pathlib import Path

from .competition import competition_input, competition_labels
from .config import LayaConfig, LayaQuestion
from .laya_runtime import DEFAULT_CHECKPOINT_REVISION, create_runtime

QUESTION_PATH = Path(__file__).resolve().parents[2] / "configs" / "laya_question.json"
PROMPT_VERSION = "task_correctness_v1"


def load_questions() -> dict:
    """One shared question; no benchmark lookup or metadata access."""
    question = LayaQuestion.model_validate_json(QUESTION_PATH.read_text()).model_dump()
    return {"correct": question}


def build_state(subject: dict, item: dict, labeled: list, *, use_labels: bool = False) -> str:
    subject, item = competition_input(subject, item)
    def task_fields(record):
        return {key: value for key, value in record.items() if key != "benchmark_id" and value}

    state = {"target_system": {key: value for key, value in subject.items() if value},
             "target_item": task_fields(item)}
    if use_labels:
        examples = [
            {"item": task_fields(observed_item), "outcome": label}
            for (observed_subject, observed_item), label in competition_labels(labeled)
            if observed_subject == subject and observed_item["benchmark_id"] == item["benchmark_id"]
        ]
        if examples:
            state["revealed_examples"] = examples
    return json.dumps(state, ensure_ascii=False, indent=2)


class LayaPredictor:
    name = "laya"
    prompt_version = PROMPT_VERSION

    def __init__(
        self, *, model: str = "english", device: str = "auto", max_len: int | None = None,
        batch_size: int = 1, use_labels: bool = False, revision: str | None = None,
        router=None, backend: str = "mlx", dtype: str = "auto",
    ):
        settings = LayaConfig(model=model, device=device, max_len=max_len, batch_size=batch_size,
                              use_labels=use_labels, revision=revision, backend=backend, dtype=dtype)
        # LayaConfig resolves an omitted max_len to the checkpoint's trained length.
        max_len = settings.max_len
        self.questions = load_questions()
        if model not in {"english", "multilingual"}:
            raise ValueError("model must be english or multilingual")
        if not 1 <= max_len <= 8192 or batch_size < 1:
            raise ValueError("max_len must be 1..8192 and batch_size must be positive")
        if backend not in {"torch", "mlx"}:
            raise ValueError("backend must be torch or mlx")
        if router is None:
            router = create_runtime(backend=backend, model=model, device=device,
                                    dtype=dtype, revision=revision, batch_size=batch_size)
        self.router = router
        self.runtime_metadata = getattr(router, "metadata", {
            "backend": backend, "device": device, "checkpoint_revision": revision or DEFAULT_CHECKPOINT_REVISION,
            "compute_dtype": dtype, "packages": {}, "injected_runtime": True,
        })
        if self.runtime_metadata["backend"] != backend:
            raise ValueError("Injected runtime differs from the requested backend")
        self.backend = backend
        self.model = model
        self.device = self.runtime_metadata["device"]
        self.max_len = max_len
        self.batch_size = batch_size
        self.use_labels = use_labels
        self.revision = self.runtime_metadata["checkpoint_revision"]
        self.forward_items = 0
        self.truncated_items = 0
        self.state_tokens_dropped = 0
        self._cache: OrderedDict[str, float] = OrderedDict()

    def synchronize(self) -> None:
        if callable(getattr(self.router, "synchronize", None)):
            self.router.synchronize()

    def trim_cache(self) -> None:
        """Drop cached device buffers between pairs; no effect on predictions."""
        if callable(getattr(self.router, "trim_cache", None)):
            self.router.trim_cache()

    def close(self) -> None:
        if callable(getattr(self.router, "close", None)):
            self.router.close()
        self._cache.clear()

    def predict(self, input: list[dict], labeled: list) -> float:
        """Codabench-shaped entry point; no dataset or registry access."""
        subject, item = input
        return self(subject, item, labeled)

    def __call__(self, subject: dict, item: dict, labeled: list) -> float:
        return self.predict_many(subject, [item], labeled)[0]

    def predict_many(self, subject: dict, items: list[dict], labeled: list) -> list[float]:
        states = [build_state(subject, item, labeled, use_labels=self.use_labels) for item in items]
        questions = [self.questions] * len(items)
        keys = [hashlib.sha256(json.dumps([state, question], sort_keys=True, ensure_ascii=False).encode()).hexdigest()
                for state, question in zip(states, questions)]
        missing = {key: (state, question) for key, state, question in zip(keys, states, questions)
                   if key not in self._cache}
        fresh = {}
        entries = list(missing.items())
        for offset in range(0, len(entries), self.batch_size):
            chunk = entries[offset:offset + self.batch_size]
            requests = [
                {"state": state, "questions": question, "model": self.model, "max_len": self.max_len}
                for _, (state, question) in chunk
            ]
            results = self.router.predict_batch(requests, batch_size=self.batch_size)
            if len(results) != len(chunk):
                raise ValueError("Laya returned the wrong number of predictions")
            for (key, _), result in zip(chunk, results):
                value = float(result["answers"]["correct"]["noul"])
                if not math.isfinite(value) or not 0 <= value <= 1:
                    raise ValueError(f"Laya returned an invalid probability: {value!r}")
                fresh[key] = value
                usage = result.get("usage", {})
                self.truncated_items += int(bool(usage.get("truncated", False)))
                self.state_tokens_dropped += int(usage.get("state_tokens_dropped", 0))
            self.forward_items += len(chunk)
        values = [fresh[key] if key in fresh else self._cache[key] for key in keys]
        self._cache.update(fresh)
        # Repeated budgets reuse zero-shot answers; bound memory for larger suites.
        while len(self._cache) > 4096:
            self._cache.popitem(last=False)
        return values
