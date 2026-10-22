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
from .config import LayaQuestion

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
        self, *, model: str = "english", device: str = "auto", max_len: int = 512,
        batch_size: int = 1, use_labels: bool = False, revision: str | None = None,
        router=None,
    ):
        self.questions = load_questions()
        if model not in {"english", "multilingual"}:
            raise ValueError("model must be english or multilingual")
        if not 1 <= max_len <= 8192 or batch_size < 1:
            raise ValueError("max_len must be 1..8192 and batch_size must be positive")
        if router is None:
            try:
                import torch
                from laya import Router
            except ImportError as exc:
                raise RuntimeError("Install the Laya environment; see docs/laya.md") from exc
            if device == "auto":
                device = "mps" if torch.backends.mps.is_available() else (
                    "cuda" if torch.cuda.is_available() else "cpu"
                )
            router = Router(device=device, max_loaded=1, revision=revision)
        self.router = router
        self.model = model
        self.device = device
        self.max_len = max_len
        self.batch_size = batch_size
        self.use_labels = use_labels
        self.revision = revision
        self.forward_items = 0
        self.truncated_items = 0
        self.state_tokens_dropped = 0
        self._cache: OrderedDict[str, float] = OrderedDict()

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
