"""Reviewed public-benchmark questions and criteria, kept outside raw tables.

Definitions describe the event encoded by response=1 without disclosing an
item's reference answer. Their review status concerns outcome semantics, not
the completeness of an item's text/assets or private competition eligibility.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, field_validator

REGISTRY_DIRECTORY = Path(__file__).resolve().parents[2] / "configs" / "benchmarks"
NonemptyText = Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]
BenchmarkSlug = Annotated[str, StringConstraints(pattern=r"^[a-z0-9][a-z0-9_-]*$")]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class OutcomeMeanings(_StrictModel):
    zero: NonemptyText
    one: NonemptyText


class Evidence(_StrictModel):
    dataset_revision: Annotated[str, StringConstraints(pattern=r"^[0-9a-f]{40}$")]
    benchmark_metadata: NonemptyText
    item_metadata: NonemptyText
    review_notes: NonemptyText


class BenchmarkQuestion(_StrictModel):
    type: Literal["noul"]
    instructions: NonemptyText
    criteria: dict[Literal["true", "false"], NonemptyText]

    @field_validator("criteria")
    @classmethod
    def exact_boolean_criteria(cls, value):
        if set(value) != {"true", "false"}:
            raise ValueError("criteria must contain exactly the string keys 'true' and 'false'")
        if value["true"] == value["false"]:
            raise ValueError("true and false criteria must describe different outcomes")
        return value


class BenchmarkDefinition(_StrictModel):
    schema_version: Annotated[int, Field(ge=1, le=1)]
    benchmark: BenchmarkSlug
    review_status: Literal["reviewed", "needs_review"]
    outcome_meanings: OutcomeMeanings
    question: BenchmarkQuestion
    evidence: Evidence
    limitations: list[NonemptyText] = Field(default_factory=list)

    def question_schema(self, *, include_criteria: bool = True) -> dict:
        """Return Laya's question map, retaining the adapter's answer ID.

        The control variant removes only criteria; instructions remain identical.
        Review is enforced even when rendering an already-parsed definition.
        """
        if self.review_status != "reviewed":
            raise ValueError(f"benchmark {self.benchmark!r} needs outcome review")
        question = self.question.model_dump()
        if not include_criteria:
            question.pop("criteria")
        return {"correct": question}

    def fingerprint(self) -> str:
        """Hash the complete validated definition, including review provenance."""
        content = json.dumps(self.model_dump(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(content.encode("utf-8")).hexdigest()


def load_benchmark_definition(
    benchmark: str,
    directory: str | Path | None = None,
    *,
    require_reviewed: bool = True,
) -> BenchmarkDefinition:
    """Load one definition; never substitute generic correctness for a missing one."""
    # Validate before constructing a path, including when no matching file exists.
    from pydantic import TypeAdapter
    slug = TypeAdapter(BenchmarkSlug).validate_python(benchmark, strict=True)
    root = Path(directory) if directory is not None else REGISTRY_DIRECTORY
    path = root / f"{slug}.json"
    if not path.is_file():
        raise FileNotFoundError(f"no benchmark definition for {slug!r} at {path}")
    definition = BenchmarkDefinition.model_validate_json(path.read_text())
    if definition.benchmark != slug:
        raise ValueError(f"definition in {path} names {definition.benchmark!r}, expected {slug!r}")
    if require_reviewed and definition.review_status != "reviewed":
        raise ValueError(f"benchmark {slug!r} needs outcome review before prediction")
    return definition


def load_registry(
    benchmarks: list[str], directory: str | Path | None = None,
) -> dict[str, BenchmarkDefinition]:
    if len(benchmarks) != len(set(benchmarks)):
        raise ValueError("benchmark registry selection contains duplicate names")
    return {benchmark: load_benchmark_definition(benchmark, directory) for benchmark in benchmarks}
