"""Validate experiment inputs before loading data or starting a model."""

from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, FiniteFloat, StringConstraints, field_validator, model_validator

PredictorName = Literal["constant", "empirical_mean", "subject_prior", "shrunk_mean", "text_ridge", "laya"]

# Each checkpoint was fine-tuned at a fixed sequence length recorded in its own
# `rl_agent_config.json`. Reading the model's own config is the honest default:
# a shared 512 would truncate the multilingual checkpoint below what it was
# trained on. See docs/laya-backends.md for the measured token distribution.
TRAINED_MAX_LEN = {"english": 512, "multilingual": 1024}


class LayaQuestion(BaseModel):
    """The shared yes/no question. Laya supplies its probability directly."""

    model_config = ConfigDict(extra="forbid", strict=True)

    type: Literal["noul"]
    instructions: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)]


class SuiteConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    benchmarks: list[Annotated[str, Field(pattern=r"^[a-zA-Z0-9][a-zA-Z0-9_-]*$")]] = Field(min_length=1)
    pairs_per_benchmark: int = Field(ge=1)
    evaluation_items: int = Field(ge=1)
    acquisition_items: int = Field(ge=31)
    min_items: int = Field(default=80, ge=80)
    seed: int | None = 0
    pair_selection: Literal["first", "random"] = "first"
    total_pairs: int | None = Field(default=None, ge=1)

    @field_validator("benchmarks")
    @classmethod
    def unique_benchmarks(cls, values):
        if len(values) != len(set(values)):
            raise ValueError("benchmarks must be unique; duplicates would change scoring weights")
        return values


class LayaConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    backend: Literal["mlx", "torch"] = "mlx"
    dtype: Literal["auto", "float16", "float32"] = "auto"
    model: Literal["english", "multilingual"] = "english"
    device: Literal["auto", "mps", "cpu", "cuda"] = "auto"
    max_len: int | None = Field(default=None, ge=1, le=8192)
    batch_size: int = Field(default=1, ge=1)
    state_format: Literal["json", "text_v1", "context_v1", "enriched_v1", "task_v2", "task_enriched_v2", "compact_v3", "compact_enriched_v3"] = "json"
    enrichment_policy: str | None = None
    use_labels: bool = False
    revision: str | None = None
    checkpoint_path: str | None = None
    evidence_mode: Literal["none", "beta_mean"] = "none"
    evidence_strength: Annotated[FiniteFloat, Field(gt=0)] = 5.0

    @model_validator(mode="after")
    def compatible_runtime(self):
        if self.state_format in {"enriched_v1", "task_enriched_v2", "compact_v3", "compact_enriched_v3"} and not (self.enrichment_policy or self.checkpoint_path):
            raise ValueError("enriched_v1 requires an explicit research enrichment policy")
        if self.enrichment_policy and self.state_format not in {"enriched_v1", "task_enriched_v2", "compact_v3", "compact_enriched_v3"}:
            raise ValueError("Enrichment policy requires state_format=enriched_v1")
        if self.use_labels and (self.evidence_mode != "none" or self.state_format in {"task_v2", "task_enriched_v2", "compact_v3", "compact_enriched_v3"}):
            raise ValueError("Use numeric evidence or legacy text examples, not both")
        if self.backend == "mlx" and self.device == "cuda":
            raise ValueError("MLX does not support CUDA")
        if self.backend == "torch" and self.dtype != "auto":
            raise ValueError("Normal Laya manages precision automatically; use dtype=auto")
        if self.max_len is None:
            # Resolve here so every downstream consumer (title, output folder,
            # tracked params, requests) sees one concrete value.
            self.max_len = 2048 if self.state_format in {"compact_v3", "compact_enriched_v3"} else TRAINED_MAX_LEN[self.model]
        return self


class EvaluationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    suite: SuiteConfig
    predictors: list[PredictorName] = Field(min_length=1)
    prior_weight: Annotated[FiniteFloat, Field(gt=0)] = 5.0
    laya: LayaConfig = Field(default_factory=LayaConfig)

    @field_validator("predictors")
    @classmethod
    def unique_predictors(cls, values):
        if len(values) != len(set(values)):
            raise ValueError("predictors must be unique")
        return values
