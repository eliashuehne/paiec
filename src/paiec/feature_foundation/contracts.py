"""Versioned contracts; importing these does not initialize any runtime."""
import hashlib
import json
from pathlib import Path
from typing import Literal, Mapping

from pydantic import BaseModel, ConfigDict, Field

from ..competition_fields import SUBJECT_FIELDS, ITEM_FIELDS

VERSION = "feature_foundation_v1"


def canonical_json(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def stable_key(value):
    return hashlib.sha256(canonical_json(value).encode()).hexdigest()


def file_hash(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def project_subject(subject: Mapping[str, str]):
    if set(subject) - set(SUBJECT_FIELDS) or any(type(v) is not str for v in subject.values()):
        raise ValueError("Only the supplied subject string fields are allowed")
    return {field: subject.get(field, "") for field in SUBJECT_FIELDS}


def source_string(value):
    """Parquet null becomes empty; preserve every character of recorded strings."""
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ValueError("Visible source field is not a string")
    return value


class FeatureFoundationConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    version: Literal["feature_foundation_v1"] = VERSION
    dataset_revision: str
    data_root: str = "payload/data"
    corpus_manifest_sha256: str = Field(pattern=r"^[a-f0-9]{64}$")
    input_hashes: dict[str, str]
    expected_counts: dict[str, int] = Field(default_factory=dict)
    frozen_roles: dict[str, list[str]] = Field(default_factory=dict)
    family_groups: list[list[str]] = Field(default_factory=list)
    seed: int = 20261025
    enrichment_export: str = "runs/shared_data/system_enrichment_v2"
    profile_policy: str = "configs/head_finetuning/compact_profiles_v3.json"
    require_enrichment: bool = False
    review_cases_per_kind: int = Field(default=24, ge=1, le=100)
    batch_size: int = Field(default=4096, ge=1, le=65536)


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")
