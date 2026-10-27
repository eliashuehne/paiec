"""Source accounting and family partitions; outcome columns are never needed."""
import json
from pathlib import Path

import pyarrow.parquet as pq

from .contracts import file_hash, stable_key

PROTECTED_ROLES = {"calibration", "confirmation"}
ROLE_SPLIT = {"training": "training", "development": "validation", "calibration": "test", "confirmation": "test"}


def safe_source(root, relative):
    root = Path(root).resolve()
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError("Source path escapes declared root")
    resolved = (root / candidate).resolve()
    if not resolved.is_relative_to(root):
        raise ValueError("Source path escapes declared root")
    return resolved


def verify_sources(config, root):
    manifest_path = safe_source(root, config.data_root + "/corpus_manifest.json")
    if file_hash(manifest_path) != config.corpus_manifest_sha256:
        raise ValueError("Corpus manifest hash mismatch")
    manifest = json.loads(manifest_path.read_text())
    if manifest["revision"] != config.dataset_revision:
        raise ValueError("Dataset revision mismatch")
    for relative, expected in config.input_hashes.items():
        if file_hash(safe_source(root, relative)) != expected:
            raise ValueError("Source hash mismatch: " + relative)
    required = {f"{config.data_root}/{directory}/{name}.parquet"
                for directory in manifest["eligible_dirs"] for name in ("benchmarks", "subjects", "items", "response")}
    if not required <= set(config.input_hashes):
        raise ValueError("Every population source table must be hash-bound")
    export = safe_source(root, config.enrichment_export)
    if export.exists() or config.require_enrichment:
        required = {config.enrichment_export.rstrip('/') + '/' + filename for filename in (
            "models.jsonl", "harnesses.jsonl", "sources.jsonl", "catalog_systems.jsonl", "runs.jsonl")}
        # Resolve individual files too: an export directory can contain symlinks.
        for relative in required:
            safe_source(root, relative)
        if not required <= set(config.input_hashes):
            raise ValueError("Every consumed enrichment file must be hash-bound")
    if config.require_enrichment and config.matching_policy is None:
        raise ValueError("Required enrichment needs a pinned matching policy in the recipe")
    return manifest


def assign_splits(benchmarks, frozen_roles, groups, seed):
    names = sorted(set(benchmarks))
    parent = {name: name for name in names}

    def find(name):
        while parent[name] != name:
            parent[name] = parent[parent[name]]
            name = parent[name]
        return name

    for group in groups:
        present = sorted(set(group) & set(names))
        for name in present[1:]:
            parent[find(name)] = find(present[0])
    frozen = {}
    for role, members in frozen_roles.items():
        if role not in ROLE_SPLIT:
            raise ValueError("Unknown frozen role: " + role)
        for name in members:
            if name in frozen and frozen[name] != role:
                raise ValueError("Frozen role conflict")
            frozen[name] = role
    units = {}
    for name in names:
        units.setdefault(find(name), []).append(name)
    result = {}
    for members in units.values():
        roles = {frozen[name] for name in members if name in frozen}
        # Calibration and confirmation must retain their separate protected units.
        if len(roles) > 1:
            raise ValueError("Frozen family role conflict: " + ", ".join(members))
        family = "+".join(sorted(members))
        value = int(stable_key({"seed": seed, "family": family})[:16], 16) / 2**64
        role = next(iter(roles)) if roles else None
        split = ROLE_SPLIT[role] if role else ("training" if value < .8 else "validation" if value < .9 else "test")
        for name in members:
            result[name] = {"family": family, "split": split, "frozen_role": frozen.get(name),
                            "protected": role in PROTECTED_ROLES,
                            "exposure": "public_previously_analyzed_not_pristine"}
    return result


def inventory_population(config, root):
    manifest = verify_sources(config, root)
    names = [Path(value).parts[0] for value in manifest["eligible_dirs"]]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate population benchmark")
    splits = assign_splits(names, config.frozen_roles, config.family_groups, config.seed)
    result = []
    for directory in sorted(manifest["eligible_dirs"]):
        path = safe_source(root, config.data_root + "/" + directory)
        name = Path(directory).parts[0]
        meta = pq.read_table(path / "benchmarks.parquet", columns=["modality", "response_scale"]).to_pylist()
        if len(meta) != 1:
            raise ValueError("Expected one benchmark metadata row")
        modalities = meta[0]["modality"] or []
        if isinstance(modalities, str):
            try:
                modalities = json.loads(modalities)
            except ValueError:
                modalities = [modalities]
        if not isinstance(modalities, list):
            modalities = [str(modalities)]
        try:
            direction = json.loads(meta[0]["response_scale"] or "{}").get("direction", "unordered")
        except (ValueError, AttributeError):
            direction = "unordered"
        counts = {kind: pq.ParquetFile(path / (filename + ".parquet")).metadata.num_rows
                  for kind, filename in [("subjects", "subjects"), ("items", "items"), ("responses", "response")]}
        result.append({"benchmark": name, "directory": directory, **counts, **splits[name],
                       "direction": direction, "modalities": modalities,
                       "feature_status": "withheld_protected" if splits[name]["protected"] else
                           "partial_nontext" if set(modalities) - {"text"} else "supported_text"})
    totals = {"benchmarks": len(result), **{key: sum(row[key] for row in result) for key in ("subjects", "items", "responses")}}
    if config.expected_counts and totals != config.expected_counts:
        raise ValueError("Population counts differ from pinned recipe: " + str(totals))
    return result
