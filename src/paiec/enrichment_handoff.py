"""Export collected system facts independently of any predictor or training set."""
import copy
import csv
import json
from pathlib import Path
import shutil
import zipfile

from .competition_fields import SUBJECT_FIELDS
from .feature_preparation import read_jsonl, write_json, write_jsonl, stable_hash
from .head_training import file_hash

TABLES = (
    "systems.jsonl", "typed_run_features.jsonl", "reviewed_static_features.jsonl",
    "supplemented_model_profiles.jsonl", "supplemented_harness_profiles.jsonl",
    "qualified_model_observations.jsonl", "qualified_harness_observations.jsonl",
    "profile_sources.jsonl", "supplemental_sources.jsonl", "source_review_status.jsonl",
    "run_setting_reviews.jsonl", "identity_review_status.jsonl",
)


def flat_facts(value, prefix=""):
    if isinstance(value, dict):
        if {"value", "status"} <= set(value):
            yield prefix, value
        else:
            for key, child in value.items():
                yield from flat_facts(child, prefix + "." + key if prefix else key)


def write_csv(path, fields, rows):
    with Path(path).open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def verify_handoff(folder):
    folder = Path(folder).resolve()
    manifest = json.loads((folder / "manifest.json").read_text())
    if manifest.get("version") not in {"system_enrichment_handoff_v1", "system_enrichment_handoff_v2"} or manifest.get("source_policy") != "research_only":
        raise ValueError("Unknown enrichment handoff")
    for name, digest in manifest["artifacts"].items():
        path = (folder / name).resolve()
        if not path.is_relative_to(folder) or not path.is_file() or file_hash(path) != digest:
            raise ValueError("Enrichment artifact changed or escaped its bundle: " + name)
    return manifest


def export_enrichment(source, out, *, expected_manifest_sha256, guide, collection=None):
    source, out, guide = Path(source).resolve(), Path(out).resolve(), Path(guide).resolve()
    if out.is_relative_to(source) or source.is_relative_to(out):
        raise ValueError("Keep exports separate from immutable source tables")
    archive = out.with_suffix(".zip")
    receipt_path = out.with_name(out.name + "_receipt.json")
    if out.exists() or archive.exists() or receipt_path.exists():
        raise FileExistsError("Use a fresh enrichment export; existing bundles are immutable")
    if file_hash(source / "manifest.json") != expected_manifest_sha256:
        raise ValueError("Reviewed feature source manifest changed")
    source_manifest = json.loads((source / "manifest.json").read_text())
    if source_manifest.get("version") != "feature_preparation_v1":
        raise ValueError("Use the reviewed feature-preparation source")
    tables = {}
    for name in TABLES:
        if file_hash(source / name) != source_manifest["artifacts"][name]:
            raise ValueError("Reviewed source table changed: " + name)
        tables[name] = list(read_jsonl(source / name))
    # The optional full collection supplements the scoped preparation view.
    # An explicit, hash-pinned inventory prevents unrelated run data entering it.
    collection_files, catalog_profiles, catalog_systems = {}, [], []
    if collection is not None:
        root = Path(collection["root"]).resolve()
        inventory = Path(collection["inventory"]).resolve()
        if file_hash(inventory) != collection["inventory_sha256"]:
            raise ValueError("Collection inventory changed")
        for name, digest in json.loads(inventory.read_text())["artifacts"].items():
            if Path(name).is_absolute() or ".." in Path(name).parts:
                raise ValueError("Collection inventory needs relative paths")
            path = (root / name).resolve()
            if (not path.is_relative_to(root) or not path.is_file()
                    or file_hash(path) != digest):
                raise ValueError("Collection source changed or escaped its root: " + name)
            if out.is_relative_to(path) or path.is_relative_to(out):
                raise ValueError("Keep collection sources separate from the export")
            collection_files[name] = path
        latest = collection["latest_catalog"].rstrip("/")
        catalog_profiles = list(read_jsonl(collection_files[latest + "/profiles.jsonl"]))
        catalog_systems = list(read_jsonl(collection_files[latest + "/subjects.jsonl"]))
        ids = {p["profile_id"] for p in catalog_profiles}
        run_ids = {p["profile_id"] for p in catalog_profiles if p["entity_type"] == "recorded_run"}
        if len(ids) != len(catalog_profiles) or {r["profile_id"] for r in catalog_systems} != run_ids:
            raise ValueError("Catalog profiles/system mappings differ")
        for row in catalog_systems:
            if any(row.get(k) is not None and row[k] not in ids for k in ["model_profile_id", "harness_profile_id"]):
                raise ValueError("Catalog system references an absent profile")
    sources = {}
    for name in ["profile_sources.jsonl", "supplemental_sources.jsonl"]:
        for row in tables[name]:
            key = row["source_id"]
            if key in sources and sources[key] != row:
                raise ValueError("Conflicting source identifier")
            sources[key] = row
    reviewed = {}
    for row in tables["reviewed_static_features.jsonl"]:
        fields = reviewed.setdefault(row["profile_id"], {})
        if row["field"] in fields:
            raise ValueError("Duplicate scoped fact")
        fact = row["fact"]
        if fact["value"] is not None and (row["review_status"] != "reviewed" or row["review_disposition"] != "value_confirmed"):
            raise ValueError("Unconfirmed scoped fact cannot become a shared value")
        fields[row["field"]] = dict(copy.deepcopy(fact), review_status=row["review_status"],
                                  review_disposition=row["review_disposition"])
    profiles, csv_facts, excluded = {}, [], 0
    for filename, kind in [("supplemented_model_profiles.jsonl", "model"), ("supplemented_harness_profiles.jsonl", "harness")]:
        for profile in tables[filename]:
            key = profile["profile_id"]
            if key in profiles or profile["entity_type"] != kind:
                raise ValueError("Duplicate or mismatched entity profile")
            facts = copy.deepcopy(reviewed.get(key, {}))
            for field, fact in flat_facts(profile["specifications"]):
                if field in facts or fact["value"] is None:
                    continue
                if fact["status"] == "documented" and fact["evidence"]:
                    facts[field] = dict(copy.deepcopy(fact), review_status="not_in_scoped_review",
                                        review_disposition="previously_documented")
                else:
                    excluded += 1
            row = {"profile_id": key, "entity_type": kind, "identity": profile["identity"],
                   "valid_from": profile["valid_from"], "valid_until": profile["valid_until"],
                   "usage": "research_only", "facts": dict(sorted(facts.items()))}
            profiles[key] = row
    for profile in catalog_profiles:
        kind, key = profile["entity_type"], profile["profile_id"]
        if kind == "recorded_run":
            continue
        if kind not in {"model", "harness"}:
            raise ValueError("Unknown collection entity")
        if key not in profiles:
            profiles[key] = {"profile_id": key, "entity_type": kind, "identity": profile["identity"],
                "valid_from": profile["valid_from"], "valid_until": profile["valid_until"],
                "usage": "research_only", "facts": {}}
        # Scoped reviewed values (including reviewed unknowns) take precedence.
        # Raw originals, rejected claims and earlier versions remain in provenance.
        for field, fact in flat_facts(profile["specifications"]):
            if field not in profiles[key]["facts"]:
                profiles[key]["facts"][field] = dict(copy.deepcopy(fact),
                    review_status="not_in_scoped_review", review_disposition="catalog_only")
    for key, row in profiles.items():
        kind = row["entity_type"]
        for field, fact in row["facts"].items():
            if any(e["source_id"] not in sources for e in fact["evidence"]):
                raise ValueError("Fact evidence lacks a source record")
            csv_facts.append({"profile_id": key, "entity_type": kind, "field": field,
                "value_json": json.dumps(fact["value"], ensure_ascii=False), "status": fact["status"],
                "review_status": fact["review_status"], "review_disposition": fact["review_disposition"],
                "note": fact.get("note", ""), "evidence_json": json.dumps(fact["evidence"], ensure_ascii=False)})
    if not set(reviewed) <= set(profiles):
        raise ValueError("Scoped facts reference an absent entity")
    systems, runs = {}, {}
    for row in tables["systems.jsonl"]:
        key = row["system_key"]
        if key in systems or set(row["recorded"]) != set(SUBJECT_FIELDS) or stable_hash(row["recorded"]) != key:
            raise ValueError("Ambiguous or changed supplied-system mapping")
        for field in ["model_profile_id", "harness_profile_id"]:
            pid = row["resolution"][field]
            if pid is not None and pid not in profiles:
                raise ValueError("System references an absent entity profile")
        systems[key] = row
    for row in tables["typed_run_features.jsonl"]:
        key = row["system_key"]
        if key in runs or key not in systems or row["recorded"] != systems[key]["recorded"]:
            raise ValueError("Run lacks an identical supplied-system match")
        if any(f["value"] is not None and f["status"] != "recorded" for f in row["features"].values()):
            raise ValueError("Run settings must remain recorded values, not capability defaults")
        runs[key] = row
    if set(systems) != set(runs):
        raise ValueError("Run/system mappings differ")
    observations = [r for n in ["qualified_model_observations.jsonl", "qualified_harness_observations.jsonl"] for r in tables[n]]
    if any(r["profile_id"] not in profiles or r["source_id"] not in sources for r in observations):
        raise ValueError("Observation lacks a profile/source")
    out.mkdir(parents=True)
    for kind, filename in [("model", "models.jsonl"), ("harness", "harnesses.jsonl")]:
        write_jsonl(out / filename, [profiles[k] for k in sorted(profiles) if profiles[k]["entity_type"] == kind])
    write_jsonl(out / "systems.jsonl", [systems[k] for k in sorted(systems)])
    write_jsonl(out / "runs.jsonl", [runs[k] for k in sorted(runs)])
    write_jsonl(out / "sources.jsonl", [sources[k] for k in sorted(sources)])
    write_jsonl(out / "observations.jsonl", observations)
    fields = ["profile_id", "entity_type", "field", "value_json", "status", "review_status", "review_disposition", "note", "evidence_json"]
    write_csv(out / "facts.csv", fields, sorted(csv_facts, key=lambda r: (r["profile_id"], r["field"])))
    write_csv(out / "run_settings.csv", ["system_key", "field", "value_json", "status", "collection_status", "note", "evidence_json"],
        [{"system_key": key, "field": field, "value_json": json.dumps(fact["value"], ensure_ascii=False),
          "status": fact["status"], "collection_status": fact.get("collection_status", ""),
          "note": fact.get("note", ""), "evidence_json": json.dumps(fact["evidence"], ensure_ascii=False)}
         for key, row in sorted(runs.items()) for field, fact in sorted(row["features"].items())])
    write_csv(out / "system_mapping.csv", ["system_key", *SUBJECT_FIELDS, "model_profile_id", "harness_profile_id", "model_identity_basis", "exact_evaluated_checkpoint_verified"],
        [dict(system_key=k, **r["recorded"], **{f: r["resolution"][f] for f in
            ["model_profile_id", "harness_profile_id", "model_identity_basis", "exact_evaluated_checkpoint_verified"]}) for k, r in sorted(systems.items())])
    raw = out / "provenance/source_tables"
    raw.mkdir(parents=True)
    for name in TABLES:
        shutil.copyfile(source / name, raw / name)
    if collection is not None:
        for name, path in collection_files.items():
            target = out / "provenance" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
        shutil.copyfile(inventory, out / "provenance/collection_inventory.json")
        write_jsonl(out / "catalog_systems.jsonl", catalog_systems)
        write_jsonl(out / "catalog_runs.jsonl", [p for p in catalog_profiles if p["entity_type"] == "recorded_run"])
    shutil.copyfile(guide, out / "README.md")
    counts = {"models": sum(p["entity_type"] == "model" for p in profiles.values()),
        "harnesses": sum(p["entity_type"] == "harness" for p in profiles.values()), "recorded_systems": len(systems),
        "scoped_fields": len(tables["reviewed_static_features.jsonl"]), "populated_scoped_fields": sum(r["fact"]["value"] is not None for r in tables["reviewed_static_features.jsonl"]),
        "additional_documented_fields": sum(r["review_disposition"] == "previously_documented" for r in csv_facts),
        "inferred_additional_fields_excluded": excluded, "populated_facts_by_status": {status: sum(r["status"] == status and r["value_json"] != "null" for r in csv_facts) for status in ["documented", "inferred"]},
        "sources": len(sources), "observations": len(observations),
        "populated_run_settings": sum(f["value"] is not None for r in runs.values() for f in r["features"].values())}
    if collection is not None:
        counts.update(catalog_system_configurations=len(catalog_systems),
                      collection_provenance_files=len(collection_files))
    manifest = {"version": "system_enrichment_handoff_v2" if collection else "system_enrichment_handoff_v1", "source_policy": "research_only", "counts": counts,
        "source_manifest_sha256": expected_manifest_sha256, "source_table_sha256": {n: source_manifest["artifacts"][n] for n in TABLES},
        "exporter_sha256": file_hash(Path(__file__)), "question_inputs_included": False, "outcome_labels_included": False,
        "feature_matrix_included": False, "model_weights_included": False,
        "artifacts": {str(p.relative_to(out)): file_hash(p) for p in sorted(out.rglob("*")) if p.is_file()}}
    if collection is not None:
        manifest["collection_inventory_sha256"] = collection["inventory_sha256"]
        manifest["profile_precedence"] = "scoped reviewed facts, supplemented documented facts, latest catalog; originals retained in provenance"
    write_json(out / "manifest.json", manifest)
    verify_handoff(out)
    with zipfile.ZipFile(archive, "x", compression=zipfile.ZIP_DEFLATED) as zipped:
        for path in sorted(out.rglob("*")):
            if path.is_file():
                info = zipfile.ZipInfo(out.name + "/" + str(path.relative_to(out)))
                zipped.writestr(info, path.read_bytes(), compress_type=zipfile.ZIP_DEFLATED)
    receipt = {"version": "system_enrichment_archive_v1", "file_name": archive.name, "bytes": archive.stat().st_size,
               "sha256": file_hash(archive), "manifest_sha256": file_hash(out / "manifest.json"), "counts": counts,
               "source_policy": "research_only", "outcome_labels_included": False, "feature_matrix_included": False}
    write_json(receipt_path, receipt)
    return receipt
