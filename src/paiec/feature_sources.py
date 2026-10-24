"""Validated additive model-documentation facts; never rewrite pinned catalogs."""
import json
from pathlib import Path

from .feature_preparation import read_jsonl, write_json, write_jsonl, stable_hash
from .head_training import file_hash
from .system_profiles import Profile


def source_document_key(source_id, sha256):
    """Keep new snapshots distinct from original catalogs' source identities."""
    return f"feature.{sha256}.{source_id}"


def supplement_models(root, models, out, *, source_registry="configs/pipelines/feature_sources_v1.json"):
    return supplement_profiles(root, models, out, kind="model", source_registry=source_registry)


def supplement_harnesses(root, harnesses, out, *, source_registry="configs/pipelines/feature_sources_v1.json"):
    return supplement_profiles(root, harnesses, out, kind="harness", source_registry=source_registry)


def supplement_profiles(root, models, out, *, kind, source_registry):
    if kind not in {"model", "harness"}:
        raise ValueError("Only model and harness documentation can be supplemented")
    root, out = Path(root), Path(out)
    registry = (root / source_registry).resolve()
    if not registry.is_relative_to((root / "configs/pipelines").resolve()):
        raise ValueError("Source registry escapes pipeline configuration storage")
    if source_registry != "configs/pipelines/feature_sources_v1.json" and not registry.is_file():
        raise FileNotFoundError("Selected source registry is missing: " + str(registry))
    settings = json.loads(registry.read_text()) if registry.exists() else {}
    bundles = settings["bundles"] if registry.exists() else [
        {"catalog": "research/feature_collection", "cache": "runs/feature_sources/primary_v1", "assertions": "assertions_v1.jsonl"}]
    sources, entries, bundle_hashes, archives = {}, [], [], {}
    for bundle in bundles:
        catalog, cache_root = (root / bundle["catalog"]).resolve(), (root / bundle["cache"]).resolve()
        if not catalog.is_relative_to(root.resolve()) or not cache_root.is_relative_to((root / "runs").resolve()):
            raise ValueError("Source bundle escapes repository storage")
        local_sources = {r["source_id"]: r for r in read_jsonl(catalog / "sources.jsonl")}
        for sid, source in local_sources.items():
            if sid in sources and sources[sid] != source:
                raise ValueError("Conflicting source ID in collection bundles")
            sources[sid] = source
        snapshots = {r["source_id"]: r for r in json.loads((catalog / "snapshots.json").read_text())}
        archives.update({sid: (snapshots[sid], cache_root) for sid in local_sources})
        assertions_path = (catalog / bundle["assertions"]).resolve()
        if not assertions_path.is_relative_to(catalog):
            raise ValueError("Assertion ledger escapes source catalog")
        bundle_hashes.append({"catalog": bundle["catalog"], "assertions_sha256": file_hash(assertions_path),
                              "sources_sha256": file_hash(catalog / "sources.jsonl"),
                              "snapshots_sha256": file_hash(catalog / "snapshots.json")})
        entries.extend((a, local_sources[a["source_id"]], snapshots[a["source_id"]], cache_root)
                       for a in read_jsonl(assertions_path))
    records = {r["profile_id"]: r for r in read_jsonl(models["path"])}
    if any(profile["entity_type"] != kind for profile in records.values()):
        raise ValueError("Selected profile table has the wrong entity type")
    applied, existing, conflicts, inapplicable = [], [], [], []
    for assertion, source, snapshot, cache_root in entries:
        pid, field = assertion["profile_id"], assertion["field"]
        if pid not in records:
            inapplicable.append(assertion)
            continue
        if source.get("eligibility") == "not_permitted":
            raise ValueError("Source assertion uses an ineligible source")
        cache = (cache_root / snapshot["cache_file"]).resolve()
        if not cache.is_relative_to(cache_root):
            raise ValueError("Source archive path escapes metadata cache")
        if file_hash(cache) != source["sha256"] or source["sha256"] != snapshot["sha256"]:
            raise ValueError("Primary source snapshot changed")
        text = cache.read_text()
        evidence = []
        for quote in [assertion["quote"], *assertion.get("supporting_quotes", [])]:
            start = text.find(quote)
            if not quote or start < 0:
                raise ValueError("Source assertion lacks exact supporting text")
            evidence.append({"source_id": source_document_key(source["source_id"], source["sha256"]),
                             "locator": f"characters {start}:{start+len(quote)}; revision {source['revision']}",
                             "excerpt": quote})
        container = records[pid]["specifications"]
        parts = field.split(".")
        for part in parts[:-1]:
            container = container[part]
        current = container[parts[-1]]
        fact = {"value": assertion["value"], "status": assertion["status"],
                "evidence": evidence, "note": assertion["note"]}
        if current["status"] != "unknown":
            (existing if stable_hash(current["value"]) == stable_hash(fact["value"]) else conflicts).append(
                dict(assertion, output_source_id=source_document_key(source["source_id"], source["sha256"])))
            continue
        container[parts[-1]] = fact
        Profile.model_validate_json(json.dumps(records[pid]))
        applied.append(assertion)
    path = out / f"supplemented_{kind}_profiles.jsonl"
    write_jsonl(path, records.values())
    # Keep the source ledger with the feature outputs so every new evidence ID resolves.
    write_jsonl(out / ("supplemental_sources.jsonl" if kind == "model" else "harness_supplemental_sources.jsonl"),
                (dict(source, source_id=source_document_key(sid, source["sha256"])) for sid, source in sources.items()))
    nominal_contexts = []
    nominal_ledger = settings.get("nominal_context_ledger") if kind == "model" else None
    if nominal_ledger:
        ledger = (root / nominal_ledger).resolve()
        if not ledger.is_relative_to((root / "research").resolve()):
            raise ValueError("Nominal-context ledger escapes research storage")
        for observation in read_jsonl(ledger):
            if observation["profile_id"] not in records:
                continue
            if (observation["field"] != "documented_context_window" or
                observation["status"] != "documented_nominal" or not isinstance(observation["value"], str)):
                raise ValueError("Nominal context must preserve a qualified string value")
            source = sources[observation["source_id"]]
            snapshot, cache_root = archives[observation["source_id"]]
            archive = (cache_root / snapshot["cache_file"]).resolve()
            if not archive.is_relative_to(cache_root) or file_hash(archive) != source["sha256"] or source["sha256"] != snapshot["sha256"]:
                raise ValueError("Nominal context source snapshot changed")
            if source.get("eligibility") == "not_permitted" or not observation["quote"] or observation["quote"] not in archive.read_text():
                raise ValueError("Nominal context lacks eligible exact source support")
            nominal_contexts.append(dict(observation, source_id=source_document_key(source["source_id"], source["sha256"])))
    observations = list(nominal_contexts)
    observation_ledgers = []
    for name in settings.get("observation_ledgers", []):
        ledger_path = (root / name).resolve()
        if not ledger_path.is_relative_to((root / "research").resolve()):
            raise ValueError("Observation ledger escapes research storage")
        observation_ledgers.append({"path": name, "sha256": file_hash(ledger_path)})
        for observation in read_jsonl(ledger_path):
            if observation["profile_id"] not in records:
                continue
            if (observation["status"] != "documented_supported" or not observation.get("note") or
                observation.get("field") not in {"prompting_modes", "inference_backends", "answer_handling",
                    "context_options", "workflow_options", "optional_assistance", "provider_execution",
                    "language_coverage_description", "training_relationships", "input_modalities",
                    "accepted_document_formats"} or
                not observation["value"] or not isinstance(observation["value"], (str, list)) or
                isinstance(observation["value"], list) and not all(isinstance(v, str) for v in observation["value"])):
                raise ValueError("Supported observation requires qualified string values")
            source = sources[observation["source_id"]]
            snapshot, cache_root = archives[observation["source_id"]]
            archive = (cache_root / snapshot["cache_file"]).resolve()
            if not archive.is_relative_to(cache_root) or len({file_hash(archive), source["sha256"], snapshot["sha256"]}) != 1:
                raise ValueError("Supported observation source snapshot changed")
            if source.get("eligibility") == "not_permitted":
                raise ValueError("Supported observation uses ineligible evidence")
            text = archive.read_text()
            for quote in [observation["quote"], *observation.get("supporting_quotes", [])]:
                if not quote or quote not in text:
                    raise ValueError("Supported observation lacks exact source support")
            observations.append(dict(observation, source_id=source_document_key(source["source_id"], source["sha256"])))
    write_jsonl(out / f"qualified_{kind}_observations.jsonl", observations)
    audit = {"applied": len(applied), "already_known": len(existing), "conflicts": conflicts,
             "not_selected": len(inapplicable), "assertion_bundles": bundle_hashes,
             "source_registry_sha256": file_hash(registry) if registry.exists() else None,
             "nominal_context_ledger_sha256": file_hash(ledger) if nominal_ledger else None,
             "qualified_context_observations": len(nominal_contexts),
             "qualified_observations": len(observations), "observation_ledgers": observation_ledgers,
             "source_snapshots": len(sources),
             "new_fact_scope": f"named_{kind}_documentation; not actual run settings",
             "original_catalog_changed": False}
    write_json(out / ("source_collection_audit.json" if kind == "model" else "harness_source_collection_audit.json"), audit)
    return dict(models, path=str(path), source_collection=audit)
