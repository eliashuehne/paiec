"""Validated additive model-documentation facts; never rewrite pinned catalogs."""
import json
from pathlib import Path

from .feature_preparation import read_jsonl, write_json, write_jsonl, stable_hash
from .head_training import file_hash
from .system_profiles import Profile


def supplement_models(root, models, out):
    root, out = Path(root), Path(out)
    registry = root / "configs/pipelines/feature_sources_v1.json"
    bundles = json.loads(registry.read_text())["bundles"] if registry.exists() else [
        {"catalog": "research/feature_collection", "cache": "runs/feature_sources/primary_v1", "assertions": "assertions_v1.jsonl"}]
    sources, entries, bundle_hashes = {}, [], []
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
        assertions_path = (catalog / bundle["assertions"]).resolve()
        if not assertions_path.is_relative_to(catalog):
            raise ValueError("Assertion ledger escapes source catalog")
        bundle_hashes.append({"catalog": bundle["catalog"], "assertions_sha256": file_hash(assertions_path)})
        entries.extend((a, local_sources[a["source_id"]], snapshots[a["source_id"]], cache_root)
                       for a in read_jsonl(assertions_path))
    records = {r["profile_id"]: r for r in read_jsonl(models["path"])}
    applied, existing, conflicts, inapplicable = [], [], [], []
    for assertion, source, snapshot, cache_root in entries:
        pid, field = assertion["profile_id"], assertion["field"]
        if pid not in records:
            inapplicable.append(assertion)
            continue
        cache = (cache_root / snapshot["cache_file"]).resolve()
        if not cache.is_relative_to(cache_root):
            raise ValueError("Source archive path escapes metadata cache")
        if file_hash(cache) != source["sha256"] or source["sha256"] != snapshot["sha256"]:
            raise ValueError("Primary source snapshot changed")
        text = cache.read_text()
        start = text.find(assertion["quote"])
        if start < 0:
            raise ValueError("Source assertion lacks exact supporting text")
        container = records[pid]["specifications"]
        parts = field.split(".")
        for part in parts[:-1]:
            container = container[part]
        current = container[parts[-1]]
        fact = {"value": assertion["value"], "status": assertion["status"],
                "evidence": [{"source_id": source["source_id"], "locator": f"characters {start}:{start+len(assertion['quote'])}; revision {source['revision']}",
                              "excerpt": assertion["quote"]}], "note": assertion["note"]}
        if current["status"] != "unknown":
            (existing if stable_hash(current["value"]) == stable_hash(fact["value"]) else conflicts).append(assertion)
            continue
        container[parts[-1]] = fact
        Profile.model_validate_json(json.dumps(records[pid]))
        applied.append(assertion)
    path = out / "supplemented_model_profiles.jsonl"
    write_jsonl(path, records.values())
    # Keep the source ledger with the feature outputs so every new evidence ID resolves.
    write_jsonl(out / "supplemental_sources.jsonl", sources.values())
    audit = {"applied": len(applied), "already_known": len(existing), "conflicts": conflicts,
             "not_selected": len(inapplicable), "assertion_bundles": bundle_hashes,
             "source_snapshots": len(sources),
             "new_fact_scope": "named_model_documentation; not actual run settings",
             "original_catalog_changed": False}
    write_json(out / "source_collection_audit.json", audit)
    return dict(models, path=str(path), source_collection=audit)
