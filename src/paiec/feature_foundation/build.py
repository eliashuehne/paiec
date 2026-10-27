"""Atomic, versioned local publication and independent artifact validation."""
from collections import Counter, defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import sqlite3
import tempfile
from itertools import zip_longest
import platform
from importlib.metadata import version as package_version

import pyarrow.parquet as pq

from .contracts import VERSION, SUBJECT_FIELDS, canonical_json, file_hash, stable_key, write_json
from .enrichment import load_export, read_jsonl, resolve_enrichment
from .report import coverage_report, feature_dictionary, render_report
from .review import select_review_cases, table_rows, validate_reviews
from .settings import parse_run_settings, POLICY_VERSION, FIELDS, ALIASES, CONTAINERS, SETTING_BRANCHES
from .sources import inventory_population, safe_source, verify_sources
from .streaming import stream_population
from .tables import SCHEMAS, write_table


@dataclass
class FeatureBundle:
    directory: Path
    manifest: dict
    receipt: dict


def jsonl(path, rows):
    with Path(path).open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(canonical_json(row) + "\n")


def source_field_inventory(config, source_root, population):
    rows = []
    permitted = {"subjects": set(SUBJECT_FIELDS) | {"subject_id"}, "items": {"item_id", "content", "item_features"},
                 "response": {"subject_id", "item_id", "response_id", "trial", "interactors"}, "benchmarks": {"modality", "response_scale"}}
    for pop in population:
        directory = safe_source(source_root, config.data_root + "/" + pop["directory"])
        for table, allowed in permitted.items():
            for field in pq.ParquetFile(directory / (table + ".parquet")).schema_arrow:
                protected_content = pop["protected"] and table != "benchmarks" and field.name not in {"subject_id", "item_id", "response_id", "trial"}
                disposition = "withheld_protected" if protected_content else "projected_input_or_audit" if field.name in allowed else "excluded_outcome" if table == "response" and field.name == "response" else "excluded_unpublished_or_reference_field"
                rows.append({"benchmark": pop["benchmark"], "table": table, "field": field.name, "source_type": str(field.type),
                             "disposition": disposition, "value_deserialized": field.name in allowed and not protected_content})
    return rows


def build_bundle(config, *, source_root, out, progress=None):
    source_root, out = Path(source_root).resolve(), Path(out).resolve()
    code_root = Path(__file__).parent
    code_hashes = {p.name: file_hash(p) for p in sorted(code_root.glob("*.py"))}
    dependency_paths = [code_root.parent / name for name in ("schema.py", "split.py", "feature_join.py")]
    dependency_hashes = {p.name: file_hash(p) for p in dependency_paths}
    # Check physical overlap before touching either input or output contents.
    for relative in (config.data_root, config.enrichment_export):
        source = safe_source(source_root, relative)
        if out == source_root or source.is_relative_to(out) or out.is_relative_to(source):
            raise ValueError("Feature outputs overlap immutable source data")
    if out.exists():
        raise FileExistsError("Use a fresh feature output directory")
    population = inventory_population(config, source_root)
    out.mkdir(parents=True)
    (out / "INCOMPLETE").write_text("No completed manifest is published until validation succeeds.\n")
    logical = {"population": write_table(out, "population", population, config.batch_size)}
    result = stream_population(config, source_root, out, population, progress=progress)
    logical.update(result["tables"])
    systems = list(table_rows(out / "systems.parquet"))
    observations, resolved = [], []
    for subject in systems:
        parsed = parse_run_settings({k: subject[k] for k in SUBJECT_FIELDS})
        observations.extend(parsed["observations"])
        resolved.extend(parsed["resolved"])
    logical["setting_observations"] = write_table(out, "setting_observations", observations, config.batch_size)
    logical["run_settings"] = write_table(out, "run_settings", resolved, config.batch_size)
    export_root = safe_source(source_root, config.enrichment_export)
    export = load_export(export_root, required=config.require_enrichment)
    resolver = None
    if config.require_enrichment:
        from .matching import ProfileMatcher
        resolver = ProfileMatcher(config.matching_policy, export)
    enriched = resolve_enrichment(systems, export, resolver)
    for name in ("profile_matches", "research_system_features"):
        logical[name] = write_table(out, name, enriched[name], config.batch_size)
    jsonl(out / "facts.jsonl", enriched["facts"])
    jsonl(out / "sources.jsonl", [export["sources"][key] for key in sorted(export["sources"])])
    jsonl(out / "catalog_mappings.jsonl", export["catalog_mappings"])
    jsonl(out / "recorded_run_facts.jsonl", export["recorded_runs"])
    prefix = config.enrichment_export.rstrip("/") + "/provenance/"
    for relative in config.input_hashes:
        if relative.startswith(prefix):
            suffix = relative[len(config.enrichment_export.rstrip("/") + "/"):]
            destination = out / suffix
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(safe_source(source_root, relative), destination)
    cases = select_review_cases(out, config, population)
    jsonl(out / "review_inputs.jsonl", cases)
    jsonl(out / "review_cases.jsonl", [{k: v for k, v in row.items() if k != "input"} for row in cases])
    reviews = []
    if config.review_ledger:
        repository_root = Path(__file__).resolve().parents[3]
        reviews = list(read_jsonl(safe_source(repository_root, config.review_ledger)))
    review = validate_reviews(cases, reviews)
    review["expected"] = config.review_cases_per_kind * 2
    review["shortfall"] = max(0, review["expected"] - len(cases))
    review["complete"] = review["complete"] and not review["shortfall"]
    jsonl(out / "reviewed_examples.jsonl", reviews)
    write_json(out / "feature_dictionary.json", feature_dictionary(enriched["facts"]))
    write_json(out / "settings_policy.json", {"version": POLICY_VERSION, "fields": FIELDS, "aliases": ALIASES, "containers": sorted(CONTAINERS), "trusted_nested_branches": sorted(SETTING_BRANCHES),
        "ambiguities": "Generic max_tokens and method names remain literal; support/defaults do not establish enabled settings"})
    fields = source_field_inventory(config, source_root, population)
    write_json(out / "source_fields.json", fields)
    accesses = result["accesses"] + [{"path": config.data_root + "/" + row["directory"] + "/benchmarks.parquet", "columns": ["modality", "response_scale"], "protected": row["protected"]} for row in population]
    write_json(out / "accesses.json", accesses)
    report = coverage_report(out, population, enriched["facts"], review)
    report["source_fields"] = dict(Counter(row["disposition"] for row in fields))
    report["enrichment_inventory"] = {"model_profiles": sum(p["entity_type"] == "model" for p in export["profiles"].values()),
        "harness_profiles": sum(p["entity_type"] == "harness" for p in export["profiles"].values()),
        "catalog_configurations": len(export["catalog_mappings"]), "preparation_descriptions": len(export["recorded_runs"])}
    write_json(out / "coverage.json", report)
    (out / "report.html").write_text(render_report(report), encoding="utf-8")
    # Inputs are immutable by contract. Refuse persistent changes between the
    # verified reads and publication rather than certifying stale provenance.
    verify_sources(config, source_root)
    if code_hashes != {p.name: file_hash(p) for p in sorted(code_root.glob("*.py"))}:
        raise ValueError("Feature implementation changed during the build; retain incomplete output")
    if dependency_hashes != {p.name: file_hash(p) for p in dependency_paths}:
        raise ValueError("Feature dependency changed during the build; retain incomplete output")
    receipt = {"version": VERSION, "created_at": datetime.now(timezone.utc).isoformat(), "source_root": str(source_root),
               "config": config.model_dump(), "input_hashes": config.input_hashes,
               "code_hashes": code_hashes,
               "dependency_code_hashes": dependency_hashes,
               "runtime_versions": {"python": platform.python_version(), **{name: package_version(name) for name in ("pyarrow", "pydantic")}},
               "matching_policy": resolver.metadata if resolver else None,
               "actions": {"model_fitting": False, "model_inference": False, "network_calls": False,
                           "outcome_columns_deserialized": False, "protected_content_loaded": False,
                           "trace_tables_loaded": False, "new_semantic_annotations_published": False}}
    write_json(out / "receipt.json", receipt)
    manifest = {"version": VERSION, "dataset_revision": config.dataset_revision, "population": report["population"],
                "logical_tables": logical, "artifacts": {}, "review_complete": review["complete"]}
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name != "INCOMPLETE":
            manifest["artifacts"][str(path.relative_to(out))] = {"sha256": file_hash(path), "bytes": path.stat().st_size}
    # Provisional manifest remains visibly incomplete until independent validation.
    write_json(out / "manifest.json", manifest)
    try:
        validate_bundle(out, allow_incomplete=True)
    except Exception:
        (out / "manifest.json").unlink(missing_ok=True)
        raise
    (out / "INCOMPLETE").unlink()
    return FeatureBundle(out, manifest, receipt)


def validate_bundle(path, *, require_review=False, allow_incomplete=False):
    path = Path(path).resolve()
    if (path / "INCOMPLETE").exists() and not allow_incomplete:
        raise ValueError("Incomplete feature bundle")
    manifest = json.loads((path / "manifest.json").read_text())
    if manifest["version"] != VERSION:
        raise ValueError("Unsupported feature bundle schema")
    required = {name + ".parquet" for name in SCHEMAS} | {
        "coverage.json", "feature_dictionary.json", "source_fields.json", "settings_policy.json", "report.html",
        "receipt.json", "facts.jsonl", "sources.jsonl", "catalog_mappings.jsonl", "recorded_run_facts.jsonl",
        "review_inputs.jsonl", "review_cases.jsonl", "reviewed_examples.jsonl", "accesses.json"}
    if not required <= set(manifest["artifacts"]):
        raise ValueError("Required bundle artifacts are missing from the manifest")
    for relative, artifact in manifest["artifacts"].items():
        actual = safe_source(path, relative)
        if file_hash(actual) != artifact["sha256"]:
            raise ValueError("Artifact hash mismatch: " + relative)
    for name, schema in SCHEMAS.items():
        reader = pq.ParquetFile(path / (name + ".parquet"))
        if reader.schema_arrow != schema or reader.metadata.num_rows != manifest["logical_tables"][name]["rows"]:
            raise ValueError("Artifact schema or count mismatch: " + name)
        digest = hashlib.sha256(schema.to_string().encode())
        for batch in reader.iter_batches(batch_size=65536, columns=schema.names):
            for row in batch.to_pylist():
                digest.update(canonical_json(row).encode() + b"\n")
        if digest.hexdigest() != manifest["logical_tables"][name]["logical_sha256"]:
            raise ValueError("Logical table digest mismatch: " + name)
    population = list(table_rows(path / "population.parquet"))
    pop = {r["benchmark"]: r for r in population}
    expected = {"benchmarks": len(pop), **{k: sum(r[k] for r in population) for k in ("subjects", "items", "responses")}}
    if expected != manifest["population"] or len(pop) != len(population):
        raise ValueError("Population totals or keys mismatch")
    system_keys = set()
    for row in table_rows(path / "systems.parquet"):
        key = row.pop("system_key")
        if key in system_keys or key != stable_key(row):
            raise ValueError("Invalid or duplicate visible system key")
        system_keys.add(key)
    subject_counts = Counter()
    for row in table_rows(path / "subject_links.parquet"):
        subject_counts[row["benchmark"]] += 1
        if row["system_key"] is not None and row["system_key"] not in system_keys:
            raise ValueError("Subject has an orphan system key")
    # Disk-backed item and task keys keep validation memory bounded.
    with tempfile.TemporaryDirectory(prefix="feature-validation-") as temporary:
        db = sqlite3.connect(Path(temporary) / "keys.sqlite")
        db.execute("PRAGMA journal_mode=OFF")
        db.execute("CREATE TABLE items (key TEXT PRIMARY KEY) WITHOUT ROWID")
        db.execute("CREATE TABLE tasks (key TEXT PRIMARY KEY) WITHOUT ROWID")
        db.execute("CREATE TABLE features (key TEXT PRIMARY KEY) WITHOUT ROWID")
        item_counts = Counter()
        for batch in pq.ParquetFile(path / "items.parquet").iter_batches(batch_size=8192):
            keys = []
            for row in batch.to_pylist():
                if row["source_row"] != item_counts[row["benchmark"]] or row["item_key"] != stable_key({"benchmark": row["benchmark"], "item_id": row["item_id"]}):
                    raise ValueError("Item source ordering or key mismatch")
                if pop[row["benchmark"]]["protected"] and (row["content"] is not None or row["item_features"] is not None):
                    raise ValueError("Protected item content was published")
                item_counts[row["benchmark"]] += 1
                keys.append((row["item_key"],))
            try:
                db.executemany("INSERT INTO items VALUES (?)", keys)
            except sqlite3.IntegrityError as error:
                raise ValueError("Duplicate source item key") from error
        for batch in pq.ParquetFile(path / "tasks.parquet").iter_batches(batch_size=8192):
            keys = []
            for row in batch.to_pylist():
                key = row.pop("task_key")
                if key != stable_key(row):
                    raise ValueError("Task input hash mismatch")
                keys.append((key,))
            try:
                db.executemany("INSERT INTO tasks VALUES (?)", keys)
            except sqlite3.IntegrityError as error:
                raise ValueError("Duplicate task key") from error
        for row in table_rows(path / "task_features.parquet", ["task_key"]):
            key = row["task_key"]
            if not db.execute("SELECT 1 FROM tasks WHERE key=?", (key,)).fetchone():
                raise ValueError("Orphan task feature")
            try:
                db.execute("INSERT INTO features VALUES (?)", (key,))
            except sqlite3.IntegrityError as error:
                raise ValueError("Duplicate task feature") from error
        if manifest["logical_tables"]["task_features"]["rows"] != manifest["logical_tables"]["tasks"]["rows"]:
            raise ValueError("Missing task features")
        db.execute("CREATE TABLE variants (item TEXT, interactor TEXT, PRIMARY KEY(item,interactor)) WITHOUT ROWID")
        for row in table_rows(path / "item_input_links.parquet", ["item_key", "task_key", "interactor_key"]):
            if not db.execute("SELECT 1 FROM items WHERE key=?", (row["item_key"],)).fetchone():
                raise ValueError("Orphan item input link")
            if row["task_key"] and not db.execute("SELECT 1 FROM tasks WHERE key=?", (row["task_key"],)).fetchone():
                raise ValueError("Orphan task input link")
            try:
                db.execute("INSERT INTO variants VALUES (?,?)", (row["item_key"], row["interactor_key"] or ""))
            except sqlite3.IntegrityError as error:
                raise ValueError("Duplicate item input variant") from error
        db.execute("CREATE TABLE groups (key TEXT PRIMARY KEY, mask INTEGER) WITHOUT ROWID")
        db.executemany("INSERT INTO groups VALUES (?,?)", ((r["group_key"], r["split_mask"]) for r in table_rows(path / "content_groups.parquet")))
        links = table_rows(path / "item_input_links.parquet")
        eligibility = table_rows(path / "training_eligibility.parquet")
        for link, row in zip_longest(links, eligibility):
            fields = ("benchmark", "item_key", "task_key", "split")
            if link is None or row is None or any(link[k] != row[k] for k in fields):
                raise ValueError("Orphan or misaligned training eligibility")
            mask = db.execute("SELECT mask FROM groups WHERE key=?", (link["group_key"],)).fetchone() if link["group_key"] else None
            conflict = bool(mask and mask[0] & (mask[0] - 1))
            reason = "cross_split_content" if link["split"] == "training" and conflict else link["feature_status"] if link["feature_status"] != "supported_text" else "nontraining_split" if link["split"] != "training" else "eligible"
            if row["reason"] != reason or row["eligible"] != (reason == "eligible"):
                raise ValueError("Invalid training eligibility disposition")
        response_counts = Counter()
        for batch in pq.ParquetFile(path / "response_input_links.parquet").iter_batches(batch_size=65536, columns=["benchmark", "source_row", "system_key", "item_key", "interactor_key", "status"]):
            rows = batch.to_pylist()
            for row in rows:
                if row["source_row"] != response_counts[row["benchmark"]]:
                    raise ValueError("Response source row ordering mismatch")
                response_counts[row["benchmark"]] += 1
                if row["system_key"] and row["system_key"] not in system_keys:
                    raise ValueError("Orphan response system link")
                if pop[row["benchmark"]]["protected"] and (row["interactor_key"] is not None or row["system_key"] is not None):
                    raise ValueError("Protected response input was published")
            # Validate distinct foreign keys once per batch, not once per trial.
            for key in {row["item_key"] for row in rows if row["item_key"]}:
                if not db.execute("SELECT 1 FROM items WHERE key=?", (key,)).fetchone():
                    raise ValueError("Orphan response item link")
            for item, interactor in {(r["item_key"], r["interactor_key"]) for r in rows
                                     if r["item_key"] and not pop[r["benchmark"]]["protected"]}:
                if not interactor or not db.execute("SELECT 1 FROM variants WHERE item=? AND interactor=?", (item, interactor)).fetchone():
                    raise ValueError("Orphan response input variant")
        for benchmark, row in pop.items():
            if (subject_counts[benchmark], item_counts[benchmark], response_counts[benchmark]) != (row["subjects"], row["items"], row["responses"]):
                raise ValueError("Population record accounting mismatch")
        db.close()
    sources = {r["source_id"] for r in read_jsonl(path / "sources.jsonl")}
    facts = list(read_jsonl(path / "facts.jsonl"))
    fact_keys = {(r["profile_id"], r["field"]) for r in facts}
    if len(fact_keys) != len(facts):
        raise ValueError("Duplicate research fact key")
    for row in facts:
        if any(ref["source_id"] not in sources for ref in row["fact"].get("evidence", [])):
            raise ValueError("Unresolvable fact evidence")
    observed = {}
    for row in table_rows(path / "setting_observations.parquet"):
        if row["system_key"] not in system_keys or row["observation_id"] in observed:
            raise ValueError("Orphan or duplicate setting observation")
        observed[row["observation_id"]] = (row["system_key"], row["field"])
    run_keys = set()
    for row in table_rows(path / "run_settings.parquet"):
        key = (row["system_key"], row["field"])
        if key in run_keys or row["system_key"] not in system_keys or row["field"] not in FIELDS:
            raise ValueError("Duplicate or orphan resolved setting")
        run_keys.add(key)
        if any(observed.get(ref) != key and not (row["reason"] == "unresolved_container" and observed.get(ref) == (row["system_key"], None)) for ref in row["observation_ids"]):
            raise ValueError("Resolved setting has orphan or foreign observations")
        if (row["status"] == "known") != (row["value_json"] is not None):
            raise ValueError("Setting null/status mismatch")
    if len(run_keys) != len(system_keys) * len(FIELDS):
        raise ValueError("Missing declared system/setting cells")
    matches = {}
    profiles = {row["profile_id"] for row in facts}
    for row in table_rows(path / "profile_matches.parquet"):
        key = (row["system_key"], row["entity_type"])
        if key in matches or row["system_key"] not in system_keys or row["entity_type"] not in {"model", "harness"}:
            raise ValueError("Duplicate or orphan profile match")
        if row["profile_id"] and row["profile_id"] not in profiles:
            raise ValueError("Profile match references absent research inventory")
        matches[key] = row["profile_id"]
    if len(matches) != 2 * len(system_keys):
        raise ValueError("Missing explicit profile-match dispositions")
    research_keys = set()
    for row in table_rows(path / "research_system_features.parquet"):
        key = (row["system_key"], row["entity_type"], row["field"])
        if key in research_keys or matches.get(key[:2]) != row["profile_id"] or (row["profile_id"], row["field"]) not in fact_keys:
            raise ValueError("Duplicate or orphan research feature")
        if any(ref not in sources for ref in row["source_ids"]) or row["eligibility"] != "research_only":
            raise ValueError("Research feature provenance or eligibility mismatch")
        if (row["status"] == "known") != (row["value_json"] is not None):
            raise ValueError("Research fact null/status mismatch")
        research_keys.add(key)
    for access in json.loads((path / "accesses.json").read_text()):
        if set(access["columns"]) & {"response", "grading_criterion", "verifier", "display_name", "test_condition"}:
            raise ValueError("Forbidden source column accessed")
        if access["protected"] and not access["path"].endswith("benchmarks.parquet") and set(access["columns"]) - {"subject_id", "item_id", "response_id", "trial"}:
            raise ValueError("Protected content was accessed")
    review = validate_reviews(list(read_jsonl(path / "review_inputs.jsonl")), list(read_jsonl(path / "reviewed_examples.jsonl")))
    if require_review and (not review["complete"] or not manifest["review_complete"]):
        raise ValueError("Review is not complete")
    return {"valid": True, "population": expected, "review_complete": manifest["review_complete"], "artifacts": len(manifest["artifacts"])}
