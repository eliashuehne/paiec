"""Feature definitions and denominator-specific source quality, never model scores."""
from collections import Counter, defaultdict
from html import escape

import pyarrow.parquet as pq

from .contracts import SUBJECT_FIELDS, VERSION
from .review import table_rows
from .settings import FIELDS, POLICY_VERSION
from .tables import SCHEMAS
from .enrichment import read_jsonl

FACETS = ("language", "domain", "skills", "reasoning_structure", "modalities", "answer_format", "information_needs", "input_issues")


def feature_dictionary(facts):
    result = []
    def add(name, kind, source, mechanism, *, eligibility="supplied_only", status="implemented", grain="system"):
        result.append({"name": name, "version": VERSION, "kind": kind, "grain": grain,
            "definition": source, "source_fields": [source], "mechanism_hypothesis": mechanism,
            "missing_behavior": "Raw empty strings stay empty; typed unknown/conflicting values are null with explicit status; zero/false/empty lists remain known.",
            "availability": "supplied_at_prediction" if eligibility == "supplied_only" else "offline_research_join_only",
            "eligibility": eligibility, "quality": "deterministic_observable_not_predictive_validation" if eligibility == "supplied_only" else "source_and_applicability_qualified",
            "status": status, "fit_scope": "none" if status == "implemented" else "separate_training_only_design_required"})
    for field in SUBJECT_FIELDS:
        add("system." + field, "string", "Supplied subject." + field + ", preserved verbatim", "Candidate model/configuration context; missing identity remains missing.")
    for field, kind in FIELDS.items():
        add("run." + field, kind, "Recorded setting observations resolved by " + POLICY_VERSION + ": " + field,
            "Actual prompting, sampling or resource configuration may affect target-system behavior; ambiguous aliases are withheld.")
    for field in SCHEMAS["task_features"]:
        if field.name != "task_key":
            add("task." + field.name, str(field.type), "Full visible task observable: " + field.name,
                "Surface structure/resource demand hypothesis; a syntax or script hint is not certified skill, language or difficulty.", grain="visible_task")
    for entity, field in sorted({(r["entity_type"], r["field"]) for r in facts}):
        add("research." + entity + "." + field, "qualified_json", "Combined shared-export " + entity + " fact: " + field,
            "Candidate functional capacity or architecture context, conditional on identity and historical applicability.", eligibility="research_only")
    for field in FACETS:
        add("semantic." + field, "candidate_labels", "Candidate task taxonomy facet: " + field,
            "Reviewed task requirements could support cross-domain transfer; no bulk labels are approved.", status="deferred_unreviewed", grain="visible_task")
    for field in ("language_match", "domain_match", "context_to_capacity", "required_tool_enabled", "semantic_embedding", "demonstrated_ability", "revealed_evidence"):
        add("candidate." + field, "not_materialized", "Proposed compatibility/representation/evidence family: " + field,
            "Requires reviewed operands, representation choice or separately authorized fitting; no zero stand-in column exists.", status="deferred", grain="system_task")
    return result


def counts(directory, table, column):
    result = Counter()
    for batch in pq.ParquetFile(directory / (table + ".parquet")).iter_batches(batch_size=65536, columns=[column]):
        result.update(batch.column(0).to_pylist())
    return dict(sorted(result.items(), key=lambda pair: str(pair[0])))


def coverage_report(directory, population, facts, review):
    systems = {r["system_key"]: r for r in table_rows(directory / "systems.parquet")}
    run = defaultdict(dict)
    run_coverage = {field: Counter() for field in FIELDS}
    for row in table_rows(directory / "run_settings.parquet"):
        run[row["system_key"]][row["field"]] = row["status"]
        run_coverage[row["field"]][row["status"]] += 1
    per_benchmark = {r["benchmark"]: {**r, "run_settings": {field: Counter() for field in FIELDS}, "supplied_fields": {field: Counter() for field in SUBJECT_FIELDS}} for r in population}
    for row in table_rows(directory / "subject_links.parquet"):
        bench = per_benchmark[row["benchmark"]]
        subject = systems.get(row["system_key"])
        for field in FIELDS:
            bench["run_settings"][field][run[row["system_key"]].get(field, "withheld_protected")] += 1
        for field in SUBJECT_FIELDS:
            status = "withheld_protected" if subject is None else "nonempty" if subject[field] else "empty"
            bench["supplied_fields"][field][status] += 1
    matched = {entity: Counter() for entity in ("model", "harness")}
    system_matches = {}
    for row in table_rows(directory / "profile_matches.parquet"):
        matched[row["entity_type"]][row["status"]] += 1
        system_matches[(row["system_key"], row["entity_type"])] = row["status"]
    research = defaultdict(Counter)
    research_reasons = defaultdict(Counter)
    system_research = {}
    for row in table_rows(directory / "research_system_features.parquet"):
        key = row["entity_type"] + "." + row["field"]
        research[key][row["status"]] += 1
        research_reasons[key][row["reason"]] += 1
        system_research[(row["system_key"], key)] = row["status"]
    research_fields = sorted({(r["entity_type"], r["field"]) for r in facts})
    for entity, field in research_fields:
        key = entity + "." + field
        research[key]["missing_profile_field"] = matched[entity]["matched"] - sum(research[key].values())
        research[key]["unmatched_systems"] = matched[entity]["unresolved"]
    for bench in per_benchmark.values():
        bench["research_features"] = {entity + "." + field: Counter() for entity, field in research_fields}
    for row in table_rows(directory / "subject_links.parquet"):
        for entity, field in research_fields:
            key = entity + "." + field
            status = "withheld_protected" if row["system_key"] is None else system_research.get(
                (row["system_key"], key), "missing_profile_field" if system_matches[(row["system_key"], entity)] == "matched" else "unmatched_systems")
            per_benchmark[row["benchmark"]]["research_features"][key][status] += 1
    task_population = defaultdict(Counter)
    for batch in pq.ParquetFile(directory / "items.parquet").iter_batches(batch_size=65536, columns=["split", "status"]):
        for row in batch.to_pylist():
            task_population[row["split"]][row["status"]] += 1
    task_reviews = [r for r in read_jsonl(directory / "reviewed_examples.jsonl") if r["kind"] == "task"]
    taxonomy = {field: {"candidate": sum(bool(r.get("taxonomy_review", {}).get("candidate_facets", {}).get(field)) for r in task_reviews),
                        "unknown": sum(not r.get("taxonomy_review", {}).get("candidate_facets", {}).get(field) for r in task_reviews),
                        "validated": 0} for field in FACETS}
    item_status = counts(directory, "items", "status")
    item_split = counts(directory, "items", "split")
    eligibility = counts(directory, "training_eligibility", "reason")
    tasks = pq.ParquetFile(directory / "tasks.parquet").metadata.num_rows
    gaps = []
    def gap(key, cause, population_name, count, priority, action, closure):
        gaps.append({"gap_id": key, "cause": cause, "population": population_name, "affected_count": count,
                     "priority": priority, "priority_rationale": "Source reach and contract completeness, not fitted feature importance",
                     "next_action": action, "cost_category": "mechanical_review" if priority == "P0" else "separate_research_or_review",
                     "eligibility": "does_not_change_source_policy", "closure_criterion": closure})
    gap("identity", "unresolved_identity", "distinct_processed_visible_systems", matched["model"]["unresolved"], "P0",
        "Review supplied reported/base identifiers and applicability before extending aliases", "Known relation or documented unresolved reason; never a guessed checkpoint")
    gap("settings", "absent_ambiguous_or_unparsed_settings", "system_setting_cells", sum(v["unknown"] + v["invalid"] + v["conflict"] for v in run_coverage.values()), "P0",
        "Review frequent raw keys and containers; distinguish missing source data from interpretation gaps", "Typed value with source or explicit unresolved/unknown disposition")
    gap("protected", "protected_role", "original_item_rows", item_status.get("withheld_protected", 0), "P0",
        "Keep metadata-only census and frozen roles", "Accounted for as withheld; no access is authorized by this gap")
    gap("modalities", "unsupported_nontext_assets", "original_item_rows", item_status.get("partial_nontext", 0), "P1",
        "Design modality handling separately; current text counts are partial", "Explicit partial support or a reviewed new modality contract")
    gap("overlap", "cross_split_normalized_content", "item_input_variants", eligibility.get("cross_split_content", 0), "P0",
        "Keep blocked training rows visible and review grouping before future modeling", "No training leakage or a justified conservative exclusion; never move frozen roles")
    gap("semantics", "unreviewed_semantic_annotations", "unique_processed_visible_tasks", tasks, "P1",
        "Use the small training audit to refine the eight-facet taxonomy before any bulk annotation", "Independent semantic quality policy and reviewed coverage; not claimed by this PR")
    gap("historical", "unverified_historical_applicability", "known_research_feature_cells", sum(v["known"] for v in research.values()), "P1",
        "Bind facts to recorded versions where evidence permits; retain other cases as qualified", "Version applicability established or explicitly unknown")
    gap("eligibility", "external_source_eligibility_unreviewed", "research_profiles", len({r["profile_id"] for r in facts}), "P1",
        "Resolve competition-source permission separately; keep research view segregated", "Explicit eligibility decision without changing supplied-only features")
    return {"version": VERSION,
        "population": {"benchmarks": len(population), **{key: sum(r[key] for r in population) for key in ("subjects", "items", "responses")}},
        "distinct_processed_visible_systems": len(systems), "distinct_processed_visible_tasks": tasks,
        "denominators": {"population": "Original source rows, including withheld and unsupported records", "run_settings": "One cell per field per distinct processed visible system", "per_benchmark": "Original source subject rows, with protected rows explicit", "research": "Distinct processed visible systems; unmatched and withheld fact values separate", "task_features": "Deduplicated exact visible input variants; not original item rows or response trials"},
        "item_status": item_status, "item_split": item_split, "training_eligibility": eligibility,
        "task_feature_population": dict(task_population), "taxonomy_audit": taxonomy,
        "response_status": counts(directory, "response_input_links", "status"),
        "run_settings": {k: dict(v) for k, v in run_coverage.items()},
        "profile_matches": {k: dict(v) for k, v in matched.items()},
        "research_features": {k: dict(v) for k, v in research.items()},
        "research_withholding_reasons": {k: dict(v) for k, v in research_reasons.items()},
        "task_features": {"known_observations_per_feature": tasks, "protected_content": "withheld", "semantic_features": "not_materialized"},
        "per_benchmark": list(per_benchmark.values()), "review": review, "gaps": gaps,
        "limitations": ["No predictive usefulness measured", "Public development corpus was previously analyzed; split names do not create a pristine test", "Protected content overlap is unmeasured", "Normalized content is a conservative exact screen, not paraphrase/translation detection", "Multimodal assets were not read", "Semantic labels are candidates outside the default projection"]}


def render_report(report):
    pop = report["population"]
    def table(headers, rows):
        return '<table><thead><tr>' + ''.join('<th>' + escape(str(h)) + '</th>' for h in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join('<td>' + escape(str(c)) + '</td>' for c in row) + '</tr>' for row in rows) + '</tbody></table>'
    html = '<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Feature foundation: population and gaps</title><style>body{font:16px/1.55 system-ui,sans-serif;max-width:1160px;margin:40px auto;padding:0 24px;color:#202b34;background:#fbfaf7}h1,h2{line-height:1.2}table{border-collapse:collapse;width:100%;margin:20px 0;font-size:14px}td,th{text-align:left;padding:9px;border-bottom:1px solid #d6dadb;vertical-align:top}th{background:#e9efec}section{margin:40px 0}p{max-width:85ch}details{margin:20px 0;overflow:auto}.notice{border-left:4px solid #3d7662;padding:12px 18px;background:#edf3ee}</style><body><h1>Feature foundation</h1><p>Full public population, traceable features and explicit information gaps.</p>'
    html += '<p class="notice">No predictive model was trained or evaluated. External enrichment remains research-only. Protected input content and all response outcome columns remain unopened.</p>'
    html += '<section><h2>Population</h2>' + table(['Unit', 'Original count'], [(key, f'{value:,}') for key, value in pop.items()]) + '<p>Counts refer to different units. Visible task variants and distinct system descriptions are deduplicated feature entities, not replacements for the original population.</p></section>'
    html += '<section><h2>Item processing and splits</h2>' + table(['Disposition', 'Item rows'], report['item_status'].items()) + table(['Split', 'Item rows'], report['item_split'].items()) + '<p>The historical training-monitor bucket is preserved at item level. Cross-split content excludes training use without moving frozen roles.</p></section>'
    html += '<section><h2>Recorded run settings</h2><p>Denominator: distinct processed visible systems. Known zero and empty lists are retained.</p>' + table(['Setting', 'Known', 'Unknown', 'Invalid', 'Conflict'], [(k, *(v.get(s, 0) for s in ['known', 'unknown', 'invalid', 'conflict'])) for k, v in report['run_settings'].items()]) + '</section>'
    html += '<section><h2>Reviewed examples</h2>' + table(['Measure', 'Value'], [(k, v) for k, v in report['review'].items() if k != 'selection_limitations']) + '<p>' + escape(report['review']['selection_limitations']) + '</p></section>'
    html += '<section><h2>Task taxonomy audit</h2><p>Counts describe the small reviewed task sample. Candidate labels are outside the feature tables; zero facets have established semantic quality.</p>' + table(['Facet', 'Candidate', 'Unknown', 'Validated'], [(k, v['candidate'], v['unknown'], v['validated']) for k, v in report['taxonomy_audit'].items()]) + '</section>'
    html += '<section><h2>Prioritized gaps</h2>' + table(['Priority', 'Gap', 'Affected count', 'Next action'], [(r['priority'], r['cause'], r['affected_count'], r['next_action']) for r in report['gaps']]) + '</section>'
    html += '<details><summary>Per-benchmark population and support</summary>' + table(['Benchmark', 'Split', 'Frozen role', 'Items', 'Responses', 'Status'], [(r['benchmark'], r['split'], r['frozen_role'] or '', r['items'], r['responses'], r['feature_status']) for r in report['per_benchmark']]) + '</details>'
    html += '<section><h2>Limits and interpretation</h2><ul>' + ''.join('<li>' + escape(v) + '</li>' for v in report['limitations']) + '</ul><p>Machine-readable definitions, coverage, source hashes and review records accompany this report.</p></section></body></html>'
    return html
