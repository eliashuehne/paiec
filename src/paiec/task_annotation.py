"""Outcome-blind task annotation plans, exact evidence and candidate imports.

This module performs no network calls, model loading, fitting or dispatch.
Response validation establishes grounding/shape, not semantic annotation quality.
"""
import json
from collections import Counter
from html import escape
from pathlib import Path
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from .feature_preparation import read_jsonl, write_json, write_jsonl, stable_hash
from .head_training import file_hash

ITEM_KEYS = {"item_content", "item_features", "interactors"}


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)


class Quote(Strict):
    field: Literal["item_content", "item_features", "interactors"]
    quote: str = Field(min_length=1)
    occurrence: int = Field(ge=0)


class Facet(Strict):
    status: Literal["estimated", "unknown"]
    values: list[str]
    evidence: list[Quote]
    rationale: str = Field(min_length=1)

    @model_validator(mode="after")
    def consistent(self):
        if self.status == "unknown":
            if self.values or self.evidence:
                raise ValueError("Unknown facets need empty values/evidence and an explanation")
        elif not self.values or not self.evidence:
            raise ValueError("Estimated facets require values and exact evidence")
        if len(self.values) != len(set(self.values)):
            raise ValueError("Duplicate facet values")
        return self


class Facets(Strict):
    language: Facet
    domain: Facet
    skills: Facet
    reasoning_structure: Facet
    modalities: Facet
    answer_format: Facet
    information_needs: Facet
    input_issues: Facet


class AnnotationResponse(Strict):
    schema_version: Literal["task_annotation_v2"]
    task_key: str = Field(pattern="^[a-f0-9]{64}$")
    facets: Facets
    difficulty: None


def visible_item(task):
    item = task["recorded"]
    if set(item) != ITEM_KEYS or any(not isinstance(v, str) for v in item.values()):
        raise ValueError("Annotation input must contain exactly the three permitted visible item fields")
    if stable_hash(item) != task["task_key"]:
        raise ValueError("Task input identity changed")
    return dict(item)


def response_schema(taxonomy):
    """All fields required; per-facet enum values also constrained at the API."""
    schema = AnnotationResponse.model_json_schema()
    for name in Facets.model_fields:
        facet = json.loads(json.dumps(schema["$defs"]["Facet"]))
        values = facet["properties"]["values"]["items"]
        if name == "language":
            values["pattern"] = r"^[a-z]{2,3}(?:-[A-Za-z0-9]{2,8})*$"
        else:
            values["enum"] = taxonomy["facets"][name]
        schema["$defs"]["Facets"]["properties"][name] = facet
    return schema


def compile_response(raw, task, taxonomy):
    response = AnnotationResponse.model_validate(raw)
    item = visible_item(task)
    if response.task_key != task["task_key"]:
        raise ValueError("Annotation belongs to a different task")
    facets = {}
    for name in Facets.model_fields:
        facet = getattr(response.facets, name)
        for value in facet.values:
            valid = bool(re.fullmatch(r"[a-z]{2,3}(?:-[A-Za-z0-9]{2,8})*", value)) if name == "language" else value in taxonomy["facets"][name]
            if not valid:
                raise ValueError("Annotation label is outside the versioned taxonomy: " + name)
        evidence = []
        for quote in facet.evidence:
            start = -1
            for _ in range(quote.occurrence + 1):
                start = item[quote.field].find(quote.quote, start + 1)
                if start < 0:
                    raise ValueError("Annotation evidence is not an exact supplied-item quote")
            evidence.append({"field": quote.field, "start": start, "end": start + len(quote.quote), "quote": quote.quote})
        facets[name] = {**facet.model_dump(mode="json"), "evidence": evidence}
    return {"schema_version": "task_annotation_v2", "task_key": response.task_key,
            "input_sha256": stable_hash(item), "facets": facets, "difficulty": None,
            "feature_projection": {name: {"status": facet["status"], "values": facet["values"]} for name, facet in facets.items()},
            "rationales_are_review_metadata_only": True,
            "review_status": "candidate", "independently_reviewed": False}


def prepare_requests(tasks, examples, contract, out):
    """Provider-neutral requests; retain every supplied character and no outcomes."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    contract_key = stable_hash(contract)
    write_json(out / "contract.json", contract)
    membership = {}
    for row in examples:
        if row["role"] not in {"training", "monitor", "development"}:
            raise ValueError("Sealed roles cannot enter annotation preparation")
        membership.setdefault(row["task_key"], set()).add(row["role"])
    seen, index, characters, input_bytes = set(), [], 0, 0
    with (out / "requests.jsonl").open("w") as handle:
        for task in tasks:
            item = visible_item(task)
            key = task["task_key"]
            if key in seen or key not in membership or len(membership[key]) != 1:
                raise ValueError("Duplicate, orphaned or cross-role task in annotation preparation")
            seen.add(key)
            request_id = stable_hash({"contract": contract_key, "task": key})
            request = {"request_id": request_id, "task_key": key, "item": item}
            handle.write(json.dumps(request, ensure_ascii=False, sort_keys=True) + "\n")
            size = sum(len(value.encode("utf-8")) for value in item.values())
            characters += sum(map(len, item.values()))
            input_bytes += size
            index.append({"request_id": request_id, "task_key": key, "role": next(iter(membership[key])), "visible_utf8_bytes": size})
    if seen != set(membership):
        raise ValueError("Some selected tasks have no annotation request")
    write_jsonl(out / "request_index.jsonl", index)
    summary = {"version": "task_annotation_plan_v2", "contract_sha256": contract_key,
               "tasks": len(index), "roles": dict(Counter(r["role"] for r in index)),
               "supplied_characters": characters, "supplied_utf8_bytes": input_bytes,
               "maximum_item_utf8_bytes": max((r["visible_utf8_bytes"] for r in index), default=0),
               "request_file_sha256": file_hash(out / "requests.jsonl"),
               "index_sha256": file_hash(out / "request_index.jsonl"),
               "truncation_performed": False, "model_api_calls": 0,
               "target_outcomes_exported": False, "benchmark_ids_exported": False,
               "token_counts": "not_measured", "semantic_quality_established": False}
    write_json(out / "plan.json", summary)
    return summary


def validate_plan(path):
    path = Path(path)
    plan = json.loads((path / "plan.json").read_text())
    contract = json.loads((path / "contract.json").read_text())
    if stable_hash(contract) != plan["contract_sha256"] or file_hash(path / "requests.jsonl") != plan["request_file_sha256"] or file_hash(path / "request_index.jsonl") != plan["index_sha256"]:
        raise ValueError("Annotation plan changed")
    return plan, contract


def import_responses(plan_path, responses, out, *, producer):
    """Reject extra/foreign/duplicate responses; incomplete imports stay incomplete."""
    plan_path, out = Path(plan_path), Path(out)
    plan, contract = validate_plan(plan_path)
    out.mkdir(parents=True, exist_ok=False)
    requests = {r["request_id"]: r for r in read_jsonl(plan_path / "requests.jsonl")}
    accepted, rejected, seen = {}, [], set()
    for envelope in responses:
        request_id = envelope.get("request_id")
        if request_id in seen:
            raise ValueError("Duplicate annotation response")
        seen.add(request_id)
        if set(envelope) != {"request_id", "response"} or request_id not in requests:
            raise ValueError("Foreign annotation response or unexpected response fields")
        request = requests[request_id]
        try:
            task = {"task_key": request["task_key"], "recorded": request["item"]}
            candidate = compile_response(envelope["response"], task, contract["taxonomy"])
        except ValueError as error:
            # Avoid copying an accidentally generated answer into feature artifacts.
            rejected.append({"request_id": request_id, "response_sha256": stable_hash(envelope["response"]), "reason": type(error).__name__})
            continue
        accepted[request_id] = {**candidate, "request_id": request_id, "producer": producer,
                                "contract_sha256": plan["contract_sha256"],
                                "response_sha256": stable_hash(envelope["response"])}
    write_jsonl(out / "candidates.jsonl", accepted.values())
    write_jsonl(out / "rejected.jsonl", rejected)
    write_jsonl(out / "pending.jsonl", ({"request_id": key, "task_key": request["task_key"]}
        for key, request in requests.items() if key not in accepted))
    report = {"tasks": len(requests), "candidate_annotations": len(accepted), "rejected": len(rejected),
              "pending": len(requests) - len(accepted), "response_coverage_complete": len(accepted) == len(requests),
              "independently_reviewed": 0, "semantic_quality_established": False,
              "training_ready": False, "empirical_difficulty_fitted": False}
    write_json(out / "coverage.json", report)
    return report


def openai_batch_request(request, contract, *, model, max_output_tokens):
    """The entire allowed provider request, also used to verify imported batches."""
    if not isinstance(model, str) or not model.strip() or type(max_output_tokens) is not int or not 1 <= max_output_tokens <= 16384:
        raise ValueError("A named model and bounded output budget are required")
    body = {"model": model, "store": False, "reasoning": {"effort": "none"},
            "instructions": contract["instructions"] + "\nTaxonomy:\n" + json.dumps(contract["taxonomy"], ensure_ascii=False),
            "input": json.dumps({"task_key": request["task_key"], "item": request["item"]}, ensure_ascii=False),
            "max_output_tokens": max_output_tokens,
            "text": {"format": {"type": "json_schema", "name": "task_annotation_v2",
                                  "schema": contract["response_schema"], "strict": True}}}
    return {"custom_id": request["request_id"], "method": "POST", "url": "/v1/responses", "body": body}


def export_openai_batch(plan_path, out, *, task_keys, model="gpt-5.4-mini", max_output_tokens=4096):
    """Build a reviewable Batch API input file locally; no submission occurs."""
    plan_path, out = Path(plan_path), Path(out)
    plan, contract = validate_plan(plan_path)
    if not task_keys or len(task_keys) != len(set(task_keys)) or not 1 <= max_output_tokens <= 16384:
        raise ValueError("Use unique explicitly selected tasks and a bounded output budget")
    selected, found = set(task_keys), set()
    available = {r["task_key"] for r in read_jsonl(plan_path / "request_index.jsonl")}
    if not selected <= available:
        raise ValueError("Selected tasks are absent from the bound annotation plan")
    with out.open("x") as handle:
        for request in read_jsonl(plan_path / "requests.jsonl"):
            if request["task_key"] not in selected:
                continue
            found.add(request["task_key"])
            row = openai_batch_request(request, contract, model=model, max_output_tokens=max_output_tokens)
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    if found != selected:
        raise ValueError("Selected tasks are absent from the bound annotation plan")
    return {"requests": len(found), "model": model, "max_output_tokens_per_request": max_output_tokens,
            "batch_file_sha256": file_hash(out), "submitted": False, "model_api_calls": 0}


def prepare_pipeline_annotations(root, tasks, out, *, config_path):
    root, out = Path(root).resolve(), Path(out)
    config_file = (root / config_path).resolve()
    if not config_file.is_relative_to(root / "configs/pipelines"):
        raise ValueError("Annotation configuration escapes pipeline storage")
    config = json.loads(config_file.read_text())
    def checked(name, directory):
        path = (root / config[name]).resolve()
        if not path.is_relative_to(root / directory):
            raise ValueError("Annotation resource escapes " + directory)
        return path
    taxonomy = json.loads(checked("taxonomy", "configs").read_text())
    prompt = checked("instructions", "configs").read_text()
    seeds_path = checked("manual_candidates", "research")
    seeds = list(read_jsonl(seeds_path))
    contract = {"version": "task_annotation_contract_v2", "instructions": prompt,
                "taxonomy": taxonomy, "response_schema": response_schema(taxonomy),
                "coordinate_system": "zero_based_unicode_codepoints", "difficulty": "outcome_based_stage_only"}
    destination = out / "task_annotations_v2"
    plan = prepare_requests(read_jsonl(tasks["path"]), read_jsonl(out / "examples.jsonl"), contract, destination)
    sample = list(read_jsonl(out / "task_review_sample.jsonl"))
    sample_keys = {r["task"]["task_key"] for r in sample}
    if {r["task_key"] for r in seeds} != sample_keys:
        raise ValueError("Manual annotation pilot must match the exact training-only review sample")
    ids = {r["task_key"]: r["request_id"] for r in read_jsonl(destination / "request_index.jsonl")}
    result = import_responses(destination, ({"request_id": ids[r["task_key"]], "response": r} for r in seeds),
                              destination / "manual", producer="codex_manual_v2")
    pilot = export_openai_batch(destination, destination / "pilot_openai_batch.jsonl", task_keys=sorted(sample_keys),
                                model=config["proposed_teacher"], max_output_tokens=config["max_output_tokens"])
    write_json(destination / "pilot_task_keys.json", sorted(sample_keys))
    report = {**plan, **result, "prepared": True, "manual_candidates_sha256": file_hash(seeds_path),
              "configuration_sha256": file_hash(config_file), "pilot": pilot,
              "network_submission_performed": False, "full_objective_achieved": False}
    write_json(destination / "summary.json", report)
    candidates = {r["task_key"]: r for r in read_jsonl(destination / "manual/candidates.jsonl")}
    html = '<!doctype html><meta charset="utf-8"><title>Task annotation pilot</title><style>body{font:16px system-ui;max-width:1100px;margin:40px auto;padding:20px}pre{white-space:pre-wrap;overflow-wrap:anywhere}details{border:1px solid #ddd;padding:12px;margin:12px 0}</style><h1>Task annotation pilot</h1><p>Full input preserved. Agent candidates, not independent reviews. No target answers, model success labels or fitted difficulty. API requests are prepared locally and have not been sent.</p><pre>' + escape(json.dumps(report, indent=2)) + '</pre>'
    for row in sample:
        key = row["task"]["task_key"]
        html += '<details><summary>' + escape(key[:16]) + '</summary><h2>Supplied item</h2><pre>' + escape(json.dumps(row["task"]["recorded"], ensure_ascii=False, indent=2)) + '</pre><h2>Candidate annotation</h2><pre>' + escape(json.dumps(candidates.get(key, {"status": "rejected"}), ensure_ascii=False, indent=2)) + '</pre></details>'
    (destination / "review.html").write_text(html)
    write_json(destination / "manifest.json", {"artifacts": {str(p.relative_to(destination)): file_hash(p) for p in sorted(destination.rglob("*")) if p.is_file()}})
    return report
