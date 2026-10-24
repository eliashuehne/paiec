"""Offline import of downloaded OpenAI Batch Responses files.

Local file integrity and request binding are verified. This does not authenticate
the network origin of supplied files or establish semantic annotation quality.
"""
from collections import Counter
import json
from pathlib import Path

from .feature_preparation import read_jsonl, stable_hash, write_json, write_jsonl
from .head_training import file_hash
from .task_annotation import import_responses, openai_batch_request, validate_plan


def checked_batch(plan_path, batch_path, expected_sha256):
    plan, contract = validate_plan(plan_path)
    if file_hash(batch_path) != expected_sha256:
        raise ValueError("Batch input hash differs from the recorded export")
    requests = {r["request_id"]: r for r in read_jsonl(Path(plan_path) / "requests.jsonl")}
    selected, settings = {}, set()
    for row in read_jsonl(batch_path):
        key = row.get("custom_id")
        if key not in requests or key in selected:
            raise ValueError("Foreign or duplicate batch input request")
        body = row.get("body", {})
        model, limit = body.get("model"), body.get("max_output_tokens")
        if row != openai_batch_request(requests[key], contract, model=model, max_output_tokens=limit):
            raise ValueError("Batch request differs from the bound annotation contract")
        settings.add((model, limit))
        selected[key] = row
    if not selected or len(settings) != 1:
        raise ValueError("Use a nonempty, single-model batch with one output budget")
    return plan, selected


def unique_object(pairs):
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("Duplicate JSON field")
        result[key] = value
    return result


def response_annotation(body):
    """Reject refusals, truncation, tool output and ambiguous multiple messages."""
    if not isinstance(body, dict) or body.get("status") != "completed" or body.get("error") is not None or body.get("incomplete_details") is not None:
        raise ValueError("response_not_completed")
    output = body.get("output")
    if not isinstance(output, list) or not output or any(not isinstance(r, dict) or r.get("type") not in {"message", "reasoning"} for r in output):
        raise ValueError("unexpected_output")
    messages = [r for r in output if r["type"] == "message"]
    if len(messages) != 1 or messages[0].get("role") != "assistant" or messages[0].get("status") != "completed":
        raise ValueError("ambiguous_message")
    content = messages[0].get("content")
    if not isinstance(content, list) or len(content) != 1 or not isinstance(content[0], dict) or content[0].get("type") != "output_text" or not isinstance(content[0].get("text"), str):
        raise ValueError("refusal_or_missing_text")
    try:
        parsed = json.loads(content[0]["text"], object_pairs_hook=unique_object)
    except ValueError:
        raise ValueError("invalid_annotation_json") from None
    if not isinstance(parsed, dict):
        raise ValueError("invalid_annotation_json")
    return parsed


def import_openai_batch(plan_path, batch_path, results_paths, out, *, batch_sha256):
    """Join unordered output/error files by custom_id; never retry or submit."""
    plan_path, batch_path, out = Path(plan_path), Path(batch_path), Path(out)
    plan, selected = checked_batch(plan_path, batch_path, batch_sha256)
    paths = [Path(p) for p in results_paths]
    if not paths or len({p.resolve() for p in paths}) != len(paths):
        raise ValueError("Provide unique downloaded result/error files")
    normalized, seen, receipts, failures = [], set(), [], []
    usage_totals, models = Counter(), Counter()
    for path in paths:
        for row in read_jsonl(path):
            key = row.get("custom_id")
            if key not in selected or key in seen:
                raise ValueError("Foreign or duplicate batch result")
            seen.add(key)
            response = row.get("response")
            body = response.get("body") if isinstance(response, dict) else None
            model = body.get("model") if isinstance(body, dict) else None
            usage = body.get("usage") if isinstance(body, dict) else None
            valid_usage = isinstance(usage, dict) and all(type(usage.get(k)) is int and usage[k] >= 0 for k in ("input_tokens", "output_tokens", "total_tokens"))
            if valid_usage:
                valid_usage = usage["input_tokens"] + usage["output_tokens"] == usage["total_tokens"]
            if valid_usage:
                usage_totals.update({k: usage[k] for k in ("input_tokens", "output_tokens", "total_tokens")})
            if isinstance(model, str):
                models[model] += 1
            receipt = {"request_id": key, "row_sha256": stable_hash(row),
                       "requested_model": selected[key]["body"]["model"], "reported_model": model if isinstance(model, str) else None,
                       "reported_usage": usage if valid_usage else None,
                       "usage_valid": valid_usage}
            try:
                if row.get("error") is not None or not isinstance(response, dict) or type(response.get("status_code")) is not int or response["status_code"] != 200:
                    raise ValueError("provider_request_failed")
                if not isinstance(model, str) or not model:
                    raise ValueError("missing_reported_model")
                normalized.append({"request_id": key, "response": response_annotation(body)})
            except ValueError as error:
                failures.append({"request_id": key, "row_sha256": receipt["row_sha256"], "reason": str(error)})
            receipts.append(receipt)
    # Foreign/duplicate rows and changed requests fail before publishing anything.
    out.mkdir(parents=True, exist_ok=False)
    report = import_responses(plan_path, normalized, out / "annotations", producer="openai_batch_local_import_v1")
    write_jsonl(out / "provider_failures.jsonl", failures)
    write_jsonl(out / "provider_receipts.jsonl", receipts)
    write_jsonl(out / "missing_batch_results.jsonl", ({"request_id": k} for k in sorted(set(selected) - seen)))
    summary = {**report, "version": "task_annotation_batch_import_v1", "contract_sha256": plan["contract_sha256"],
               "batch_input_sha256": batch_sha256,
               "result_files": [{"sha256": file_hash(p), "name": p.name} for p in paths],
               "selected_requests": len(selected), "received_results": len(seen),
               "missing_results": len(selected) - len(seen), "provider_failures": len(failures),
               "reported_models": dict(models), "observed_usage": dict(usage_totals),
               "results_with_valid_usage": sum(r["usage_valid"] for r in receipts),
               "network_origin_verified": False, "model_api_calls": 0,
               "billing_total_verified": False, "automatic_retries": 0}
    write_json(out / "summary.json", summary)
    write_json(out / "manifest.json", {"artifacts": {str(p.relative_to(out)): file_hash(p) for p in sorted(out.rglob("*")) if p.is_file()}})
    return summary
