"""Typed observations of supplied settings, independent of model encodings."""
import json
import math
from urllib.parse import unquote

from ..state_rendering import split_settings
from .contracts import canonical_json, project_subject, stable_key

FIELDS = {
    "few_shot_count": "integer", "prompting_method": "string", "selection_method": "string",
    "reasoning_effort": "string", "max_input_tokens": "integer", "max_output_tokens": "integer",
    "max_reasoning_tokens": "integer", "max_turns": "integer", "max_tool_calls": "integer",
    "max_retries": "integer", "candidate_count": "integer", "temperature": "number", "top_p": "number",
    "enabled_tools": "string_list", "quantization": "string", "serving_configuration": "string",
    "sampling_enabled": "boolean", "reported_method": "string",
}
ALIASES = {**{name: name for name in FIELDS}, "n_shots": "few_shot_count", "shot_size": "few_shot_count",
           "prompting": "prompting_method", "prompt_style": "prompting_method",
           "max_new_tokens": "max_output_tokens", "do_sample": "sampling_enabled",
           "selection": "selection_method", "method": "reported_method"}
CONTAINERS = {"agent_configuration", "tool_configuration", "configuration", "config", "settings", "subject_features_extra", "released_configuration"}
SETTING_BRANCHES = CONTAINERS | {"llm", "generation_parameters", "generation_config", "sampling_params"}
POLICY_VERSION = "recorded_settings_v3"


class _ObservedObject(dict):
    """A JSON object retaining repeated members for observation/conflict checks."""
    def __init__(self, pairs):
        super().__init__(pairs)
        self.pairs = pairs

    def items(self):
        return self.pairs

    def values(self):
        return [value for _, value in self.pairs]


def _json(text):
    return json.loads(text, object_pairs_hook=_ObservedObject,
                      parse_constant=lambda _: (_ for _ in ()).throw(ValueError("nonfinite number")))


def _typed(value, kind):
    if kind == "boolean":
        if type(value) is not bool:
            raise ValueError("expected boolean")
        return value
    if kind == "string":
        if not isinstance(value, str) or not value.strip():
            raise ValueError("expected nonempty string")
        return value.strip()
    if kind == "string_list":
        if not isinstance(value, list) or any(not isinstance(v, str) for v in value):
            raise ValueError("expected string list")
        return value
    if type(value) not in {int, float} or not math.isfinite(value) or value < 0:
        raise ValueError("expected nonnegative finite number")
    if kind == "integer":
        if type(value) is not int:
            raise ValueError("expected integer")
        return value
    return float(value)


def parse_run_settings(subject, *, max_depth=12, max_nodes=4096):
    """Preserve raw occurrences; duplicate contradictions abstain per field.

    Generic max_tokens stays ambiguous; method is only a reported label. Only
    explicitly named containers are decoded; no task text is accepted here.
    """
    subject = project_subject(subject)
    system_key = stable_key(subject)
    observations, candidates = [], {field: [] for field in FIELDS}
    visited = 0

    def observe(key, raw, source_field, path, *, status="literal", reason="unrecognized_key", value=None):
        occurrence = len(observations)
        row = {"system_key": system_key, "source_field": source_field, "source_path": path,
               "occurrence": occurrence, "raw_key": key, "raw_value": raw,
               "field": ALIASES.get(key), "status": status, "reason": reason,
               "value_json": None if value is None else canonical_json(value),
               "source_scope": "released_configuration" if "/released_configuration" in path else
                    "selection_scope_unspecified" if key == "selection" else
                    "reported_method_label_only" if key == "method" else
                    "dedicated_subject_field" if source_field == "reasoning_effort" else "supplied_setting"}
        row["observation_id"] = stable_key(row)
        observations.append(row)
        if row["field"]:
            candidates[row["field"]].append(row)
        return row

    def visit(key, raw, source_field, path, depth=0, decoded_value=None, trusted_path=True):
        nonlocal visited
        visited += 1
        if not trusted_path:
            observe(key, raw, source_field, path, reason="unsupported_setting_path")
            return
        if depth > max_depth or visited > max_nodes:
            observe(key, raw, source_field, path, status="invalid", reason="decoding_limit")
            return
        if key in CONTAINERS:
            try:
                candidate = raw
                if candidate.lstrip().lower().startswith(("%7b", "%5b")):
                    import re
                    if re.search(r"%(?![a-fA-F0-9]{2})", candidate):
                        raise ValueError("invalid URL encoding")
                    candidate = unquote(candidate, errors="strict")
                for _ in range(max_depth):
                    parsed = _json(candidate)
                    if not isinstance(parsed, str):
                        break
                    candidate = parsed
                else:
                    raise ValueError("quoted container decoding limit")
                if not isinstance(parsed, (dict, list)):
                    raise ValueError("expected JSON container")
                # Check the entire shape before emitting trusted leaves.
                def check(node, level=0):
                    if level > max_depth:
                        raise ValueError("decoding_limit")
                    if isinstance(node, dict):
                        return 1 + sum(check(v, level + 1) for v in node.values())
                    if isinstance(node, list):
                        return 1 + sum(check(v, level + 1) for v in node)
                    return 1
                if check(parsed) + visited > max_nodes:
                    raise ValueError("decoding_limit")
                observe(key, raw, source_field, path, status="container", reason="decoded_settings")
                def flatten(node, prefix, trusted_path=True):
                    if isinstance(node, dict):
                        for child_key, child in node.items():
                            child_path = prefix + "/" + child_key.replace("~", "~0").replace("/", "~1")
                            if isinstance(child, dict) and child_key not in ALIASES:
                                flatten(child, child_path, trusted_path and child_key in SETTING_BRANCHES)
                            else:
                                visit(child_key, canonical_json(child), source_field, child_path, depth + 1, child, trusted_path)
                    else:
                        observe(key, canonical_json(node), source_field, prefix, status="invalid", reason="uninterpreted_container_list")
                flatten(parsed, path)
            except (ValueError, TypeError, UnicodeError, RecursionError):
                observe(key, raw, source_field, path, status="invalid", reason="invalid_or_oversized_container")
            return
        field = ALIASES.get(key)
        if not field:
            observe(key, raw, source_field, path)
            return
        try:
            value = decoded_value
            if value is None:
                if FIELDS[field] == "string":
                    try:
                        value = _json(raw)
                    except ValueError:
                        value = raw
                else:
                    value = _json(raw)
            value = _typed(value, FIELDS[field])
            if field == "top_p" and value > 1:
                raise ValueError("top_p outside [0,1]")
            observe(key, raw, source_field, path, status="known", reason="explicit_recorded_value", value=value)
        except (ValueError, TypeError, OverflowError):
            observe(key, raw, source_field, path, status="invalid", reason="invalid_type_or_range")

    if subject["reasoning_effort"]:
        visit("reasoning_effort", subject["reasoning_effort"], "reasoning_effort", "/reasoning_effort")
    raw = subject["subject_features_extra"]
    if raw:
        pairs = split_settings(raw)
        if pairs is None:
            if raw.lstrip().startswith(("{", "[")):
                visit("subject_features_extra", raw, "subject_features_extra", "/")
            else:
                observe("subject_features_extra", raw, "subject_features_extra", "/", reason="unparseable_literal")
        else:
            for index, (key, value) in enumerate(pairs):
                visit(key, value, "subject_features_extra", "/" + str(index) + "/" + key)
    resolved = []
    blockers = [row for row in observations if row["raw_key"] in CONTAINERS and row["status"] == "invalid"]
    for field, kind in FIELDS.items():
        claims = candidates[field]
        known = {row["value_json"] for row in claims if row["status"] == "known"}
        status = "invalid" if blockers else "unknown" if not claims else "invalid" if any(row["status"] == "invalid" for row in claims) else "conflict" if len(known) > 1 else "known" if known else "unknown"
        resolved.append({"system_key": system_key, "field": field, "value_type": kind,
                         "value_json": next(iter(known)) if status == "known" else None,
                         "status": status, "reason": "unresolved_container" if blockers else "not_recorded" if not claims else status,
                         "source_scope": "not_recorded" if not claims else next(iter({r["source_scope"] for r in claims})) if len({r["source_scope"] for r in claims}) == 1 else "mixed_recorded_scopes",
                         "observation_ids": [row["observation_id"] for row in claims + blockers], "policy_version": POLICY_VERSION})
    return {"observations": observations, "resolved": resolved,
            "diagnostics": {"observations": len(observations), "unrecognized": sum(row["field"] is None for row in observations)}}
