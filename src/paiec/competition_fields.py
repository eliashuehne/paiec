"""Portable supplied-field boundary and compatible-evidence matching."""
SUBJECT_FIELDS = ("normalized_name", "provider", "release_date", "access_date", "harness",
                  "harness_version", "reasoning_effort", "subject_features_extra")
ITEM_FIELDS = ("item_content", "item_features", "interactors", "benchmark_id")


def checked_fields(subject, item):
    for record, keys in [(subject, SUBJECT_FIELDS), (item, ITEM_FIELDS)]:
        if not isinstance(record, dict) or set(record) != set(keys) or any(type(v) is not str for v in record.values()):
            raise ValueError("Expected the exact supplied competition string fields")
    return subject, item


def normalize_fields(subject, item):
    """Restore missing known fields as empty strings, preserving supplied text."""
    result = []
    for record, keys in [(subject, SUBJECT_FIELDS), (item, ITEM_FIELDS)]:
        if not isinstance(record, dict) or not set(record) <= set(keys) or any(type(v) is not str for v in record.values()):
            raise ValueError("Unexpected competition fields or non-string values")
        result.append({key: record.get(key, "") for key in keys})
    return result


def matching_evidence(subject, item, labeled):
    checked_fields(subject, item)
    matched = []
    if not item["benchmark_id"]:
        return matched
    for entry in labeled or []:
        if not isinstance(entry, (tuple, list)) or len(entry) != 2:
            raise ValueError("Malformed revealed example")
        records, label = entry
        if not isinstance(records, (tuple, list)) or len(records) != 2:
            raise ValueError("Malformed revealed input")
        observed_subject, observed_item = checked_fields(*records)
        if type(label) not in {bool, int} or label not in (0, 1):
            raise ValueError("Revealed labels must be binary")
        if observed_subject == subject and observed_item["benchmark_id"] == item["benchmark_id"] and observed_item != item:
            matched.append((observed_item, label))
    return matched
