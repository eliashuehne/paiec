"""Outcome-blind training samples and honestly qualified review records."""
from collections import defaultdict
import heapq
import json

import pyarrow.parquet as pq

from .contracts import SUBJECT_FIELDS, stable_key


def table_rows(path, columns=None):
    reader = pq.ParquetFile(path)
    for batch in reader.iter_batches(batch_size=8192, columns=columns or reader.schema_arrow.names):
        yield from batch.to_pylist()


def select_review_cases(directory, config, population):
    """Bounded reservoirs before stratification; no task labels are consulted.

    Semantic inspection is capped at 8192 characters per complete visible input.
    Longer inputs remain fully extracted and explicitly outside this review.
    """
    training = {row["benchmark"] for row in population if row["split"] == "training"}
    system_keys = {row["system_key"] for row in table_rows(directory / "subject_links.parquet")
                   if row["benchmark"] in training and row["system_key"]}
    system_candidates = []
    for row in table_rows(directory / "systems.parquet"):
        if row["system_key"] in system_keys:
            raw = {key: row[key] for key in SUBJECT_FIELDS}
            rank = stable_key({"seed": config.seed, "system": row["system_key"]})
            system_candidates.append((rank, row["harness"], row["system_key"], raw))
    # Round-robin harness strata, with seeded stratum ordering.
    groups = defaultdict(list)
    for rank, harness, key, raw in sorted(system_candidates):
        groups[harness].append((rank, key, raw))
    systems = []
    strata = sorted(groups, key=lambda k: stable_key({"seed": config.seed, "harness": k}))
    while len(systems) < config.review_cases_per_kind and any(groups.values()):
        for stratum in strata:
            if groups[stratum] and len(systems) < config.review_cases_per_kind:
                _, key, raw = groups[stratum].pop(0)
                systems.append({"case_id": stable_key({"kind": "system", "key": key}), "kind": "system",
                                "split": "training", "input_hash": stable_key(raw), "system_key": key,
                                "source": {"table": "systems.parquet", "key": key}, "input": raw})
    reservoirs = defaultdict(list)
    for row in table_rows(directory / "training_eligibility.parquet"):
        if row["split"] != "training" or not row["task_key"] or row["reason"] in {"cross_split_content", "withheld_protected", "missing_content"}:
            continue
        rank = int(stable_key({"seed": config.seed, "task": row["task_key"]})[:16], 16)
        entry = (-rank, row["task_key"], row["item_key"])
        heap = reservoirs[row["benchmark"]]
        if len(heap) < 12:
            heapq.heappush(heap, entry)
        elif entry > heap[0]:
            heapq.heapreplace(heap, entry)
    candidates = {}
    for benchmark, entries in reservoirs.items():
        for rank, task, item in entries:
            candidates.setdefault(task, {"rank": -rank, "benchmark": benchmark, "item_key": item})
    strata = defaultdict(list)
    for row in table_rows(directory / "task_features.parquet"):
        if row["task_key"] not in candidates or row["content_characters"] > 8192:
            continue
        item = candidates[row["task_key"]]
        length = "short" if row["content_characters"] <= 512 else "medium" if row["content_characters"] <= 2048 else "long"
        script = "cjk" if row["cjk_characters"] else "arabic" if row["arabic_characters"] else "cyrillic" if row["cyrillic_characters"] else "other"
        fmt = "code" if row["has_code_fence"] else "options" if row["option_marker_count_hint"] else row["structure_hint"]
        stratum = (item["benchmark"], length, script, fmt)
        strata[stratum].append((item["rank"], row["task_key"]))
    selected = []
    ordered = sorted(strata, key=lambda key: stable_key({"seed": config.seed, "stratum": key}))
    for entries in strata.values():
        entries.sort()
    while len(selected) < config.review_cases_per_kind and any(strata.values()):
        for stratum in ordered:
            if strata[stratum] and len(selected) < config.review_cases_per_kind:
                _, key = strata[stratum].pop(0)
                selected.append((key, stratum))
    wanted = dict(selected)
    tasks = []
    for row in table_rows(directory / "tasks.parquet"):
        key = row["task_key"]
        if key in wanted:
            raw = {k: row[k] for k in ("item_content", "item_features", "interactors")}
            if sum(map(len, raw.values())) > 8192:
                continue
            tasks.append({"case_id": stable_key({"kind": "task", "key": key}), "kind": "task", "split": "training",
                          "input_hash": stable_key(raw), "task_key": key, "stratum": list(wanted[key]),
                          "source": {"table": "tasks.parquet", "key": key, **candidates[key]}, "input": raw})
    return sorted(systems + tasks, key=lambda row: (row["kind"], row["case_id"]))


def validate_reviews(cases, reviews):
    case_map = {row["case_id"]: row for row in cases}
    if len(case_map) != len(cases) or any(row["split"] != "training" for row in cases):
        raise ValueError("Review cases must be unique and training-only")
    seen = set()
    for row in reviews:
        case = case_map.get(row["case_id"])
        if case is None or row["case_id"] in seen or row["input_hash"] != case["input_hash"]:
            raise ValueError("Review references duplicate, foreign or changed inputs")
        if row.get("status") != "reviewed" or not row.get("checks") or not row.get("disposition"):
            raise ValueError("Review has no completed checks/disposition")
        if row.get("method") not in {"agent_structural_and_taxonomy_audit", "independent_human"}:
            raise ValueError("Unqualified reviewer method")
        if row["method"] == "independent_human" and "agent" in row.get("reviewer", "").lower():
            raise ValueError("Agent review cannot be recorded as human review")
        for evidence in row.get("evidence", []):
            raw = case.get("input", {})
            text = raw.get(evidence["field"])
            if text is None or text[evidence["start"]:evidence["end"]] != evidence["quote"]:
                raise ValueError("Review evidence is not an exact visible-input span")
        if "difficulty" in row or "answer" in row or "solution" in row:
            raise ValueError("Review must not generate answers or numeric difficulty")
        seen.add(row["case_id"])
    return {"selected": len(cases), "reviewed": len(seen), "complete": bool(cases) and len(seen) == len(cases),
            "independent_human_reviews": sum(row["method"] == "independent_human" for row in reviews),
            "semantic_quality_established": False,
            "selection_limitations": "Seeded bounded reservoirs; training inputs only; complete review inputs <=8192 characters. Not a representative corpus quality estimate."}
