#!/usr/bin/env python
"""Audit outcome meanings without changing data or rerunning any judge."""

import json
from pathlib import Path

import pandas as pd
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "payload/data"
OUT = ROOT / "runs/outcome_audit"


def native_bool(value):
    if value is True or value == "True":
        return 1
    if value is False or value == "False":
        return 0
    raise ValueError(f"Unexpected native boolean: {value!r}")


def joined_traces(benchmark):
    directory = DATA / benchmark / "formatted_tables"
    responses = pq.read_table(directory / "response.parquet", columns=["response_id", "response"]).to_pandas()
    traces = pq.read_table(directory / "traces.parquet", columns=["response_id", "trace"]).to_pandas()
    return responses.merge(traces, on="response_id", validate="one_to_one")


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    manifest = json.loads((DATA / "corpus_manifest.json").read_text())
    rows = []
    for entry in manifest["eligible_dirs"]:
        directory = DATA / entry
        meta = pq.read_table(directory / "benchmarks.parquet").to_pylist()[0]
        scale = json.loads(meta["response_scale"])
        rows.append({
            "benchmark": entry.split("/")[0], "direction": scale.get("direction"),
            "meaning_0": scale.get("meanings", {}).get("0"),
            "meaning_1": scale.get("meanings", {}).get("1"),
            "response_type": meta["response_type"], "description": meta["description"],
            "nonstandard_direction": scale.get("direction") in {"lower_is_better", "unordered"},
        })
    catalog = pd.DataFrame(rows)
    catalog.to_csv(OUT / "outcome_meanings.csv", index=False)
    # These are archived native fields, not a new safety/correctness judgment.
    adv = joined_traces("advprompter")
    native_score = adv["trace"].map(lambda text: json.loads(text)["final_score"])
    brave = joined_traces("braveguard")
    records = brave["trace"].map(lambda text: json.loads(text)["record"])
    harmful = records.map(lambda record: native_bool(record["harmful"]))
    correct = records.map(lambda record: native_bool(record["correct"]))
    checks = {
        "dataset_revision": manifest["revision"],
        "direction_counts": catalog["direction"].value_counts().to_dict(),
        "flagged_by_direction": catalog.loc[catalog.nonstandard_direction, "benchmark"].tolist(),
        "advprompter": {
            "joined_rows": len(adv), "native_score_matches_response": int(adv.response.eq(native_score).sum()),
            "positive_outcomes": int(adv.response.sum()),
        },
        "braveguard": {
            "joined_rows": len(brave), "native_harmful_matches_response": int(brave.response.eq(harmful).sum()),
            "native_correct_differs_from_response": int(brave.response.ne(correct).sum()),
            "inverted_correct_differs_from_response": int(brave.response.ne(1 - correct).sum()),
        },
        "limits": "Direction flags identify clear counterexamples, not an exhaustive correctness classification. Native trace reconciliation is not an independent rejudging of outputs. Hidden evaluator label transformations remain unverified.",
    }
    assert checks["advprompter"]["native_score_matches_response"] == len(adv)
    assert checks["braveguard"]["native_harmful_matches_response"] == len(brave)
    pd.crosstab(brave.response, correct, rownames=["stored_response"], colnames=["native_correct"]).to_csv(OUT / "braveguard_cross_tab.csv")
    (OUT / "checks.json").write_text(json.dumps(checks, indent=2))
    print(json.dumps(checks, indent=2))


if __name__ == "__main__":
    main()
