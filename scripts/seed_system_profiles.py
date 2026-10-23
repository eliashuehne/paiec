#!/usr/bin/env python3
"""Seed a research-only catalog from one actual recorded subject; no web facts."""

import argparse
import hashlib
import json
from datetime import date
from pathlib import Path
import pandas as pd
from paiec.schema import subject_record
from paiec.system_profiles import (
    Evidence,
    Fact,
    Identity,
    Profile,
    Source,
    ModelSpecifications,
    HarnessSpecifications,
    RunSpecifications,
)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    path = root / "payload/data/adaptivestep/formatted_tables/subjects.parquet"
    revision = json.loads((root / "payload/data/corpus_manifest.json").read_text())[
        "revision"
    ]
    row = pd.read_parquet(path).set_index("subject_id").loc["50e726a69a130d06"]
    source = Source(
        source_id="measurement-db.adaptivestep.subjects",
        source_type="benchmark_table",
        location="https://huggingface.co/datasets/aims-foundations/measurement-db/blob/"
        + revision
        + "/adaptivestep/formatted_tables/subjects.parquet",
        revision=revision,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        retrieved_on=date.today().isoformat(),
        eligibility="inventory_source",
        eligibility_note="Existing pinned public measurement-db release; no external facts added.",
    )

    def fact(value, field):
        return Fact[str](
            value=value,
            status="recorded",
            evidence=[
                Evidence(
                    source_id=source.source_id,
                    locator="subject_id=50e726a69a130d06; column=" + field,
                    excerpt=str(row[field]),
                )
            ],
            note="Reported identifier only; exact architecture and checkpoint have not been resolved.",
        )

    model = Profile(
        profile_id="model.metamath-mistral.reported",
        entity_type="model",
        identity=Identity(
            name=fact("MetaMath-Mistral", "subject_features_extra"),
            resolution="reported_identifier_only",
        ),
        specifications=ModelSpecifications(),
    )
    harness = Profile(
        profile_id="harness.asprm.reported",
        entity_type="harness",
        identity=Identity(
            name=fact(str(row.harness), "harness"),
            resolution="reported_identifier_only",
        ),
        specifications=HarnessSpecifications(),
    )
    run = Profile(
        profile_id="run.adaptivestep.50e726a69a130d06",
        entity_type="recorded_run",
        specifications=RunSpecifications(
            model_profile_id=model.profile_id,
            harness_profile_id=harness.profile_id,
            raw_supplied_subject=subject_record(row),
        ),
    )
    output = Path(args.out)
    output.mkdir(parents=True, exist_ok=False)
    (output / "sources.jsonl").write_text(source.model_dump_json() + "\n")
    (output / "profiles.jsonl").write_text(
        "".join(profile.model_dump_json() + "\n" for profile in [model, harness, run])
    )
