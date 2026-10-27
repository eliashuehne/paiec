"""Small real Parquet inputs; outcomes exist only to catch accidental reads."""
import hashlib
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value), encoding="utf-8")


def corpus(root):
    root = Path(root)
    directories = []
    hashes = {}
    for name in ("alpha", "beta", "sealed"):
        directory = root / "payload/data" / name / "formatted_tables"
        directory.mkdir(parents=True)
        directories.append(f"{name}/formatted_tables")
        tables = {
            "benchmarks": [{"benchmark_id": name, "modality": ["text"] if name != "beta" else ["text", "image"], "response_scale": '{"direction":"higher_is_better"}'}],
            "subjects": [{"subject_id": "s1", "display_name": "PRIVATE NAME", "normalized_name": "Example 7B", "provider": "", "release_date": "", "access_date": "", "harness": "Example", "harness_version": "", "reasoning_effort": "", "subject_features_extra": "n_shots=5;temperature=0"}],
            "items": [{"item_id": "i1", "content": "What is 2 + 2?", "item_features": "format=mcq", "grading_criterion": "SECRET ANSWER", "verifier": "SECRET VERIFIER"}, {"item_id": "i2", "content": "def add(a, b):\n    return a + b", "item_features": "", "grading_criterion": "SECRET", "verifier": "SECRET"}],
            "response": [{"response_id": f"{name}-r1", "subject_id": "s1", "item_id": "i1", "trial": 0, "interactors": "[]", "response": 1}, {"response_id": f"{name}-r2", "subject_id": "s1", "item_id": "i1", "trial": 1, "interactors": "[]", "response": 0}, {"response_id": f"{name}-r3", "subject_id": "s1", "item_id": "i2", "trial": 0, "interactors": "python", "response": 1}],
        }
        for table, rows in tables.items():
            path = directory / (table + ".parquet")
            pq.write_table(pa.Table.from_pylist(rows), path)
            hashes[str(path.relative_to(root))] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest = {"revision": "fixture-revision", "eligible_dirs": directories, "files": list(hashes)}
    write_json(root / "payload/data/corpus_manifest.json", manifest)
    return {
        "version": "feature_foundation_v1", "dataset_revision": "fixture-revision",
        "corpus_manifest_sha256": hashlib.sha256((root / "payload/data/corpus_manifest.json").read_bytes()).hexdigest(),
        "input_hashes": hashes,
        "expected_counts": {"benchmarks": 3, "subjects": 3, "items": 6, "responses": 9},
        "frozen_roles": {"training": ["alpha"], "confirmation": ["sealed"]},
        "family_groups": [["alpha", "beta"]], "seed": 20261025,
    }
