"""Selective original-development outcomes for saved-head evidence comparisons."""
import json

import pandas as pd

from .pairs import Pair
from .schema import benchmark_dir, subject_record
from .split import make_split, MAX_BUDGET
from .head_training import file_hash


def original_development_pairs(root, manifest, development, seed):
    """Keep the full eligible item universe, read labels only for scored/revealed items.

    Full item IDs are needed to reproduce the frozen 50/50 split. Reading that
    universe does not require returning outcomes for other items or subjects.
    Parquet predicates exclude supplemental systems before label materialization.
    """
    split = json.loads((root / manifest["data_config"]["split"]).read_text())
    if not set(development.benchmark).issubset(split["roles"]["development"]):
        raise ValueError("Evidence comparison may use only original development benchmarks")
    source_manifest = root / split["artifact_directory"] / "source_file_hashes.json"
    if file_hash(source_manifest) != split["artifact_fingerprints"]["source_file_hashes.json"]:
        raise ValueError("Frozen development source manifest changed")
    source_hashes = json.loads(source_manifest.read_text())
    result = []
    for benchmark in development.benchmark.unique():
        expected = development[development.benchmark.eq(benchmark)]
        selected = expected.subject_id.unique().tolist()
        directory = benchmark_dir(root / "payload/data", benchmark)
        for name in ["subjects.parquet", "items.parquet", "response.parquet"]:
            path = directory / name
            if file_hash(path) != source_hashes[str(path.relative_to(root / "payload/data"))]:
                raise ValueError("Frozen development source changed: " + name)
        subjects = pd.read_parquet(directory / "subjects.parquet",
                                   filters=[("subject_id", "in", selected)]).set_index("subject_id", drop=False)
        items = pd.read_parquet(directory / "items.parquet", columns=["item_id", "content", "item_features"])
        items = items.drop_duplicates("item_id").set_index("item_id", drop=False)
        # Outcome values are a predicate, not a returned field; preserve the
        # same binary-response eligibility and original table order as the scorer.
        metadata = pd.read_parquet(directory / "response.parquet",
            columns=["subject_id", "item_id", "interactors"],
            filters=[("subject_id", "in", selected), ("response", "in", [0., 1.])])
        metadata = metadata[metadata.item_id.isin(items.index)]
        for subject, group in metadata.groupby("subject_id", sort=True):
            ids = tuple(group.item_id.drop_duplicates().astype(str))
            local = make_split(ids, benchmark=benchmark, subject_id=str(subject), seed=seed,
                               acquisition_items=40, evaluation_items=120)
            saved = expected[expected.subject_id.eq(subject)]
            if set(local.evaluation) != set(saved.item_id):
                raise ValueError("Original development evaluation IDs changed")
            selected_items = list(local.evaluation) + list(local.acquisition[:MAX_BUDGET])
            responses = pd.read_parquet(directory / "response.parquet",
                columns=["item_id", "response"], filters=[("subject_id", "==", subject),
                    ("item_id", "in", selected_items), ("response", "in", [0., 1.])])
            responses = responses.rename(columns={"response": "label"})
            responses["label"] = responses.label.astype(int)
            paired = items.reindex(ids)[["content", "item_features"]].copy()
            paired["interactors"] = group.drop_duplicates("item_id").set_index("item_id").interactors
            result.append(Pair(benchmark, str(subject), subject_record(subjects.loc[subject]),
                               paired, responses, ids))
    if len(result) != len(development[["benchmark", "subject_id"]].drop_duplicates()):
        raise ValueError("Original development subject coverage changed")
    return result
