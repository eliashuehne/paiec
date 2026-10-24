"""Outcome-blind expansion inside frozen training families; no native frameworks."""
from collections import defaultdict, deque
import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, model_validator

from .head_data import digest, task_key
from .head_training import file_hash
from .split import stable_seed


class ExpandedDataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)
    split: str = "configs/head_finetuning/frozen_split_v1.json"
    policy: str = "configs/head_finetuning/compact_profiles_v3.json"
    prepared: str = "runs/head_finetuning/enriched_30k_v1"
    samples: int = Field(default=30000, ge=1, le=50000)
    monitor_samples: int = Field(default=1000, ge=1, le=2000)
    seed: int = 20261023
    max_len: Literal[2048] = 2048
    head_max_len: Literal[192] = 192
    input_pipeline: Literal["compact_v3"] = "compact_v3"

    @model_validator(mode="after")
    def paths(self):
        path = Path(self.prepared)
        if path.is_absolute() or ".." in path.parts or path.parts[:1] != ("runs",):
            raise ValueError("Expanded preparation must be inside runs/")
        return self


def balanced_unique_sample(frame, count, seed):
    """Family-balanced, then pair-balanced unique items; labels never drive order.

    One source trial is chosen uniformly per system/item before this function.
    Exhausted families/pairs redistribute their share without oversampling.
    """
    keys = ["benchmark", "subject_id", "item_id"]
    if frame.duplicated(keys).any():
        raise ValueError("Sampling requires one trial per system/item")
    if len(frame) < count:
        raise ValueError(f"Need {count} unique records; only {len(frame)} available")
    frame = frame.sort_values(keys).reset_index(drop=True)
    rng = np.random.default_rng(seed)
    families = defaultdict(dict)
    for (family, benchmark, subject), group in frame.groupby(["family", "benchmark", "subject_id"], sort=True):
        indices = group.index.to_numpy().copy()
        rng.shuffle(indices)
        families[family][(benchmark, subject)] = deque(indices.tolist())
    chosen = []
    pair_order = {family: deque(rng.permutation(len(pairs)).tolist()) for family, pairs in families.items()}
    pair_keys = {family: list(pairs) for family, pairs in families.items()}
    while len(chosen) < count:
        active = [family for family, order in pair_order.items() if order]
        rng.shuffle(active)
        for family in active:
            position = pair_order[family].popleft()
            group = families[family][pair_keys[family][position]]
            chosen.append(group.popleft())
            if group:
                pair_order[family].append(position)
            if len(chosen) == count:
                break
    selected = frame.iloc[chosen].copy().reset_index(drop=True)
    family_pairs = selected[["family", "benchmark", "subject_id"]].drop_duplicates().groupby("family").size()
    pair_n = selected.groupby(["benchmark", "subject_id"]).item_id.transform("size")
    family_count = selected.family.nunique()
    selected["sample_weight"] = [count / (family_count * family_pairs[f] * n)
                                 for f, n in zip(selected.family, pair_n)]
    return selected


def monitor_key(task_sha256, seed):
    """All systems and trials sharing a normalized task inherit the same fold."""
    return stable_seed(seed, "training-only-monitor", task_sha256) % 20 == 0


def verify_expanded(root, prepared):
    root, prepared = Path(root), Path(prepared)
    manifest = json.loads((prepared / "manifest.json").read_text())
    split = root / manifest["data_config"]["split"]
    if file_hash(split) != manifest["split_sha256"]:
        raise ValueError("Frozen split changed")
    for name, expected in manifest["artifacts"].items():
        path = (prepared / name).resolve()
        if not path.is_relative_to(prepared.resolve()) or file_hash(path) != expected:
            raise ValueError("Expanded artifact changed: " + name)
    for name, expected in manifest["renderer_sources"].items():
        if file_hash(root / name) != expected:
            raise ValueError("Expanded renderer changed: " + name)
    from .laya_predictor import load_questions
    if manifest["question"] != load_questions():
        raise ValueError("Active question changed")
    return manifest


def load_candidates(root, config, out):
    """Load only training outcomes. Source hashes link the earlier full-item review."""
    from .schema import benchmark_dir, clean
    root, out = Path(root), Path(out)
    split_path = root / config.split
    split = json.loads(split_path.read_text())
    corpus = json.loads((root / "payload/data/corpus_manifest.json").read_text())
    if corpus["revision"] != split["dataset_revision"]:
        raise ValueError("Expanded dataset revision changed")
    source_config = root / split["source_config"]
    if file_hash(source_config) != split["source_config_sha256"]:
        raise ValueError("Frozen admission decisions changed")
    original = root / split["artifact_directory"]
    for name in ["provenance.parquet", "source_file_hashes.json", "development.jsonl"]:
        if file_hash(original / name) != split["artifact_fingerprints"][name]:
            raise ValueError("Frozen input/review changed: " + name)
    original_hashes = json.loads((original / "source_file_hashes.json").read_text())
    reviews = json.loads(source_config.read_text())["reviews"]
    reserved = pd.read_parquet(original / "provenance.parquet")
    reserved_keys = set(reserved.loc[~reserved.role.eq("training"), "task_sha256"])
    pools, tables, admission, source_hashes = [], {}, [], {}
    for benchmark in split["roles"]["training"]:
        review = reviews[benchmark]
        if review["status"] != "eligible" or split["family_roles"][review["family"]] != "training":
            raise ValueError("Attempt to expand outside frozen training families")
        folder = benchmark_dir(root / "payload/data", benchmark)
        for name in ["items.parquet", "subjects.parquet", "response.parquet", "benchmarks.parquet"]:
            path = folder / name
            key = str(path.relative_to(root / "payload/data"))
            actual = file_hash(path)
            if actual != original_hashes[key]:
                raise ValueError("Previously reviewed source changed: " + key)
            source_hashes[str(path.relative_to(root))] = actual
        items = pd.read_parquet(folder / "items.parquet", columns=["item_id", "content", "item_features"])
        items = items[items.content.map(clean).ne("")].drop_duplicates("item_id").copy()
        required = review.get("required_feature")
        if required:
            items = items[items.item_features.map(clean).str.contains(required, regex=False)]
        items["task_sha256"] = items.content.map(lambda value: digest(task_key(clean(value))))
        if items.task_sha256.isin(reserved_keys).any():
            raise ValueError("Expanded training item overlaps a reserved role")
        subjects = pd.read_parquet(folder / "subjects.parquet").set_index("subject_id", drop=False)
        responses = pd.read_parquet(folder / "response.parquet", columns=[
            "subject_id", "item_id", "response_id", "trial", "interactors", "response"])
        responses = responses[responses.response.isin([0, 1]) & responses.item_id.isin(items.item_id)
                              & responses.subject_id.isin(subjects.index)].copy()
        if responses.response_id.duplicated().any():
            raise ValueError("Duplicate source response IDs: " + benchmark)
        responses["interactors"] = responses.interactors.map(clean)
        varying = responses.groupby(["subject_id", "item_id"]).interactors.nunique().gt(1)
        bad_subjects = set(varying[varying].index.get_level_values(0))
        sizes = responses.groupby("subject_id").item_id.nunique()
        admitted = sorted(sizes[sizes.ge(80)].index.difference(bad_subjects))
        for subject in sizes.index:
            admission.append(dict(benchmark=benchmark, subject_id=subject, distinct_items=int(sizes[subject]),
                admitted=subject in admitted, reason="reviewed binary/text inputs; stable supplied per-item tools; >=80 items"
                if subject in admitted else "insufficient items or varying per-trial input tools",
                identity_required=False, missing_identity_not_filled=True))
        responses = responses[responses.subject_id.isin(admitted)].sort_values("response_id")
        # Uniform trial selection, outcome-blind. Preserve its raw binary outcome.
        responses = responses.sample(frac=1, random_state=stable_seed(config.seed, benchmark) % (2**32))
        responses = responses.drop_duplicates(["subject_id", "item_id"]).copy()
        responses["benchmark"], responses["family"] = benchmark, review["family"]
        responses = responses.merge(items[["item_id", "task_sha256"]], on="item_id", validate="many_to_one")
        pools.append(responses)
        tables[benchmark] = (subjects, items.set_index("item_id", drop=False))
        print(f"Admitted {benchmark}: {len(admitted)} pairs, {len(responses)} unique system/items", flush=True)
    pd.DataFrame(admission).to_csv(out / "subject_admission.csv", index=False)
    frame = pd.concat(pools, ignore_index=True)
    return frame, tables, split, source_hashes
