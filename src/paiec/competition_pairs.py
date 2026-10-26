"""Competition-shaped pair matrix over the whole public benchmark pool.

The competition scores one prediction per subject-benchmark pair, so a pair is
the unit of sampling and of scoring. The earlier protocol panel held 170 pairs
in which no system appeared twice, which makes system ability and benchmark
difficulty indistinguishable. This builds the real unit for every public
benchmark at once.

Only ``payload/data`` is read: the public training corpus at the revision its
own manifest records. No sealed role, prepared development state or
calibration/confirmation outcome is opened. Direction is recorded as a column
and never silently flipped; ``lower_is_better`` benchmarks keep their native
labels.
"""
from collections import Counter
import ast
import hashlib
import json
from pathlib import Path
from typing import Literal

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field

from .competition import anonymous_benchmark_id
from .feature_preparation import read_jsonl, stable_hash, write_json, write_jsonl
from .head_training import file_hash
from .schema import SUBJECT_FIELDS, benchmark_dir, clean, subject_record

FORMAT = "competition_pairs_v1"
TEXT_MODALITIES = {"text"}
NON_TEXT_MODALITIES = {"image", "audio", "video"}


class PairMatrixConfig(BaseModel):
    """Declared rules for admitting benchmarks and forming pairs."""

    model_config = ConfigDict(extra="forbid", strict=True)

    version: Literal["competition_pairs_v1"] = "competition_pairs_v1"
    data_root: str = "payload/data"
    dataset_revision: str = "54074453da7b9911be0f48e4c560be547bc84d21"
    min_items: int = Field(default=80, ge=2)
    require_binary: Literal[True] = True
    require_item_granularity: Literal[True] = True
    require_text_modality: Literal[True] = True
    keep_lower_is_better: Literal[True] = True

    def data_directory(self, root):
        path = (Path(root) / self.data_root).resolve()
        if not path.is_dir():
            raise FileNotFoundError("Public corpus directory not found: " + self.data_root)
        if (Path(root) / "runs" / "competition_sources").resolve() in path.parents:
            raise ValueError("The pair matrix never reads a prepared competition source role")
        return path


def parse_direction(response_scale) -> str:
    """Read the recorded direction; anything unrecognized stays ``unordered``."""
    if not isinstance(response_scale, str) or not response_scale.strip():
        return "unordered"
    try:
        parsed = json.loads(response_scale)
    except json.JSONDecodeError:
        return "unordered"
    if not isinstance(parsed, dict):
        return "unordered"
    direction = parsed.get("direction")
    return direction if direction in {"higher_is_better", "lower_is_better"} else "unordered"


def parse_modalities(value) -> list[str]:
    """Modalities arrive as a sequence, a list literal or a plain string."""
    if value is None:
        return []
    if isinstance(value, str):
        if not value.strip():
            return []
        try:
            parsed = ast.literal_eval(value.strip())
        except (ValueError, SyntaxError):
            return [value.strip().lower()]
        value = parsed
    if isinstance(value, (list, tuple, set)):
        items = list(value)
    elif hasattr(value, "tolist"):
        items = list(value.tolist())
    else:
        items = [value]
    return [str(item).strip().lower() for item in items if str(item).strip()]


def identity_tier(subject) -> str:
    """Named identity is a stronger capability key than a display name alone."""
    return "named" if clean(subject.get("normalized_name")) else "display_only"


def one_benchmark_row(path):
    table = pd.read_parquet(path)
    if len(table) != 1:
        raise ValueError("Expected exactly one benchmark row in the metadata table")
    return {key: value for key, value in table.iloc[0].to_dict().items()}


def benchmark_admission(benchmark, meta, cells, config):
    """Declared admission rules; every rejection carries its reason."""
    reasons = []
    if clean(meta.get("response_type")).lower() != "binary":
        reasons.append("response_type_is_not_binary")
    if clean(meta.get("granularity")).lower() != "item":
        reasons.append("granularity_is_not_item")
    modalities = parse_modalities(meta.get("modality"))
    if config.require_text_modality and (not modalities or NON_TEXT_MODALITIES & set(modalities)):
        reasons.append("modality_is_not_text_only")
    if not cells:
        reasons.append("no_subject_reaches_the_item_threshold")
    direction = parse_direction(meta.get("response_scale"))
    return {
        "benchmark": benchmark,
        "benchmark_id": anonymous_benchmark_id(benchmark),
        "admitted": not reasons,
        "reasons": reasons,
        "direction": direction,
        "response_type": clean(meta.get("response_type")),
        "granularity": clean(meta.get("granularity")),
        "modalities": modalities,
        "recorded_subjects": int(meta.get("n_subjects") or 0),
        "recorded_items": int(meta.get("n_items") or 0),
        "recorded_responses": int(meta.get("n_responses") or 0),
        "recorded_coverage": None if meta.get("coverage") is None else float(meta.get("coverage")),
        "max_trial": None if meta.get("max_trial") is None else int(meta.get("max_trial")),
        "cells": len(cells),
    }


def read_cell_sources(directory, min_items):
    """Return per-cell item labels plus the item texts those cells use.

    Responses outside {0, 1} are dropped and counted, never reinterpreted. All
    recorded trials of an item are kept together: the platform counts every
    recorded response and keeps an item's responses in one pool.
    """
    subjects = pd.read_parquet(directory / "subjects.parquet").set_index("subject_id", drop=False)
    items = pd.read_parquet(directory / "items.parquet", columns=["item_id", "content", "item_features"])
    items = items.drop_duplicates("item_id").set_index("item_id", drop=False)
    responses = pd.read_parquet(directory / "response.parquet",
                               columns=["subject_id", "item_id", "response", "trial"])
    dropped = int((~responses["response"].isin([0.0, 1.0])).sum())
    responses = responses[responses["response"].isin([0.0, 1.0])]
    missing_text = int((~responses["item_id"].isin(items.index)).sum())
    responses = responses[responses["item_id"].isin(items.index)]
    if not len(responses):
        return [], items.iloc[0:0], {"dropped_non_binary": dropped, "dropped_missing_text": missing_text}
    grouped = responses.groupby(["subject_id", "item_id"], observed=True)["response"]
    table = grouped.agg(n_responses="size", label_sum="sum").reset_index()
    per_subject = table.groupby("subject_id")["item_id"].nunique()
    retained = per_subject[per_subject >= min_items].index
    table = table[table["subject_id"].isin(retained)]
    trials = responses.groupby("subject_id")["trial"].max()
    cells = []
    for subject_id, group in table.groupby("subject_id", observed=True):
        if subject_id not in subjects.index:
            raise ValueError("Response subject is missing from the subject table")
        recorded = trials.get(subject_id)
        cells.append({
            "subject_id": str(subject_id),
            "subject": subject_record(subjects.loc[subject_id]),
            "items": group.reset_index(drop=True),
            "max_trial": 1 if recorded is None or pd.isna(recorded) else int(recorded),
        })
    return cells, items.loc[sorted(set(table["item_id"]))], {
        "dropped_non_binary": dropped, "dropped_missing_text": missing_text}


def content_report(items, benchmark):
    """Length and format report for one benchmark's retained item texts."""
    lengths = [len(text) for text in items["content"].astype(str)] if len(items) else []
    if not lengths:
        return {"benchmark": benchmark, "items": 0}
    ordered = sorted(lengths)
    return {
        "benchmark": benchmark,
        "items": len(ordered),
        "characters_total": int(sum(ordered)),
        "characters_min": int(ordered[0]),
        "characters_median": int(ordered[len(ordered) // 2]),
        "characters_max": int(ordered[-1]),
    }


def build_pair_matrix(root, config, out, *, log=None):
    """Write the pair matrix, a per-benchmark report and a hashed manifest."""
    root = Path(root).resolve()
    config = PairMatrixConfig.model_validate(config) if not isinstance(config, PairMatrixConfig) else config
    out = Path(out).resolve()
    if out.exists():
        raise ValueError("Use a fresh output directory for the pair matrix")
    if not out.is_relative_to(root / "runs"):
        raise ValueError("Pair matrix output belongs under runs/")
    data = config.data_directory(root)
    manifest_path = data / "corpus_manifest.json"
    corpus = json.loads(manifest_path.read_text())
    if corpus.get("revision") != config.dataset_revision:
        raise ValueError("Corpus revision differs from the declared public training release")
    directories = sorted(corpus["eligible_dirs"])

    out.mkdir(parents=True)
    cells, reports, item_reports, counters = [], [], [], Counter()
    items_path = out / "items.parquet"
    cell_items_path = out / "cell_items.parquet"
    item_writer = pq.ParquetWriter(items_path, pa.schema([
        ("benchmark", pa.string()), ("item_id", pa.string()), ("content", pa.string())]))
    cell_writer = pq.ParquetWriter(cell_items_path, pa.schema([
        ("cell_index", pa.int32()), ("item_id", pa.string()),
        ("n_responses", pa.int32()), ("label_sum", pa.int32())]))
    sources = {}
    try:
        for entry in directories:
            benchmark = entry.split("/")[0]
            directory = benchmark_dir(data, entry)
            meta_path = directory / "benchmarks.parquet"
            meta = one_benchmark_row(meta_path)
            source_hashes = {name: file_hash(directory / name) for name in
                             ["benchmarks.parquet", "subjects.parquet", "items.parquet", "response.parquet"]}
            sources[benchmark] = source_hashes
            raw_cells, used_items, drops = read_cell_sources(directory, config.min_items)
            report = benchmark_admission(benchmark, meta, raw_cells, config)
            report.update(drops)
            record = {"benchmark": benchmark, "benchmark_id": report["benchmark_id"],
                      "source_sha256": stable_hash(source_hashes)}
            reports.append(report)
            if log:
                log(benchmark, len(raw_cells) if report["admitted"] else 0, report["admitted"])
            if not report["admitted"]:
                # A rejected benchmark contributes no pair: its items may be
                # images or audio, which no text-only predictor can see.
                continue
            for cell in raw_cells:
                items = cell["items"]
                n_responses = int(items["n_responses"].sum())
                label_sum = int(round(items["label_sum"].sum()))
                cell_index = len(cells)
                key = stable_hash({"benchmark": benchmark, "subject_id": cell["subject_id"]})
                cells.append({
                    "cell_index": cell_index,
                    "cell_key": key,
                    "benchmark": benchmark,
                    "benchmark_id": report["benchmark_id"],
                    "subject_id": cell["subject_id"],
                    "subject": cell["subject"],
                    "identity_tier": identity_tier(cell["subject"]),
                    "direction": report["direction"],
                    "n_items": int(len(items)),
                    "n_responses": n_responses,
                    "label_sum": label_sum,
                    "rate": label_sum / n_responses,
                    "trials": int(cell["max_trial"]),
                    "coverage": (int(len(items)) / report["recorded_items"]
                                 if report["recorded_items"] else None),
                    "item_sha256": stable_hash(sorted(str(value) for value in items["item_id"])),
                    "source_sha256": record["source_sha256"],
                })
                cell_writer.write_table(pa.table({
                    "cell_index": pa.array([cell_index] * len(items), type=pa.int32()),
                    "item_id": pa.array([str(value) for value in items["item_id"]], type=pa.string()),
                    "n_responses": pa.array([int(v) for v in items["n_responses"]], type=pa.int32()),
                    "label_sum": pa.array([int(round(v)) for v in items["label_sum"]], type=pa.int32()),
                }))
                counters["labelled_responses"] += n_responses
                counters["cells"] += 1
                counters[f"direction:{report['direction']}"] += 1
            if raw_cells:
                item_writer.write_table(pa.table({
                    "benchmark": pa.array([benchmark] * len(used_items), type=pa.string()),
                    "item_id": pa.array([str(value) for value in used_items["item_id"]], type=pa.string()),
                    "content": pa.array([clean(value) for value in used_items["content"]], type=pa.string()),
                }))
                item_reports.append(content_report(used_items, benchmark))
    finally:
        item_writer.close()
        cell_writer.close()

    write_jsonl(out / "cells.jsonl", cells)
    write_jsonl(out / "benchmarks.jsonl", reports)
    write_json(out / "item_lengths.json", {"benchmarks": item_reports})
    write_json(out / "source_contract.json", {
        "version": FORMAT,
        "source_policy": "provided_public_corpus_only",
        "data_root": str(data),
        "dataset_revision": config.dataset_revision,
        "corpus_manifest_sha256": file_hash(manifest_path),
        "corpus_scope": corpus.get("corpus_scope"),
        "eligible_directories": len(directories),
        "config": config.model_dump(),
        "sources": sources,
        "sealed_roles_opened": False,
    })
    summary = summarise(cells, reports, counters, config)
    write_json(out / "summary.json", summary)
    write_json(out / "manifest.json", {
        "version": FORMAT,
        "source_contract_sha256": file_hash(out / "source_contract.json"),
        "artifacts": {str(path.relative_to(out)): file_hash(path)
                      for path in sorted(out.rglob("*")) if path.is_file()},
    })
    return summary


def summarise(cells, reports, counters, config):
    """Totals a later admission decision needs, including model-name overlap."""
    per_name = Counter()
    for cell in cells:
        name = clean(cell["subject"].get("normalized_name")).lower()
        if name:
            per_name[(cell["benchmark"], name)] += 1
    benchmarks_by_name = Counter()
    for (_, name) in per_name:
        benchmarks_by_name[name] += 1
    admitted = [row for row in reports if row["admitted"]]
    excluded = [row for row in reports if not row["admitted"]]
    return {
        "version": FORMAT,
        "source_policy": "provided_public_corpus_only",
        "dataset_revision": config["dataset_revision"] if isinstance(config, dict) else config.dataset_revision,
        "min_items": config["min_items"] if isinstance(config, dict) else config.min_items,
        "benchmarks_read": len(reports),
        "benchmarks_admitted": len(admitted),
        "benchmarks_excluded": len(excluded),
        "exclusion_reasons": dict(Counter(reason for row in excluded for reason in row["reasons"])),
        "cells": len(cells),
        "cells_by_benchmark": dict(Counter(cell["benchmark"] for cell in cells)),
        "labelled_responses": counters["labelled_responses"],
        "directions": {key.split(":", 1)[1]: value for key, value in counters.items()
                       if key.startswith("direction:")},
        "cells_named_identity": sum(cell["identity_tier"] == "named" for cell in cells),
        "distinct_named_models": len(benchmarks_by_name),
        "named_models_in_multiple_benchmarks": sum(value > 1 for value in benchmarks_by_name.values()),
        "named_models_in_at_least_five_benchmarks": sum(value >= 5 for value in benchmarks_by_name.values()),
        "mean_cell_rate": (sum(cell["rate"] for cell in cells) / len(cells)) if cells else None,
        "mean_bernoulli_variance": (sum(cell["rate"] * (1 - cell["rate"]) for cell in cells) / len(cells))
        if cells else None,
        "labels_reinterpreted": False,
        "direction_flipped": False,
        "gpu_jobs": 0,
    }


def refresh_manifest(directory):
    """Rewrite the artifact manifest after a script adds its own records."""
    directory = Path(directory)
    write_json(directory / "manifest.json", {
        "version": FORMAT,
        "source_contract_sha256": file_hash(directory / "source_contract.json")
        if (directory / "source_contract.json").is_file() else None,
        "artifacts": {str(path.relative_to(directory)): file_hash(path)
                      for path in sorted(directory.rglob("*")) if path.is_file()
                      and path.name != "manifest.json"},
    })


def load_pair_matrix(directory):
    """Load cells and their item labels; the item texts load lazily."""
    directory = Path(directory)
    cells = list(read_jsonl(directory / "cells.jsonl"))
    if not cells:
        raise ValueError("Pair matrix has no cells")
    table = pq.read_table(directory / "cell_items.parquet").to_pandas()
    labels = {}
    for cell_index, group in table.groupby("cell_index", sort=True):
        labels[int(cell_index)] = group[["item_id", "n_responses", "label_sum"]].reset_index(drop=True)
    missing = [cell["cell_index"] for cell in cells if cell["cell_index"] not in labels]
    if missing:
        raise ValueError("Pair matrix cell has no item labels")
    return {"directory": str(directory), "cells": cells, "labels": labels,
            "manifest_sha256": file_hash(directory / "manifest.json")}


def load_item_text(directory):
    """Item texts keyed by (benchmark, item_id) for task-structure features."""
    directory = Path(directory)
    table = pq.read_table(directory / "items.parquet", columns=["benchmark", "item_id", "content"]).to_pandas()
    return {(row.benchmark, row.item_id): row.content for row in table.itertuples(index=False)}


__all__ = ["FORMAT", "PairMatrixConfig", "benchmark_admission", "build_pair_matrix", "content_report",
           "identity_tier", "load_item_text", "load_pair_matrix", "parse_direction", "parse_modalities",
           "refresh_manifest", "summarise"]
