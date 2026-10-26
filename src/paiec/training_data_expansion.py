"""Does admitting unused public benchmarks improve cross-benchmark transfer?

This is a bounded, CPU-only sensitivity study for a data-admission decision. It
fits no candidate predictor and promotes nothing.

Instrument: the repository's existing portable lexical/system ridge control
(``conditional_features`` plus ``fit_conditional_ridge``) with fixed L2 and
interactions, reused unchanged as a *relative* instrument. Every other choice is
held fixed so two fitting sets can be compared; its absolute error is not a
quality claim.

Part A measures cold-start Brier against the number of training families over
every subset of the non-held families. Part B adds rows drawn outcome-blind from
unused public benchmarks and compares them with the unmodified panel fitting
side and with an equal-row control sampled from the panel's own families, which
separates row count from benchmark diversity.

Evaluation is leave-one-training-family-out on the prepared protocol panel.
Repeated query task groups are removed from the fitting side. Development,
calibration and confirmation outcomes are never opened, and benchmarks assigned
to a non-training role in the frozen split are refused.
"""
from __future__ import annotations

from collections import defaultdict
import itertools
import json
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, field_validator, model_validator

from .competition import anonymous_benchmark_id
from .conditional_analysis import training_panel, prepare_feature_matrix
from .conditional_features import DIMS, VERSION, feature_vector
from .conditional_training import fit_conditional_ridge, predict_matrix
from .feature_preparation import local_path, stable_hash, write_json, write_jsonl
from .head_data import task_key
from .head_training import file_hash
from .schema import SUBJECT_FIELDS, clean


class AddedBenchmark(BaseModel):
    """One unused public benchmark admitted to this diagnostic."""

    model_config = ConfigDict(extra="forbid", strict=True)
    name: str
    cap: int
    domain: str

    @field_validator("name")
    @classmethod
    def plain_directory(cls, value: str) -> str:
        if not value or "/" in value or "\\" in value or value in {".", ".."}:
            raise ValueError("Use a plain benchmark directory name")
        return value

    @field_validator("cap")
    @classmethod
    def positive_cap(cls, value: int) -> int:
        if value <= 0:
            raise ValueError("Row cap must be positive")
        return value


class TrainingExpansionConfig(BaseModel):
    """Predeclared recipe for the expansion sensitivity study."""

    model_config = ConfigDict(extra="forbid", strict=True)
    version: Literal["training_data_expansion_v1"] = "training_data_expansion_v1"
    prepared: str
    data_root: str
    l2: float = 100.0
    interactions: bool = True
    rows_per_system: int = 100
    min_pair_items: int = 80
    repeats: int = 3
    seed: int = 20261024
    added_benchmarks: list[AddedBenchmark]

    @model_validator(mode="after")
    def coherent(self):
        if not self.added_benchmarks:
            raise ValueError("Select at least one unused benchmark")
        names = [b.name for b in self.added_benchmarks]
        if len(set(names)) != len(names):
            raise ValueError("Add each benchmark once")
        if not np.isfinite(self.l2) or self.l2 <= 0:
            raise ValueError("L2 must be positive and finite")
        if self.rows_per_system <= 0 or self.min_pair_items < 2:
            raise ValueError("Use positive row and pair limits")
        if self.repeats <= 0:
            raise ValueError("Run at least one repeat")
        if not isinstance(self.seed, int):
            raise ValueError("Seed must be an integer")
        return self


def resolve_data_root(root, value):
    """Resolve the public data root, which is not a run-artifact path."""
    root = Path(root).resolve()
    path = Path(value)
    path = (path if path.is_absolute() else root / path).resolve()
    if not path.is_relative_to(root):
        raise ValueError("Data root must stay inside the repository")
    return path


def frozen_roles(root) -> dict[str, str]:
    """Benchmark-to-role assignments from the frozen split."""
    path = Path(root) / "configs/head_finetuning/frozen_split_v1.json"
    split = json.loads(path.read_text())
    roles = split.get("roles")
    if not isinstance(roles, dict) or "training" not in roles:
        raise ValueError("Frozen split does not declare benchmark roles")
    return {str(name): str(role) for role, names in roles.items() for name in names}


def verify_expansion_inputs(root, config: TrainingExpansionConfig):
    """Resolve paths and refuse sealed, non-text or wrong-direction benchmarks."""
    root = Path(root).resolve()
    config = TrainingExpansionConfig.model_validate(config.model_dump())
    prepared = local_path(root, config.prepared)
    data_root = resolve_data_root(root, config.data_root)
    if not (prepared / "manifest.json").is_file():
        raise ValueError("Prepared panel bundle is missing")
    if not (data_root / "corpus_manifest.json").is_file():
        raise ValueError("Public data root is missing its corpus manifest")
    roles = frozen_roles(root)
    selected = {}
    for benchmark in config.added_benchmarks:
        role = roles.get(benchmark.name)
        if role is not None and role != "training":
            raise ValueError(f"Benchmark {benchmark.name!r} belongs to sealed role {role!r}")
        directory = data_root / benchmark.name / "formatted_tables"
        if not (directory / "benchmarks.parquet").is_file():
            raise ValueError(f"Missing benchmark tables for {benchmark.name!r}")
        metadata = pd.read_parquet(directory / "benchmarks.parquet",
                                   columns=["modality", "response_scale", "response_type", "granularity"])
        if len(metadata) != 1:
            raise ValueError(f"Expected one benchmark record for {benchmark.name!r}")
        row = metadata.iloc[0]
        scale = json.loads(row["response_scale"])
        if list(row["modality"]) != ["text"]:
            raise ValueError(f"{benchmark.name!r} is not a text-only benchmark")
        if scale.get("direction") != "higher_is_better":
            raise ValueError(f"{benchmark.name!r} does not record success as higher_is_better")
        if row["response_type"] != "binary" or row["granularity"] != "item":
            raise ValueError(f"{benchmark.name!r} is not a binary item-level benchmark")
        selected[benchmark.name] = {
            "domain": benchmark.domain, "cap": benchmark.cap,
            "benchmarks_sha256": file_hash(directory / "benchmarks.parquet"),
            "subjects_sha256": file_hash(directory / "subjects.parquet"),
            "items_sha256": file_hash(directory / "items.parquet"),
            "responses_sha256": file_hash(directory / "response.parquet"),
        }
    return {"prepared": prepared, "data_root": data_root, "roles": roles, "added": selected,
            "corpus_manifest_sha256": file_hash(data_root / "corpus_manifest.json")}


def held_split(family, groups, held_name):
    """Training mask that keeps a whole family and any repeated task out."""
    family, groups = np.asarray(family), np.asarray(groups)
    if family.shape != groups.shape or family.ndim != 1 or held_name not in family:
        raise ValueError("Invalid held family")
    held = family == held_name
    held_groups = set(groups[held])
    fit = (~held) & np.array([group not in held_groups for group in groups])
    if not fit.any():
        raise ValueError("No independent training rows remain")
    return held, fit


def _fit_predict(x_fit, y_fit, scopes_fit, x_held, *, l2, interactions):
    if len(x_fit) < 2:
        raise ValueError("Fitting side is too small")
    counts = pd.Series(scopes_fit).value_counts()
    weights = np.array([1.0 / counts[scope] for scope in scopes_fit])
    state = fit_conditional_ridge(np.asarray(x_fit), np.asarray(y_fit), weights,
                                  interactions=interactions, l2=l2)
    return predict_matrix(state, np.asarray(x_held))


def _scope_errors(prediction, labels, scopes):
    return {str(scope): float(np.mean((prediction[scopes == scope] - labels[scopes == scope]) ** 2))
            for scope in sorted(set(scopes))}


def diversity_curve(labels, family, scopes, groups, matrix, *, l2, interactions):
    """Equal-family and equal-scope cold-start curves over every family subset."""
    labels, family, scopes = np.asarray(labels), np.asarray(family), np.asarray(scopes)
    families = sorted(set(family))
    if len(families) < 2:
        raise ValueError("A diversity curve needs at least two families")
    records = []
    for held_name in families:
        held, base_fit = held_split(family, groups, held_name)
        others = [name for name in families if name != held_name]
        for size in range(1, len(others) + 1):
            for combo in itertools.combinations(others, size):
                mask = base_fit & np.isin(family, combo)
                prediction = _fit_predict(matrix[mask], labels[mask], scopes[mask], matrix[held],
                                          l2=l2, interactions=interactions)
                records.append({"held_family": held_name, "families_used": size,
                                "subset": "+".join(combo),
                                "errors": _scope_errors(prediction, labels[held], scopes[held])})
    summary = []
    for size in sorted({record["families_used"] for record in records}):
        chosen = [record for record in records if record["families_used"] == size]
        per_family = {}
        for name in families:
            values = [float(np.mean(list(record["errors"].values())))
                      for record in chosen if record["held_family"] == name]
            per_family[name] = float(np.mean(values))
        averaged = defaultdict(lambda: defaultdict(list))
        for record in chosen:
            for scope, value in record["errors"].items():
                averaged[record["held_family"]][scope].append(value)
        equal_scope = float(np.mean([np.mean(values) for family_values in averaged.values()
                                     for values in family_values.values()]))
        summary.append({"families_used": size, "subsets": len(chosen),
                        "equal_family_brier": float(np.mean(list(per_family.values()))),
                        "equal_scope_brier": equal_scope, **per_family})
    return summary, records


def sample_added_rows(data_root, benchmarks, panel_groups, *, rows_per_system, min_pair_items, seed,
                      pool_multiplier=1):
    """One outcome-blind response per subject-item from each unused benchmark.

    Selection uses hashed identifiers only: no label, order or score influences
    which rows enter. Repeated content already present in the panel is dropped,
    and content may belong to only one added benchmark. ``pool_multiplier``
    samples a larger pool so repeats can draw independent equal-size subsets.
    """
    if type(pool_multiplier) is not int or pool_multiplier <= 0:
        raise ValueError("Pool multiplier must be a positive integer")
    panel_groups = set(panel_groups)
    used_by_benchmark = {}
    sampled = []
    for benchmark in benchmarks:
        directory = Path(data_root) / benchmark.name / "formatted_tables"
        subjects = pd.read_parquet(directory / "subjects.parquet").set_index("subject_id")
        items = pd.read_parquet(directory / "items.parquet").set_index("item_id")
        responses = pd.read_parquet(
            directory / "response.parquet",
            columns=["subject_id", "item_id", "response_id", "response", "interactors"])
        responses = responses[responses.response.isin([0, 1])]
        eligible = responses.groupby("subject_id").item_id.nunique()
        eligible = eligible[eligible >= min_pair_items].index
        order = sorted(eligible, key=lambda subject: stable_hash(
            {"seed": seed, "benchmark": benchmark.name, "subject": subject}))
        rows = []
        for subject_id in order:
            group = responses[responses.subject_id == subject_id].copy()
            group["_response"] = [stable_hash({"seed": seed, "benchmark": benchmark.name,
                                              "response": response})
                                  for response in group.response_id]
            group = group.sort_values("_response").drop_duplicates("item_id")
            group["_item"] = [stable_hash({"seed": seed, "benchmark": benchmark.name,
                                          "subject": str(subject_id), "item": item})
                              for item in group.item_id]
            group = group.sort_values("_item").head(rows_per_system)
            subject = {field: clean(subjects.loc[subject_id].get(field)) for field in SUBJECT_FIELDS}
            for _, response in group.iterrows():
                item_row = items.loc[response.item_id]
                content = clean(item_row.get("content"))
                group_key = stable_hash(task_key(content))
                if group_key in panel_groups:
                    continue
                if used_by_benchmark.setdefault(group_key, benchmark.name) != benchmark.name:
                    continue
                item = {"item_content": content,
                        "item_features": clean(item_row.get("item_features")),
                        "interactors": clean(response.get("interactors")),
                        "benchmark_id": anonymous_benchmark_id(benchmark.name)}
                rows.append({"benchmark": benchmark.name, "subject_id": str(subject_id),
                             "item_id": str(response.item_id), "label": int(response.response),
                             "subject": subject, "item": item,
                             "scope_key": stable_hash({"benchmark": benchmark.name,
                                                       "subject": str(subject_id)}),
                             "task_group": group_key})
                if len(rows) >= benchmark.cap * pool_multiplier:
                    break
            if len(rows) >= benchmark.cap * pool_multiplier:
                break
        sampled.extend(rows)
    return sampled


def repeat_subset(added, benchmarks, repeat, seed):
    """Draw one equal-size-per-benchmark subset from the sampled pool."""
    pools = defaultdict(list)
    for row in added:
        pools[row["benchmark"]].append(row)
    subset = []
    for benchmark in benchmarks:
        pool = sorted(pools[benchmark.name], key=lambda row: stable_hash({
            "seed": seed + repeat, "benchmark": benchmark.name,
            "subject": row["subject_id"], "item": row["item_id"]}))
        subset.extend(pool[:benchmark.cap])
    return subset


def expansion_arms(labels, family, scopes, groups, matrix, added, benchmarks, *, l2, interactions, repeats, seed):
    """Compare the panel fitting side with equal rows and with new benchmarks."""
    labels, family, scopes = np.asarray(labels), np.asarray(family), np.asarray(scopes)
    x_add = np.empty((len(added), DIMS), dtype=np.float64)
    y_add = np.array([row["label"] for row in added], dtype=float)
    s_add = np.array([row["scope_key"] for row in added])
    for index, row in enumerate(added):
        x_add[index] = feature_vector(row["subject"], row["item"])
    position = {(row["benchmark"], row["subject_id"], row["item_id"]): index
                for index, row in enumerate(added)}
    families = sorted(set(family))
    records = []
    for repeat in range(repeats):
        subset = repeat_subset(added, benchmarks, repeat, seed)
        if not subset:
            raise ValueError("A repeat drew no added rows")
        added_index = np.array([position[(row["benchmark"], row["subject_id"], row["item_id"])]
                                for row in subset])
        rng = np.random.default_rng(seed + repeat)
        for held_name in families:
            held, base_fit = held_split(family, groups, held_name)
            base_index = np.flatnonzero(base_fit)
            if len(added_index) > len(base_index):
                raise ValueError("Equal-row control needs more panel fitting rows than added rows")
            control_index = rng.choice(base_index, size=len(added_index), replace=False)
            arms = {
                "base": (base_index, None, None),
                "base+panel_rows": (base_index, control_index, None),
                "base+unused_benchmarks": (base_index, None, added_index),
                "unused_benchmarks_only": (np.array([], dtype=int), None, added_index),
            }
            for arm, (panel_index, panel_extra, added_extra) in arms.items():
                x_fit, y_fit, s_fit = matrix[panel_index], labels[panel_index], scopes[panel_index]
                if panel_extra is not None:
                    x_fit = np.vstack([x_fit, matrix[panel_extra]])
                    y_fit = np.concatenate([y_fit, labels[panel_extra]])
                    s_fit = np.concatenate([s_fit, scopes[panel_extra]])
                if added_extra is not None:
                    x_fit = np.vstack([x_fit, x_add[added_extra]])
                    y_fit = np.concatenate([y_fit, y_add[added_extra]])
                    s_fit = np.concatenate([s_fit, s_add[added_extra]])
                prediction = _fit_predict(x_fit, y_fit, s_fit, matrix[held],
                                          l2=l2, interactions=interactions)
                records.append({"arm": arm, "repeat": repeat, "held_family": held_name,
                                "rows_fitted": int(len(x_fit)), "scopes_fitted": int(len(set(s_fit))),
                                "errors": _scope_errors(prediction, labels[held], scopes[held])})
    measured = []
    for (arm, repeat), group in pd.DataFrame(records).groupby(["arm", "repeat"], sort=True):
        merged = {}
        for _, record in group.iterrows():
            merged.update(record["errors"])
        per_family = {name: float(np.mean([merged[scope] for scope in set(scopes[family == name])]))
                      for name in families}
        measured.append({"arm": arm, "repeat": int(repeat),
                         "equal_scope_brier": float(np.mean(list(merged.values()))),
                         "equal_family_brier": float(np.mean(list(per_family.values()))),
                         "rows_fitted": int(group.rows_fitted.iloc[0]),
                         "scopes_fitted": int(group.scopes_fitted.iloc[0]), **per_family})
    return measured, x_add, y_add, s_add


def _write_csv(path, rows, fields):
    import csv
    path = Path(path)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fields})


def run_training_data_expansion(root, config, out):
    """Run the complete sensitivity study and publish its evidence."""
    root, out = Path(root).resolve(), Path(out).resolve()
    config = TrainingExpansionConfig.model_validate(config)
    inputs = verify_expansion_inputs(root, config)
    if out.exists() and any(out.iterdir()):
        raise ValueError("Create a fresh output directory before running")
    out.mkdir(parents=True, exist_ok=True)
    if out.is_relative_to(inputs["prepared"]) or inputs["prepared"].is_relative_to(out):
        raise ValueError("Keep experiment output separate from prepared data")

    _, rows, targets, provenance = training_panel(inputs["prepared"])
    labels = np.array([targets[row["row_key"]] for row in rows], dtype=float)
    family = np.array([row["validation_family"] for row in rows])
    scopes = np.array([row["scope_key"] for row in rows])
    groups = np.array([row["task_group"] for row in rows])
    matrix = prepare_feature_matrix(inputs["prepared"], rows, out)

    curve, _ = diversity_curve(labels, family, scopes, groups, matrix,
                               l2=config.l2, interactions=config.interactions)
    panel_groups = set(groups)
    added = sample_added_rows(inputs["data_root"], config.added_benchmarks, panel_groups,
                              rows_per_system=config.rows_per_system,
                              min_pair_items=config.min_pair_items, seed=config.seed,
                              pool_multiplier=config.repeats)
    if not added:
        raise ValueError("No usable rows were sampled from the selected benchmarks")
    measured, _, _, _ = expansion_arms(labels, family, scopes, groups, matrix, added,
                                       config.added_benchmarks,
                                       l2=config.l2, interactions=config.interactions,
                                       repeats=config.repeats, seed=config.seed)

    families = sorted(set(family))
    inventory = []
    for benchmark in config.added_benchmarks:
        chosen = [row for row in added if row["benchmark"] == benchmark.name]
        per_repeat = [repeat_subset(added, [benchmark], repeat, config.seed)
                      for repeat in range(config.repeats)]
        labels_here = [row["label"] for rows in per_repeat for row in rows]
        inventory.append({"benchmark": benchmark.name, "domain": benchmark.domain,
                          "requested_cap": benchmark.cap, "pool_rows": len(chosen),
                          "rows_per_repeat": float(np.mean([len(rows) for rows in per_repeat])),
                          "scopes_per_repeat": float(np.mean([len({row["scope_key"] for row in rows})
                                                              for rows in per_repeat])),
                          "label_mean": float(np.mean(labels_here)) if labels_here else None,
                          **inputs["added"][benchmark.name]})
    arm_summary = []
    for arm in ["base", "base+panel_rows", "base+unused_benchmarks", "unused_benchmarks_only"]:
        values = [row for row in measured if row["arm"] == arm]
        if not values:
            raise ValueError(f"Missing arm {arm}")
        scores = [row["equal_scope_brier"] for row in values]
        arm_summary.append({"arm": arm, "repeats": len(values),
                            "equal_scope_brier": float(np.mean(scores)),
                            "equal_scope_min": float(np.min(scores)),
                            "equal_scope_max": float(np.max(scores)),
                            "equal_family_brier": float(np.mean([row["equal_family_brier"] for row in values])),
                            "equal_family_min": float(np.min([row["equal_family_brier"] for row in values])),
                            "equal_family_max": float(np.max([row["equal_family_brier"] for row in values])),
                            "rows_fitted": int(round(np.mean([row["rows_fitted"] for row in values]))),
                            "scopes_fitted": int(round(np.mean([row["scopes_fitted"] for row in values])))})
    per_family = []
    for name in families:
        base = float(np.mean([row[name] for row in measured if row["arm"] == "base"]))
        added_mean = float(np.mean([row[name] for row in measured if row["arm"] == "base+unused_benchmarks"]))
        control = float(np.mean([row[name] for row in measured if row["arm"] == "base+panel_rows"]))
        per_family.append({"held_family": name, "base": base, "base+panel_rows": control,
                           "base+unused_benchmarks": added_mean,
                           "delta_unused_minus_base": added_mean - base})
    delta = {row["held_family"]: row["delta_unused_minus_base"] for row in per_family}
    largest = max(delta, key=lambda name: abs(delta[name]))
    remaining = [value for name, value in delta.items() if name != largest]
    added_rows = [row for row in measured if row["arm"] == "base+unused_benchmarks"]
    control_rows = [row for row in measured if row["arm"] == "base+panel_rows"]
    arm = {row["arm"]: row for row in arm_summary}

    full = curve[-1]
    gains = {size: float(curve[0]["equal_family_brier"] - row["equal_family_brier"]) for size, row in
             enumerate(curve, start=1)}
    report = {
        "version": "training_data_expansion_v1",
        "purpose": "compare training-benchmark diversity against row count for held-out-family cold-start transfer",
        "instrument": {"features": VERSION, "dimensions": DIMS, "model": "conditional_ridge",
                       "l2": config.l2, "interactions": config.interactions,
                       "note": "already-rejected control reused as a relative instrument, not a candidate"},
        "metrics": {"equal_scope": "per-scope mean squared error over every panel scope, averaged equally",
                    "equal_family": "per-family means averaged equally; composition-neutral diagnostic"},
        "fitting_role": "training_only", "development_outcomes_used": False,
        "monitor_outcomes_used": False, "sealed_roles_opened": False,
        "prepared_bundle_manifest_sha256": file_hash(inputs["prepared"] / "manifest.json"),
        "panel": {"rows": len(rows), "families": len(families), "scopes": len(set(scopes)),
                  "label_mean": float(labels.mean()),
                  "family_attribution": provenance, "feature_dimensions": DIMS},
        "family_diversity_curve": curve,
        "diversity_gain_equal_family": {str(size): gain for size, gain in gains.items()},
        "full_panel_equal_scope_reproduction": float(full["equal_scope_brier"]),
        "added_benchmarks": inventory,
        "added_pool_rows": len(added),
        "added_rows_per_repeat": arm["base+unused_benchmarks"]["rows_fitted"] - arm["base"]["rows_fitted"],
        "added_scopes_per_repeat": arm["base+unused_benchmarks"]["scopes_fitted"] - arm["base"]["scopes_fitted"],
        "sampling_sensitivity": {
            "added_equal_scope_range": float(np.ptp([row["equal_scope_brier"] for row in added_rows])),
            "control_equal_scope_range": float(np.ptp([row["equal_scope_brier"] for row in control_rows])),
            "added_equal_family_range": float(np.ptp([row["equal_family_brier"] for row in added_rows])),
            "control_equal_family_range": float(np.ptp([row["equal_family_brier"] for row in control_rows])),
        },
        "expansion_arms": arm_summary,
        "expansion_per_family": per_family,
        "largest_per_family_abs_delta": largest,
        "mean_delta_excluding_largest": float(np.mean(remaining)) if remaining else None,
        "model_selection_performed": False, "candidate_promoted": False,
        "limitations": [
            "Lexical ridge control only; a tree or text model may use the same rows differently.",
            "One dataset revision, seven training families and cold-start Brier only.",
            "Added rows use one outcome-blind response per subject-item and a per-system cap.",
            "higher_is_better is a coarse success proxy; external eligibility is unreviewed.",
            "Fixed L2 with no tuning and no bootstrap interval across families.",
            "Repeats vary the hashed row draw; they do not resample systems, tasks or trajectories.",
            "One dataset revision, seven training families and a single ridge instrument.",
        ],
        "implementation_sha256": file_hash(Path(__file__)),
    }

    _write_csv(out / "family_diversity_curve.csv", curve,
               ["families_used", "subsets", "equal_family_brier", "equal_scope_brier", *families])
    _write_csv(out / "expansion_arms.csv", measured,
               ["arm", "repeat", "equal_scope_brier", "equal_family_brier",
                "rows_fitted", "scopes_fitted", *families])
    _write_csv(out / "expansion_per_family.csv", per_family,
               ["held_family", "base", "base+panel_rows", "base+unused_benchmarks",
                "delta_unused_minus_base"])
    write_jsonl(out / "added_rows_index.jsonl", (
        {key: value for key, value in row.items() if key not in {"subject", "item"}}
        for row in added))
    write_json(out / "summary.json", report)
    write_json(out / "manifest.json", {"artifacts": {
        str(path.relative_to(out)): file_hash(path)
        for path in sorted(out.rglob("*")) if path.is_file() and path.name != "manifest.json"}})
    return report
