#!/usr/bin/env python
"""Prepare reviewed family splits and pilot files; never trains or scores a model."""

from __future__ import annotations

import argparse
from collections import defaultdict
import hashlib
import json
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

from paiec.competition import competition_input
from paiec.head_data import HeadDataConfig, ROLES, assign_roles, digest, near_duplicates, pair_balanced_sample, task_key
from paiec.laya_predictor import build_state, load_questions
from paiec.laya_runtime import DEFAULT_CHECKPOINT_REVISION
from paiec.schema import clean, subject_record
from paiec.split import make_split, stable_seed
from paiec.tracking import git_context, package_versions
from measure_tokens import load_tokenizer

ROOT = Path(__file__).resolve().parents[1]


def file_hash(path):
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def review_and_group(data, cfg, out):
    """Review evidence and exact source-task overlaps before assigning roles."""
    parents = {review.family: review.family for review in cfg.reviews.values()}

    def root(name):
        while parents[name] != name:
            name = parents[name]
        return name

    catalogue, first, overlaps, hashes = [], {}, [], {}
    tokenizer = load_tokenizer("english")
    from laya_mlx.common import build_head
    prefix, _, _ = build_head(tokenizer, {"t": "noul", "ins": load_questions()["correct"]["instructions"], "crit": None}, 192)
    room = 512 - len(prefix) - 1
    for benchmark, review in sorted(cfg.reviews.items()):
        path = data / benchmark / "formatted_tables"
        meta = pd.read_parquet(path / "benchmarks.parquet").iloc[0]
        catalogue.append({"benchmark": benchmark, **review.model_dump(),
                          "description": meta.description, "source_url": meta.source_url,
                          "response_scale": meta.response_scale, "modality": str(meta.modality),
                          "review_basis": "Pinned benchmark metadata, native grading definitions, and two seed-219 item inspections for candidate benchmarks; no traces or new model scores."})
        if review.status not in {"eligible", "monitoring", "monitoring_related"}:
            continue
        for name in ("benchmarks.parquet", "items.parquet", "subjects.parquet", "response.parquet"):
            hashes[str((path / name).relative_to(data))] = file_hash(path / name)
        items = pd.read_parquet(path / "items.parquet", columns=["item_id", "content", "item_features"])
        if review.required_feature:
            items = items[items.item_features.fillna("").str.contains(review.required_feature, regex=False)]
        if review.status == "eligible":
            # Outcome-blind length probe, using random real subject profiles.
            subjects = pd.read_parquet(path / "subjects.parquet")
            profiles = subjects.sample(min(4, len(subjects)), random_state=219)
            probe = items.sample(min(32, len(items)), random_state=219)
            fits = []
            for index, row in enumerate(probe.itertuples()):
                profile = subject_record(profiles.iloc[index % len(profiles)])
                state = build_state(profile, {"item_content": clean(row.content),
                    "item_features": clean(row.item_features), "benchmark_id": benchmark}, [])
                fits.append(len(tokenizer(state.replace(tokenizer.mask_token, " "))["input_ids"]) <= room)
            catalogue[-1]["visibility_probe_n"] = len(fits)
            catalogue[-1]["visibility_probe_fit_fraction"] = float(np.mean(fits))
            catalogue[-1]["has_short"] = bool(np.mean(fits) >= .5)
        for item_id, content in zip(items.item_id, items.content):
            key = task_key(clean(content))
            if not key:
                continue
            fingerprint = digest(key)
            if fingerprint not in first:
                first[fingerprint] = (benchmark, str(item_id), review.family)
                continue
            other, other_id, family = first[fingerprint]
            left, right = root(family), root(review.family)
            if left != right:
                parents[max(left, right)] = min(left, right)
                overlaps.append({"benchmark_a": other, "item_a": other_id,
                                 "benchmark_b": benchmark, "item_b": str(item_id),
                                 "normalized_task_sha256": fingerprint})
    components = defaultdict(list)
    for family in parents:
        components[root(family)].append(family)
    aliases = {family: "+".join(sorted(members)) for members in components.values() for family in members}
    reserved = {aliases[r.family] for r in cfg.reviews.values()
                if r.status in {"monitoring", "monitoring_related"}}
    families = {}
    for row in catalogue:
        row["source_family"] = row["family"]
        row["family"] = aliases[row["family"]]
        if row["status"] == "eligible" and row["family"] in reserved:
            row["status"] = "monitoring_related"
            row["reason"] += " Exact normalized task overlap links this family to historical monitoring."
        if row["status"] == "eligible":
            entry = families.setdefault(row["family"], {"domain": row["domain"], "has_english": False, "has_short": False})
            entry["has_english"] |= row["has_english"]
            entry["has_short"] |= row["has_short"]
    assigned, allocation = assign_roles(families, cfg.seed)
    for row in catalogue:
        row["role"] = assigned[row["family"]] if row["status"] == "eligible" else row["status"]
    pd.DataFrame(catalogue).to_csv(out / "benchmark_review.csv", index=False)
    pd.DataFrame(overlaps, columns=["benchmark_a", "item_a", "benchmark_b", "item_b", "normalized_task_sha256"]).to_csv(out / "source_overlap_edges.csv", index=False)
    (out / "source_file_hashes.json").write_text(json.dumps(hashes, indent=2))
    return catalogue, assigned, allocation


def candidate_rows(data, benchmark, cfg):
    path = data / benchmark / "formatted_tables"
    items = pd.read_parquet(path / "items.parquet", columns=["item_id", "content", "item_features"])
    review = cfg.reviews[benchmark]
    if review.required_feature:
        items = items[items.item_features.fillna("").str.contains(review.required_feature, regex=False)]
    items = items.drop_duplicates("item_id").set_index("item_id", drop=False)
    subjects = pd.read_parquet(path / "subjects.parquet").set_index("subject_id", drop=False)
    counts = pd.read_parquet(path / "response.parquet", columns=["subject_id", "item_id", "response"])
    counts = counts[counts.response.isin([0, 1]) & counts.item_id.isin(items.index)]
    sizes = counts.groupby("subject_id").item_id.nunique()
    eligible = sorted(sizes[(sizes >= cfg.min_items) & sizes.index.isin(subjects.index)].index)
    rng = np.random.default_rng(stable_seed(cfg.seed, benchmark, "head-subjects"))
    rng.shuffle(eligible)
    selected = eligible[:cfg.subjects_per_benchmark]
    del counts
    if not selected:
        raise ValueError(f"No qualifying subjects after input filtering: {benchmark}")
    rows = pd.read_parquet(path / "response.parquet",
                           columns=["subject_id", "item_id", "response_id", "trial", "interactors", "response"],
                           filters=[("subject_id", "in", selected)])
    rows = rows[rows.response.isin([0, 1]) & rows.item_id.isin(items.index)].copy()
    rows["interactors"] = rows.interactors.map(clean)
    if (rows.groupby(["subject_id", "item_id"]).interactors.nunique() > 1).any():
        raise ValueError(f"Per-trial input tools vary for the same subject/item: {benchmark}; explicit review needed")
    if rows.response_id.duplicated().any():
        raise ValueError(f"Duplicate source response IDs: {benchmark}")
    rows["benchmark"] = benchmark
    return rows, items, subjects


def prepare(data, cfg, catalogue, out):
    pools, tables, evaluation_manifest, selection = defaultdict(list), {}, [], []
    for review in catalogue:
        if review["status"] != "eligible":
            continue
        benchmark, role = review["benchmark"], review["role"]
        rows, items, subjects = candidate_rows(data, benchmark, cfg)
        tables[benchmark] = (items, subjects)
        rows["family"] = review["family"]
        rows["role"] = role
        if role in {"development", "confirmation"}:
            chosen = []
            for subject, group in rows.groupby("subject_id", sort=True):
                split = make_split(tuple(group.item_id.unique()), benchmark=benchmark,
                                   subject_id=subject, seed=cfg.seed,
                                   acquisition_items=cfg.acquisition_items,
                                   evaluation_items=cfg.evaluation_items)
                for pool, ids in (("acquisition", split.acquisition), ("evaluation", split.evaluation)):
                    evaluation_manifest.extend({"role": role, "family": review["family"],
                        "benchmark": benchmark, "subject_id": subject, "item_id": item,
                        "pool": pool, "position": i} for i, item in enumerate(ids))
                chosen.append(group[group.item_id.isin(split.acquisition + split.evaluation)])
            rows = pd.concat(chosen, ignore_index=True)
        pools[role].append(rows)
        selection.append({"benchmark": benchmark, "family": review["family"], "role": role,
                          "selected_subjects": rows.subject_id.nunique(), "candidate_responses": len(rows),
                          "candidate_items": rows.item_id.nunique()})
        print(f"{role}: {benchmark}, {rows.subject_id.nunique()} subjects, {len(rows):,} candidate responses", flush=True)
    pd.DataFrame(selection).to_csv(out / "selection_summary.csv", index=False)
    pd.DataFrame(evaluation_manifest).to_csv(out / "evaluation_manifest.csv", index=False)
    selected = {}
    for role in ROLES:
        frame = pd.concat(pools[role], ignore_index=True)
        if role in {"training", "calibration"}:
            count = cfg.training_rows if role == "training" else cfg.calibration_rows
            frame = pair_balanced_sample(frame, count, stable_seed(cfg.seed, role))
        selected[role] = frame
    return selected, tables


def export(selected, tables, cfg, out):
    tokenizer = load_tokenizer("english")
    question = load_questions()
    from laya_mlx.common import build_head
    prefix, _, _ = build_head(tokenizer, {"t": "noul", "ins": question["correct"]["instructions"], "crit": None}, 192)
    room = 512 - len(prefix) - 1
    provenance, cases, unique, role_keys = [], {}, {}, defaultdict(set)
    encoded_cache = {}
    question_hash = digest(json.dumps(question, sort_keys=True))
    for role, frame in selected.items():
        for row in frame.itertuples():
            items, subjects = tables[row.benchmark]
            subject = subject_record(subjects.loc[row.subject_id])
            record = items.loc[row.item_id]
            item = {"item_content": clean(record.content), "item_features": clean(record.item_features),
                    "interactors": row.interactors, "benchmark_id": row.benchmark}
            state = build_state(subject, item, [])
            state_hash = digest(state)
            clean_text = state.replace(tokenizer.mask_token, " ")
            if state_hash not in encoded_cache:
                encoded_cache[state_hash] = tokenizer(clean_text)["input_ids"]
            ids = encoded_cache[state_hash]
            effective = digest(json.dumps([DEFAULT_CHECKPOINT_REVISION, prefix, ids[:room]]))
            normalized = task_key(item["item_content"])
            key = digest(normalized)
            # Confirmation is input-only: never export target outcomes for it.
            case = {"state": state, "questions": question}
            if role != "confirmation":
                case["expected"] = {"correct": bool(row.response)}
            row_id = digest(f"{row.benchmark}/{row.response_id}")
            cases[row_id] = case
            provenance.append({"row_id": row_id, "role": role, "family": row.family,
                "benchmark": row.benchmark, "subject_id": row.subject_id, "item_id": row.item_id,
                "response_id": row.response_id, "trial": row.trial,
                "sample_weight": getattr(row, "sample_weight", 1.0),
                "state_sha256": state_hash, "effective_input_sha256": effective,
                "task_sha256": key, "question_sha256": question_hash,
                "state_tokens": len(ids), "state_tokens_used": min(len(ids), room),
                "truncated": len(ids) > room,
                "missing_model_name": not bool(competition_input(subject, item)[0]["normalized_name"]),
                "pool": ""})
            role_keys[("effective", effective)].add(role)
            role_keys[("task", key)].add(role)
            unique.setdefault((role, key), (normalized, role, row_id))
    conflicts = [{"kind": k[0], "sha256": k[1], "roles": sorted(v)}
                 for k, v in role_keys.items() if len(v) > 1]
    examples = list(unique.values())
    near = near_duplicates([e[0] for e in examples], [e[1] for e in examples])
    near_rows = [{"row_a": examples[a][2], "role_a": examples[a][1],
                  "row_b": examples[b][2], "role_b": examples[b][1], "jaccard": score}
                 for a, b, score in near]
    (out / "cross_role_exact_conflicts.json").write_text(json.dumps(conflicts, indent=2))
    pd.DataFrame(near_rows, columns=["row_a", "role_a", "row_b", "role_b", "jaccard"]).to_csv(out / "cross_role_near_candidates.csv", index=False)
    sidecar = pd.DataFrame(provenance)
    sidecar.to_parquet(out / "provenance.parquet", index=False)
    if conflicts or near_rows:
        raise ValueError(f"Split not frozen: {len(conflicts)} exact conflicts, {len(near_rows)} near candidates require review; no JSONL exported")
    evaluation = pd.read_csv(out / "evaluation_manifest.csv", dtype={"item_id": str, "subject_id": str})
    pools = {(r.benchmark, r.subject_id, r.item_id): r.pool for r in evaluation.itertuples()}
    sidecar["pool"] = [pools.get((r.benchmark, r.subject_id, r.item_id), "") for r in sidecar.itertuples()]
    sidecar["exported"] = ~sidecar.role.isin(["development", "confirmation"]) | sidecar.pool.eq("evaluation")
    # Confirmation needs one probability per subject/item, not repeated copies
    # for every trial. Keep all trial provenance for response-correct scoring.
    confirmation = sidecar.role.eq("confirmation") & sidecar.exported
    duplicate = sidecar[confirmation].duplicated(["benchmark", "subject_id", "item_id"])
    sidecar.loc[sidecar[confirmation].index[duplicate], "exported"] = False
    sidecar["jsonl_line"] = -1
    for role in ROLES:
        rows = sidecar[sidecar.role.eq(role) & sidecar.exported]
        sidecar.loc[rows.index, "jsonl_line"] = np.arange(len(rows))
        filename = "confirmation_inputs.jsonl" if role == "confirmation" else f"{role}.jsonl"
        with (out / filename).open("w") as handle:
            for row_id in rows.row_id:
                handle.write(json.dumps(cases[row_id], ensure_ascii=False) + "\n")
    sidecar.to_parquet(out / "provenance.parquet", index=False)
    return sidecar


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default=str(ROOT / "configs/head_finetuning/data_split_v1.json"))
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    cfg = HeadDataConfig.model_validate_json(Path(args.config).read_text())
    data = ROOT / "payload/data"
    corpus = json.loads((data / "corpus_manifest.json").read_text())
    if corpus["revision"] != cfg.dataset_revision:
        raise ValueError("Dataset revision does not match reviewed configuration")
    if {x.split('/')[0] for x in corpus["eligible_dirs"]} != set(cfg.reviews):
        raise ValueError("Every downloaded benchmark must have an explicit review decision")
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=False)
    with zipfile.ZipFile(out / "source.zip", "w", zipfile.ZIP_DEFLATED) as archive:
        for path in [*ROOT.glob("src/paiec/*.py"), ROOT / "scripts/prepare_head_data.py",
                     ROOT / "scripts/measure_tokens.py", ROOT / "tests/test_head_data.py",
                     Path(args.config)]:
            archive.write(path, str(path.resolve().relative_to(ROOT)))
    (out / "resolved_config.json").write_text(cfg.model_dump_json(indent=2))
    catalogue, assigned, allocation = review_and_group(data, cfg, out)
    (out / "family_roles.json").write_text(json.dumps(assigned, indent=2))
    selected, tables = prepare(data, cfg, catalogue, out)
    sidecar = export(selected, tables, cfg, out)
    for role in ROLES[1:]:
        rows = sidecar[sidecar.role.eq(role) & sidecar.exported]
        if (~rows.truncated).mean() < .1:
            raise ValueError(f"Split not frozen: {role} has fewer than 10% fully visible inputs")
    summary = sidecar.groupby("role").agg(rows=("row_id", "size"),
        exported_rows=("exported", "sum"),
        benchmarks=("benchmark", "nunique"), families=("family", "nunique"),
        truncated=("truncated", "sum"), missing_model_names=("missing_model_name", "sum"))
    summary.to_csv(out / "summary.csv")
    (out / "preparation.json").write_text(json.dumps({"status": "prepared_not_trained",
        "allocation": allocation, "seed": cfg.seed, "dataset_revision": cfg.dataset_revision,
        "checkpoint_revision": DEFAULT_CHECKPOINT_REVISION,
        "checks": {"cross_role_exact_conflicts": 0, "cross_role_near_candidates": 0,
                   "near_screen": "Approximate 16-minhash LSH; exact word-trigram Jaccard >=0.9 on candidates. Not exhaustive paraphrase or translation detection."},
        "git": git_context(ROOT), "packages": package_versions()}, indent=2))
    hashes = {str(p.relative_to(out)): file_hash(p) for p in sorted(out.iterdir()) if p.is_file()}
    (out / "artifact_hashes.json").write_text(json.dumps(hashes, indent=2))
    print(summary.to_string(), flush=True)
    print(f"Prepared {out}; no training or model inference performed.", flush=True)


if __name__ == "__main__":
    main()
