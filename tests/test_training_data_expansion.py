import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from paiec.feature_preparation import write_json, write_jsonl
from paiec.head_training import file_hash
from paiec.training_data_expansion import (
    TrainingExpansionConfig,
    held_split,
    run_training_data_expansion,
    sample_added_rows,
    verify_expansion_inputs,
)


def refresh_manifest(path):
    write_json(path / "manifest.json", {"version": "competition_data_v1", "artifacts": {
        str(p.relative_to(path)): file_hash(p)
        for p in sorted(path.rglob("*")) if p.is_file() and p.name != "manifest.json"}})


class ExpansionFixture:
    """Synthetic panel bundle, public tables and frozen roles."""

    def __init__(self, root: Path):
        self.root = root
        self.source = root / "runs/source_bundle"
        self.prepared = root / "runs/prepared"

    def write_public_benchmark(self, name, *, direction="higher_is_better", modality=("text",), labels=(1, 0)):
        directory = self.root / "payload/data" / name / "formatted_tables"
        directory.mkdir(parents=True)
        scale = json.dumps({"direction": direction, "kind": "discrete", "values": [0.0, 1.0]})
        pd.DataFrame([{"benchmark_id": name, "modality": list(modality), "response_scale": scale,
                       "response_type": "binary", "granularity": "item"}]).to_parquet(
            directory / "benchmarks.parquet", index=False)
        subjects = [{"subject_id": f"{name}_s1", "normalized_name": f"Model {name} one",
                     "provider": "Org", "release_date": "2024-01-01", "access_date": None,
                     "harness": "ToyHarness", "harness_version": None, "reasoning_effort": None,
                     "subject_features_extra": "shot_size=0"},
                    {"subject_id": f"{name}_s2", "normalized_name": f"Model {name} two",
                     "provider": "Org", "release_date": "2024-02-01", "access_date": None,
                     "harness": "ToyHarness", "harness_version": None, "reasoning_effort": None,
                     "subject_features_extra": None}]
        pd.DataFrame(subjects).to_parquet(directory / "subjects.parquet", index=False)
        items = [{"item_id": f"{name}_i{i}", "benchmark_id": name,
                  "content": f"{name} question number {i}",
                  "item_features": f"dataset={name}" if i % 2 else None}
                 for i in range(4)]
        pd.DataFrame(items).to_parquet(directory / "items.parquet", index=False)
        responses = []
        for subject in [f"{name}_s1", f"{name}_s2"]:
            for i, label in enumerate(labels):
                responses.append({"subject_id": subject, "item_id": f"{name}_i{i}",
                                  "response_id": f"{name}_{subject}_{i}", "benchmark_id": name,
                                  "trial": 1, "interactors": None, "response": float(label)})
        pd.DataFrame(responses).to_parquet(directory / "response.parquet", index=False)
        return directory

    def write(self):
        self.write_public_benchmark("toy_a")
        self.write_public_benchmark("toy_c", direction="lower_is_better")
        self.write_public_benchmark("toy_image", modality=("text", "image"))
        (self.root / "configs/head_finetuning").mkdir(parents=True, exist_ok=True)
        write_json(self.root / "configs/head_finetuning/frozen_split_v1.json", {
            "roles": {"training": ["toy_a"], "calibration": ["toy_sealed"]},
            "family_roles": {"fam_a": "training", "fam_b": "training"}})
        write_json(self.root / "payload/data/corpus_manifest.json", {"revision": "fixture"})
        write_json(self.root / "split.json", {"dataset_revision": "fixture",
                                               "family_roles": {"fam_a": "training", "fam_b": "training"}})

        systems = {
            "sys_a": {"normalized_name": "Toy Model A", "provider": "Org", "release_date": "2024-03-01",
                      "access_date": "", "harness": "ToyHarness", "harness_version": "",
                      "reasoning_effort": "", "subject_features_extra": "shot_size=2"},
            "sys_b": {"normalized_name": "Toy Model B", "provider": "", "release_date": "",
                      "access_date": "", "harness": "", "harness_version": "",
                      "reasoning_effort": "", "subject_features_extra": ""},
        }
        tasks = {f"t{i}": {"item_content": f"toy task {i}", "item_features": f"dataset=toy{i}",
                           "interactors": "", "benchmark_id": "benchmark_toy"} for i in range(1, 8)}
        rows = [("fam_a", "scope_a1", "g1", "sys_a", "t1", 0),
                ("fam_a", "scope_a1", "g2", "sys_a", "t2", 1),
                ("fam_a", "scope_a2", "g3", "sys_b", "t3", 0),
                ("fam_b", "scope_b1", "g4", "sys_b", "t4", 1),
                ("fam_b", "scope_b1", "g5", "sys_b", "t5", 0),
                ("fam_b", "scope_b2", "g6", "sys_a", "t6", 1),
                ("fam_b", "scope_b2", "g7", "sys_a", "t7", 0)]
        examples, targets, provenance = [], [], []
        for line, (family, scope, group, system, task, label) in enumerate(rows):
            example = {"role": "training", "source_line": line, "system_key": system,
                       "task_key": task, "benchmark_id": "benchmark_toy",
                       "task_group": group, "scope_key": scope, "row_key": f"row{line}"}
            examples.append(example)
            targets.append({"row_key": f"row{line}", "native_binary_value": bool(label)})
            provenance.append({"role": "training", "jsonl_line": line, "family": family})
        self.prepared.mkdir(parents=True)
        write_jsonl(self.prepared / "examples.jsonl", examples)
        write_jsonl(self.prepared / "training_targets.jsonl", targets)
        write_jsonl(self.prepared / "systems.jsonl",
                    ({"system_key": key, "recorded": value} for key, value in systems.items()))
        write_jsonl(self.prepared / "tasks.jsonl",
                    ({"task_key": key, "recorded": value} for key, value in tasks.items()))
        write_json(self.prepared / "summary.json", {"source_policy": "provided_dataset_only",
                                                    "external_profile_facts_used": 0})
        (self.prepared / "evidence_plan_v1").mkdir()
        write_jsonl(self.prepared / "evidence_plan_v1/training_row_index.jsonl", examples)
        write_jsonl(self.prepared / "evidence_plan_v1/evidence_episode_links.jsonl", [])

        self.source.mkdir()
        pd.DataFrame(provenance).to_parquet(self.source / "family_provenance.parquet", index=False)
        (self.source / "source").mkdir()
        (self.source / "source/training.jsonl").write_text("{}\n" * len(rows))
        write_json(self.source / "manifest.json", {
            "version": "enriched_expanded_v1", "dataset_revision": "fixture",
            "family_provenance_path": "family_provenance.parquet",
            "data_config": {"split": "split.json"},
            "artifacts": {"family_provenance.parquet": file_hash(self.source / "family_provenance.parquet"),
                          "source/training.jsonl": file_hash(self.source / "source/training.jsonl")}})
        write_json(self.prepared / "source_contract.json", {
            "root": str(self.root), "prepared": str(self.source),
            "source_manifest_sha256": file_hash(self.source / "manifest.json"),
            "split_sha256": file_hash(self.root / "split.json"), "dataset_revision": "fixture",
            "config": {"source_policy": "provided_dataset_only", "roles": ["training"]}})
        refresh_manifest(self.prepared)
        return self

    def config(self, **overrides):
        values = {"prepared": "runs/prepared", "data_root": "payload/data",
                  "l2": 100.0, "interactions": True, "rows_per_system": 1,
                  "min_pair_items": 2, "repeats": 1, "seed": 7,
                  "added_benchmarks": [{"name": "toy_a", "cap": 2, "domain": "toy"}]}
        values.update(overrides)
        return TrainingExpansionConfig.model_validate(values)


class TrainingDataExpansionTests(unittest.TestCase):
    def test_sealed_non_text_and_wrong_direction_benchmarks_are_refused(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ExpansionFixture(Path(temporary)).write()
            with self.assertRaisesRegex(ValueError, "sealed role"):
                verify_expansion_inputs(fixture.root, fixture.config(
                    added_benchmarks=[{"name": "toy_sealed", "cap": 1, "domain": "sealed"}]))
            with self.assertRaisesRegex(ValueError, "higher_is_better"):
                verify_expansion_inputs(fixture.root, fixture.config(
                    added_benchmarks=[{"name": "toy_c", "cap": 1, "domain": "wrong target"}]))
            with self.assertRaisesRegex(ValueError, "text-only"):
                verify_expansion_inputs(fixture.root, fixture.config(
                    added_benchmarks=[{"name": "toy_image", "cap": 1, "domain": "multimodal"}]))
            with self.assertRaisesRegex(ValueError, "plain benchmark"):
                fixture.config(added_benchmarks=[{"name": "../toy_a", "cap": 1, "domain": "escape"}])

    def test_sampling_uses_hashes_not_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ExpansionFixture(Path(temporary)).write()
            config = fixture.config(added_benchmarks=[{"name": "toy_a", "cap": 4, "domain": "toy"}],
                                    rows_per_system=2)
            inputs = verify_expansion_inputs(fixture.root, config)
            first = sample_added_rows(inputs["data_root"], config.added_benchmarks, set(),
                                      rows_per_system=config.rows_per_system,
                                      min_pair_items=config.min_pair_items, seed=config.seed)
            self.assertTrue(1 <= len(first) <= 4)
            self.assertEqual(len({(row["subject_id"], row["item_id"]) for row in first}), len(first))
            directory = fixture.root / "payload/data/toy_a/formatted_tables"
            frame = pd.read_parquet(directory / "response.parquet")
            frame["response"] = 1.0 - frame.response
            frame = frame.sample(frac=1, random_state=3).reset_index(drop=True)
            frame.to_parquet(directory / "response.parquet", index=False)
            second = sample_added_rows(inputs["data_root"], config.added_benchmarks, set(),
                                       rows_per_system=config.rows_per_system,
                                       min_pair_items=config.min_pair_items, seed=config.seed)
            self.assertEqual([(row["subject_id"], row["item_id"]) for row in first],
                             [(row["subject_id"], row["item_id"]) for row in second])
            self.assertEqual([row["label"] for row in first], [1 - row["label"] for row in second])

    def test_whole_family_and_repeated_task_are_excluded_from_fitting(self):
        held, fit = held_split(["A", "A", "B", "B", "C"], ["x", "y", "x", "z", "z"], "A")
        self.assertEqual(np.flatnonzero(held).tolist(), [0, 1])
        self.assertEqual(np.flatnonzero(fit).tolist(), [3, 4])
        with self.assertRaises(ValueError):
            held_split(["A", "A"], ["x", "y"], "missing")

    def test_run_publishes_evidence_and_rejects_a_changed_bundle(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = ExpansionFixture(Path(temporary)).write()
            out = fixture.root / "runs/out"
            report = run_training_data_expansion(fixture.root, fixture.config().model_dump(), out)
            self.assertEqual(report["panel"]["rows"], 7)
            self.assertEqual(report["panel"]["families"], 2)
            self.assertEqual(len(report["expansion_arms"]), 4)
            self.assertEqual(len(report["expansion_per_family"]), 2)
            self.assertEqual(report["added_rows_per_repeat"], 2)
            self.assertFalse(report["candidate_promoted"])
            self.assertFalse(report["sealed_roles_opened"])
            for name in ["summary.json", "manifest.json", "family_diversity_curve.csv",
                         "expansion_arms.csv", "expansion_per_family.csv",
                         "added_rows_index.jsonl", "feature_contract.json"]:
                self.assertTrue((out / name).is_file(), name)
            manifest = json.loads((out / "manifest.json").read_text())["artifacts"]
            self.assertIn("summary.json", manifest)

            examples = (fixture.prepared / "examples.jsonl").read_text() + "\n"
            (fixture.prepared / "examples.jsonl").write_text(examples)
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                run_training_data_expansion(fixture.root, fixture.config().model_dump(),
                                            fixture.root / "runs/second")
            with self.assertRaisesRegex(ValueError, "fresh output directory"):
                run_training_data_expansion(fixture.root, fixture.config().model_dump(), out)


if __name__ == "__main__":
    unittest.main()
