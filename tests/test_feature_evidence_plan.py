from pathlib import Path
import tempfile
import unittest

from paiec.feature_evidence_plan import prepare_evidence_plan
from paiec.feature_preparation import read_jsonl


class FeatureEvidencePlanTests(unittest.TestCase):
    def fixture(self):
        tasks = [{"task_key": str(i), "normalized_content_sha256": str(i // 2)} for i in range(16)]
        examples = [{"role": "training", "source_line": i, "task_key": str(i),
                     "system_key": "system", "benchmark_id": "anonymous", "expected": i % 2}
                    for i in range(16)]
        return tasks, examples

    def test_supports_exclude_query_groups_and_other_systems_and_use_nested_budgets(self):
        tasks, examples = self.fixture()
        examples += [dict(r, source_line=r["source_line"] + 16, system_key="other") for r in examples]
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "plan"
            result = prepare_evidence_plan(tasks, examples, out, folds=3)
            rows = {r["row_key"]: r for r in read_jsonl(out / "training_row_index.jsonl")}
            self.assertEqual(result["training_rows"], 32)
            for pool in read_jsonl(out / "out_of_fold_support_pools.jsonl"):
                for key in pool["support_row_keys"]:
                    self.assertNotEqual(rows[key]["fold"], pool["query_fold"])
                    self.assertEqual(rows[key]["scope_key"], pool["scope_key"])
            for episode in read_jsonl(out / "evidence_episode_links.jsonl"):
                self.assertFalse(set(episode["query_task_groups"]) & set(episode["acquisition_task_groups"]))
                self.assertEqual(len(episode["query_task_groups"]), len(episode["acquisition_task_groups"]))
                for key in episode["query_row_keys"]:
                    self.assertIn(rows[key]["task_group"], episode["query_task_groups"])
                order = episode["support_order_row_keys"]
                self.assertEqual(len(order), len(set(rows[k]["task_group"] for k in order)))
                for budget in episode["budgets"]:
                    self.assertEqual(budget["available"], min(budget["requested"], len(order)))

    def test_inputs_ignore_labels_private_ids_and_order(self):
        tasks, examples = self.fixture()
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            prepare_evidence_plan(tasks, examples, root / "one", folds=3)
            changed = [dict(r, expected="SECRET", subject_id="PRIVATE") for r in reversed(examples)]
            prepare_evidence_plan(list(reversed(tasks)), changed, root / "two", folds=3)
            for path in (root / "one").iterdir():
                self.assertEqual(path.read_bytes(), (root / "two" / path.name).read_bytes())
                self.assertNotIn(b"SECRET", path.read_bytes())

    def test_all_variants_and_systems_share_a_capability_fold(self):
        tasks, examples = self.fixture()
        examples += [dict(r, source_line=r["source_line"] + 16, system_key="other") for r in examples]
        with tempfile.TemporaryDirectory() as temporary:
            out = Path(temporary) / "plan"
            prepare_evidence_plan(tasks, examples, out, folds=3)
            folds = {}
            for row in read_jsonl(out / "training_row_index.jsonl"):
                folds.setdefault(row["task_group"], set()).add(row["fold"])
            self.assertTrue(all(len(f) == 1 for f in folds.values()))

    def test_sealed_or_overlapping_roles_cannot_enter_the_plan(self):
        tasks, examples = self.fixture()
        with tempfile.TemporaryDirectory() as temporary:
            for role in ["development", "confirmation"]:
                with self.subTest(role=role), self.assertRaises(ValueError):
                    prepare_evidence_plan(tasks, examples + [dict(examples[0], role=role)], Path(temporary) / role, folds=3)
