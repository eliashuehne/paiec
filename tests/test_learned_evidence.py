import unittest
import importlib.util
import json
from pathlib import Path
import tempfile

from paiec.learned_evidence import FEATURE_NAMES, corrected_probability, LearnedEvidencePredictor, fit_from_training_monitor


class LearnedEvidenceTests(unittest.TestCase):
    def fitted(self):
        return {"feature_names":list(FEATURE_NAMES), "coefficients":[1., 1., 0., 0., 0., 0.], "intercept":0.}

    def test_zero_budget_is_exact_and_opposite_outcomes_move_in_opposite_directions(self):
        model = self.fitted()
        self.assertEqual(corrected_probability(.712345, 0, 0, model), .712345)
        self.assertGreater(corrected_probability(.7, 7, 7, model), .7)
        self.assertLess(corrected_probability(.7, 7, 0, model), .7)
        with self.assertRaises(ValueError):
            corrected_probability(.7, 3, 4, model)

    def test_only_matching_visible_subject_and_benchmark_evidence_is_used(self):
        class Base:
            use_labels = False
            runtime_metadata = {}
            def predict_many(self, subject, items, labeled):
                if labeled:
                    raise AssertionError("Base must not receive outcome text")
                return [.7] * len(items)
        subject = {"normalized_name":"model-a"}
        query = {"item_content":"query", "benchmark_id":"benchmark_abc"}
        support = {"item_content":"support", "benchmark_id":"benchmark_abc"}
        predictor = LearnedEvidencePredictor(Base(), self.fitted())
        with self.assertRaisesRegex(ValueError, "different checkpoint"):
            LearnedEvidencePredictor(Base(), dict(self.fitted(), checkpoint_identity="local-other"))
        irrelevant = [([{"normalized_name":"model-b"}, support], 1),
                      ([subject, dict(support, benchmark_id="benchmark_def")], 1),
                      ([subject, query], 1)]
        self.assertEqual(predictor(subject, query, irrelevant), .7)
        self.assertGreater(predictor(subject, query, irrelevant + [([subject, support], 1)]), .7)

    @unittest.skipUnless(importlib.util.find_spec("sklearn"), "Optional fitting dependency required")
    def test_fit_uses_disjoint_training_queries_and_rejects_query_support_overlap(self):
        import pandas as pd
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "source").mkdir()
            (root / "enriched").mkdir()
            (root / "manifest.json").write_text('{}')
            (root / "report.json").write_text(json.dumps(dict(status="finished", checkpoint_identity="local-fixture")))
            records = []
            for role in ["training", "monitor"]:
                rows = []
                for label in [0, 1]:
                    state = json.dumps(dict(target_system={"normalized_name":str(label)},
                        target_item={"benchmark_id":"benchmark_abc", "item_content":f"{role}-{label}"}))
                    rows.append(dict(state=state, expected={"correct":bool(label)}))
                    records.append(dict(role=role, row_id=f"{role}-{label}", task_sha256=f"{role}-{label}", jsonl_line=label, sample_weight=1.))
                (root / "source" / (role+'.jsonl')).write_text(''.join(json.dumps(r)+'\n' for r in rows))
            frame = pd.DataFrame(records)
            frame.to_parquet(root / "enriched/provenance.parquet", index=False)
            pd.DataFrame(dict(row_id=['monitor-0','monitor-1'], probability=[.5,.5])).to_csv(root / "monitor_predictions.csv", index=False)
            model = fit_from_training_monitor(root, root, root)
            self.assertEqual(model['unique_query_records'], 2)
            self.assertGreater(corrected_probability(.5, 1, 1, model), corrected_probability(.5, 1, 0, model))
            frame.loc[frame.role.eq('monitor'), 'task_sha256'] = ['training-0','training-1']
            frame.to_parquet(root / "enriched/provenance.parquet", index=False)
            with self.assertRaisesRegex(ValueError, "query/support task overlap"):
                fit_from_training_monitor(root, root, root)


if __name__ == "__main__":
    unittest.main()
