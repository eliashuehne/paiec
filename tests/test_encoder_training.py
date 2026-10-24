import importlib.util
import json
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from paiec.expanded_data import balanced_unique_sample, verify_expanded
from paiec.encoder_training import EncoderTrainingConfig, ComputeSettings, grouped_batches, set_training_policy, train_window
from paiec.head_training import file_hash
from paiec.laya_predictor import load_questions
from paiec.execution.jobs import JobSpec, ExecutionProfile, command, input_files


class ExpansionTests(unittest.TestCase):
    def test_unique_sampling_is_label_blind_and_balances_families_before_pairs(self):
        rows = [dict(family=family, benchmark=family, subject_id=str(subject), item_id=str(item), response=(item % 2))
                for family, subjects in [("a", 1), ("b", 4)] for subject in range(subjects) for item in range(10)]
        frame = pd.DataFrame(rows)
        sampled = balanced_unique_sample(frame, 16, 7)
        reversed_labels = balanced_unique_sample(frame.assign(response=1-frame.response), 16, 7)
        self.assertEqual(sampled.item_id.tolist(), reversed_labels.item_id.tolist())
        self.assertEqual(sampled.subject_id.tolist(), reversed_labels.subject_id.tolist())
        self.assertEqual(sampled.groupby("family").size().to_dict(), {"a": 8, "b": 8})
        for _, group in sampled.groupby("family"):
            self.assertAlmostEqual(group.sample_weight.sum(), 8.)
        self.assertFalse(sampled.duplicated(["benchmark", "subject_id", "item_id"]).any())
        with self.assertRaises(ValueError):
            balanced_unique_sample(pd.concat([frame, frame.iloc[:1]]), 16, 7)

    def test_expanded_transport_preserves_hashes_and_excludes_sealed_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = EncoderTrainingConfig(prepared="runs/prepared", samples=8)
            (root / "configs").mkdir()
            (root / "configs/recipe.json").write_text(config.model_dump_json())
            (root / "configs/split.json").write_text('{}')
            prepared = root / config.prepared
            (prepared / "enriched").mkdir(parents=True)
            for name in ["enriched/training.jsonl", "native_tokens.jsonl.gz", "native_tokens_manifest.json", "calibration.jsonl", "confirmation.jsonl"]:
                (prepared / name).write_text('fixture')
            manifest = dict(data_config={"split":"configs/split.json"},
                split_sha256=file_hash(root / "configs/split.json"), renderer_sources={}, question=load_questions(),
                artifacts={"enriched/training.jsonl":file_hash(prepared / "enriched/training.jsonl")})
            (prepared / "manifest.json").write_text(json.dumps(manifest))
            job = JobSpec(task="qualify-encoder", encoder_config="configs/recipe.json", out="runs/probe")
            files = input_files(root, job)
            self.assertFalse(any('calibration' in str(p) or 'confirmation' in str(p) for p in files))
            self.assertIn(Path("runs/prepared/native_tokens.jsonl.gz"), files)
            (prepared / "enriched/training.jsonl").write_text('changed')
            with self.assertRaisesRegex(ValueError, "Expanded artifact changed"):
                verify_expanded(root, prepared)
        with self.assertRaises(ValueError):
            JobSpec(task="train-encoder", out="runs/train")
        with self.assertRaises(ValueError):
            command(job, ExecutionProfile(executor="local", backend="mlx", device="mps"), "python")

    def test_grouping_preserves_the_optimizer_window_and_short_final_batch(self):
        batches = grouped_batches([0, 1, 2, 3, 4], [90, 10, 200, 30, 50], 2)
        self.assertEqual(batches, [[1, 3], [4, 0], [2]])
        self.assertEqual(sorted(sum(batches, [])), list(range(5)))
        self.assertEqual(EncoderTrainingConfig().samples, 30000)
        with self.assertRaises(ValueError):
            ComputeSettings(precision="auto", micro_batch=4)
        with self.assertRaisesRegex(ValueError, "gradient scaling"):
            train_window(None, None, None, None, None, None, {"precision":"float16"}, 2)


@unittest.skipUnless(importlib.util.find_spec("torch"), "Torch environment required")
class EncoderPolicyTests(unittest.TestCase):
    def test_zero_lr_fused_optimizer_normalizes_signed_zero_without_numeric_weight_changes(self):
        import torch
        from paiec.encoder_qualification import zero_canonical_weight_hash
        model = torch.nn.Linear(2, 1)
        with torch.no_grad():
            model.weight.copy_(torch.tensor([[-0., 1.]]))
        before = zero_canonical_weight_hash(model)
        optimizer = torch.optim.AdamW(model.parameters(), lr=0., fused=True)
        for value in model.parameters():
            value.grad = -torch.ones_like(value)
        optimizer.step()
        self.assertEqual(zero_canonical_weight_hash(model), before)
        with torch.no_grad():
            model.weight[0, 1] += 1e-6
        self.assertNotEqual(zero_canonical_weight_hash(model), before)

    def test_fp32_scoring_boundary_preserves_export_keys_and_gradients(self):
        import torch
        from paiec.encoder_training import keep_scorer_float32
        model = self.model()
        keys = list(model.state_dict())
        keep_scorer_float32(model)
        result = model.scorer(torch.ones(1, 2, dtype=torch.bfloat16))
        self.assertEqual(result.dtype, torch.float32)
        result.sum().backward()
        self.assertIsNotNone(model.scorer.weight.grad)
        self.assertEqual(list(model.state_dict()), keys)

    def model(self):
        import torch
        model = torch.nn.Module()
        model.encoder = torch.nn.Module()
        model.encoder.layers = torch.nn.ModuleList([torch.nn.Linear(2, 2) for _ in range(4)])
        model.encoder.final_norm = torch.nn.LayerNorm(2)
        model.encoder.embeddings = torch.nn.Linear(2, 2)
        for name in ["head", "type_emb", "scorer", "act_head"]:
            setattr(model, name, torch.nn.Linear(2, 2))
        return model

    def test_policy_unfreezes_only_intended_layers_and_never_detaches_their_gradients(self):
        import torch
        model = self.model()
        set_training_policy(model, 2)
        x = torch.ones(2, 2)
        x = model.encoder.embeddings(x)
        for layer in model.encoder.layers:
            x = layer(x)
        model.scorer(model.encoder.final_norm(x)).sum().backward()
        for name, value in model.named_parameters():
            intended = name.startswith(("head.", "type_emb.", "scorer.", "encoder.layers.2.", "encoder.layers.3.", "encoder.final_norm."))
            self.assertEqual(value.requires_grad, intended, name)
            if name.startswith(("encoder.layers.2.", "encoder.layers.3.")):
                self.assertIsNotNone(value.grad)
            if not intended:
                self.assertIsNone(value.grad)
        self.assertFalse(model.encoder.layers[0].training)
        self.assertTrue(model.encoder.layers[3].training)
        self.assertFalse(model.act_head.training)
        set_training_policy(model, 0)
        self.assertFalse(any(p.requires_grad for p in model.encoder.parameters()))


if __name__ == "__main__":
    unittest.main()
