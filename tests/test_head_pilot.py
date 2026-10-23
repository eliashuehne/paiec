import importlib.util
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import pandas as pd

from paiec.head_pilot import HeadPilotConfig, epoch_windows, pair_weights
from tests.fixtures import write_pilot_fixture


class PilotPlanTests(unittest.TestCase):
    def test_single_enriched_cloud_report_is_completed_without_inventing_comparison(self):
        import runpy
        root = Path(__file__).resolve().parents[1]
        render = runpy.run_path(str(root / "scripts/report_head_pilot.py"))["render"]
        with tempfile.TemporaryDirectory() as temporary:
            directory = Path(temporary)
            (directory / "run_plan.json").write_text(json.dumps(dict(variants=["enriched"],
                config=dict(device="cuda", samples=5000, epochs=4, micro_batch=2, accumulation=32))))
            (directory / "comparison.json").write_text(json.dumps(dict(runs=[dict(variant="enriched")],
                enriched_minus_baseline=None)))
            self.assertFalse(render(directory))
            html = (directory / "report.html").read_text()
            self.assertIn("Head training: Completed", html)
            self.assertIn("PyTorch/CUDA", html)
            self.assertIn("Selected variants: enriched.", html)
            self.assertNotIn("Enriched minus baseline", html)

    def test_budget_and_last_partial_window_cover_each_record_once(self):
        config = HeadPilotConfig()
        self.assertEqual(config.updates, 316)
        self.assertEqual(config.effective_batch, 64)
        for epoch in range(config.epochs):
            windows = epoch_windows(5000, 64, 0, epoch)
            flat = [row for window in windows for row in window]
            self.assertEqual(sorted(flat), list(range(5000)))
            self.assertEqual(len(windows[-1]), 8)
            self.assertEqual(windows, epoch_windows(5000, 64, 0, epoch))

    def test_pair_weights_preserve_outcomes_and_equal_pair_mass(self):
        frame = pd.DataFrame([dict(row_id=str(i), benchmark="a", subject_id=subject, outcome=i % 2)
                              for i, subject in enumerate(["one"] * 6 + ["two"] * 2)])
        weights = pair_weights(frame)
        self.assertAlmostEqual(sum(weights[:6]), sum(weights[6:]))
        self.assertEqual(weights, pair_weights(frame.assign(outcome=1 - frame.outcome)))

    def test_pilot_has_an_explicit_bounded_recipe(self):
        self.assertEqual(HeadPilotConfig(device="cuda").device, "cuda")
        for settings in [dict(epochs=5), dict(samples=5001), dict(device="cpu"),
                         dict(max_len=2048), dict(precision="float16")]:
            with self.assertRaises(ValueError):
                HeadPilotConfig(**settings)

    def test_placement_override_preserves_prepared_recipe_and_rejects_optimizer_drift(self):
        from paiec.head_pilot import verify_prepared
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            config = write_pilot_fixture(root).model_copy(update={"device": "cuda", "output": "runs/cloud_fresh"})
            _, manifest = verify_prepared(root, config)
            self.assertEqual(manifest["config"]["device"], "mps")
            with self.assertRaisesRegex(ValueError, "Prepared configuration differs"):
                verify_prepared(root, config.model_copy(update={"learning_rate": .0002}))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch device boundary")
    def test_cuda_memory_and_synchronization_do_not_use_mps(self):
        from unittest.mock import patch
        import torch
        from paiec.head_pilot_training import trim_device, device_memory
        with patch("torch.mps.synchronize") as mps, patch("torch.cuda.synchronize") as cuda, \
             patch("torch.cuda.memory_allocated", return_value=10), \
             patch("torch.cuda.memory_reserved", return_value=20), \
             patch("torch.cuda.max_memory_allocated", return_value=15):
            device = torch.device("cuda")
            trim_device(device)
            self.assertEqual(device_memory(device), dict(cuda_allocated_bytes=10,
                cuda_reserved_bytes=20, cuda_peak_allocated_bytes=15))
            cuda.assert_called_once_with(device)
            mps.assert_not_called()

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch MPS shape check")
    def test_bounded_padding_preserves_all_original_tokens_masks_and_targets(self):
        import torch
        from laya.common import collate_items
        from paiec.head_pilot_training import collate_pilot
        items = [dict(ids=[4, 5, 6], markers=[0, 1], qtype=2, target=[0., 1.]),
                 dict(ids=[4, 9], markers=[0, 1], qtype=2, target=[1., 0.])]
        original = collate_items([items], 0)
        bucketed = collate_pilot(items, 0, 1024)
        self.assertEqual(bucketed["input_ids"].shape, (2, 128))
        self.assertTrue(torch.equal(bucketed["input_ids"][:, :3], original["input_ids"]))
        self.assertTrue(torch.equal(bucketed["attention_mask"][:, :3], original["attention_mask"]))
        self.assertEqual(int(bucketed["attention_mask"][:, 3:].sum()), 0)
        for key in ["marker_pos", "marker_mask", "target", "qtype"]:
            self.assertTrue(torch.equal(bucketed[key], original[key]))
        # Masked padding also preserves real-token attention outputs at inference.
        torch.manual_seed(0)
        embedding = torch.nn.Embedding(10, 4)
        layer = torch.nn.TransformerEncoderLayer(4, 1, 8, dropout=0., batch_first=True).eval()
        with torch.no_grad():
            old = layer(embedding(original["input_ids"]),
                        src_key_padding_mask=~original["attention_mask"].bool())
            new = layer(embedding(bucketed["input_ids"]),
                        src_key_padding_mask=~bucketed["attention_mask"].bool())
        visible = original["attention_mask"].bool()
        self.assertTrue(torch.allclose(old[visible], new[:, :3][visible], atol=1e-6))

    @unittest.skipUnless(importlib.util.find_spec("torch"), "PyTorch export check")
    def test_export_bundles_profile_contract_and_detects_later_drift(self):
        from types import SimpleNamespace
        from unittest.mock import patch
        import torch
        from paiec.head_pilot_training import save_pilot_checkpoint
        from paiec.head_pilot import snapshot_profiles, input_contract
        from paiec.input_contract import resolve_input_contract
        from paiec.laya_runtime import local_checkpoint_files

        root = Path(__file__).resolve().parents[1]
        class Config:
            def save_pretrained(self, folder):
                Path(folder).mkdir(parents=True)
                Path(folder, "config.json").write_text("{}")
        class Tokenizer:
            def save_pretrained(self, folder):
                Path(folder).mkdir(parents=True)
                Path(folder, "tokenizer.json").write_text("{}")
        model = torch.nn.Linear(2, 2)
        model.encoder = SimpleNamespace(config=Config())
        config = HeadPilotConfig()
        with tempfile.TemporaryDirectory() as folder:
            prepared = Path(folder) / "prepared"
            checkpoint = Path(folder) / "checkpoint"
            snapshot_profiles(root / config.policy, prepared, root)
            (prepared / "enriched").mkdir()
            (prepared / "enriched/input_contract.json").write_text(json.dumps(input_contract(prepared, "enriched", config)))
            with patch("paiec.head_pilot_training.normalize_tokenizer_export"):
                save_pilot_checkpoint(model, Tokenizer(), {}, checkpoint, prepared, config, "enriched", {"epoch": 1})
            _, identity = local_checkpoint_files(str(checkpoint), "english")
            self.assertIn("profile_bundle/policy.json", identity["checkpoint_sha256"])
            self.assertEqual(resolve_input_contract(checkpoint)["max_len"], 1024)
            policy = checkpoint / "profile_bundle/policy.json"
            policy.write_text(policy.read_text() + " ")
            with self.assertRaises(ValueError):
                local_checkpoint_files(str(checkpoint), "english")
