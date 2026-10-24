import importlib.util
from pathlib import Path
import tempfile
import unittest

from paiec.multilingual_recipe import token_spans, MultilingualRecipe, verify_qualification_inputs
from paiec.execution.jobs import JobSpec, ExecutionProfile, command, input_files
from paiec.feature_preparation import write_json
from paiec.head_training import file_hash


class MultilingualRecipeTests(unittest.TestCase):
    def test_window_coverage_and_weights_include_every_task_token_once(self):
        for size in [1, 200, 8192, 50000]:
            spans = token_spans(size, 7000, 128)
            coverage = set()
            for span in spans:
                coverage.update(range(span["start"], span["end"]))
                self.assertLessEqual(span["end"] - span["start"], 7000)
            self.assertEqual(coverage, set(range(size)))
            self.assertEqual(sum(s["new_tokens"] for s in spans), size)
        for count, capacity, overlap in [(0, 100, 1), (100, 128, 128), (100, 100, -1)]:
            with self.assertRaises(ValueError):
                token_spans(count, capacity, overlap)

    def test_qualification_transport_is_slim_and_cannot_dispatch_a_pilot(self):
        profile = ExecutionProfile(executor="modal", backend="torch", device="cuda")
        job = JobSpec(task="qualify-multilingual", out="runs/qualification", multilingual_config="configs/recipe.json")
        args = command(job, profile, "python")
        self.assertIn("scripts/qualify_multilingual.py", args)
        self.assertNotIn("--train", args)
        with self.assertRaises(ValueError):
            command(job, ExecutionProfile(executor="local", backend="mlx", device="mps"), "python")
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            (root / "configs").mkdir()
            config = MultilingualRecipe(prepared="runs/prepared")
            write_json(root / "configs/recipe.json", config.model_dump())
            prepared = root / config.prepared
            prepared.mkdir(parents=True)
            write_json(prepared / "fixtures.json", {"synthetic": "fixture"})
            write_json(prepared / "manifest.json", {"version": "multilingual_qualification_inputs_v1", "config": config.model_dump(),
                "artifacts": {"fixtures.json": file_hash(prepared / "fixtures.json")}})
            files = input_files(root, job)
            self.assertEqual(set(files), {Path("configs/recipe.json"), Path("runs/prepared/manifest.json"), Path("runs/prepared/fixtures.json")})
            (prepared / "fixtures.json").write_text("changed")
            with self.assertRaisesRegex(ValueError, "fixture changed"):
                verify_qualification_inputs(root, config.model_dump())


@unittest.skipUnless(importlib.util.find_spec("torch"), "Torch needed for gradient reference")
class GroupedGradientTests(unittest.TestCase):
    def test_bounded_memory_replay_matches_joint_loss_and_dropout_rng(self):
        import torch
        from paiec.grouped_learning import backward_group
        torch.manual_seed(7)
        model = torch.nn.Sequential(torch.nn.Linear(3, 8), torch.nn.Dropout(.3), torch.nn.Linear(8, 2))
        views = [torch.randn(1, 3), torch.randn(1, 3), torch.randn(1, 3)]
        weights = [.4, .4, .2]
        before = torch.get_rng_state()
        logits = sum(w * model(v) for w, v in zip(weights, views))
        reference = torch.nn.functional.cross_entropy(logits, torch.tensor([1]))
        reference.backward()
        gradients = [p.grad.clone() for p in model.parameters()]
        after = torch.get_rng_state()
        model.zero_grad(set_to_none=True)
        torch.set_rng_state(before)
        value = backward_group(model, views, weights, 1)
        self.assertAlmostEqual(value, float(reference.detach()), places=6)
        for p, expected in zip(model.parameters(), gradients):
            torch.testing.assert_close(p.grad, expected, atol=1e-6, rtol=1e-5)
        self.assertTrue(torch.equal(torch.get_rng_state(), after))
        with self.assertRaises(ValueError):
            backward_group(model, views, [1., 1., 1.], 1)
