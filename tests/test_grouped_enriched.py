from copy import deepcopy
import importlib.util
import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

from paiec.competition_fields import SUBJECT_FIELDS, ITEM_FIELDS
from paiec.feature_preparation import write_json, write_jsonl
from paiec.grouped_checkpoint import snapshot_collection, renderer_fingerprints, verify_grouped_checkpoint
from paiec.grouped_data import GroupedTrainingConfig, TokenCorpus, family_scope_weights, checked_artifacts
from paiec.grouped_engine import probability_from_logits
from paiec.grouped_inputs import GroupedStateBuilder, SNAPSHOT_FILES, FORMAT, AGGREGATION
from paiec.grouped_predictor import GroupedLayaPredictor
from paiec.head_training import file_hash
from paiec.laya_predictor import load_questions


ROOT = Path(__file__).resolve().parents[1]


def snapshot_fixture(folder):
    folder.mkdir()
    subject = {k: "" for k in SUBJECT_FIELDS}
    subject.update(normalized_name="Synthetic model", subject_features_extra="generation_limit=512")
    item = {k: "" for k in ITEM_FIELDS}
    item.update(item_content="  Can the target compute 三 + two?  ", item_features="literal None null", benchmark_id="anonymous")
    rows = {n: [] for n in SNAPSHOT_FILES}
    rows["systems.jsonl"] = [{"system_key": "subject-key", "recorded": subject, "resolution": {
        "model_profile_id": "model.test", "harness_profile_id": "harness.test", "model_identity_basis": "base_model",
        "exact_evaluated_checkpoint_verified": False}}]
    rows["reviewed_static_features.jsonl"] = [{"profile_id": "model.test", "field": "architecture.total_parameters",
        "fact": {"value": 8000, "status": "documented", "note": "Rounded, not exact."},
        "review_status": "reviewed", "review_disposition": "value_confirmed"}]
    rows["typed_run_features.jsonl"] = [{"system_key": "subject-key", "recorded": subject,
        "features": {"generation_limit": {"value": 512, "status": "recorded"},
                     "enabled_tools": {"value": None, "status": "unknown"}}}]
    rows["qualified_harness_observations.jsonl"] = [{"profile_id": "harness.test", "field": "supported_tools",
        "value": ["Python"], "status": "documented_supported"}]
    for name, data in rows.items():
        write_jsonl(folder / name, data)
    write_json(folder / "manifest.json", {"version": "feature_preparation_v1",
        "artifacts": {name: file_hash(folder / name) for name in rows}})
    return subject, item


def checkpoint_fixture(path, snapshot):
    path.mkdir()
    for name in ["model.safetensors", "rl_agent_config.json", "encoder/config.json", "tokenizer/tokenizer.json"]:
        f = path / name
        f.parent.mkdir(parents=True, exist_ok=True)
        f.write_text("synthetic fixture")
    snapshot_collection(snapshot, path / "collection_snapshot")
    write_json(path / "questions.json", load_questions())
    contract = dict(GroupedTrainingConfig().input_contract(), state_format=FORMAT, aggregation=AGGREGATION,
        source_policy="research_only", model="multilingual", question_sha256=file_hash(ROOT / "configs/laya_question.json"),
        renderer_sources=renderer_fingerprints(), snapshot="collection_snapshot", probability="raw_grouped_softmax", evidence_mode="none")
    write_json(path / "paiec_training.json", {"version": "grouped_enriched_checkpoint_v1", "model": "multilingual",
        "input_contract": contract, "artifact_sha256": {str(f.relative_to(path)): file_hash(f) for f in path.rglob("*") if f.is_file()}})


class GroupedBoundaryTests(unittest.TestCase):
    def test_shared_builder_retains_facts_scope_unknowns_and_literal_task(self):
        with tempfile.TemporaryDirectory() as tmp:
            subject, item = snapshot_fixture(Path(tmp) / "source")
            builder = GroupedStateBuilder(Path(tmp) / "source")
            state = builder.build(subject, item)
            self.assertEqual(state["target_task"]["item_content"], item["item_content"])
            self.assertNotIn("benchmark_id", json.dumps(state))
            c = state["collected_information"]
            self.assertEqual(c["documented_model_or_base"]["facts"]["values_by_status"]["documented"]["architecture.total_parameters"], 8000)
            self.assertEqual(c["documented_model_or_base"]["source_scope_qualifiers"]["architecture.total_parameters"], "Rounded, not exact.")
            self.assertIn("enabled_tools", c["recorded_run_settings"]["unknown_or_unconfirmed"])
            self.assertNotIn("Python", json.dumps(c["recorded_run_settings"]))
            self.assertEqual(builder.build(subject, dict(item, benchmark_id="different")), state)
            with self.assertRaises(ValueError):
                builder.build(dict(subject, display_name="private"), item)
            with self.assertRaises(ValueError):
                builder.build(subject, dict(item, expected=1))
            unknown = builder.build(dict(subject, harness_version="unseen run"), item)
            self.assertEqual(unknown["collected_information"]["documented_model_or_base"]["identity_basis"], "unresolved")
            self.assertEqual(unknown["collected_information"]["recorded_run_settings"]["values_by_status"], {})
            self.assertEqual(unknown["target_task"], state["target_task"])

    def test_weighted_mean_logits_is_not_mean_probability(self):
        logits = [[0., 8.], [0., -2.]]
        p = probability_from_logits(logits, [.2, .8])
        self.assertAlmostEqual(p, .5)
        self.assertGreater(abs(p - (.2 / (1 + math.exp(-8)) + .8 / (1 + math.exp(2)))), .1)
        self.assertEqual(p, probability_from_logits(list(reversed(logits)), [.8, .2]))
        self.assertAlmostEqual(probability_from_logits([[10000., 0.]], [1.]), 0.)
        for weights in [[1., 1.], [float("nan"), .5], [0., 1.], [-1., 2.]]:
            with self.assertRaises(ValueError):
                probability_from_logits(logits, weights)

    def test_checkpoint_tampering_and_contract_overrides_fail_before_native_load(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            snapshot_fixture(root / "source")
            checkpoint_fixture(root / "checkpoint", root / "source")
            verify_grouped_checkpoint(root / "checkpoint")
            for kwargs in [{"backend": "mlx"}, {"max_len": 2048}, {"use_labels": True}, {"evidence_mode": "beta_mean"}]:
                with self.assertRaises(ValueError):
                    GroupedLayaPredictor(checkpoint_path=root / "checkpoint", **kwargs)
            (root / "checkpoint/collection_snapshot/systems.jsonl").write_text("changed")
            with self.assertRaisesRegex(ValueError, "artifact changed"):
                GroupedLayaPredictor(checkpoint_path=root / "checkpoint")

    def test_predictor_uses_one_group_and_ignores_labels_and_scope_in_cache(self):
        class Engine:
            tokenizer = object()
            calls = 0
            def predict_group(self, group):
                self.calls += 1
                return probability_from_logits([[0., 2.], [0., -2.]], group["weights"])
            def close(self):
                pass
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subject, item = snapshot_fixture(root / "source")
            checkpoint_fixture(root / "checkpoint", root / "source")
            engine = Engine()
            predictor = GroupedLayaPredictor(checkpoint_path=root / "checkpoint", engine=engine)
            group = {"views": [{"ids": [1]}, {"ids": [2]}], "weights": [.5, .5]}
            with patch("paiec.grouped_predictor.encode_group", return_value=group) as encode:
                self.assertEqual(predictor.predict([subject, item], labeled=["never read"]), .5)
                self.assertEqual(predictor(subject, dict(item, benchmark_id="other"), labeled=[False]), .5)
            self.assertEqual(engine.calls, 1)
            self.assertEqual(predictor.forward_groups, 1)
            self.assertEqual(predictor.forward_items, 2)
            state = encode.call_args.args[1]
            self.assertNotIn("expected", state)
            self.assertEqual(state["target_task"]["item_content"], item["item_content"])

    def test_family_scope_weights_equal_mass_without_label_balancing(self):
        rows = [{"validation_family": "a", "scope_key": "a1"}] * 3 + [
            {"validation_family": "a", "scope_key": "a2"}, {"validation_family": "b", "scope_key": "b1"}]
        w = family_scope_weights(rows)
        self.assertAlmostEqual(sum(w), len(rows))
        self.assertAlmostEqual(sum(w[:4]), w[4])
        self.assertAlmostEqual(sum(w[:3]), w[3])

    def test_token_corpus_reconstructs_original_groups_and_rejects_bad_boundaries(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            np.array([1, 2, 3, 4, 5, 6], dtype="<i4").tofile(root / "tokens.bin")
            record = {"role": "training", "row_key": "row", "label": 1, "sample_weight": 1.,
                "weights": [.5, .5], "views": [{"offset": 0, "length": 3, "markers": [0, 1]},
                                               {"offset": 3, "length": 3, "markers": [0, 1]}]}
            write_jsonl(root / "groups.jsonl", [record])
            corpus = TokenCorpus(root)
            self.assertEqual(len(corpus), 1)
            self.assertEqual(corpus.group(0)["views"][1]["ids"], [4, 5, 6])
            self.assertEqual(corpus.group(0)["views"][0]["target"], [0., 1.])
            record["views"][1]["offset"] = 2
            write_jsonl(root / "groups.jsonl", [record])
            with self.assertRaisesRegex(ValueError, "offsets"):
                TokenCorpus(root)
            with self.assertRaisesRegex(ValueError, "escaped"):
                checked_artifacts(root, {"../outside": "bad"})

    def test_zenml_routes_are_explicit_and_enriched_only(self):
        from paiec.execution.jobs import JobSpec, ExecutionProfile, command
        from paiec.pipeline_workflows import workflow_plan
        job = JobSpec(task="train-grouped-enriched", out="runs/new")
        args = command(job, ExecutionProfile(executor="modal", backend="torch", device="cuda"), "python")
        self.assertIn("--train", args)
        self.assertNotIn("--variant", args)
        with self.assertRaises(ValueError):
            JobSpec(task="train-grouped-enriched", out="runs/new", variant="baseline")
        with self.assertRaises(ValueError):
            command(job, ExecutionProfile(executor="local", backend="mlx", device="mps"), "python")
        plan = workflow_plan(ROOT, "training", "configs/pipelines/grouped_enriched_training_v1.json", "runs/grouped_plan_test")
        self.assertFalse(plan["execution_started"])
        self.assertFalse(plan["opens_development_outcomes"])
        plan = workflow_plan(ROOT, "inputs", "configs/pipelines/grouped_enriched_preparation_v1.json", "runs/grouped_enriched/prepared_v1")
        self.assertEqual(plan["graph"], "laya_grouped_input_preparation")
        self.assertFalse(plan["may_launch_native_job"])

    def test_execution_uses_grouped_graph_rather_than_historical_input_graph(self):
        from types import SimpleNamespace
        from unittest.mock import Mock
        from paiec.pipeline_workflows import execute_workflow
        with tempfile.TemporaryDirectory() as tmp:
            graph = Mock()
            zenml_setup = SimpleNamespace(configure_environment=Mock(), setup_stack=Mock())
            plan = {"workflow": "inputs", "root": tmp, "out": str(Path(tmp) / "out"),
                    "config": {"version": "grouped_enriched_preparation_workflow_v1"}}
            with patch.dict("sys.modules", {"paiec.zenml_pipeline": zenml_setup}), \
                    patch("importlib.import_module", return_value=SimpleNamespace(laya_grouped_input_preparation=graph)) as modules:
                execute_workflow(plan)
            modules.assert_called_once_with("paiec.grouped_steps")
            graph.assert_called_once()


@unittest.skipUnless(importlib.util.find_spec("torch"), "Native Torch required")
class NativeGroupedTests(unittest.TestCase):
    def tokenizer(self):
        from tokenizers import Tokenizer, models, pre_tokenizers
        from transformers import PreTrainedTokenizerFast
        words = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "false", "true", "noul", "correct", "target", "task", "x", "y", "three"]
        native = Tokenizer(models.WordLevel({w: i for i, w in enumerate(words)}, unk_token="[UNK]"))
        native.pre_tokenizer = pre_tokenizers.Whitespace()
        return PreTrainedTokenizerFast(tokenizer_object=native, pad_token="[PAD]", unk_token="[UNK]", cls_token="[CLS]",
            sep_token="[SEP]", mask_token="[MASK]", model_max_length=8192)

    def test_native_windows_keep_all_task_tokens_and_label_does_not_change_inputs(self):
        from paiec.multilingual_recipe import encode_group
        tok = self.tokenizer()
        state = {"target_system_recorded": {"normalized_name": "Synthetic"}, "target_task": {"item_content": "x " * 800}}
        a = encode_group(tok, state, load_questions(), 0, 256, 96, 8)
        b = encode_group(tok, state, load_questions(), 1, 256, 96, 8)
        self.assertGreater(len(a["views"]), 1)
        self.assertEqual(sum(s["new_tokens"] for s in a["layout"]["task_spans"]), 800)
        self.assertAlmostEqual(sum(a["weights"]), 1.)
        for first, second in zip(a["views"], b["views"]):
            self.assertEqual(first["ids"], second["ids"])
            self.assertEqual(first["markers"], second["markers"])
            self.assertLessEqual(len(first["ids"]), 256)
            self.assertNotEqual(first["target"], second["target"])

    def test_weighted_original_example_gradient_matches_joint_reference(self):
        import torch
        from types import SimpleNamespace
        from paiec.grouped_training import backward_window
        torch.manual_seed(4)
        model = torch.nn.Linear(3, 2)
        engine = SimpleNamespace(view_logits=model, device=torch.device("cpu"))
        groups = [{"views": [torch.randn(3), torch.randn(3)], "weights": [.7, .3], "label": 1},
                  {"views": [torch.randn(3)], "weights": [1.], "label": 0}]
        weights = [2., .5]
        losses = [torch.nn.functional.cross_entropy(sum(w * model(v) for w, v in zip(g["weights"], g["views"])).reshape(1, 2),
                  torch.tensor([g["label"]])) for g in groups]
        reference = sum(w * loss for w, loss in zip(weights, losses)) / sum(weights)
        reference.backward()
        grads = [p.grad.clone() for p in model.parameters()]
        model.zero_grad(set_to_none=True)
        actual = backward_window(engine, groups, weights)
        self.assertAlmostEqual(actual, float(reference.detach()), places=6)
        for p, gradient in zip(model.parameters(), grads):
            torch.testing.assert_close(p.grad, gradient)

    def test_native_checkpoint_predictor_probability_and_token_roundtrip(self):
        import torch
        from transformers import ModernBertConfig, ModernBertModel
        from laya.common import DecisionModel
        from paiec.encoder_training import set_training_policy
        from paiec.grouped_engine import TorchGroupedEngine
        from paiec.grouped_checkpoint import save_grouped_checkpoint
        from paiec.multilingual_recipe import encode_group
        torch.manual_seed(9)
        cfg = ModernBertConfig(vocab_size=32, hidden_size=64, intermediate_size=128, num_hidden_layers=2,
            num_attention_heads=1, max_position_embeddings=8192, pad_token_id=0, cls_token_id=2, sep_token_id=3,
            bos_token_id=2, eos_token_id=3)
        model = DecisionModel(ModernBertModel(cfg), head_layers=1, n_act=1)
        tok = self.tokenizer()
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            subject, item = snapshot_fixture(root / "source")
            state = GroupedStateBuilder(root / "source").build(subject, item)
            engine = TorchGroupedEngine(model, tok)
            group = encode_group(tok, state, load_questions(), 1, 8192)
            set_training_policy(model, 2)
            before_frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
            from paiec.grouped_training import backward_window
            optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=.001)
            backward_window(engine, [group], [1.])
            optimizer.step()
            for name, expected in before_frozen.items():
                self.assertTrue(torch.equal(dict(model.named_parameters())[name], expected))
            expected = engine.predict_group(group)
            save_grouped_checkpoint(model, tok, {"head_layers": 1, "act_costs": {}}, root / "checkpoint", root / "source",
                                    contract=GroupedTrainingConfig().input_contract(), training={"synthetic_test_only": True})
            predictor = GroupedLayaPredictor(checkpoint_path=root / "checkpoint")
            prepared = predictor.prepare_input(subject, item)
            self.assertEqual(prepared["views"][0]["ids"], group["views"][0]["ids"])
            self.assertEqual(prepared["weights"], group["weights"])
            self.assertLess(abs(predictor(subject, item) - expected), .005)
            predictor.close()
