import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest
from types import SimpleNamespace
from unittest.mock import patch, Mock

from paiec.pipeline_workflows import workflow_plan


class FeatureWorkflowTests(unittest.TestCase):
    def test_feature_execution_selects_the_compiler_client_local_stack(self):
        from paiec.pipeline_workflows import execute_workflow
        root = Path(__file__).resolve().parents[1]
        plan = workflow_plan(root, "features", None, "runs/feature-plan-test")
        client = Mock()
        with patch.dict(sys.modules, {
            "paiec.zenml_pipeline": SimpleNamespace(configure_environment=Mock(), setup_stack=Mock(return_value={"stack":"paiec_local"})),
            "zenml.client": SimpleNamespace(Client=Mock(return_value=client)),
            "paiec.feature_foundation_steps": SimpleNamespace(feature_foundation_preparation=Mock()),
        }):
            execute_workflow(plan)
        client.activate_stack.assert_called_once_with("paiec_local")

    def test_feature_plan_is_source_only_and_accepts_external_physical_root(self):
        root = Path(__file__).resolve().parents[1]
        try:
            plan = workflow_plan(root, "features", None, "runs/feature-plan-test", source_root="/not-present/input-root")
        except (KeyError, TypeError):
            self.fail("Full-population feature workflow route is missing")
        self.assertFalse(plan["execution_started"])
        self.assertFalse(plan["may_launch_native_job"])
        self.assertFalse(plan["opens_development_outcomes"])
        self.assertEqual(plan["config"]["expected_counts"]["items"], 1356713)
        self.assertEqual(plan["source_root"], "/not-present/input-root")
        self.assertFalse(Path(plan["out"]).exists())

    def test_external_source_root_cannot_change_other_workflows(self):
        root = Path(__file__).resolve().parents[1]
        try:
            with self.assertRaisesRegex(ValueError, "features"):
                workflow_plan(root, "data", None, "runs/feature-plan-test", source_root="/other")
        except TypeError:
            self.fail("Workflow source-root restriction is missing")

    def test_cli_plan_does_not_initialize_tracking_or_open_data(self):
        root = Path(__file__).resolve().parents[1]
        result = subprocess.run([sys.executable, "scripts/run_pipeline.py", "features", "--source-root", "/not-present/input-root", "--out", "runs/feature-plan-test"], cwd=root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertFalse(json.loads(result.stdout)["execution_started"])

    def test_core_import_needs_no_native_or_tracking_libraries(self):
        root = Path(__file__).resolve().parents[1]
        code = "import sys; sys.modules.update({name:None for name in ['torch','mlx','laya','laya_mlx','transformers','zenml','mlflow']}); import paiec.feature_foundation.build; print('OK')"
        result = subprocess.run([sys.executable, "-c", code], cwd=root, capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
