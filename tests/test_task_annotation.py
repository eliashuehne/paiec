import json
from pathlib import Path
import tempfile
import unittest

from paiec.feature_preparation import read_jsonl, stable_hash, write_json, write_jsonl
from paiec.head_training import file_hash
from paiec.task_annotation import (Facets, compile_response, prepare_requests,
    import_responses, response_schema, export_openai_batch)
from paiec.task_annotation_batch import checked_batch, import_openai_batch, response_annotation


class TaskAnnotationTests(unittest.TestCase):
    def task(self, text="猫 2 + 2? 猫"):
        item = {"item_content": text, "item_features": "", "interactors": ""}
        return {"task_key": stable_hash(item), "recorded": item}

    def contract(self):
        taxonomy = {"facets": {name: ["fixture"] for name in Facets.model_fields if name != "language"}}
        return {"version": "fixture", "instructions": "Annotate; do not solve.",
                "taxonomy": taxonomy, "response_schema": response_schema(taxonomy)}

    def response(self, task):
        facets = {name: {"status": "unknown", "values": [], "evidence": [],
                        "rationale": "The supplied item does not establish this facet."} for name in Facets.model_fields}
        facets["skills"] = {"status": "estimated", "values": ["fixture"], "rationale": "A grounded fixture requirement.",
                            "evidence": [{"field": "item_content", "quote": "猫", "occurrence": 1}]}
        return {"schema_version": "task_annotation_v2", "task_key": task["task_key"], "facets": facets, "difficulty": None}

    def plan(self, root, tasks):
        return prepare_requests(tasks, [{"role": "training", "task_key": t["task_key"]} for t in tasks], self.contract(), root)

    def test_unicode_evidence_is_exact_and_bound_to_original_fields(self):
        task = self.task()
        raw = self.response(task)
        result = compile_response(raw, task, self.contract()["taxonomy"])
        span = result["facets"]["skills"]["evidence"][0]
        self.assertEqual(span["start"], task["recorded"]["item_content"].rfind("猫"))
        self.assertEqual(task["recorded"]["item_content"][span["start"]:span["end"]], "猫")
        self.assertEqual(result["review_status"], "candidate")
        self.assertNotIn("rationale", result["feature_projection"]["skills"])
        for mutation in ["absent quote", "different task"]:
            bad = json.loads(json.dumps(raw))
            if mutation == "absent quote":
                bad["facets"]["skills"]["evidence"][0]["quote"] = "UNSUPPLIED"
            else:
                bad["task_key"] = "a" * 64
            with self.assertRaises(ValueError):
                compile_response(bad, task, self.contract()["taxonomy"])

    def test_answers_difficulty_self_approval_and_unknown_values_are_rejected(self):
        task = self.task()
        raw = self.response(task)
        for extra in [{"difficulty": .9}, {"answer": "SECRET"}, {"review_status": "reviewed"}]:
            with self.assertRaises(ValueError):
                compile_response(dict(raw, **extra), task, self.contract()["taxonomy"])
        raw["facets"]["domain"]["values"] = ["fixture"]
        with self.assertRaises(ValueError):
            compile_response(raw, task, self.contract()["taxonomy"])

    def test_requests_preserve_input_and_exclude_outcomes_and_private_metadata(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task = self.task(' literal \\n and Unicode 猫\n ')
            self.plan(root / "first", [dict(task, expected=True, benchmark_id="PRIVATE")])
            self.plan(root / "second", [dict(task, expected=False, benchmark_id="OTHER")])
            self.assertEqual((root / "first/requests.jsonl").read_bytes(), (root / "second/requests.jsonl").read_bytes())
            request = next(read_jsonl(root / "first/requests.jsonl"))
            self.assertEqual(request["item"], task["recorded"])
            self.assertNotIn("expected", request)
            self.assertNotIn("benchmark_id", request)
            polluted = self.task()
            polluted["recorded"]["reference_answer"] = "SECRET"
            with self.assertRaises(ValueError):
                self.plan(root / "polluted", [polluted])

    def test_sealed_and_cross_role_tasks_cannot_enter_a_plan(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task = self.task()
            for roles in [["confirmation"], ["training", "development"]]:
                with self.subTest(roles=roles), self.assertRaises(ValueError):
                    prepare_requests([task], [{"role": r, "task_key": task["task_key"]} for r in roles],
                                     self.contract(), root / roles[-1])

    def test_partial_import_is_not_full_coverage_and_duplicates_are_rejected(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tasks = [self.task(), self.task("other 猫 猫")]
            self.plan(root / "plan", tasks)
            request = next(read_jsonl(root / "plan/requests.jsonl"))
            envelope = {"request_id": request["request_id"], "response": self.response(tasks[0])}
            result = import_responses(root / "plan", [envelope], root / "import", producer="fixture")
            self.assertEqual((result["candidate_annotations"], result["pending"]), (1, 1))
            self.assertFalse(result["response_coverage_complete"])
            self.assertFalse(result["training_ready"])
            with self.assertRaisesRegex(ValueError, "Duplicate"):
                import_responses(root / "plan", [envelope, envelope], root / "duplicate", producer="fixture")

    def test_changed_plan_and_foreign_export_selection_fail_before_publication(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task = self.task()
            self.plan(root / "plan", [task])
            with self.assertRaisesRegex(ValueError, "absent"):
                export_openai_batch(root / "plan", root / "foreign.jsonl", task_keys=["b" * 64])
            self.assertFalse((root / "foreign.jsonl").exists())
            result = export_openai_batch(root / "plan", root / "pilot.jsonl", task_keys=[task["task_key"]])
            self.assertFalse(result["submitted"])
            body = next(read_jsonl(root / "pilot.jsonl"))["body"]
            self.assertEqual(json.loads(body["input"])["item"], task["recorded"])
            self.assertTrue(body["text"]["format"]["strict"])
            write_json(root / "plan/contract.json", {"version": "changed"})
            with self.assertRaisesRegex(ValueError, "plan changed"):
                export_openai_batch(root / "plan", root / "changed.jsonl", task_keys=[task["task_key"]])

    def provider_result(self, key, annotation):
        return {"custom_id": key, "error": None, "response": {"status_code": 200, "body": {
            "model": "fixture-model-snapshot", "status": "completed", "error": None,
            "usage": {"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
            "output": [{"type": "message", "status": "completed", "role": "assistant",
                        "content": [{"type": "output_text", "text": json.dumps(annotation)}]}]}}}

    def test_unordered_batch_results_failures_and_missing_rows_preserve_gaps(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            tasks = [self.task(str(i) + " 猫 猫") for i in range(4)]
            self.plan(root / "plan", tasks)
            export_openai_batch(root / "plan", root / "batch.jsonl", task_keys=[t["task_key"] for t in tasks])
            requests = list(read_jsonl(root / "plan/requests.jsonl"))
            good = self.provider_result(requests[0]["request_id"], self.response(tasks[0]))
            bad_schema = self.provider_result(requests[1]["request_id"], dict(self.response(tasks[1]), answer="UNWANTED"))
            failure = {"custom_id": requests[2]["request_id"], "response": None, "error": {"code": "batch_expired"}}
            write_jsonl(root / "output.jsonl", [bad_schema, good])
            write_jsonl(root / "errors.jsonl", [failure])
            summary = import_openai_batch(root / "plan", root / "batch.jsonl", [root / "output.jsonl", root / "errors.jsonl"],
                                          root / "imported", batch_sha256=file_hash(root / "batch.jsonl"))
            self.assertEqual((summary["candidate_annotations"], summary["rejected"], summary["pending"]), (1, 1, 3))
            self.assertEqual((summary["provider_failures"], summary["missing_results"]), (1, 1))
            self.assertEqual(summary["observed_usage"]["total_tokens"], 30)
            self.assertFalse(summary["training_ready"])
            self.assertFalse(summary["network_origin_verified"])
            for path in (root / "imported").rglob("*.jsonl"):
                self.assertNotIn("UNWANTED", path.read_text())

    def test_batch_import_rejects_foreign_and_duplicate_results_before_publication(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task = self.task()
            self.plan(root / "plan", [task])
            export_openai_batch(root / "plan", root / "batch.jsonl", task_keys=[task["task_key"]])
            key = next(read_jsonl(root / "plan/requests.jsonl"))["request_id"]
            good = self.provider_result(key, self.response(task))
            for name, rows in [("duplicate", [good, good]), ("foreign", [dict(good, custom_id="foreign")])]:
                write_jsonl(root / (name + ".jsonl"), rows)
                with self.subTest(name=name), self.assertRaisesRegex(ValueError, "Foreign or duplicate"):
                    import_openai_batch(root / "plan", root / "batch.jsonl", [root / (name + ".jsonl")], root / name,
                                        batch_sha256=file_hash(root / "batch.jsonl"))
                self.assertFalse((root / name).exists())

    def test_batch_cannot_change_prompt_tools_input_or_hash(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            task = self.task()
            self.plan(root / "plan", [task])
            export_openai_batch(root / "plan", root / "batch.jsonl", task_keys=[task["task_key"]])
            original_hash = file_hash(root / "batch.jsonl")
            original = next(read_jsonl(root / "batch.jsonl"))
            for field, value in [("input", "changed"), ("instructions", "Solve it"), ("tools", [{"type": "web_search"}])]:
                row = json.loads(json.dumps(original))
                row["body"][field] = value
                write_jsonl(root / "changed.jsonl", [row])
                with self.subTest(field=field), self.assertRaisesRegex(ValueError, "contract"):
                    checked_batch(root / "plan", root / "changed.jsonl", file_hash(root / "changed.jsonl"))
                with self.assertRaisesRegex(ValueError, "hash"):
                    checked_batch(root / "plan", root / "changed.jsonl", original_hash)

    def test_refusal_incomplete_tools_and_duplicate_json_are_not_annotations(self):
        body = self.provider_result("fixture", self.response(self.task()))["response"]["body"]
        variants = []
        variants.append(dict(body, status="incomplete"))
        variants.append(dict(body, output=[{"type": "function_call"}]))
        variants.append(dict(body, output=body["output"] * 2))
        for content in [[{"type": "refusal", "refusal": "No"}], [{"type": "output_text", "text": '{"a":1,"a":2}'}]]:
            variants.append(dict(body, output=[dict(body["output"][0], content=content)]))
        for variant in variants:
            with self.assertRaises(ValueError):
                response_annotation(variant)
