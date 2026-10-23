import json
import math
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from paiec.task_inputs import TaskStateBuilder, extract_features, structured
from paiec.evidence import evidence_counts, EvidencePredictor, update_probability


class CharacterTokenizer:
    mask_token = "[MASK]"
    def __call__(self, text, **kwargs):
        return {"input_ids": list(text.encode("utf-8"))}


class TaskPipelineTests(unittest.TestCase):
    def setUp(self):
        self.builder = TaskStateBuilder(CharacterTokenizer(), 950)
        self.subject = {"harness": "test", "subject_features_extra": "model_identifier=model-a;n_shots=5"}

    def test_distinct_cases_survive_a_common_long_instruction_prefix(self):
        def item(case):
            return {"item_content": json.dumps({"messages": [
                {"role":"system", "content":"Generic instructions. "*800},
                {"role":"user", "content":"Shared preamble. "*500+case}]})}
        a = self.builder.build(self.subject, item("A constitutional complaint?"))
        b = self.builder.build(self.subject, item("A criminal liability dispute?"))
        self.assertNotEqual(a.state,b.state)
        self.assertIn("constitutional complaint",a.state)
        self.assertIn("criminal liability",b.state)
        self.assertTrue(a.visibility["task_omitted"])
        self.assertLessEqual(self.builder.count(a.state),950)

    def test_sql_question_and_unknown_fields_remain_visible(self):
        question = "Which schools have free meals?"
        item = {"item_content":json.dumps({"schema":"CREATE TABLE x (a int);"*300,
            "question":question,"evidence":"recorded hint", "unfamiliar":False})}
        built = self.builder.build(self.subject,item)
        self.assertIn(question,built.state)
        self.assertIn("false",built.state)
        self.assertTrue(built.visibility["task_omitted"])

    def test_projection_blocks_private_fields_and_benchmark_identity(self):
        built = self.builder.build({"harness":"H", "display_name":"SECRET MODEL", "outcome":1},
            {"item_content":"中文问题\nA: 一\nB: 二", "benchmark_id":"SECRET BENCHMARK", "reference":"SECRET ANSWER"})
        self.assertNotIn("SECRET",built.state)
        self.assertTrue(built.features["system"]["identity_unknown"])
        self.assertEqual(built.features["task"]["script_hint"],"cjk")
        self.assertEqual(built.features["task"]["option_count_hint"],2)
        self.assertIn("中文问题",built.state)

    def test_duplicate_json_keys_and_literals_are_not_silently_decoded(self):
        raw='{"question":"one","question":"two"}'
        self.assertEqual(structured(raw),raw)
        self.assertEqual(structured("false"),"false")
        self.assertIn(raw,self.builder.build(self.subject,{"item_content":raw}).state)

    def test_large_structures_fall_back_to_literal_with_explicit_loss_flags(self):
        raw=json.dumps({f"attribute_{i}":"value" for i in range(300)})
        built=self.builder.build(self.subject,{"item_content":raw})
        self.assertTrue(built.visibility["structure_fallback"])
        self.assertTrue(built.visibility["task_omitted"])
        self.assertLessEqual(self.builder.count(built.state),950)

    def test_feature_extraction_does_not_change_when_private_labels_change(self):
        item={"item_content":"What?", "reference":"A"}
        self.assertEqual(extract_features(self.subject,item),
                         extract_features(dict(self.subject,outcome=1),dict(item,reference="B",label=0)))

    def test_evidence_matching_excludes_other_profiles_benchmarks_and_target(self):
        target={"item_content":"target", "benchmark_id":"b"}
        labels=[[[self.subject,dict(target,item_content="one")],1],
                [[self.subject,dict(target,item_content="two")],0],
                [[self.subject,target],1],
                [[dict(self.subject,harness="other"),dict(target,item_content="three")],1],
                [[self.subject,dict(target,benchmark_id="other",item_content="four")],1]]
        self.assertEqual(evidence_counts(self.subject,target,labels),(2,1))
        self.assertEqual(evidence_counts(self.subject,{"item_content":"target"},labels),(0,0))
        self.assertAlmostEqual(update_probability(.5,7,6,4),8/11)
        for strength in [0,-1,math.nan,math.inf]:
            with self.assertRaises(ValueError):update_probability(.5,0,0,strength)

    def test_numeric_evidence_changes_predictions_without_changing_base_inputs(self):
        class Base:
            use_labels=False
            runtime_metadata={}
            calls=[]
            def predict_many(self,subject,items,labeled):
                self.calls.append(labeled)
                return [.5]*len(items)
        base=Base(); predictor=EvidencePredictor(base,4)
        target={"item_content":"target","benchmark_id":"b"}
        labels=[[[self.subject,dict(target,item_content="example")],1]]
        self.assertEqual(predictor.predict([self.subject,target],[]),.5)
        self.assertEqual(predictor.predict([self.subject,target],labels),.6)
        self.assertEqual(base.calls,[[],[]])

    def test_task_runtime_uses_native_room_and_logs_loss_separately(self):
        from paiec.task_predictor import TaskLayaPredictor
        class Runtime:
            agent=SimpleNamespace(tok=CharacterTokenizer())
            metadata={"backend":"mlx", "device":"cpu", "packages":{}}
            calls=[]
            def predict_batch(self,requests,batch_size):
                self.calls.append(requests)
                return [{"answers":{"correct":{"noul":.4}},"usage":{"truncated":False}} for _ in requests]
        runtime=Runtime()
        with patch("paiec.task_predictor.native_budget",return_value=(len,950)):
            p=TaskLayaPredictor(max_len=1024,router=runtime)
        item={"item_content":"question "*1000}
        self.assertEqual(p.predict([self.subject,item],[]),.4)
        self.assertEqual(p.predict([self.subject,item],[]),.4)
        self.assertEqual(p.forward_items,1)
        self.assertEqual(p.visibility_counts["task_omitted"],2)
        self.assertEqual(p.truncated_items,0)
        self.assertLessEqual(p.builder.count(runtime.calls[0][0]["state"]),950)

    def test_unicode_control_keeps_chinese_and_respects_pair_weights(self):
        import numpy as np
        from paiec.feature_control import feature_vector, FeatureRidge
        a=[{}, {"item_content":"中文甲"}]; b=[{}, {"item_content":"中文乙"}]
        self.assertFalse(np.array_equal(feature_vector(*a),feature_vector(*b)))
        p=FeatureRidge()
        with self.assertRaises(RuntimeError):p(*a,[])
        p.fit([a,a,b],[1,1,0],[.5,.5,1.])
        self.assertAlmostEqual(p.intercept,.5)
        self.assertGreater(p(*a,[]),p(*b,[]))

    def test_execution_command_keeps_evidence_separate_from_renderer(self):
        from paiec.execution.jobs import JobSpec, ExecutionProfile, command
        job=JobSpec(task="evaluate",out="runs/future_eval",state_format="task_v2",
                    evidence_mode="beta_mean",evidence_strength=4.)
        args=command(job,ExecutionProfile(executor="local",backend="torch",device="mps"),"python")
        self.assertIn("task_v2",args)
        self.assertIn("--laya-evidence",args)
        self.assertEqual(args[args.index("--laya-evidence-strength")+1],"4.0")

    def test_shared_evidence_does_not_reveal_private_subject_grouping(self):
        import pandas as pd
        from paiec.pairs import Pair
        from paiec.evaluate import evaluate
        ids=tuple(str(i) for i in range(80))
        pairs=[Pair(benchmark="b",subject_id=name,subject={"harness":"indistinguishable"},
            items=pd.DataFrame({"content":ids,"item_features":[""]*80,"interactors":[""]*80},index=ids),
            responses=pd.DataFrame({"item_id":ids,"label":[label]*80}),item_ids=ids)
            for name,label in [("private-a",0),("private-b",1)]]
        seen=[]
        def predictor(subject,item,labeled):
            seen.append(labeled)
            return .5
        result=evaluate(pairs,predictor,shared_evidence=True,acquisition_items=31,evaluation_items=1)
        self.assertEqual(result.brier_alc,.25)
        self.assertEqual(len(seen[1]),2)
        self.assertEqual({entry[1] for entry in seen[1]},{0,1})
        self.assertEqual(seen[1],seen[7])

    def test_input_mismatch_is_rejected_before_loading_runtime(self):
        import tempfile
        from pathlib import Path
        from paiec.task_predictor import TaskLayaPredictor, make_laya_predictor
        with tempfile.TemporaryDirectory() as directory:
            (Path(directory)/"paiec_training.json").write_text('{"model":"english"}')
            with patch("paiec.task_predictor.create_runtime") as loader:
                with self.assertRaisesRegex(ValueError,"matching task-v2 checkpoint"):
                    TaskLayaPredictor(checkpoint_path=directory)
                loader.assert_not_called()
        with patch("paiec.task_predictor.create_runtime") as loader:
            with self.assertRaises(ValueError):make_laya_predictor(state_format="task_v2",evidence_mode="beta_mean",evidence_strength=-1)
            loader.assert_not_called()


if __name__ == "__main__":
    unittest.main()
