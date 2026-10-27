"""Main-only integration: no legacy model or orchestration dependencies."""
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import unittest


class MainPortTests(unittest.TestCase):
    def test_core_import_is_independent_of_legacy_models_and_tracking(self):
        code = "import sys; sys.modules.update({k:None for k in ['torch','mlx','transformers','laya','laya_mlx','zenml','mlflow','paiec.head_data','paiec.expanded_data','paiec.state_rendering','paiec.compact_profiles','paiec.profile_enrichment','paiec.system_profiles']}); import paiec.feature_foundation.build; print('OK')"
        result = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True)
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertEqual(result.stdout.strip(), 'OK')

    def test_plan_requires_no_corpus_or_tracking_and_creates_no_output(self):
        with tempfile.TemporaryDirectory() as temp:
            out = Path(temp)/'new'
            result = subprocess.run([sys.executable, 'scripts/prepare_feature_foundation.py', '--source-root', str(Path(temp)/'absent'), '--out', str(out)], capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            plan = json.loads(result.stdout)
            self.assertFalse(plan['execution_started'])
            self.assertEqual(plan['config']['expected_counts']['items'], 1356713)
            self.assertFalse(out.exists())

    def test_cli_builds_and_validates_fixture_without_tracking(self):
        from feature_foundation_fixtures import corpus
        with tempfile.TemporaryDirectory() as root, tempfile.TemporaryDirectory() as dest:
            config=Path(dest)/'recipe.json'
            config.write_text(json.dumps(corpus(root)))
            out=Path(dest)/'bundle'
            build=subprocess.run([sys.executable,'scripts/prepare_feature_foundation.py','--config',str(config),'--source-root',root,'--out',str(out),'--execute'],capture_output=True,text=True)
            self.assertEqual(build.returncode,0,build.stderr)
            self.assertEqual(json.loads(build.stdout)['population']['responses'],9)
            validation=subprocess.run([sys.executable,'scripts/validate_feature_foundation.py','--bundle',str(out)],capture_output=True,text=True)
            self.assertEqual(validation.returncode,0,validation.stderr)
            self.assertTrue(json.loads(validation.stdout)['valid'])

    def test_historical_content_normalization_is_preserved(self):
        from paiec.feature_foundation.text import normalized_content
        self.assertEqual(normalized_content('  FULL-width Ａ + 2! '), 'full width a 2')
        self.assertEqual(normalized_content('[{"role":"user","content":"First"},{"role":"assistant","content":"reply"},{"role":"user","content":"Last task!"}]'), 'last task')
        self.assertEqual(normalized_content('{"context":"Keep me","question":"Q"}'), 'context keep me question q')

    def test_matching_uses_supplied_identifiers_and_rejects_conflicts(self):
        from paiec.feature_foundation.matching import MatchingPolicy, ProfileMatcher
        policy = MatchingPolicy.model_validate({'models':[{'profile_id':'m1','identifiers':['Model One']},{'profile_id':'m2','identifiers':['Model Two']}], 'harnesses':[{'profile_id':'h','identifiers':['Runner']}], 'scoped_aliases':[]})
        export = {'profiles': {'m1':{'entity_type':'model'},'m2':{'entity_type':'model'},'h':{'entity_type':'harness'}}, 'sources':{}}
        matcher = ProfileMatcher(policy, export)
        self.assertEqual(matcher.resolve({'normalized_name':'Model One','harness':'Runner'})['model_profile_id'],'m1')
        self.assertIsNone(matcher.resolve({'display_name':'Model One'})['model_profile_id'])
        self.assertIsNone(matcher.resolve({'normalized_name':'Model One','subject_features_extra':'model_identifier=Model Two'})['model_profile_id'])
        self.assertIsNone(matcher.resolve({'subject_features_extra':'model_identifier=unknown;base_model=Model One'})['model_profile_id'])
        self.assertEqual(matcher.resolve({'subject_features_extra':'base_model=Model One'})['model_identity_basis'],'base_model')

    def test_scoped_alias_needs_pinned_allowed_evidence_and_exact_unversioned_scope(self):
        from paiec.feature_foundation.matching import MatchingPolicy, ProfileMatcher
        raw={'models':[], 'harnesses':[], 'scoped_aliases':[{'harness':'Runner','source_model':'short','profile_id':'m','basis':'base_family','evidence':[{'source_id':'s','locator':'dispatch','excerpt':'source'}], 'note':'Reviewed base family'}]}
        policy=MatchingPolicy.model_validate(raw)
        export={'profiles':{'m':{'entity_type':'model'}},'sources':{'s':{'eligibility':'unreviewed','sha256':'a'*64,'revision':'r','source_type':'harness_code'}}}
        matcher=ProfileMatcher(policy,export)
        subject={'harness':'Runner','subject_features_extra':'source_model=short'}
        self.assertEqual(matcher.resolve(subject)['model_identity_basis'],'documented_harness_scoped_base_family')
        for extra in [{'normalized_name':'Other'},{'harness_version':'v1'},{'harness':'Other'}]:
            self.assertIsNone(matcher.resolve({**subject,**extra})['model_profile_id'])
        export['sources']['s']['eligibility']='not_permitted'
        with self.assertRaises(ValueError): ProfileMatcher(policy,export)

    def test_scoped_alias_rejects_malformed_source_qualification(self):
        from paiec.feature_foundation.matching import MatchingPolicy, ProfileMatcher
        policy=MatchingPolicy.model_validate({'models':[], 'harnesses':[], 'scoped_aliases':[{'harness':'Runner','source_model':'short','profile_id':'m','evidence':[{'source_id':'s','locator':'dispatch','excerpt':'source'}], 'note':'Reviewed identity'}]})
        good={'eligibility':'unreviewed','sha256':'a'*64,'revision':'r','source_type':'harness_code'}
        for bad in [{'sha256':'not-a-hash'},{'sha256':123},{'revision':123},{'revision':''},{'eligibility':'unsupported'},{'source_type':'unsupported'}]:
            with self.subTest(source=bad), self.assertRaises(ValueError):
                ProfileMatcher(policy,{'profiles':{'m':{'entity_type':'model'}},'sources':{'s':{**good,**bad}}})
