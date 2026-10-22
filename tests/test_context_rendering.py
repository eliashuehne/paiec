import json
import unittest
from urllib.parse import quote
from paiec.laya_predictor import build_state, LayaPredictor
from paiec.state_rendering import split_settings, setting_value
from test_laya_predictor import FakeRouter, SUBJECT, ITEM


class ContextRenderingTests(unittest.TestCase):
    def test_quoted_values_duplicates_unknowns_and_json_semicolons(self):
        text = 'base_model=X;note="a;b=c";config={"x":"a;b=c"};temperature=0;temperature=1;unknown=false;empty='
        entries = split_settings(text)
        self.assertEqual(entries,[('base_model','X'),('note','"a;b=c"'),('config','{"x":"a;b=c"}'),
                                  ('temperature','0'),('temperature','1'),('unknown','false'),('empty','')])
        state = build_state(dict(SUBJECT,subject_features_extra=text), ITEM, [],state_format='context_v1')
        for value in ['base_model: X','note: "a;b=c"','config: {"x":"a;b=c"}',
                      'temperature: 0','temperature: 1','unknown: false','empty: ']:
            self.assertIn(value,state)

    def test_malformed_settings_remain_literal(self):
        for text in ['ordinary description; literal text', 'key="unclosed', 'key={"unclosed":1',
                     'key=x;unparseable remainder','key=]']:
            self.assertIsNone(split_settings(text))
            state = build_state(dict(SUBJECT,subject_features_extra=text),ITEM,[],state_format='context_v1')
            self.assertIn(text,state)

    def test_only_explicit_config_containers_decode_once_and_keep_leaf_strings(self):
        encoded=quote(json.dumps({'model_name':'provider/model+variant','kwargs':{'temperature':0,'enabled':False},
                                  'literal':'%7B%22x%22%3A1%7D'}),safe='')
        value=setting_value('agent_configuration',encoded)
        self.assertIn('[URL-decoded JSON]',value)
        parsed=json.loads(value.removeprefix('[URL-decoded JSON] '))
        self.assertEqual(parsed['model_name'],'provider/model+variant')
        self.assertEqual(parsed['kwargs'],{'temperature':0,'enabled':False})
        self.assertEqual(parsed['literal'],'%7B%22x%22%3A1%7D')
        self.assertEqual(setting_value('unknown',encoded),encoded)
        for raw in ['%7Bbroken','{"x":1,"x":2}',quote('{"x":1,"x":2}',safe='')]:
            self.assertEqual(setting_value('agent_configuration',raw),raw)

    def test_hidden_fields_never_enter_and_task_is_not_reinterpreted(self):
        literal='{"question":"literal %7B%22x%22%3A1%7D;temperature=99"}'
        subject=dict(SUBJECT,provider='',display_name='SECRET',reference_answer='SECRET',subject_features_extra='model_identifier=M')
        item=dict(ITEM,item_content=literal,grading_criterion='SECRET')
        state=build_state(subject,item,[],state_format='context_v1')
        self.assertNotIn('SECRET',state)
        self.assertNotIn('benchmark_',state)
        self.assertIn(literal,state)
        self.assertIn('model_identifier: M',state)
        self.assertNotIn('provider:',state)

    def test_revealed_labels_still_match_before_rendering(self):
        labels=[[[SUBJECT,dict(ITEM,item_content='visible')],0],
                [[dict(SUBJECT,normalized_name='other'),dict(ITEM,item_content='wrong')],1]]
        state=build_state(SUBJECT,ITEM,labels,use_labels=True,state_format='context_v1')
        self.assertIn('outcome: 0',state)
        self.assertIn('visible',state)
        self.assertNotIn('wrong',state)
        self.assertNotIn('Revealed example',build_state(SUBJECT,ITEM,labels,state_format='context_v1'))

    def test_predictor_uses_same_format_on_injected_backend(self):
        router=FakeRouter()
        predictor=LayaPredictor(router=router,state_format='context_v1')
        predictor(SUBJECT,ITEM,[])
        self.assertEqual(router.requests[0]['state'],build_state(SUBJECT,ITEM,[],state_format='context_v1'))
