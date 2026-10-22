import unittest
import pandas as pd
from paiec.head_evaluation import score_frozen_development


class FrozenHeadEvaluationTests(unittest.TestCase):
    def manifest(self):
        return pd.DataFrame([dict(benchmark='a',subject_id='s',item_id='one',role='development',pool='evaluation',exported=True),
                             dict(benchmark='a',subject_id='s',item_id='one',role='development',pool='evaluation',exported=True),
                             dict(benchmark='b',subject_id='t',item_id='two',role='development',pool='evaluation',exported=True)])

    def test_trials_and_equal_pair_weights_are_preserved(self):
        result, items = score_frozen_development(self.manifest(),[0,1,1],[.2,.2,.8])
        # A: mean(.04,.64)=.34. B: .04. Equal pair mean=.19, not response mean=.24.
        self.assertAlmostEqual(result.brier_alc,.19)
        self.assertTrue(all(abs(v-.19)<1e-12 for v in result.per_budget))
        self.assertEqual(items[(items.budget==0)&(items.benchmark=='a')].n_responses.tolist(),[2])

    def test_confirmation_and_acquisition_cannot_enter_scoring(self):
        for key,value in [('role','confirmation'),('pool','acquisition'),('exported',False)]:
            frame = self.manifest()
            frame.loc[0,key]=value
            with self.assertRaisesRegex(ValueError,'Only exported development'):
                score_frozen_development(frame,[0,1,1],[.2,.2,.8])

    def test_invalid_predictions_and_trial_mismatch_are_rejected(self):
        for probabilities in [[.2,float('nan'),.8],[.2,1.1,.8],[.2,.3,.8]]:
            with self.assertRaises(ValueError):
                score_frozen_development(self.manifest(),[0,1,1],probabilities)
