import unittest

from paiec.acquisition import POLICIES, random_policy, streamed_evidence, uncertainty_policy
from paiec.evaluate import evaluate, pair_split, score_pair
from paiec.metrics import BUDGETS
from paiec.split import first_labels

from test_evaluate import constant_predictor, make_pair


def bought(pair, policy, predict=constant_predictor(0.5), **caps):
    split = pair_split(pair, **caps)
    evidence = streamed_evidence(pair, split, predict, policy, first_labels(pair.responses))
    return split, {budget: [entry[0][1]["item_content"] for entry in labeled]
                   for budget, labeled in evidence.items()}, evidence


class StreamedEvidenceTests(unittest.TestCase):
    def test_random_fills_every_budget_with_nested_acquisition_items(self):
        pair = make_pair()
        split, _, evidence = bought(pair, random_policy)
        acquisition = {pair.item(i)["item_content"] for i in split.acquisition}
        previous = []
        for budget in BUDGETS:
            labeled = evidence[budget]
            self.assertEqual(len(labeled), budget)
            self.assertEqual(labeled[: len(previous)], previous)
            self.assertTrue({entry[0][1]["item_content"] for entry in labeled} <= acquisition)
            previous = labeled

    def test_random_is_reproducible_and_not_the_fixed_order(self):
        pair = make_pair(n=200)
        first, second = bought(pair, random_policy)[2], bought(pair, random_policy)[2]
        self.assertEqual(first[31], second[31])
        split = pair_split(pair)
        fixed = [pair.item(i) for i in split.acquisition[:31]]
        self.assertNotEqual([entry[0][1] for entry in first[31]], fixed)

    def test_uncertainty_buys_only_uncertain_items_when_there_are_enough(self):
        pair = make_pair(n=400)
        # Item content is its label: items labelled 1 look uncertain, 0 look certain.
        def predict(subject, item, labeled):
            return 0.55 if item["item_content"] == "1" else 0.9

        _, contents, _ = bought(pair, uncertainty_policy(0.1), predict)
        self.assertEqual(len(contents[31]), 31)
        self.assertEqual(set(contents[31]), {"1"})

    def test_uncertainty_still_fills_the_budget_when_nothing_is_uncertain(self):
        _, contents, _ = bought(make_pair(), uncertainty_policy(0.1), constant_predictor(0.99),
                                acquisition_items=40)
        self.assertEqual([len(contents[b]) for b in BUDGETS], list(BUDGETS))

    def test_predictions_see_the_labels_bought_so_far(self):
        sizes = []

        def recorder(subject, item, labeled):
            sizes.append(len(labeled))
            return 0.5

        bought(make_pair(), uncertainty_policy(0.1), recorder)
        self.assertEqual(sorted(set(sizes)), list(range(31)))


class PolicyEvaluationTests(unittest.TestCase):
    def test_policies_score_like_the_fixed_order_for_a_constant_predictor(self):
        pair = make_pair()
        for name in POLICIES:
            rows = score_pair(pair, constant_predictor(0.5), policy=POLICIES[name](),
                              evaluation_items=50)
            self.assertTrue(all(abs(row["brier"] - 0.25) < 1e-12 for row in rows))

    def test_policy_with_pooling_is_rejected(self):
        with self.assertRaises(ValueError):
            evaluate([make_pair()], constant_predictor(0.5), pool_size=2, policy=random_policy)


if __name__ == "__main__":
    unittest.main()
