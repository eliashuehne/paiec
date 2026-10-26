import unittest

import numpy as np

from paiec.catboost_uncertainty import (clustered_bootstrap, equal_scope, family_jackknife,
                                        margin_assessment, paired_scope_errors, scope_means)


class PairedScopeTests(unittest.TestCase):
    def test_equal_scope_weights_scopes_not_rows(self):
        values = np.asarray([0., 0., 0., 1.])
        scopes = np.asarray(["a", "a", "a", "b"], dtype=object)
        self.assertEqual(scope_means(values, scopes)[0].tolist(), ["a", "b"])
        self.assertAlmostEqual(equal_scope(values, scopes), .5)
        # A pooled mean would be 0.25 here, so the weighting choice is visible.
        self.assertAlmostEqual(float(values.mean()), .25)

    def test_paired_difference_sign_follows_the_candidate(self):
        labels = np.asarray([1., 0., 1., 0.])
        scopes = np.asarray(["a", "a", "b", "b"], dtype=object)
        reference = np.asarray([.5, .5, .5, .5])
        better = paired_scope_errors(np.asarray([.9, .1, .9, .1]), reference, labels, scopes)
        worse = paired_scope_errors(np.asarray([.1, .9, .1, .9]), reference, labels, scopes)
        self.assertLess(equal_scope(better, scopes), 0)
        self.assertGreater(equal_scope(worse, scopes), 0)
        with self.assertRaises(ValueError):
            paired_scope_errors(np.asarray([.5, .5]), reference, labels, scopes)
        with self.assertRaises(ValueError):
            paired_scope_errors(reference, reference, np.asarray([1., 2., 0., 1.]), scopes)
        with self.assertRaises(ValueError):
            paired_scope_errors(reference, reference, np.asarray([np.nan] * 4), scopes)


class BootstrapTests(unittest.TestCase):
    def fixture(self):
        rng = np.random.default_rng(5)
        scopes = np.asarray([f"s{i}" for i in range(40) for _ in range(5)], dtype=object)
        # Every scope is worse by one constant, so the interval cannot cross zero.
        differences = np.full(len(scopes), .02) + rng.normal(scale=1e-6, size=len(scopes))
        return differences, scopes

    def test_seed_determinism_and_interval_shape(self):
        differences, scopes = self.fixture()
        first = clustered_bootstrap(differences, scopes, draws=200, seed=9)
        again = clustered_bootstrap(differences, scopes, draws=200, seed=9)
        other = clustered_bootstrap(differences, scopes, draws=200, seed=10)
        self.assertEqual(first, again)
        self.assertNotEqual(first["lower"], other["lower"])
        self.assertAlmostEqual(first["point"], .02, places=5)
        self.assertLessEqual(first["lower"], first["point"])
        self.assertGreaterEqual(first["upper"], first["point"])
        self.assertEqual(first["scopes"], 40)
        self.assertGreater(first["lower"], 0)
        self.assertEqual(first["share_candidate_worse"], 1.)

    def test_resampling_uses_whole_scopes(self):
        # Two scopes with opposite signs and no within-scope spread: any draw that
        # mixed individual rows would land between the two values and could produce
        # an interval narrower than the gap between scope means.
        differences = np.asarray([-.5] * 10 + [.5] * 10)
        scopes = np.asarray(["a"] * 10 + ["b"] * 10, dtype=object)
        interval = clustered_bootstrap(differences, scopes, draws=500, seed=1)
        self.assertAlmostEqual(interval["point"], 0.)
        self.assertAlmostEqual(interval["lower"], -.5)
        self.assertAlmostEqual(interval["upper"], .5)
        # Both draws land on the same scope a quarter of the time; mixed draws are
        # exactly zero and never count as worse. Monte Carlo noise allows a band.
        self.assertGreater(interval["share_candidate_worse"], .20)
        self.assertLess(interval["share_candidate_worse"], .30)

    def test_invalid_requests_fail_closed(self):
        differences, scopes = self.fixture()
        for kwargs in [{"draws": 0}, {"draws": 2.0}, {"seed": .5}, {"alpha": 0.}, {"alpha": 1.}]:
            with self.assertRaises(ValueError):
                clustered_bootstrap(differences, scopes, **kwargs)
        with self.assertRaises(ValueError):
            clustered_bootstrap(np.asarray([.1, .2]), np.asarray(["a", "a"], dtype=object), draws=10)
        with self.assertRaises(ValueError):
            clustered_bootstrap(np.asarray([.1, np.inf]), np.asarray(["a", "b"], dtype=object), draws=10)


class JackknifeTests(unittest.TestCase):
    def test_leave_one_family_out_recomputes_equal_scope(self):
        differences = np.asarray([.1, .1, .4, .4, -.2])
        scopes = np.asarray(["a1", "a2", "b1", "b2", "c1"], dtype=object)
        families = np.asarray(["a", "a", "b", "b", "c"], dtype=object)
        rows = family_jackknife(differences, scopes, families)
        self.assertAlmostEqual(rows["a"], (.4 + .4 + -.2) / 3)
        self.assertAlmostEqual(rows["b"], (.1 + .1 + -.2) / 3)
        self.assertAlmostEqual(rows["c"], (.1 + .1 + .4 + .4) / 4)
        with self.assertRaises(ValueError):
            family_jackknife(differences, scopes, np.asarray(["a"] * 5, dtype=object))


class MarginAssessmentTests(unittest.TestCase):
    def interval(self, lower, upper, point):
        return {"point": point, "lower": lower, "upper": upper}

    def test_both_conditions_are_required(self):
        clear = margin_assessment(.014, self.interval(.004, .024, .014), {"a": .012, "b": .018})
        self.assertTrue(clear["distinguishable_from_zero"])
        self.assertEqual(clear["reading"], "distinguishable")
        crossing = margin_assessment(.014, self.interval(-.003, .031, .014), {"a": .012, "b": .018})
        self.assertFalse(crossing["distinguishable_from_zero"])
        self.assertEqual(crossing["reading"], "not_distinguishable_at_this_design")
        disagreement = margin_assessment(.014, self.interval(.004, .024, .014), {"a": .012, "b": -.002})
        self.assertTrue(disagreement["interval_excludes_zero"])
        self.assertFalse(disagreement["family_jackknife_same_sign"])
        self.assertFalse(disagreement["distinguishable_from_zero"])
        with self.assertRaises(ValueError):
            margin_assessment(.014, self.interval(.004, .024, .014), {})
        with self.assertRaises(ValueError):
            margin_assessment(np.nan, self.interval(.004, .024, .014), {"a": .01})


if __name__ == "__main__":
    unittest.main()
