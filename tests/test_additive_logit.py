import unittest

import numpy as np
import pandas as pd

from paiec.additive_logit import (DEFAULT_GROUPS, block_columns, checked_penalties, fit_additive,
                                  predict_additive, select_penalties)


def synthetic(rows=400, seed=4):
    rng = np.random.default_rng(seed)
    frame = pd.DataFrame({
        "system_strength": rng.normal(size=rows),
        "task_difficulty": rng.normal(size=rows),
        "harness": rng.choice(["alpha", "beta", "gamma"], size=rows),
        "task_characters_log": rng.normal(size=rows),
    })
    linear = 1.2 * frame.system_strength - 1.5 * frame.task_difficulty
    linear = linear + np.where(frame.harness == "alpha", .4, -.2)
    labels = (rng.random(rows) < 1 / (1 + np.exp(-linear))).astype(float)
    return frame, labels


COLUMNS = {"numeric": ["system_strength", "task_difficulty", "task_characters_log"],
           "categorical": ["harness"],
           "system": ["system_strength", "harness"],
           "task": ["task_difficulty", "task_characters_log"]}


class FitTests(unittest.TestCase):
    def fit(self, frame, labels, **changes):
        options = {"l2_system": 10., "l2_task": 10., "max_iterations": 60, "tolerance": 1e-9}
        options.update(changes)
        return fit_additive(frame, labels, np.ones(len(frame)), COLUMNS, **options)

    def test_fit_recovers_directions_and_converges(self):
        frame, labels = synthetic()
        state = self.fit(frame, labels)
        self.assertLessEqual(state["iterations"], 60)
        weights = dict(zip(state["names"], state["coefficients"]))
        self.assertGreater(weights["system_strength"], 0)
        self.assertLess(weights["task_difficulty"], 0)
        probabilities = predict_additive(state, frame)
        self.assertEqual(probabilities.shape, (len(frame),))
        self.assertTrue(((probabilities > 0) & (probabilities < 1)).all())
        self.assertLess(float(np.mean((probabilities - labels) ** 2)), .25)

    def test_model_is_additive_across_blocks(self):
        frame, labels = synthetic()
        state = self.fit(frame, labels)
        first = frame.iloc[[0, 1]].copy()
        second = frame.iloc[[0, 1]].copy()
        # Change only the task column: the logit difference between the two
        # systems must stay identical because no cross-block term exists.
        first["task_difficulty"] = -1.0
        second["task_difficulty"] = 2.0
        logits = np.log(predict_additive(state, first) / (1 - predict_additive(state, first)))
        shifted = np.log(predict_additive(state, second) / (1 - predict_additive(state, second)))
        self.assertAlmostEqual(float(logits[0] - logits[1]), float(shifted[0] - shifted[1]), places=9)

    def test_block_penalties_act_on_their_own_block(self):
        frame, labels = synthetic()
        strong_system = self.fit(frame, labels, l2_system=1e6, l2_task=10.)
        strong_task = self.fit(frame, labels, l2_system=10., l2_task=1e6)
        system_weights = [abs(value) for name, value in
                          zip(strong_system["names"], strong_system["coefficients"])
                          if COLUMNS["system"] and name in COLUMNS["system"] or name.startswith("harness=")]
        task_weights = [abs(value) for name, value in
                        zip(strong_system["names"], strong_system["coefficients"])
                        if name in COLUMNS["task"]]
        self.assertLess(max(system_weights), .05)
        self.assertGreater(max(task_weights), .2)
        system_weights = [abs(value) for name, value in
                          zip(strong_task["names"], strong_task["coefficients"])
                          if name in COLUMNS["system"] or name.startswith("harness=")]
        task_weights = [abs(value) for name, value in
                        zip(strong_task["names"], strong_task["coefficients"])
                        if name in COLUMNS["task"]]
        self.assertGreater(max(system_weights), .2)
        self.assertLess(max(task_weights), .05)

    def test_unseen_category_falls_back_to_the_reference_level(self):
        frame, labels = synthetic()
        state = self.fit(frame, labels)
        baseline = frame.iloc[[0]].copy()
        baseline["harness"] = "alpha"
        unseen = frame.iloc[[0]].copy()
        unseen["harness"] = "never-seen"
        np.testing.assert_allclose(predict_additive(state, baseline), predict_additive(state, unseen))

    def test_invalid_inputs_fail_closed(self):
        frame, labels = synthetic()
        with self.assertRaises(ValueError):
            fit_additive(frame, labels[:-1], np.ones(len(frame)), COLUMNS,
                         l2_system=1., l2_task=1.)
        with self.assertRaises(ValueError):
            fit_additive(frame, np.full(len(frame), 2.), np.ones(len(frame)), COLUMNS,
                         l2_system=1., l2_task=1.)
        with self.assertRaises(ValueError):
            fit_additive(frame, labels, np.zeros(len(frame)), COLUMNS, l2_system=1., l2_task=1.)
        with self.assertRaisesRegex(ValueError, "did not converge"):
            fit_additive(frame.iloc[:40], labels[:40], np.ones(40), COLUMNS,
                         l2_system=1., l2_task=1., max_iterations=1, tolerance=0.)
        state = self.fit(frame, labels)
        with self.assertRaises(ValueError):
            predict_additive({**state, "coefficients": state["coefficients"][:-1]}, frame)


class SelectionTests(unittest.TestCase):
    def test_penalties_are_selected_on_inner_families_only(self):
        frame, labels = synthetic(rows=240)
        families = np.asarray(["a"] * 60 + ["b"] * 60 + ["c"] * 60 + ["d"] * 60)
        scopes = np.asarray([f"s{index % 8}" for index in range(len(frame))])
        groups = np.asarray([f"g{index}" for index in range(len(frame))])
        grid = checked_penalties([{"l2_system": 10., "l2_task": 10.},
                                 {"l2_system": 500., "l2_task": 10.}])
        outer = families != "d"
        first = select_penalties(frame, labels, scopes, families, groups, outer, grid, COLUMNS,
                                 max_iterations=60, tolerance=1e-9)
        flipped = labels.copy()
        flipped[families == "d"] = 1 - flipped[families == "d"]
        second = select_penalties(frame, flipped, scopes, families, groups, outer, grid, COLUMNS,
                                  max_iterations=60, tolerance=1e-9)
        self.assertEqual(first["selected_config"], second["selected_config"])
        self.assertEqual([row["mean_inner_logloss"] for row in first["candidates"]],
                         [row["mean_inner_logloss"] for row in second["candidates"]])
        self.assertIn(first["selected_index"], (0, 1))


class ValidationTests(unittest.TestCase):
    def test_penalty_grid_and_block_partition(self):
        self.assertEqual(len(checked_penalties([{"l2_system": 1., "l2_task": 2.}])), 1)
        for invalid in [[], [{"l2_system": 1.}], [{"l2_system": 0., "l2_task": 1.}],
                        [{"l2_system": 1., "l2_task": 1.}, {"l2_system": 1., "l2_task": 1.}]]:
            with self.assertRaises(ValueError):
                checked_penalties(invalid)
        columns = block_columns(("system_attributes", "task_structure"))
        self.assertIn("provider", columns["system"])
        self.assertIn("task_characters_log", columns["task"])
        self.assertEqual(set(columns["system"]) & set(columns["task"]), set())
        self.assertEqual(sorted(columns["system"] + columns["task"]), sorted(DEFAULT_GROUPS and
                                                                            columns["numeric"] + columns["categorical"]))


if __name__ == "__main__":
    unittest.main()
