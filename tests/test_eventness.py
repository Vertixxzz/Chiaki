import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from types import SimpleNamespace
import numpy as np

from app.ml.eventness import (UNKNOWN, fit_eventness_model, grouped_video_folds,
                              known_window_labels, localization_error,
                              nearest_event_window, supervised_rows,
                              threshold_tradeoff)


class _Label:
    def __init__(self, violation, timestamp=None):
        self.violation = violation
        self.timestamp = timestamp


def small_data():
    return SimpleNamespace(
        x=np.zeros((6, 10), dtype=np.float32),
        groups=np.array(["1", "1", "1", "2", "2", "2"]),
        starts=np.array([0., 5., 10., 0., 5., 10.]),
        ends=np.array([10., 15., 20., 10., 15., 20.]),
    )


class EventnessTests(unittest.TestCase):
    def test_positive_window_is_nearest_centre(self):
        self.assertEqual(nearest_event_window(np.array([0., 5., 10.]), np.array([10., 15., 20.]), 12.), 1)
        # Equal distances consistently choose the earliest cached window.
        self.assertEqual(nearest_event_window(np.array([0., 10.]), np.array([10., 20.]), 10.), 0)

    def test_unknown_violation_windows_are_not_negatives(self):
        data = small_data(); labels = {"1": _Label(True, 12.), "2": _Label(False)}
        weak = known_window_labels(data, labels)
        self.assertEqual(weak.tolist(), [UNKNOWN, 1, UNKNOWN, 0, 0, 0])
        fit = supervised_rows(data, labels, ["1", "2"])
        self.assertEqual(fit.labels.tolist(), [UNKNOWN, 1, UNKNOWN, 0, 0, 0])
        self.assertEqual(fit.sample_weight[0], 0)

    def test_grouped_folds_have_no_video_leakage(self):
        ids = [str(i) for i in range(8)]
        labels = [0, 0, 0, 0, 1, 1, 1, 1]
        for train, validation in grouped_video_folds(ids, labels, n_splits=2):
            self.assertFalse(set(np.asarray(ids)[train]) & set(np.asarray(ids)[validation]))

    def test_event_score_direction(self):
        x = np.vstack([np.zeros((4, 10)), np.full((4, 10), 5)]).astype(np.float32)
        y = np.array([0] * 4 + [1] * 4)
        model = fit_eventness_model(x, y, np.ones(len(y)), "combined", "logistic_regression")
        self.assertGreater(model.score(np.full((1, 10), 5, dtype=np.float32))[0],
                           model.score(np.zeros((1, 10), dtype=np.float32))[0])

    def test_threshold_selection_uses_only_given_development_scores(self):
        # The function accepts no test labels/scores: changing an imagined
        # holdout cannot affect the selected threshold.
        threshold, rows = threshold_tradeoff(np.array([.1, .2, .8, .9]), np.array([0, 0, 1, 1]), .25)
        self.assertTrue(any(row["threshold"] == threshold for row in rows))
        self.assertTrue(all(row["normal_video_fpr"] <= 1 for row in rows))

    def test_localization_error(self):
        self.assertEqual(localization_error(12.5, 10), 2.5)
        self.assertTrue(np.isnan(localization_error(12.5, None)))


if __name__ == "__main__":
    unittest.main()
