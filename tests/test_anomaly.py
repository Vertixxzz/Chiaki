import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from app.ml.anomaly import aggregate_scores, fit_normal_model, percentile_threshold
from app.ml.anomaly_data import make_split
from app.data.dataset import VideoLabel

class AnomalyTests(unittest.TestCase):
    def setUp(self):
        self.normal = np.vstack([np.zeros((12, 10)), np.full((12, 10), .05)]).astype(np.float32)

    def test_normal_only_model_scores_far_sample_higher(self):
        model = fit_normal_model(self.normal, 'combined', 'knn', pca_components=4)
        near = np.zeros((1, 10), dtype=np.float32); far = np.full((1, 10), 10, dtype=np.float32)
        self.assertGreater(model.score(far)[0], model.score(near)[0])

    def test_threshold_is_normal_validation_percentile(self):
        self.assertEqual(percentile_threshold(np.array([1., 2., 3., 4.]), 50), 2.5)

    def test_temporal_aggregations(self):
        values = np.array([.1, .2, .9, .8, .3])
        self.assertEqual(aggregate_scores(values, 'max'), .9)
        self.assertAlmostEqual(aggregate_scores(values, 'top3_mean'), (0.9 + .8 + .3) / 3)
        self.assertAlmostEqual(aggregate_scores(values, 'top5_mean'), values.mean())

    def test_split_keeps_test_videos_out_of_normal_training(self):
        videos = [VideoLabel(str(i), Path(f'{i}.mov'), False, None, f'X-{i}', None) for i in range(1, 9)]
        videos += [VideoLabel('9', Path('9.mov'), True, 3., 'X-9', '3 секунда')]
        split = make_split(videos, {'1', '9'})
        self.assertNotIn('1', split.normal_train_ids)
        self.assertNotIn('1', split.normal_validation_ids)
        self.assertNotIn('9', split.normal_train_ids)
        self.assertFalse(set(split.normal_train_ids) & set(split.normal_validation_ids))

if __name__ == '__main__': unittest.main()
