import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.data.dataset import parse_timestamp
from app.video.windows import inference_windows, labeled_windows

class WindowGridTests(unittest.TestCase):
    def test_timestamp_formats(self):
        self.assertEqual(parse_timestamp('22 секунда'), 22)
        self.assertEqual(parse_timestamp('1минута 8 секунда'), 68)
        self.assertEqual(parse_timestamp('1 минута 8 секунд'), 68)

    def test_normal_and_event_videos_share_identical_grid(self):
        normal = labeled_windows(40, None, 10, 5, 2)
        event = labeled_windows(40, 22, 10, 5, 2)
        self.assertEqual([(w.start, w.end) for w in normal], [(w.start, w.end) for w in event])
        self.assertEqual([(w.start, w.end) for w in normal], inference_windows(40, 10, 5))

    def test_timestamp_changes_labels_not_boundaries(self):
        first = labeled_windows(40, 12, 10, 5, 2)
        second = labeled_windows(40, 22, 10, 5, 2)
        self.assertEqual([(w.start, w.end) for w in first], [(w.start, w.end) for w in second])
        self.assertNotEqual([w.label for w in first], [w.label for w in second])

    def test_ambiguous_windows_are_ignored(self):
        windows = labeled_windows(40, 20, 10, 5, 6)
        self.assertIn(-1, [w.label for w in windows])
        self.assertTrue(all(w.label in (-1, 0, 1) for w in windows))

    def test_event_near_beginning_and_end_stays_on_grid(self):
        self.assertEqual([(w.start, w.end) for w in labeled_windows(30, 1, 10, 5)], inference_windows(30, 10, 5))
        self.assertEqual([(w.start, w.end) for w in labeled_windows(30, 29, 10, 5)], inference_windows(30, 10, 5))

if __name__ == '__main__': unittest.main()
