import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.video.windows import labeled_windows

class LabeledWindowsTests(unittest.TestCase):
    def assert_no_negative_overlap(self, windows):
        positive = next(w for w in windows if w.label == 1)
        for window in windows:
            if window.label == 0:
                self.assertFalse(window.start < positive.end and window.end > positive.start, window)

    def test_event_in_middle_excludes_overlapping_windows(self):
        windows = labeled_windows(40, 22, 10, 5)
        self.assertEqual((17, 27), (windows[0].start, windows[0].end))
        self.assert_no_negative_overlap(windows)
        self.assertNotIn((15, 25), [(w.start, w.end) for w in windows if not w.label])

    def test_event_near_beginning(self):
        windows = labeled_windows(30, 1, 10, 5)
        self.assertEqual((0, 10), (windows[0].start, windows[0].end)); self.assert_no_negative_overlap(windows)

    def test_event_near_end(self):
        windows = labeled_windows(30, 29, 10, 5)
        self.assertEqual((20, 30), (windows[0].start, windows[0].end)); self.assert_no_negative_overlap(windows)

    def test_short_video(self):
        windows = labeled_windows(4, 2, 10, 5)
        self.assertEqual([(0, 4, 1)], [(w.start, w.end, w.label) for w in windows])

    def test_no_event_is_all_negative(self):
        self.assertTrue(all(w.label == 0 for w in labeled_windows(20, None, 10, 5)))

    def test_safety_margin_excludes_nearby_window(self):
        windows = labeled_windows(40, 20, 10, 5, safety_margin=5)
        self.assertTrue(all(not (w.start < 30 and w.end > 10) for w in windows if not w.label))

if __name__ == '__main__': unittest.main()
