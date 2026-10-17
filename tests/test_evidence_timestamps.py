import unittest
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app.debug.evidence import event_context_timestamps, window_timestamps

class EvidenceTimestampTests(unittest.TestCase):
    def test_event_context_is_aligned_to_event(self):
        self.assertEqual(event_context_timestamps(8, 20), [('event-5s', 3), ('event-2s', 6), ('event+0s', 8), ('event+2s', 10), ('event+5s', 13)])

    def test_event_context_clamps_at_video_boundaries(self):
        self.assertEqual([time for _, time in event_context_timestamps(1, 4)], [0, 0, 1, 3, 4])

    def test_window_start_middle_end_match_requested_window(self):
        self.assertEqual(window_timestamps(62.7, 72.7, 72.7), [('window_start', 62.7), ('window_middle', 67.7), ('window_end', 72.7)])

if __name__ == '__main__': unittest.main()
