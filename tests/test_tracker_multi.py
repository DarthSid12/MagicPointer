from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock

import numpy as np

from magicpointer.tracking import Tracker
from test_tracking import hand


class MultiTrackerTests(unittest.TestCase):
    def test_image_mode_retries_fresh_inference_without_video_state(self):
        tracker = Tracker.__new__(Tracker)
        tracker.timestamp, tracker.image_mode = -1, True
        tracker.mp, tracker.detector = MagicMock(), MagicMock()
        tracker.detector.detect.return_value = SimpleNamespace(hand_landmarks=[], hand_world_landmarks=[], handedness=[])
        image = np.zeros((480, 640, 3), np.uint8)
        self.assertEqual(tracker.detect_all(image, 1.), [])
        self.assertEqual(tracker.detect_all(image, 1.5), [])
        self.assertEqual(tracker.detector.detect.call_count, 2)
        tracker.detector.detect_for_video.assert_not_called()

    def test_all_model_hands_are_returned_with_corresponding_geometry_and_legacy_detect_remains_usable(self):
        tracker = Tracker.__new__(Tracker)
        tracker.timestamp = -1
        tracker.mp, tracker.detector = MagicMock(), MagicMock()
        shapes = [hand(False, [0, 0, 0, 0]), hand(False, [1, 0, 0, 0])]
        world = [[SimpleNamespace(x=p[0], y=p[1], z=p[2]) for p in shape] for shape in shapes]
        image_points = [[SimpleNamespace(x=.3+side*.3+p[0], y=.5+p[1], z=p[2]) for p in shape]
                        for side, shape in enumerate(shapes)]
        tracker.detector.detect_for_video.return_value = SimpleNamespace(
            hand_landmarks=image_points, hand_world_landmarks=world,
            handedness=[[SimpleNamespace(category_name=label, score=.95)] for label in ('Right', 'Left')])
        image = np.zeros((480, 640, 3), np.uint8)
        hands = tracker.detect_all(image, 1.)
        self.assertEqual([sample.label for sample in hands], ['Right', 'Left'])
        self.assertEqual([sample.gesture.value for sample in hands], ['fist', 'point'])
        self.assertAlmostEqual(hands[1].pixels[0, 0], .6*640)
        self.assertEqual(hands[1].handedness_score, .95)
        self.assertEqual(tracker.detect(image, .5).label, 'Right')
        self.assertEqual(tracker.timestamp, 1001)
        tracker.detector.detect_for_video.return_value = SimpleNamespace(hand_landmarks=[], hand_world_landmarks=[], handedness=[])
        self.assertEqual(tracker.detect_all(image, 2.), [])
        self.assertIsNone(tracker.detect(image, 2.))
