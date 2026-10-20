import unittest

import numpy as np

from magicpointer.engine import Gesture
from magicpointer.tracking import classify, gesture_metrics


def hand(thumb, fingers):
    points = np.zeros((21, 3), dtype=float)
    points[1:5] = [[-.025, -.015, 0.], [-.04, -.03, 0.], [-.055, -.045, 0.], [-.09, -.08, 0.]] if thumb else [
        [-.025, -.015, 0.], [-.03, -.04, 0.], [-.02, -.065, 0.], [-.02, -.075, 0.]]
    for start, x, extended in zip((5, 9, 13, 17), (-.025, 0., .025, .05), fingers):
        points[start] = [x, -.09, 0.]
        points[start+1] = [x, -.125, 0.]
        points[start+2] = [x, -.15, 0.] if extended else [x, -.09, .03]
        points[start+3] = [x, -.175, 0.] if extended else [x, -.055, .025]
    return points


class ClassifierTests(unittest.TestCase):
    def test_all_requested_gestures_survive_rigid_rotation_and_scale(self):
        poses = [(False, [0, 0, 0, 0], Gesture.FIST), (False, [1, 0, 0, 0], Gesture.POINT),
                 (True, [1, 0, 0, 0], Gesture.LEFT), (True, [1, 1, 0, 0], Gesture.RIGHT),
                 (False, [1, 1, 0, 0], Gesture.SCROLL), (True, [1, 1, 1, 1], Gesture.PALM)]
        rng = np.random.default_rng(8)
        for thumb, fingers, expected in poses:
            for _ in range(12):
                q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
                points = hand(thumb, fingers) @ q * rng.uniform(.5, 2.) + rng.normal(size=3)
                with self.subTest(expected=expected):
                    self.assertEqual(classify(points), expected)

    def test_ambiguous_or_invalid_hand_is_unknown(self):
        self.assertEqual(classify(np.zeros((21, 3))), Gesture.UNKNOWN)
        self.assertEqual(classify(hand(True, [0, 0, 0, 0])), Gesture.UNKNOWN)

    def test_naturally_bent_fingers_and_narrow_open_thumb(self):
        # Reproduce the failure mechanisms without committing personal recordings.
        # These DIP joints are 140 degrees, below the original 150-degree gate.
        poses = [(False, [1, 0, 0, 0], Gesture.POINT), (True, [1, 0, 0, 0], Gesture.LEFT),
                 (False, [1, 1, 0, 0], Gesture.SCROLL), (True, [1, 1, 0, 0], Gesture.RIGHT),
                 (True, [1, 1, 1, 1], Gesture.PALM)]
        rng = np.random.default_rng(12)
        for thumb, fingers, expected in poses:
            points = hand(thumb, fingers)
            if thumb:
                points[2:5] = [[-.04, -.03, 0.], [-.05, -.055, 0.], [-.06, -.08, 0.]]
            for start, extended in zip((5, 9, 13, 17), fingers):
                if extended:
                    points[start+2] = points[start+1] + [.0125, -.021650635, 0.]
                    points[start+3] = points[start+2] + [.023492316, -.008550504, 0.]
            for _ in range(8):
                rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
                transformed = points @ rotation * rng.uniform(.6, 1.4) + rng.normal(size=3)
                with self.subTest(expected=expected):
                    self.assertEqual(classify(transformed), expected)

    def test_bent_tucked_thumb_is_not_open_just_because_it_is_wide(self):
        for fingers, expected in [([0, 0, 0, 0], Gesture.FIST), ([1, 0, 0, 0], Gesture.POINT),
                                  ([1, 1, 0, 0], Gesture.SCROLL)]:
            points = hand(False, fingers)
            points[4] = [-.08, -.03, 0.]
            metrics = gesture_metrics(points)
            self.assertGreater(metrics['thumb']['separation_ratio'], .65)
            self.assertEqual(classify(points), expected)

    def test_uncertain_thumb_and_bad_index_remain_unrecognized(self):
        points = hand(True, [1, 0, 0, 0])
        # Set IP bend to the deliberate uncertain band, with substantial separation.
        points[2:5] = [[-.05, -.04, 0.], [-.05, -.06, 0.], [-.05-.03*np.sin(np.deg2rad(38)), -.06-.03*np.cos(np.deg2rad(38)), 0.]]
        self.assertEqual(gesture_metrics(points)['thumb']['state'], -1)
        self.assertEqual(classify(points), Gesture.UNKNOWN)
        points = hand(False, [1, 0, 0, 0])
        points[6:9] = [[-.025, -.125, 0.], [.02, -.13, 0.], [-.025, -.14, 0.]]
        self.assertEqual(classify(points), Gesture.UNKNOWN)
