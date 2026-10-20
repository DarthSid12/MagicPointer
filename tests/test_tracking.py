import unittest

import numpy as np

from magicpointer.engine import Gesture
from magicpointer.tracking import classify


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
