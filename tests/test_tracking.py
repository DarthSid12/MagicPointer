import math
import unittest

import numpy as np

from magicpointer.engine import Gesture
from magicpointer.tracking import Classifier, classify, extension


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


# Approximate adult knuckle positions (m) and phalanx lengths, index to little.
KNUCKLES = ((-.025, -.09), (0., -.093), (.022, -.088), (.042, -.078))
PHALANGES = ((.039, .022, .020), (.044, .026, .021), (.041, .025, .021), (.032, .019, .019))
STRAIGHT, RELAXED, LOOSE, CURLED = (0, 0, 0), (10, 20, 15), (30, 70, 40), (80, 100, 60)


def posed(thumb_out, bends):
    """Anatomical hand with per-finger (MCP, PIP, DIP) flexion in degrees."""
    points = np.zeros((21, 3), dtype=float)
    points[1:3] = [[-.025, -.02, .01], [-.045, -.04, .015]]
    # Abducted thumb for clicks; otherwise tucked across the curled fingers.
    points[3:5] = [[-.07, -.055, .015], [-.09, -.065, .015]] if thumb_out else [
        [-.035, -.06, .035], [-.012, -.065, .04]]
    for finger, ((x, y), lengths, flexion) in enumerate(zip(KNUCKLES, PHALANGES, bends)):
        start = 5 + 4 * finger
        points[start] = joint = np.array([x, y, 0.])
        total = 0.
        for offset, (length, bend) in enumerate(zip(lengths, flexion), 1):
            total += math.radians(bend)
            joint = joint + length * np.array([0., -math.cos(total), math.sin(total)])
            points[start + offset] = joint
    return points


REALISTIC = [
    (False, (CURLED,) * 4, Gesture.FIST),
    (False, (CURLED, LOOSE, CURLED, LOOSE), Gesture.FIST),
    (False, (RELAXED, CURLED, LOOSE, LOOSE), Gesture.POINT),
    (True, (RELAXED, LOOSE, CURLED, LOOSE), Gesture.LEFT),
    (True, (RELAXED, RELAXED, LOOSE, LOOSE), Gesture.RIGHT),
    (False, (RELAXED, RELAXED, CURLED, LOOSE), Gesture.SCROLL),
    (True, (RELAXED, STRAIGHT, RELAXED, RELAXED), Gesture.PALM),
]


def rotate(points, rng):
    q, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    return points @ q * rng.uniform(.5, 2.) + rng.normal(size=3)


class ClassifierTests(unittest.TestCase):
    def test_all_requested_gestures_survive_rigid_rotation_and_scale(self):
        poses = [(False, [0, 0, 0, 0], Gesture.FIST), (False, [1, 0, 0, 0], Gesture.POINT),
                 (True, [1, 0, 0, 0], Gesture.LEFT), (True, [1, 1, 0, 0], Gesture.RIGHT),
                 (False, [1, 1, 0, 0], Gesture.SCROLL), (True, [1, 1, 1, 1], Gesture.PALM)]
        rng = np.random.default_rng(8)
        for thumb, fingers, expected in poses:
            for _ in range(12):
                points = rotate(hand(thumb, fingers), rng)
                with self.subTest(expected=expected):
                    self.assertEqual(classify(points), expected)

    def test_relaxed_real_hand_poses_classify_in_any_orientation(self):
        # Partly bent pointing fingers and loosely curled others are normal, not ambiguous.
        rng = np.random.default_rng(3)
        for thumb, bends, expected in REALISTIC:
            for _ in range(12):
                with self.subTest(expected=expected, bends=bends):
                    self.assertEqual(classify(rotate(posed(thumb, bends), rng)), expected)

    def test_single_frames_tolerate_landmark_jitter(self):
        rng = np.random.default_rng(5)
        for thumb, bends, expected in REALISTIC:
            base = posed(thumb, bends)
            hits = np.mean([classify(base + rng.normal(scale=.003, size=base.shape)) == expected
                            for _ in range(300)])
            with self.subTest(expected=expected, bends=bends):
                self.assertGreaterEqual(hits, .95)

    def test_temporal_classifier_holds_pose_under_heavy_jitter(self):
        rng = np.random.default_rng(6)
        for thumb, bends, expected in REALISTIC:
            base, classifier = posed(thumb, bends), Classifier()
            frames = [classifier.update(extension(base + rng.normal(scale=.006, size=base.shape)), i / 30)
                      for i in range(300)]
            with self.subTest(expected=expected, bends=bends):
                self.assertGreaterEqual(np.mean([f == expected for f in frames]), .98)

    def test_hysteresis_keeps_borderline_finger_state(self):
        classifier, now = Classifier(), 0.
        point = list(extension(posed(False, (RELAXED, CURLED, CURLED, CURLED))))
        self.assertEqual(classifier.update(point, now), Gesture.POINT)
        # Index sagging into the band (1.4-1.6) stays extended until clearly curled.
        for _ in range(20):
            now += 1 / 30
            self.assertEqual(classifier.update([point[0], 1.45, *point[2:]], now), Gesture.POINT)
        for _ in range(20):
            now += 1 / 30
            last = classifier.update([point[0], 1.0, *point[2:]], now)
        self.assertEqual(last, Gesture.FIST)

    def test_reacquired_hand_does_not_inherit_stale_state(self):
        classifier = Classifier()
        classifier.update(extension(posed(True, (STRAIGHT,) * 4)), 0.)
        self.assertEqual(classifier.update(extension(posed(False, (CURLED,) * 4)), 1.), Gesture.FIST)
        self.assertEqual(classifier.update(None, 1.1), Gesture.UNKNOWN)

    def test_ambiguous_or_invalid_hand_is_unknown(self):
        self.assertEqual(classify(np.zeros((21, 3))), Gesture.UNKNOWN)
        self.assertEqual(classify(hand(True, [0, 0, 0, 0])), Gesture.UNKNOWN)
        self.assertIsNone(extension(np.zeros((21, 3))))
        self.assertIsNone(extension(np.full((21, 3), np.nan)))
