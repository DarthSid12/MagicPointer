import math
import unittest

import numpy as np

from magicpointer.engine import Gesture
from magicpointer.tracking import PoseStabilizer, classify, gesture_metrics


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
    (False, (RELAXED, CURLED, LOOSE, RELAXED), Gesture.RIGHT),
    (False, (RELAXED, RELAXED, CURLED, LOOSE), Gesture.SCROLL),
    (True, (RELAXED, STRAIGHT, RELAXED, RELAXED), Gesture.PALM),
]


def thumb_at(points, ip_degrees):
    """Same hand with the thumb IP joint set to an exact bend, well clear of the palm."""
    points = points.copy()
    bend = math.radians(180 - ip_degrees)
    points[2:4] = [[-.05, -.04, 0.], [-.05, -.06, 0.]]
    points[4] = points[3] + [-.03 * math.sin(bend), -.03 * math.cos(bend), 0.]
    return points


class StabilizerTests(unittest.TestCase):
    def test_clean_pose_changes_switch_on_the_first_frame(self):
        poses = [(posed(thumb, bends), expected) for thumb, bends, expected in REALISTIC]
        for first, _ in poses:
            for second, expected in poses:
                stabilizer = PoseStabilizer()
                stabilizer.update(first, 0.)
                with self.subTest(expected=expected):
                    self.assertEqual(stabilizer.update(second, .03), expected)

    def test_uncertain_digit_keeps_its_last_state(self):
        stabilizer = PoseStabilizer()
        left = thumb_at(hand(True, [1, 0, 0, 0]), 170.)
        self.assertEqual(stabilizer.update(left, 0.), Gesture.LEFT)
        uncertain = thumb_at(left, 142.)
        self.assertEqual(gesture_metrics(uncertain)['thumb']['state'], -1)
        self.assertEqual(stabilizer.update(uncertain, .03), Gesture.LEFT)
        self.assertEqual(PoseStabilizer().update(uncertain, 0.), Gesture.UNKNOWN)

    def test_tucked_thumb_needs_margin_to_open_but_closes_normally(self):
        point = thumb_at(hand(False, [1, 0, 0, 0]), 120.)
        stabilizer = PoseStabilizer()
        self.assertEqual(stabilizer.update(point, 0.), Gesture.POINT)
        # 147 degrees reads as open on its own, but not 5 degrees past the gate.
        self.assertEqual(classify(thumb_at(point, 147.)), Gesture.LEFT)
        self.assertEqual(stabilizer.update(thumb_at(point, 147.), .03), Gesture.POINT)
        self.assertEqual(stabilizer.update(thumb_at(point, 155.), .06), Gesture.LEFT)
        self.assertEqual(stabilizer.update(thumb_at(point, 147.), .09), Gesture.LEFT)
        # Closing is not sticky: the classifier's own tucked threshold applies.
        self.assertEqual(stabilizer.update(thumb_at(point, 138.), .12), Gesture.POINT)

    def test_reacquired_hand_does_not_inherit_stale_state(self):
        stabilizer = PoseStabilizer()
        stabilizer.update(thumb_at(hand(False, [1, 0, 0, 0]), 120.), 0.)
        self.assertEqual(stabilizer.update(thumb_at(hand(False, [1, 0, 0, 0]), 147.), 1.), Gesture.LEFT)
        self.assertEqual(stabilizer.update(np.zeros((21, 3)), 1.03), Gesture.UNKNOWN)

    def test_jitter_flickers_less_than_single_frame_classification(self):
        rng = np.random.default_rng(5)
        for thumb, bends, expected in REALISTIC:
            base, stabilizer = posed(thumb, bends), PoseStabilizer()
            raw, stable = [], []
            for frame in range(400):
                noisy = base + rng.normal(scale=.003, size=base.shape)
                raw.append(classify(noisy) == expected)
                stable.append(stabilizer.update(noisy, frame / 30) == expected)
            with self.subTest(expected=expected):
                self.assertGreaterEqual(np.mean(stable), .93)
                self.assertGreaterEqual(np.mean(stable), np.mean(raw))


class ClassifierTests(unittest.TestCase):
    def test_all_requested_gestures_survive_rigid_rotation_and_scale(self):
        poses = [(False, [0, 0, 0, 0], Gesture.FIST), (False, [1, 0, 0, 0], Gesture.POINT),
                 (True, [1, 0, 0, 0], Gesture.LEFT), (False, [1, 0, 0, 1], Gesture.RIGHT),
                 (True, [1, 0, 0, 1], Gesture.RIGHT),
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

    def test_right_click_is_index_and_little_finger_with_thumb_ignored(self):
        for thumb in (False, True):
            self.assertEqual(classify(hand(thumb, [1, 0, 0, 1])), Gesture.RIGHT)
        # The old thumb + index + middle chord no longer clicks, so it cannot race scrolling.
        self.assertEqual(classify(hand(True, [1, 1, 0, 0])), Gesture.UNKNOWN)
        # Index + middle + little, or a half-raised ring finger, is not a right click.
        self.assertEqual(classify(hand(False, [1, 1, 0, 1])), Gesture.UNKNOWN)
        points = posed(False, (RELAXED, CURLED, (20, 60, 30), RELAXED))
        self.assertEqual(gesture_metrics(points)['fingers']['ring']['state'], -1)
        self.assertEqual(classify(points), Gesture.UNKNOWN)

    def test_half_raised_ring_finger_keeps_an_established_right_click(self):
        stabilizer = PoseStabilizer()
        self.assertEqual(stabilizer.update(posed(False, (RELAXED, CURLED, LOOSE, RELAXED)), 0.), Gesture.RIGHT)
        self.assertEqual(stabilizer.update(posed(False, (RELAXED, CURLED, (20, 60, 30), RELAXED)), .03), Gesture.RIGHT)

    def test_naturally_bent_fingers_and_narrow_open_thumb(self):
        # Reproduce the failure mechanisms without committing personal recordings.
        # These DIP joints are 140 degrees, below the original 150-degree gate.
        poses = [(False, [1, 0, 0, 0], Gesture.POINT), (True, [1, 0, 0, 0], Gesture.LEFT),
                 (False, [1, 1, 0, 0], Gesture.SCROLL), (True, [1, 0, 0, 1], Gesture.RIGHT),
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
