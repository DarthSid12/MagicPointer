import unittest

import numpy as np

from magicpointer.engine import Engine, Gesture, Observation
from magicpointer.fusion import combine
from magicpointer.tracking import Hand, PoseStabilizer
from test_tracking import REALISTIC, posed

POSES = {expected: posed(thumb, bends) for thumb, bends, expected in REALISTIC}


def session(script, sigma, rng):
    """Two jittered views through fusion, per-hand stabilization and the engine."""
    engine = Engine(mirror_x=False, retain_activation_on_loss=True, activation_grace=.4)
    stabilizer, now, clicks = PoseStabilizer(), 0., []
    for gesture, seconds in script:
        for _ in range(round(seconds * 30)):
            now += 1 / 30
            views = [Hand(np.zeros((21, 2)), POSES[gesture] + rng.normal(scale=sigma, size=(21, 3)), 'Right', None)
                     for _ in range(2)]
            fused, _, _ = combine(views, bound_pair=engine.active)
            if fused is not None:
                fused.gesture = stabilizer.update(fused.world, now)
            observation = None if fused is None else Observation(fused.gesture, (0., 0., 2.), (0., 0., 2.))
            clicks += [event.value for event in engine.step(observation, now) if event.kind == 'click']
    return clicks


class JitteredSessionTests(unittest.TestCase):
    def test_held_non_click_poses_never_click_under_jitter(self):
        rng = np.random.default_rng(31)
        for pose in (Gesture.POINT, Gesture.SCROLL, Gesture.FIST):
            clicks = [session([(Gesture.FIST, 1.), (pose, 6.)], .005, rng) for _ in range(8)]
            with self.subTest(pose=pose):
                self.assertEqual(sum(map(len, clicks)), 0)

    def test_intended_clicks_complete_under_jitter(self):
        rng = np.random.default_rng(32)
        script = [(Gesture.FIST, 1.), (Gesture.POINT, .8), (Gesture.LEFT, .6), (Gesture.POINT, .8),
                  (Gesture.RIGHT, .6), (Gesture.POINT, .8)]
        results = [session(script, .004, rng) for _ in range(20)]
        self.assertGreaterEqual(np.mean([clicks == ['left', 'right'] for clicks in results]), .9)
