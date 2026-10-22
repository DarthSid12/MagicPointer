import unittest

import numpy as np

from magicpointer.app import observe
from magicpointer.approximate import ApproximateMotion
from magicpointer.engine import Engine, Gesture, Observation
from magicpointer.stereo import Stereo
from test_geometry import rig, project
from test_ownership_fusion import pose


class AvailableGeometryTests(unittest.TestCase):
    def test_right_only_observation_and_stereo_recovery(self):
        stereo = Stereo(rig())
        left, right = pose(Gesture.POINT), pose(Gesture.POINT)
        world = np.asarray(left.world)*.06 + [0., 0., 2.]
        left.pixels, right.pixels = project(stereo, world)
        mapper = ApproximateMotion()
        observation, reason = observe(None, right, stereo=stereo, approximate=mapper,
                                       reference=(.1, .2, 2.))
        self.assertEqual(observation.gesture, Gesture.POINT)
        self.assertIn('no measured depth', reason)
        self.assertEqual(mapper.source, 1)
        np.testing.assert_allclose(observation.wrist, [.1, .2, 2.])
        observation, reason = observe(left, right, stereo=stereo, approximate=mapper)
        self.assertIsNone(mapper.source)
        self.assertIn('calibrated stereo', reason)
        np.testing.assert_allclose(observation.wrist, world[0], atol=1e-8)

    def test_fallback_relative_motion_uses_fixed_scale_and_rebases_on_source_change(self):
        mapper = ApproximateMotion()
        hand = pose(Gesture.POINT)
        first, changed = mapper.map(hand, 0, [0., 0., 2.])
        self.assertTrue(changed)
        palm = np.linalg.norm(hand.pixels[0]-hand.pixels[9])
        hand.pixels += [palm, 0.]
        second, changed = mapper.map(hand, 0)
        self.assertFalse(changed)
        np.testing.assert_allclose(second-first, np.tile([.1, 0., 0.], (21, 1)))
        third, changed = mapper.map(hand, 1, second[0])
        self.assertTrue(changed)
        np.testing.assert_allclose(third[0], second[0])

    def test_engine_coordinate_rebase_preserves_cursor_and_resets_scroll_neutral(self):
        engine = Engine()
        engine.step(Observation(Gesture.FIST, (0.,0.,2.), (0.,0.,2.)), 0.)
        engine.step(Observation(Gesture.FIST, (0.,0.,2.), (0.,0.,2.)), 1.)
        engine.step(Observation(Gesture.POINT, (0.,0.,2.), (.05,0.,2.)), 1.1)
        cursor = engine.cursor
        engine.rebase_motion()
        engine.step(Observation(Gesture.POINT, (1.,0.,0.), (1.2,0.,0.)), 1.2)
        np.testing.assert_allclose(engine.cursor, cursor)
        self.assertIsNone(engine.scroll.neutral)

    def test_both_views_missing_produce_no_observation(self):
        self.assertIsNone(observe(None, None, approximate=ApproximateMotion())[0])

    def test_measured_out_of_range_is_not_replaced_with_guessed_distance(self):
        stereo = Stereo(rig())
        left, right = pose(Gesture.POINT), pose(Gesture.POINT)
        left.pixels, right.pixels = project(stereo, np.asarray(left.world)*.06+[0., 0., 5.])
        mapper = ApproximateMotion()
        observation, reason = observe(left, right, stereo=stereo, approximate=mapper)
        self.assertIsNone(observation)
        self.assertIn('beyond 12-foot', reason)
        self.assertIsNone(mapper.source)
