import math
import unittest
from magicpointer.engine import Engine, Gesture, Observation

class EngineTests(unittest.TestCase):

    def setUp(self):
        self.engine = Engine(mirror_x=False)
        self.now = 0.0

    def sample(self, gesture, *, tip=(0.0, 0.0, 2.0), wrist=(0.0, 0.0, 2.0), dt=0.03):
        self.now += dt
        obs = None if gesture is None else Observation(gesture, wrist, tip)
        return self.engine.step(obs, self.now)

    def hold(self, gesture, seconds=0.5, **kwargs):
        events = []
        for _ in range(math.ceil(seconds / 0.03)):
            events += self.sample(gesture, **kwargs)
        return events

    def activate(self):
        return self.hold(Gesture.FIST)

    def test_inactive_never_outputs_input(self):
        for gesture in (Gesture.POINT, Gesture.LEFT, Gesture.RIGHT, Gesture.SCROLL, Gesture.PALM):
            self.assertEqual(self.hold(gesture), [])

    def test_activation_centers_once_at_fist_origin(self):
        events = self.hold(Gesture.FIST, wrist=(1.0, -0.4, 3.0))
        self.assertEqual([e.kind for e in events], ['activate', 'move'])
        self.assertEqual(self.engine.origin, (1.0, -0.4, 3.0))
        self.assertEqual(events[-1].value, (0.5, 0.5))
        self.assertEqual(self.hold(Gesture.FIST), [])

    def test_loss_suppresses_actions_and_clears_origin(self):
        self.activate()
        self.assertEqual(self.sample(None), [])
        events = self.hold(None, 0.4)
        self.assertEqual([e.kind for e in events], ['deactivate'])
        self.assertIsNone(self.engine.origin)
        self.assertEqual(self.hold(Gesture.LEFT), [])

    def test_stall_with_valid_hand_requires_reactivation(self):
        self.activate()
        events = self.sample(Gesture.LEFT, dt=0.5)
        self.assertEqual([e.kind for e in events], ['deactivate'])
        self.assertFalse(self.engine.active)

    def test_short_fist_and_interrupted_candidates_do_not_activate(self):
        self.hold(Gesture.FIST, 0.3)
        self.sample(None)
        self.hold(Gesture.FIST, 0.3)
        self.assertFalse(self.engine.active)

    def test_nonfinite_observation_is_tracking_loss(self):
        self.activate()
        self.hold(Gesture.POINT, 0.5, tip=(float('nan'), 0.0, 0.0))
        self.assertFalse(self.engine.active)

    def test_invalid_configuration_and_time(self):
        for width in (0, -1, float('nan')):
            with self.assertRaises(ValueError):
                Engine(width=width)
        self.sample(None)
        with self.assertRaises(ValueError):
            self.engine.step(None, 0.0)
