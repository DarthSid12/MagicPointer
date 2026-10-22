import math
import unittest

from magicpointer.engine import Engine, Gesture, Observation


class EngineTests(unittest.TestCase):
    def setUp(self):
        self.engine = Engine(mirror_x=False)
        self.now = 0.

    def sample(self, gesture, *, tip=(0., 0., 2.), wrist=(0., 0., 2.), dt=.03):
        self.now += dt
        obs = None if gesture is None else Observation(gesture, wrist, tip)
        return self.engine.step(obs, self.now)

    def hold(self, gesture, seconds=.5, **kwargs):
        events = []
        for _ in range(math.ceil(seconds/.03)):
            events += self.sample(gesture, **kwargs)
        return events

    def activate(self):
        return self.hold(Gesture.FIST)

    def test_inactive_never_outputs_input(self):
        for gesture in (Gesture.POINT, Gesture.LEFT, Gesture.RIGHT, Gesture.SCROLL, Gesture.PALM):
            self.assertEqual(self.hold(gesture), [])

    def test_fist_centers_once_and_uses_origin_anywhere(self):
        events = self.hold(Gesture.FIST, wrist=(1., -.4, 3.))
        self.assertEqual([e.kind for e in events], ['activate', 'move'])
        self.assertEqual(events[-1].value, (.5, .5))
        self.assertEqual(self.hold(Gesture.FIST), [])
        move = self.sample(Gesture.POINT, tip=(1.1, -.4, 3.))[-1]
        self.assertAlmostEqual(move.value[0], .75)
        self.assertAlmostEqual(move.value[1], .5)

    def test_activation_centers_once_at_fist_origin(self):
        events = self.hold(Gesture.FIST, wrist=(1., -.4, 3.))
        self.assertEqual([e.kind for e in events], ['activate', 'move'])
        self.assertEqual(self.engine.origin, (1., -.4, 3.))
        self.assertEqual(events[-1].value, (.5, .5))
        self.assertEqual(self.hold(Gesture.FIST), [])

    def test_click_chord_settles_without_left_or_scroll(self):
        self.activate()
        self.hold(Gesture.POINT, .3)
        before = self.hold(Gesture.LEFT, .12)
        events = before + self.hold(Gesture.RIGHT, .6)
        self.assertEqual([e.value for e in events if e.kind == 'click'], ['right'])
        self.assertFalse(any(e.kind == 'scroll' for e in events))
        # Releasing thumb first cannot turn a held right click into scrolling.
        self.assertFalse(any(e.kind == 'scroll' for e in self.hold(Gesture.SCROLL, .4, tip=(0., .1, 2.))))
        self.assertFalse(any(e.kind == 'click' for e in self.hold(Gesture.LEFT)))

    def test_click_requires_neutral_before_repeat(self):
        self.activate()
        events = self.hold(Gesture.LEFT, 1.)
        self.assertEqual([e.kind for e in events].count('click'), 1)
        self.hold(Gesture.POINT, .3)
        events = self.hold(Gesture.LEFT, .3)
        self.assertEqual([e.kind for e in events].count('click'), 1)

    def test_palm_stops_then_fist_reactivates(self):
        self.activate()
        events = self.hold(Gesture.PALM)
        self.assertEqual([e.kind for e in events], ['deactivate'])
        self.assertFalse(self.engine.active)
        self.assertEqual(self.hold(Gesture.POINT), [])
        self.assertEqual([e.kind for e in self.activate()], ['activate', 'move'])

    def test_loss_suppresses_actions_and_clears_origin(self):
        self.activate()
        self.assertEqual(self.sample(None), [])
        events = self.hold(None, .4)
        self.assertEqual([e.kind for e in events], ['deactivate'])
        self.assertIsNone(self.engine.origin)
        self.assertEqual(self.hold(Gesture.LEFT), [])

    def test_stall_with_valid_hand_requires_reactivation(self):
        self.activate()
        events = self.sample(Gesture.LEFT, dt=.5)
        self.assertEqual([e.kind for e in events], ['deactivate'])
        self.assertFalse(self.engine.active)

    def test_mapping_clamped_and_mirrored(self):
        self.engine = Engine(mirror_x=True)
        self.activate()
        event = self.sample(Gesture.POINT, tip=(100., -100., 2.))[-1]
        self.assertEqual(event.value, (0., 0.))

    def test_short_fist_and_interrupted_candidates_do_not_activate(self):
        self.hold(Gesture.FIST, .3)
        self.hold(None, .15)
        self.hold(Gesture.FIST, .3)
        self.assertFalse(self.engine.active)
        self.setUp()
        self.hold(Gesture.FIST, .3)
        self.hold(Gesture.POINT, .15)
        self.hold(Gesture.FIST, .3)
        self.assertFalse(self.engine.active)

    def test_single_dropped_or_unrecognized_frame_does_not_restart_activation(self):
        for glitch in (None, Gesture.UNKNOWN):
            self.setUp()
            self.hold(Gesture.FIST, .3)
            self.sample(glitch)
            events = self.hold(Gesture.FIST, .24)
            with self.subTest(glitch=glitch):
                self.assertEqual([e.kind for e in events], ['activate', 'move'])
        # A different recognized pose still restarts the hold.
        self.setUp()
        self.hold(Gesture.FIST, .3)
        self.sample(Gesture.POINT)
        self.assertEqual(self.hold(Gesture.FIST, .24), [])

    def test_brief_misread_does_not_restart_or_repeat_click(self):
        for glitches in ((Gesture.UNKNOWN,), (None,), (None, Gesture.UNKNOWN)):
            self.setUp()
            self.activate()
            self.hold(Gesture.POINT, .3)
            events = self.hold(Gesture.LEFT, .09)
            for glitch in glitches:
                events += self.sample(glitch)
            # The dwell continues: less than a fresh 0.18 s hold completes it.
            events += self.hold(Gesture.LEFT, .15)
            with self.subTest(glitches=glitches):
                self.assertEqual([e.value for e in events if e.kind == 'click'], ['left'])

    def test_bridged_frames_do_not_count_toward_a_click(self):
        self.activate()
        self.hold(Gesture.POINT, .3)
        # Two observed click frames separated by misreads span 0.18 s but only
        # 0.06 s of the pose was seen, so no click may fire.
        events = self.hold(Gesture.LEFT, .06) + self.sample(Gesture.UNKNOWN) + self.sample(Gesture.UNKNOWN)
        events += self.sample(Gesture.LEFT) + self.sample(Gesture.UNKNOWN) + self.sample(Gesture.UNKNOWN)
        events += self.sample(Gesture.LEFT)
        self.assertFalse(any(e.kind == 'click' for e in events))

    def test_sustained_unrecognized_pose_still_suppresses_actions(self):
        self.activate()
        self.hold(Gesture.POINT, .3)
        self.hold(Gesture.LEFT, .09)
        events = self.hold(Gesture.UNKNOWN, .3)
        self.assertFalse(any(e.kind == 'click' for e in events))
        self.assertTrue(self.engine.active)

    def test_slow_frames_still_complete_dwell(self):
        self.activate()
        self.hold(Gesture.POINT, .3)
        events = []
        for _ in range(3):
            events += self.sample(Gesture.LEFT, dt=.13)
        self.assertEqual([e.value for e in events if e.kind == 'click'], ['left'])

    def test_click_lands_where_pointer_was_before_thumb_opened(self):
        self.activate()
        settled = self.hold(Gesture.POINT, .3, tip=(.05, 0., 2.))[-1].value
        # Opening the thumb drags the index tip before the pose is recognized.
        self.hold(Gesture.POINT, .06, tip=(.09, 0., 2.))
        events = self.hold(Gesture.LEFT, .3, tip=(.12, 0., 2.))
        self.assertEqual([e.kind for e in events], ['move', 'click'])
        self.assertEqual(events[0].value, settled)
        self.assertEqual(self.engine.cursor, settled)

    def test_pointer_filter_suppresses_jitter_but_follows_motion(self):
        self.activate()
        self.hold(Gesture.POINT, .3)
        positions = []
        for i in range(60):
            positions.append(self.sample(Gesture.POINT, tip=(.005 * (-1) ** i, 0., 2.))[-1].value[0])
        self.assertLess(max(abs(x - .5) for x in positions[20:]), .005)
        self.hold(Gesture.POINT, .5, tip=(.1, .05, 2.))
        self.assertAlmostEqual(self.engine.cursor[0], .75, delta=.01)
        self.assertAlmostEqual(self.engine.cursor[1], .7, delta=.01)

    def test_nonfinite_observation_is_tracking_loss(self):
        self.activate()
        self.hold(Gesture.POINT, .5, tip=(float('nan'), 0., 0.))
        self.assertFalse(self.engine.active)

    def test_invalid_configuration_and_time(self):
        for width in (0, -1, float('nan')):
            with self.assertRaises(ValueError):
                Engine(width=width)
        with self.assertRaises(ValueError):
            Engine(grace=0)
        self.sample(None)
        with self.assertRaises(ValueError):
            self.engine.step(None, 0.)
