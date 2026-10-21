import unittest
from unittest.mock import patch

import numpy as np

from magicpointer.app import observe
from magicpointer.engine import Engine, Gesture, Observation
from magicpointer.scrolling import PitchScroll, tilt_speed
from magicpointer.stereo import Stereo
from test_app import debug_session
from test_geometry import project, rig
from test_wrist_tilt import camera_pose


class ScrollTests(unittest.TestCase):
    def setUp(self):
        self.engine = Engine(mirror_x=False)
        self.now = 0.
        self.hold(Gesture.FIST, pitch=None)

    def hold(self, gesture=Gesture.SCROLL, pitch=0., seconds=.9, tip=(0., 0., 2.), point=False, fps=60):
        events = []
        for _ in range(round(seconds*fps)):
            self.now += 1./fps
            obs = None if gesture is None else Observation(gesture, (0., 0., 2.), tip, pitch, point)
            events += self.engine.step(obs, self.now)
        return events

    def test_auto_neutral_freezes_pointer_and_translation_does_not_scroll(self):
        self.hold(Gesture.POINT, pitch=None, tip=(.05, .03, 2.))
        cursor = self.engine.cursor
        self.assertEqual(self.hold(tip=(.1, .15, 2.)), [])
        self.assertTrue(self.engine.scrolling)
        self.assertIsNotNone(self.engine.scroll.neutral)
        self.assertEqual(self.hold(tip=(.1, -.15, 2.)), [])
        self.assertEqual(self.engine.cursor, cursor)
        for pitch in (-9., 9.):
            self.assertEqual(self.hold(pitch=pitch), [])

    def test_tilt_sign_speed_limit_and_thumb_flips_never_click(self):
        self.hold()
        for gesture, pitch, sign in [(Gesture.RIGHT, 30., -1), (Gesture.UNKNOWN, -30., 1),
                                      (Gesture.FIST, 40., -1)]:
            # A valid straight two-finger pitch wins over a fist-like label.
            self.hold(gesture, pitch=pitch)
            events = self.hold(gesture, pitch=pitch, seconds=1.)
            self.assertTrue(self.engine.scrolling)
            self.assertTrue(events)
            self.assertTrue(all(e.kind == 'scroll' and e.value*sign > 0 for e in events))
            self.assertLessEqual(abs(sum(e.value for e in events)), 12)

    def test_index_exit_rebases_pointer_without_jump_then_clicks_work(self):
        self.hold(Gesture.POINT, pitch=None, tip=(.05, 0., 2.))
        cursor = self.engine.cursor
        self.hold()
        events = self.hold(Gesture.POINT, pitch=None, point=True, tip=(.2, -.1, 2.))
        self.assertFalse(self.engine.scrolling)
        self.assertEqual(self.engine.cursor, cursor)
        self.assertTrue(all(e.kind == 'move' and e.value == cursor for e in events))
        events = self.hold(Gesture.LEFT, pitch=None, tip=(.2, -.1, 2.))
        self.assertEqual([e.value for e in events if e.kind == 'click'], ['left'])
        self.hold(Gesture.POINT, pitch=None, tip=(.21, -.1, 2.))
        self.assertGreater(self.engine.cursor[0], cursor[0])

    def test_open_thumb_index_exit_does_not_click_until_neutral(self):
        self.hold()
        events = self.hold(Gesture.LEFT, pitch=None, point=True)
        self.assertFalse(self.engine.scrolling)
        self.assertFalse(any(e.kind == 'click' for e in events))
        self.hold(Gesture.POINT, pitch=None)
        self.assertEqual([e.value for e in self.hold(Gesture.RIGHT, pitch=None) if e.kind == 'click'], ['right'])

    def test_invalid_shape_pauses_and_requires_new_neutral_after_timeout(self):
        self.hold()
        self.hold(pitch=30.)
        self.assertEqual(self.hold(Gesture.UNKNOWN, pitch=None), [])
        self.assertIsNone(self.engine.scroll.neutral)
        self.assertEqual(self.hold(Gesture.RIGHT, pitch=30.), [])
        self.assertAlmostEqual(self.engine.scroll.neutral, 30.)
        self.assertTrue(self.engine.scrolling)

    def test_palm_and_tracking_loss_stop_active_scroll(self):
        self.hold()
        self.hold(pitch=30.)
        events = self.hold(Gesture.PALM, pitch=None)
        self.assertEqual([e.kind for e in events], ['deactivate'])
        self.assertFalse(self.engine.scrolling)
        self.hold(Gesture.FIST, pitch=None)
        self.hold()
        events = self.hold(None)
        self.assertEqual([e.kind for e in events], ['deactivate'])
        self.assertFalse(self.engine.scrolling)

    def test_right_click_remains_available_outside_scroll_mode(self):
        self.hold(Gesture.POINT, pitch=None)
        events = self.hold(Gesture.RIGHT, pitch=0.)
        self.assertEqual([e.value for e in events if e.kind == 'click'], ['right'])
        self.assertFalse(self.engine.scrolling)
        self.assertEqual(self.hold(), [])  # release of clicked chord cannot enter
        self.assertFalse(self.engine.scrolling)

    def test_rate_is_independent_of_fps_and_keeps_no_invalid_backlog(self):
        totals = []
        for fps in (30, 60):
            controller = PitchScroll()
            now = 0.
            for _ in range(fps):
                now += 1./fps
                controller.update(0., now)
            for _ in range(fps):
                now += 1./fps
                controller.update(40., now)
            total = 0
            for _ in range(fps*3):
                now += 1./fps
                total += controller.update(40., now)
            totals.append(total)
            self.assertEqual(controller.update(None, now+.01), 0)
            self.assertEqual(controller.fraction, 0.)
        self.assertLessEqual(abs(totals[0]-totals[1]), 1)
        self.assertTrue(all(abs(total+36) <= 1 for total in totals))
        self.assertEqual(tilt_speed(10.), 0.)
        self.assertEqual(tilt_speed(-30.), -12.)
        self.assertEqual(tilt_speed(90.), 12.)


    def test_main_preview_displays_latched_scroll_and_no_desktop_input(self):
        poses = ([Gesture.FIST]*55 + [Gesture.POINT]*6 + [camera_pose()]*20 +
                 [camera_pose(30., thumb=True)]*30 + [Gesture.POINT]*8 + [Gesture.PALM]*8)
        # Check feedback text through the real rendering/dispatch loop.
        import magicpointer.app as app
        with patch.object(app.cv2, 'putText', wraps=app.cv2.putText) as text:
            result, displays, tracker, camera, audio = debug_session(poses)
        self.assertEqual(result, 0)
        messages = [call.args[1] for call in text.call_args_list]
        self.assertTrue(any('setting scroll neutral' in message for message in messages))
        self.assertTrue(any('Wrist scroll:' in message for message in messages))
        self.assertTrue(any('scroll -' in message for message in messages))
        tracker.close.assert_called_once()


class ObservationTests(unittest.TestCase):
    def pair(self, pitch=0., thumb=False):
        stereo = Stereo(rig())
        left, right = camera_pose(pitch), camera_pose(pitch, thumb=thumb)
        points = np.asarray(left.world)*.08 + [0., 0., 2.]
        left.pixels, right.pixels = project(stereo, points)
        return stereo, left, right

    def test_stereo_pitch_uses_triangulated_direction_and_ignores_bad_thumb(self):
        stereo, left, right = self.pair(25., thumb=True)
        right.pixels[1:5] += [0., 100.]  # deliberately unusable thumb disparities
        self.assertIsNone(observe(left, right, stereo=stereo)[0])
        obs, reason = observe(left, right, stereo=stereo, scrolling=True)
        self.assertIsNotNone(obs, reason)
        self.assertAlmostEqual(obs.scroll_pitch, 25., places=7)
        # Model directions are only shape gates; stereo gives the actual pitch.
        left.world = camera_pose(-20.).world
        right.world = camera_pose(-20., thumb=True).world
        obs, _ = observe(left, right, stereo=stereo, scrolling=True)
        self.assertAlmostEqual(obs.scroll_pitch, 25., places=7)

    def test_wrong_handedness_or_invalid_depth_never_produces_observation(self):
        stereo, left, right = self.pair()
        right.label = 'Left'
        self.assertIsNone(observe(left, right, stereo=stereo, scrolling=True)[0])
        right.label = 'Right'
        right.pixels[:] = left.pixels
        self.assertIsNone(observe(left, right, stereo=stereo, scrolling=True)[0])

    def test_both_views_must_support_two_finger_shape_and_palm_has_priority(self):
        stereo, left, right = self.pair()
        right.world = []
        obs, _ = observe(left, right, stereo=stereo, scrolling=True)
        self.assertIsNone(obs)
        right.gesture = Gesture.PALM
        obs, _ = observe(left, right, stereo=stereo, scrolling=True)
        self.assertIsNone(obs)

    def test_mono_preview_uses_model_pitch_without_depth_claim(self):
        sample = camera_pose(-25.)
        obs, _ = observe(sample)
        self.assertAlmostEqual(obs.scroll_pitch, -25.)
        self.assertEqual(obs.wrist[2], 0.)
