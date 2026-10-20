import math
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import zipfile
import json

import numpy as np

from magicpointer.cli import parser
from magicpointer.engine import Gesture
from magicpointer.tilt import Tilt, measure
from magicpointer.tilt_app import Log, render, run
from magicpointer.tracking import Hand, classify
from test_tracking import hand


def pose(degrees=0.):
    points = hand(False, [1, 1, 0, 0])
    t = math.radians(degrees)
    rotation = np.array([[1, 0, 0], [0, math.cos(t), -math.sin(t)], [0, math.sin(t), math.cos(t)]])
    for base in (5, 9):
        points[base+1:base+4] = (points[base+1:base+4]-points[base]) @ rotation.T + points[base]
    return Hand(np.zeros((21, 2)), points.tolist(), 'Right', classify(points))


class TiltTests(unittest.TestCase):
    def setUp(self):
        self.controller = Tilt()
        self.now = 1.

    def hold(self, degrees=0., seconds=.6):
        for _ in range(math.ceil(seconds/.03)):
            self.now += .03
            result = self.controller.update(pose(degrees), self.now)
        return result

    def calibrate(self, degrees=25.):
        self.hold()
        self.controller.capture_neutral()
        self.hold(degrees)
        self.controller.capture_down()

    def test_measured_pitch_ignores_whole_hand_rotation_translation_and_scale(self):
        original = pose(25.)
        baseline, _ = measure(original)
        rng = np.random.default_rng(22)
        for _ in range(20):
            rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
            rotation[:, 2] *= np.linalg.det(rotation)
            moved = np.asarray(original.world) @ rotation * rng.uniform(.5, 2.) + rng.normal(size=3)
            changed = Hand(original.pixels, moved.tolist(), 'Right', original.gesture)
            value, _ = measure(changed)
            self.assertAlmostEqual(value, baseline, places=8)
        self.assertAlmostEqual(abs(baseline), 25., places=8)

    def test_calibrated_speed_in_both_directions_neutral_and_cap(self):
        self.calibrate()
        self.assertGreater(self.hold(25.)['lines_per_second'], 11.9)
        self.assertEqual(self.hold(2.)['lines_per_second'], 0.)
        self.assertLess(self.hold(-25.)['lines_per_second'], -11.9)
        self.assertLessEqual(self.hold(50.)['lines_per_second'], 12.)
        # The opposite calibration direction also works without handedness rules.
        self.controller.reset()
        self.calibrate(-25.)
        self.assertGreater(self.hold(-25.)['lines_per_second'], 11.9)

    def test_shape_gate_can_continue_when_normal_classifier_is_unrecognized(self):
        sample = pose(25.)
        sample.gesture = Gesture.UNKNOWN
        self.assertIsNotNone(measure(sample)[0])

    def test_invalid_or_released_pose_stops_immediately_and_loss_disarms(self):
        self.calibrate()
        self.now += .03
        self.assertEqual(self.controller.update(None, self.now)['lines_per_second'], 0.)
        self.now += .4
        result = self.controller.update(pose(25.), self.now)
        self.assertEqual(result['phase'], 'neutral')
        self.assertEqual(result['lines_per_second'], 0.)
        palm = hand(True, [1, 1, 1, 1])
        self.assertIsNone(measure(Hand(None, palm.tolist(), 'Right', Gesture.PALM))[0])
        point = hand(False, [1, 0, 0, 0])
        self.assertIsNone(measure(Hand(None, point.tolist(), 'Right', Gesture.POINT))[0])

    def test_calibration_rejects_missing_unstable_and_too_small_range(self):
        with self.assertRaises(ValueError):
            self.controller.capture_neutral()
        self.hold()
        self.controller.capture_neutral()
        self.hold(4.)
        with self.assertRaises(ValueError):
            self.controller.capture_down()
        for degrees in (-15, 15)*8:
            self.now += .03
            self.controller.update(pose(degrees), self.now)
        with self.assertRaises(ValueError):
            self.controller.capture_neutral()

    def test_hand_change_and_processing_stall_clear_calibration(self):
        self.calibrate()
        other = pose(25.)
        other.label = 'Left'
        self.now += .03
        self.assertEqual(self.controller.update(other, self.now)['phase'], 'neutral')
        self.calibrate()
        self.now += .5
        self.assertEqual(self.controller.update(pose(25.), self.now)['phase'], 'neutral')

    def test_invalid_landmarks_config_and_time(self):
        bad = pose()
        bad.world[0][0] = float('nan')
        self.assertIsNone(measure(bad)[0])
        for value in (0, -1, 30, float('nan')):
            with self.assertRaises(ValueError):
                Tilt(value)
        self.hold()
        with self.assertRaises(ValueError):
            self.controller.update(pose(), self.now-1)


class TiltAppTests(unittest.TestCase):
    def test_preview_quit_saves_log_and_never_constructs_desktop(self):
        camera, tracker = MagicMock(), MagicMock()
        camera.__enter__.return_value = camera
        camera.info = [{'index': 0}]
        camera.read.return_value = [MagicMock(image=np.zeros((480, 640, 3), np.uint8), grabbed_at=1.)]
        tracker.detect.return_value = pose()
        with tempfile.TemporaryDirectory() as directory:
            args = parser().parse_args(['tilt', '--output', directory])
            from pathlib import Path
            with patch('magicpointer.tilt_app.Cameras', return_value=camera), \
                    patch('magicpointer.tilt_app.Tracker', return_value=tracker), \
                    patch('cv2.imshow'), patch('cv2.waitKey', return_value=ord('q')), \
                    patch('cv2.destroyAllWindows'), patch('magicpointer.desktop.Desktop') as desktop:
                self.assertEqual(run(args), 0)
                desktop.assert_not_called()
            with zipfile.ZipFile(next(Path(directory).glob('*.zip'))) as archive:
                self.assertEqual(set(archive.namelist()), {'session.json', 'samples.jsonl', 'summary.json'})
                self.assertEqual(json.loads(archive.read('session.json'))['kind'], 'finger-tilt')
                sample = json.loads(archive.read('samples.jsonl').splitlines()[0])
                self.assertTrue(sample['valid'])
                self.assertEqual(sample['lines_per_second'], 0.)
        tracker.close.assert_called_once()
        camera.__exit__.assert_called_once()

    def test_capture_failure_still_saves_partial_session_and_releases_resources(self):
        camera, tracker = MagicMock(), MagicMock()
        camera.__enter__.return_value = camera
        camera.info = []
        camera.read.side_effect = ValueError('Camera disconnected')
        with tempfile.TemporaryDirectory() as directory:
            args = parser().parse_args(['tilt', '--output', directory])
            with patch('magicpointer.tilt_app.Cameras', return_value=camera), \
                    patch('magicpointer.tilt_app.Tracker', return_value=tracker), patch('cv2.destroyAllWindows'):
                with self.assertRaisesRegex(ValueError, 'disconnected'):
                    run(args)
        tracker.close.assert_called_once()
        camera.__exit__.assert_called_once()
