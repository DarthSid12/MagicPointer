import json
import math
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import zipfile

import numpy as np

from magicpointer.cli import parser
from magicpointer.engine import Gesture
from magicpointer.tilt import WristTilt, measure_wrist
from magicpointer.tilt_app import run
from magicpointer.tracking import Hand, classify
from test_tracking import hand


def camera_pose(pitch=0., yaw=0., roll=0.):
    # Whole-hand rotation: local fingers point -Y; rotate them toward camera -Z.
    x = math.radians(90+pitch)
    y, z = math.radians(yaw), math.radians(roll)
    rx = np.array([[1, 0, 0], [0, math.cos(x), -math.sin(x)], [0, math.sin(x), math.cos(x)]])
    ry = np.array([[math.cos(y), 0, math.sin(y)], [0, 1, 0], [-math.sin(y), 0, math.cos(y)]])
    rz = np.array([[math.cos(z), -math.sin(z), 0], [math.sin(z), math.cos(z), 0], [0, 0, 1]])
    points = hand(False, [1, 1, 0, 0]) @ (rz @ ry @ rx).T
    return Hand(np.zeros((21, 2)), points.tolist(), 'Right', classify(points))


class WristTests(unittest.TestCase):
    def setUp(self):
        self.controller, self.now = WristTilt(), 1.

    def hold(self, pitch=0., seconds=.9):
        for _ in range(math.ceil(seconds/.03)):
            self.now += .03
            result = self.controller.update(camera_pose(pitch), self.now)
        return result

    def calibrate(self):
        self.hold()
        self.controller.capture_neutral()

    def test_camera_pitch_reacts_to_whole_hand_up_and_down(self):
        for pitch in (-40, -20, 0, 20, 40):
            self.assertAlmostEqual(measure_wrist(camera_pose(pitch))[0], pitch, places=8)

    def test_translation_scale_yaw_and_neutral_roll_do_not_create_scroll(self):
        sample = camera_pose(20., yaw=35.)
        points = np.asarray(sample.world)*1.4 + [1., 2., 3.]
        sample.world = points.tolist()
        self.assertAlmostEqual(measure_wrist(sample)[0], 20., places=8)
        for roll in (-60, 30, 60):
            self.assertAlmostEqual(measure_wrist(camera_pose(roll=roll))[0], 0., places=8)

    def test_one_capture_starts_live_with_symmetric_dead_zone_and_speed(self):
        self.calibrate()
        self.assertEqual(self.hold()['phase'], 'live')
        for pitch in (-9., 9.):
            self.assertEqual(self.hold(pitch)['lines_per_second'], 0.)
        self.assertGreater(self.hold(30.)['lines_per_second'], 11.9)
        self.assertLess(self.hold(-30.)['lines_per_second'], -11.9)
        self.assertAlmostEqual(self.hold(50.)['lines_per_second'], 12., places=6)

    def test_sideways_away_or_folded_hands_pause(self):
        self.assertIsNone(measure_wrist(camera_pose(yaw=90.))[0])
        self.assertIsNone(measure_wrist(camera_pose(yaw=180.))[0])
        for fingers, thumb, gesture in [([0, 0, 0, 0], False, Gesture.FIST),
                                        ([1, 1, 1, 1], True, Gesture.PALM)]:
            points = hand(thumb, fingers)
            self.assertIsNone(measure_wrist(Hand(None, points.tolist(), 'Right', gesture))[0])

    def test_missing_hand_stall_and_reset_require_new_neutral(self):
        self.calibrate()
        self.now += .03
        self.assertEqual(self.controller.update(None, self.now)['lines_per_second'], 0.)
        self.now += .4
        result = self.controller.update(camera_pose(), self.now)
        self.assertEqual(result['phase'], 'neutral')
        self.assertEqual(result['lines_per_second'], 0.)
        self.calibrate()
        self.controller.reset()
        self.assertEqual(self.hold()['phase'], 'neutral')

    def test_invalid_full_speed_ranges_rejected(self):
        for angle in (5., 10., 61., float('nan')):
            with self.assertRaises(ValueError):
                WristTilt(full_speed_angle=angle)


class WristAppTests(unittest.TestCase):
    def test_camera_preview_saves_distinct_logs_without_desktop_events(self):
        camera, tracker = MagicMock(), MagicMock()
        camera.__enter__.return_value = camera
        camera.info = [{'index': 0}]
        camera.read.return_value = [MagicMock(image=np.zeros((480, 640, 3), np.uint8), grabbed_at=1.)]
        tracker.detect.return_value = camera_pose()
        with tempfile.TemporaryDirectory() as directory:
            args = parser().parse_args(['wrist-tilt', '--output', directory])
            self.assertEqual(args.dead_zone, 10.)
            with patch('magicpointer.tilt_app.Cameras', return_value=camera), \
                    patch('magicpointer.tilt_app.Tracker', return_value=tracker), \
                    patch('cv2.imshow'), patch('cv2.waitKey', return_value=ord('q')), \
                    patch('cv2.destroyAllWindows'), patch('magicpointer.desktop.Desktop') as desktop:
                self.assertEqual(run(args), 0)
                desktop.assert_not_called()
            with zipfile.ZipFile(next(Path(directory).glob('wrist-tilt-*.zip'))) as archive:
                metadata = json.loads(archive.read('session.json'))
                self.assertEqual(metadata['kind'], 'wrist-tilt')
                self.assertEqual(metadata['measurement_frame'], 'camera')
                self.assertEqual(metadata['dead_zone_degrees'], 10.)
                sample = json.loads(archive.read('samples.jsonl').splitlines()[0])
                self.assertTrue(sample['valid'])
                self.assertEqual(sample['phase'], 'neutral')
        tracker.close.assert_called_once()
        camera.__exit__.assert_called_once()
