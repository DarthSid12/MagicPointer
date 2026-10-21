import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import zipfile

import numpy as np

from magicpointer import app
from magicpointer.cameras import Frame
from magicpointer.cli import parser
from magicpointer.engine import Gesture
from magicpointer.stereo import Stereo
from magicpointer.tracking import Hand
from test_geometry import project, rig
from test_tracking import hand


def dual_session(calibration='valid', *, marker_ok=True, skew=.005, disagree=False,
                 missing_right=False, capture_error=False, resolution=(1920, 1080)):
    """Real app render/engine/archive path, synthetic cameras and stereo geometry."""
    geometry = rig()
    geometry['camera_indices'] = [0, 1]
    stereo = Stereo(geometry)
    camera, audio = MagicMock(), MagicMock()
    trackers = [MagicMock(), MagicMock()]
    camera.__enter__.return_value = camera
    camera.info = [{'index': i, 'width': resolution[0], 'height': resolution[1], 'reported_fps': 30} for i in (0, 1)]
    images = [np.full((resolution[1], resolution[0], 3), color, np.uint8) for color in ((70, 30, 20), (20, 70, 30))]
    index, clock, displays = [0], [1.], []

    def tick():
        clock[0] += .03
        return clock[0]

    def read():
        if capture_error and index[0] >= 2:
            raise ValueError('Camera stream stopped')
        return [Frame(images[0], clock[0]), Frame(images[1], clock[0]+skew)]

    def detect(side):
        def result(*_):
            if side == 1 and missing_right:
                return None
            world = hand(False, [0, 0, 0, 0] if index[0] < 25 else [1, 0, 0, 0])
            pixels = project(stereo, np.asarray(world)*.06 + [0., 0., 2.])[side]
            gesture = Gesture.FIST if index[0] < 25 else Gesture.POINT
            if side == 1 and disagree:
                gesture = Gesture.PALM
            return Hand(pixels, world, 'Right', gesture)
        return result

    def key(_):
        index[0] += 1
        return ord('q') if index[0] >= 36 else -1

    camera.read.side_effect = read
    for side, tracker in enumerate(trackers):
        tracker.detect.side_effect = detect(side)
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory)/'calibration.json'
        if calibration == 'valid':
            path.write_text(json.dumps(geometry))
        elif calibration == 'invalid':
            path.write_text('{invalid')
        elif calibration == 'wrong_pair':
            geometry['camera_indices'] = [1, 0]
            path.write_text(json.dumps(geometry))
        args = parser().parse_args(['run', '--debug', '--left', '0', '--right', '1',
                                   '--calibration', str(path), '--output', directory, '--mute'])
        with patch.object(app, 'Cameras', return_value=camera), patch.object(app, 'Tracker', side_effect=trackers), \
                patch.object(app, 'Chimes', return_value=audio), patch.object(app.time, 'monotonic', side_effect=tick), \
                patch.object(Stereo, 'check_markers', return_value=(marker_ok, 'Show wall markers')) as markers, \
                patch.object(app.cv2, 'imshow', side_effect=lambda _, image: displays.append(image.copy())), \
                patch.object(app.cv2, 'waitKey', side_effect=key), patch.object(app.cv2, 'getWindowProperty', return_value=1), \
                patch.object(app.cv2, 'destroyAllWindows'), patch('magicpointer.desktop.Desktop') as desktop:
            if capture_error:
                try:
                    app.run(args)
                except ValueError as error:
                    assert 'stream stopped' in str(error)
            else:
                assert app.run(args) == 0
            desktop.assert_not_called()
        with zipfile.ZipFile(next(Path(directory).glob('debug-*.zip'))) as archive:
            metadata = json.loads(archive.read('session.json'))
            rows = [json.loads(line) for line in archive.read('samples.jsonl').splitlines()]
    return metadata, rows, displays, trackers, camera, audio, markers


class TwoCameraDebugTests(unittest.TestCase):
    def test_missing_invalid_and_wrong_pair_calibration_show_both_views_without_simulation(self):
        for mode in ('missing', 'invalid', 'wrong_pair'):
            with self.subTest(mode=mode):
                metadata, rows, panels, trackers, camera, audio, markers = dual_session(mode)
                self.assertTrue(metadata['preview_only'])
                self.assertFalse(metadata['stereo'])
                self.assertIsNone(metadata['coordinate_units'])
                self.assertTrue(metadata['calibration_issue'])
                self.assertEqual(metadata['camera_indices'], [0, 1])
                self.assertEqual(panels[-1].shape, (760, 1280, 3))
                np.testing.assert_array_equal(panels[-1][200, 20], [70, 30, 20])
                np.testing.assert_array_equal(panels[-1][200, 660], [20, 70, 30])
                self.assertTrue(all(len(row['cameras']) == 2 for row in rows))
                self.assertTrue(all(row['observation'] is None and not row['events'] for row in rows))
                self.assertTrue(all(not row['markers_verified'] for row in rows))
                self.assertTrue(all(abs(row['pair_grab_skew_ms']-5.) < .001 for row in rows))
                self.assertFalse(metadata['images_recorded'])
                markers.assert_not_called()
                for tracker in trackers:
                    self.assertEqual(tracker.detect.call_count, len(rows))
                    tracker.close.assert_called_once()
                camera.__exit__.assert_called_once()
                audio.close.assert_called_once()

    def test_calibrated_stereo_checks_markers_and_simulates_metric_pointer(self):
        metadata, rows, _, _, _, _, markers = dual_session()
        self.assertFalse(metadata['preview_only'])
        self.assertTrue(metadata['stereo'])
        self.assertEqual(metadata['coordinate_units'], 'metres')
        self.assertIsNone(rows[0]['observation'])
        self.assertIsNone(rows[1]['observation'])
        self.assertIsNotNone(rows[2]['observation'])
        self.assertGreaterEqual(markers.call_count, 3)
        self.assertTrue(any(e['kind'] == 'activate' for row in rows for e in row['events']))
        self.assertTrue(any(e['kind'] == 'move' for row in rows for e in row['events']))
        self.assertTrue(any(row['active'] for row in rows))
        self.assertAlmostEqual(rows[-1]['observation']['wrist'][2], 2.)

    def test_calibrated_preview_keeps_both_feeds_when_markers_hand_or_pair_invalid(self):
        for kwargs in ({'marker_ok': False}, {'skew': .08}, {'disagree': True}, {'missing_right': True}):
            with self.subTest(kwargs=kwargs):
                _, rows, panels, _, _, _, _ = dual_session(**kwargs)
                self.assertEqual(panels[-1].shape, (760, 1280, 3))
                self.assertTrue(all(row['observation'] is None and not row['events'] for row in rows))
                self.assertTrue(all(row['reason'] != 'Tracking' for row in rows))

    def test_wrong_resolution_falls_back_to_visible_preview_and_records_reason(self):
        metadata, rows, _, _, _, _, markers = dual_session(resolution=(640, 480))
        self.assertTrue(metadata['preview_only'])
        self.assertFalse(metadata['stereo'])
        self.assertIsNone(metadata['coordinate_units'])
        self.assertIn('resolution', metadata['calibration_issue'])
        self.assertTrue(all(row['observation'] is None for row in rows))
        markers.assert_not_called()

    def test_second_stream_failure_saves_partial_recording_and_closes_both_trackers(self):
        metadata, rows, _, trackers, camera, audio, _ = dual_session('missing', capture_error=True)
        self.assertEqual(metadata['termination'], 'ValueError')
        self.assertEqual(len(rows), 2)
        for tracker in trackers:
            tracker.close.assert_called_once()
        camera.__exit__.assert_called_once()
        audio.close.assert_called_once()

    def test_normal_control_still_rejects_missing_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            args = parser().parse_args(['run', '--right', '1', '--calibration', str(Path(directory)/'missing.json')])
            with patch.object(app, 'Cameras') as cameras, patch('magicpointer.desktop.Desktop') as desktop:
                with self.assertRaisesRegex(ValueError, 'Cannot load calibration'):
                    app.run(args)
                cameras.assert_not_called()
                desktop.assert_not_called()
