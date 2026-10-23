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
                 missing_right=False, capture_error=False, resolution=(1920, 1080), scene=None, frame_count=70, force_uncalibrated=False,
                 capture_args=(), recovery_scene=None):
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
    recovery_trackers = []

    def recovery_tracker(*args, **kwargs):
        tracker = MagicMock()
        tracker.detect_all.side_effect = lambda image, timestamp: recovery_scene(index[0]) if recovery_scene else []
        recovery_trackers.append(tracker)
        return tracker

    def tick():
        clock[0] += .03
        return clock[0]

    def read():
        if capture_error and index[0] >= 2:
            raise ValueError('Camera stream stopped')
        current_skew = skew(index[0]) if callable(skew) else skew
        return [Frame(images[0], clock[0]), Frame(images[1], clock[0]+current_skew)]

    def detect(side):
        def result(*_):
            if side == 1 and missing_right:
                return None
            world = hand(False, [0, 0, 0, 0] if index[0] < 55 else [1, 0, 0, 0])
            pixels = project(stereo, np.asarray(world)*.06 + [0., 0., 2.])[side]
            gesture = Gesture.FIST if index[0] < 55 else Gesture.POINT
            if side == 1 and disagree:
                gesture = Gesture.PALM
                world = hand(True, [1, 1, 1, 1])
            return Hand(pixels, world, 'Right', gesture)
        return result

    def key(_):
        index[0] += 1
        return ord('q') if index[0] >= frame_count else -1

    camera.read.side_effect = read
    created = iter(trackers)
    for side, tracker in enumerate(trackers):
        tracker.detect_all.side_effect = (lambda *args, side=side: scene(index[0], side)) if scene else (lambda *args, detect=detect(side): ([result] if (result := detect(*args)) is not None else []))
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
                                   '--calibration', str(path), '--output', directory, '--mute', *capture_args])
        args.uncalibrated = force_uncalibrated
        with patch.object(app, 'Cameras', return_value=camera) as open_cameras, patch.object(app, 'Tracker', side_effect=lambda *a, **k: next(created, None) or recovery_tracker(*a, **k)), \
                patch('magicpointer.adaptive_detection.Tracker', side_effect=recovery_tracker), \
                patch.object(app, 'Chimes', return_value=audio), patch.object(app.time, 'monotonic', side_effect=tick), \
                patch.object(Stereo, 'check_markers', return_value=(marker_ok, 'Show wall markers')) as markers, \
                patch.object(app.cv2, 'imshow', side_effect=lambda _, image: displays.__setitem__(slice(None), [image.copy()])), \
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
            open_cameras.assert_called_once_with(0, 1, args.fps, backend=args.camera_backend,
                                                 width=args.capture_width, height=args.capture_height)
        with zipfile.ZipFile(next(Path(directory).glob('debug-*.zip'))) as archive:
            metadata = json.loads(archive.read('session.json'))
            rows = [json.loads(line) for line in archive.read('samples.jsonl').splitlines()]
            camera.recording_summary = json.loads(archive.read('summary.json'))
        camera.recovery_trackers = recovery_trackers
        if args.adaptive_detection:
            from scripts.analyze_debug import analyze
            camera.archive_analysis = analyze(next(Path(directory).glob('debug-*.zip')))
    return metadata, rows, displays, trackers, camera, audio, markers


class TwoCameraDebugTests(unittest.TestCase):
    def test_adaptive_resources_close_when_capture_stops(self):
        metadata, _, _, _, camera, _, _ = dual_session(
            'missing', missing_right=True, capture_error=True,
            capture_args=('--adaptive-detection',))
        self.assertEqual(metadata['termination'], 'ValueError')
        self.assertEqual(len(camera.recovery_trackers), 1)
        camera.recovery_trackers[0].close.assert_called_once()

    def test_adaptive_recovery_is_logged_and_still_requires_a_matched_pair(self):
        shape = np.asarray(hand(False, [0, 0, 0, 0]))
        retry_hand = Hand(shape[:, :2]*1000+[320, 240], shape, 'Right', Gesture.FIST)
        metadata, rows, _, _, camera, _, _ = dual_session(
            'missing', frame_count=12, missing_right=True,
            # Isolate the experiment: crop recovery would also re-find the retried hand.
            capture_args=('--adaptive-detection', '--no-crop-recovery'), recovery_scene=lambda frame: [retry_hand])
        self.assertTrue(metadata['adaptive_detection']['requires_both_views'])
        retries = [row for row in rows if row['adaptive_detection']['attempted_view'] is not None]
        self.assertGreater(len(retries), 0)
        for row in rows:
            self.assertTrue(row['cameras'][0]['baseline_hands'])
            self.assertEqual(row['cameras'][1]['baseline_hands'], [])
            self.assertIn('adaptive_detection_ms', row['timings_ms'])
            if row in retries:
                self.assertEqual(row['cameras'][1]['detection_source'], 'recovery')
                self.assertIsNotNone(row['observation'])
            else:
                self.assertEqual(row['cameras'][1]['hands'], [])
                self.assertIsNone(row['observation'])
        self.assertEqual(camera.recording_summary['adaptive_attempts'], len(retries))
        self.assertEqual(camera.recording_summary['adaptive_observation_frames'], len(retries))
        self.assertEqual(camera.archive_analysis['adaptive_detection']['attempts'], len(retries))
        self.assertEqual(camera.archive_analysis['adaptive_detection']['per_camera'][1]['baseline_detected_frames'], 0)
        self.assertEqual(camera.archive_analysis['adaptive_detection']['per_camera'][1]['effective_detected_frames'], len(retries))
        self.assertEqual(len(camera.recovery_trackers), 1)
        camera.recovery_trackers[0].close.assert_called_once()

    def test_default_low_resolution_debug_routes_capture_options(self):
        metadata, rows, _, _, _, _, _ = dual_session(
            'missing', frame_count=2, resolution=(640, 480), force_uncalibrated=True,
            capture_args=('--capture-width', '640', '--capture-height', '480', '--fps', '30'))
        self.assertTrue(metadata['preview_only'])
        self.assertEqual(metadata['cameras'][0]['width'], 640)
        self.assertEqual(len(rows), 2)

    def test_missing_invalid_and_wrong_pair_calibration_support_shared_uncalibrated_testing(self):
        for mode in ('missing', 'invalid', 'wrong_pair'):
            with self.subTest(mode=mode):
                metadata, rows, panels, trackers, camera, audio, markers = dual_session(mode)
                self.assertTrue(metadata['preview_only'])
                self.assertFalse(metadata['stereo'])
                self.assertEqual(metadata['coordinate_units'], 'activation_palm_sizes')
                self.assertTrue(metadata['calibration_issue'])
                self.assertEqual(metadata['camera_indices'], [0, 1])
                self.assertEqual(panels[-1].shape, (900, 1280, 3))
                np.testing.assert_array_equal(panels[-1][370, 20], [70, 30, 20])
                np.testing.assert_array_equal(panels[-1][370, 660], [20, 70, 30])
                self.assertTrue(all(len(row['cameras']) == 2 for row in rows))
                self.assertTrue(any(row['observation'] is not None for row in rows))
                self.assertTrue(any(e['kind'] == 'activate' for row in rows for e in row['events']))
                self.assertTrue(all(not row['markers_verified'] for row in rows))
                self.assertTrue(all(abs(row['pair_grab_skew_ms']-5.) < .001 for row in rows))
                self.assertFalse(metadata['images_recorded'])
                markers.assert_not_called()
                for tracker in trackers:
                    self.assertEqual(tracker.detect_all.call_count, len(rows))
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
                self.assertEqual(panels[-1].shape, (900, 1280, 3))
                self.assertTrue(all(row['observation'] is None and not row['events'] for row in rows))
                self.assertTrue(all(row['reason'] != 'Tracking' for row in rows))

    def test_wrong_resolution_falls_back_to_visible_preview_and_records_reason(self):
        metadata, rows, _, _, _, _, markers = dual_session(resolution=(640, 480))
        self.assertTrue(metadata['preview_only'])
        self.assertFalse(metadata['stereo'])
        self.assertEqual(metadata['coordinate_units'], 'activation_palm_sizes')
        self.assertIn('resolution', metadata['calibration_issue'])
        self.assertTrue(any(row['observation'] is not None for row in rows))
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

    def test_one_owner_countdown_other_hands_loss_recovery_palm_and_reselection(self):
        from test_ownership_fusion import pose

        def scene(frame, side):
            own_gesture = Gesture.FIST if frame < 21 or 44 <= frame < 96 or 113 <= frame < 167 else Gesture.POINT
            other_gesture = Gesture.FIST if 21 <= frame < 44 or frame >= 177 else Gesture.POINT
            if 96 <= frame < 106:
                other_gesture = Gesture.PALM
            if 167 <= frame < 177:
                own_gesture = Gesture.PALM
            own = pose(own_gesture, 400.+side*70)
            other = pose(other_gesture, 1300.+side*70, 'Left')
            if side == 1 and 106 <= frame < 113:
                return [other]
            # Alternate model order every frame; order must not identify owner.
            return [own, other] if frame % 2 else [other, own]

        metadata, rows, panels, _, _, audio, _ = dual_session('missing', scene=scene, frame_count=230)
        activates = [(index, row) for index, row in enumerate(rows)
                     if any(event['kind'] == 'activate' for event in row['events'])]
        self.assertEqual(len(activates), 2)
        self.assertGreaterEqual(activates[0][0], 85)  # interrupted fists cannot accumulate
        self.assertGreaterEqual(activates[1][0], 218)
        first_owner = activates[0][1]['selection']['owner_ids']
        self.assertNotEqual(activates[1][1]['selection']['owner_ids'], first_owner)
        self.assertTrue(all(row['selection']['owner_ids'] == first_owner for row in rows[96:113]))
        self.assertTrue(all(row['observation'] is None for row in rows[106:113]))
        # The first ~0.1 s of the gap is bridged; longer loss pauses. Neither emits input.
        self.assertTrue(all(row['active'] and not row['events'] for row in rows[106:113]))
        self.assertTrue(rows[112]['engine_after']['paused'])
        self.assertFalse(any(event['kind'] == 'deactivate' for row in rows[106:113] for event in row['events']))
        self.assertTrue(any(row['selection']['owner_ids'] is None for row in rows[167:177]))
        self.assertFalse(any(event['kind'] == 'click' for row in rows for event in row['events']))
        self.assertEqual(sum(call.args == (True,) for call in audio.play.call_args_list), 2)
        self.assertTrue(all('hands' in camera for row in rows for camera in row['cameras']))
        self.assertEqual(panels[-1].shape, (900, 1280, 3))

    def test_debug_banner_uses_large_shared_gesture_and_countdown_text(self):
        with patch.object(app.cv2, 'putText', wraps=app.cv2.putText) as text:
            dual_session('missing')
        large = [call for call in text.call_args_list if call.args[4] >= 2.]
        self.assertTrue(any(call.args[1] == 'FIST' for call in large))
        self.assertTrue(any(call.args[1] == 'POINT' for call in large))
        self.assertTrue(any('HOLD FIST' in call.args[1] for call in text.call_args_list))
        self.assertTrue(any(call.args[1] == 'HAND LOCKED' for call in text.call_args_list))


    def test_explicit_uncalibrated_mode_skips_even_valid_calibration_and_markers(self):
        metadata, rows, _, _, _, _, markers = dual_session(force_uncalibrated=True)
        self.assertFalse(metadata['stereo'])
        self.assertEqual(metadata['coordinate_units'], 'activation_palm_sizes')
        self.assertIn('explicitly', metadata['calibration_issue'])
        self.assertTrue(any(event['kind'] == 'activate' for row in rows for event in row['events']))
        markers.assert_not_called()

    def test_uncalibrated_real_control_needs_no_calibration_and_drives_desktop(self):
        for cameras in (['--right', '1'], []):
            args = parser().parse_args(['run', '--uncalibrated', *cameras])
            with self.subTest(cameras=cameras), patch.object(app.Stereo, 'load') as load, \
                    patch('magicpointer.desktop.Desktop') as desktop, \
                    patch.object(app, 'Cameras', side_effect=ValueError('no cameras in test')), \
                    patch.object(app, 'Tracker'), patch.object(app, 'Chimes'), \
                    patch.object(app.cv2, 'destroyAllWindows'), patch('builtins.print'):
                with self.assertRaisesRegex(ValueError, 'no cameras in test'):
                    app.run(args)
                load.assert_not_called()
                desktop.assert_called_once()

    def test_one_camera_real_control_still_needs_uncalibrated_or_debug(self):
        with patch('magicpointer.desktop.Desktop') as desktop:
            with self.assertRaisesRegex(ValueError, '--uncalibrated'):
                app.run(parser().parse_args(['run']))
            desktop.assert_not_called()


    def test_rejected_pair_cannot_release_owner_even_with_both_palm_estimates(self):
        from test_ownership_fusion import pose
        def scene(frame, side):
            return [pose(Gesture.FIST if frame < 55 else Gesture.PALM, 400.+side*80)]
        _, rows, _, _, _, _, _ = dual_session('missing', scene=scene, frame_count=58,
                                              skew=lambda frame: .08 if frame == 55 else .005)
        self.assertTrue(rows[55]['active'])
        self.assertIsNotNone(rows[55]['selection']['owner_ids'])
        # A single rejected pair is bridged by the grace period without input.
        self.assertFalse(rows[55]['engine_after']['paused'])
        self.assertFalse(rows[55]['events'])
        self.assertIsNone(rows[56]['selection']['owner_ids'])
        self.assertTrue(any(event['kind'] == 'deactivate' and event['value'] == 'open palm' for event in rows[56]['events']))
