import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, patch
import zipfile

import numpy as np

from magicpointer.cli import parser
from magicpointer.diagnostics import Session, STEPS, record, run
from magicpointer.engine import Gesture
from magicpointer.tracking import Hand, classify, gesture_metrics
from test_tracking import hand


class DiagnosticsTests(unittest.TestCase):
    def test_sequence_covers_six_gestures_and_five_angles(self):
        self.assertEqual(len(STEPS), 30)
        self.assertEqual(len({step['gesture'] for step in STEPS}), 6)
        self.assertEqual(len({step['orientation'] for step in STEPS}), 5)

    def test_metrics_preserve_classifier_result_and_explain_uncertainty(self):
        points = hand(False, [1, 1, 0, 0])
        metrics = gesture_metrics(points)
        self.assertEqual(Gesture(metrics['gesture']), classify(points))
        self.assertEqual(metrics['signature'], [0, 1, 1, 0, 0])
        self.assertIn('pip_degrees', metrics['fingers']['middle'])
        self.assertFalse(gesture_metrics(np.zeros((21, 3)))['valid'])

    def test_recording_missing_hands_keeps_detection_denominator(self):
        image = np.zeros((200, 300, 3), dtype=np.uint8)
        points = hand(False, [0, 0, 0, 0]).tolist()
        tracked = Hand(np.zeros((21, 2)), points, 'Right', Gesture.FIST, .95, points)
        with tempfile.TemporaryDirectory() as directory:
            session = Session(directory, {'camera': [{'index': 0}]})
            session.append(record(tracked, 0, .1, image))
            session.append(record(None, 0, .2, image))
            session.finished.add(0)
            session.skipped.add(1)
            session.close()
            session.close()  # Idempotent cleanup must preserve the archive.
            report = json.loads((session.path/'summary.json').read_text())['steps']
            self.assertEqual(report[0]['detection_rate'], .5)
            self.assertEqual(report[0]['match_rate_on_detected'], 1.)
            self.assertTrue(report[0]['complete'])
            self.assertTrue(report[1]['skipped'])
            self.assertIsNone(report[1]['detection_rate'])
            samples = [json.loads(line) for line in (session.path/'samples.jsonl').read_text().splitlines()]
            self.assertEqual(samples[0]['metrics']['signature'], [0, 0, 0, 0, 0])
            self.assertFalse(samples[1]['detected'])
            with zipfile.ZipFile(session.path.with_suffix('.zip')) as archive:
                self.assertEqual(set(archive.namelist()), {'session.json', 'summary.json', 'samples.jsonl'})

    def test_session_never_overwrites_existing_collection(self):
        with tempfile.TemporaryDirectory() as directory:
            one, two = Session(directory, {}), Session(directory, {})
            self.assertNotEqual(one.path, two.path)
            one.close()
            two.close()

    def test_guided_capture_quit_preserves_partial_data_without_pointer_input(self):
        image = np.zeros((480, 640, 3), dtype=np.uint8)
        camera, tracker = MagicMock(), MagicMock()
        camera.__enter__.return_value = camera
        camera.info = [{'index': 0}]
        camera.read.return_value = [MagicMock(image=image, grabbed_at=1.)]
        tracker.detect.return_value = None
        with tempfile.TemporaryDirectory() as directory:
            args = parser().parse_args(['collect', '--output', directory])
            with patch('magicpointer.diagnostics.Cameras', return_value=camera), \
                    patch('magicpointer.diagnostics.Tracker', return_value=tracker), \
                    patch('cv2.imshow'), patch('cv2.waitKey', side_effect=[ord(' '), -1, ord('q')]), \
                    patch('cv2.getWindowProperty', return_value=1), patch('cv2.destroyAllWindows'), \
                    patch('magicpointer.desktop.Desktop') as desktop:
                self.assertEqual(run(args), 0)
                desktop.assert_not_called()
            archive = next(Path(directory).glob('*.zip'))
            with zipfile.ZipFile(archive) as bundle:
                samples = bundle.read('samples.jsonl').decode().splitlines()
                self.assertEqual(len(samples), 2)
                self.assertTrue(all(not json.loads(row)['detected'] for row in samples))
                self.assertFalse(json.loads(bundle.read('summary.json'))['steps'][0]['complete'])
        tracker.close.assert_called_once()
        camera.__exit__.assert_called_once()
