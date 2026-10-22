from contextlib import redirect_stderr
import io
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from magicpointer.adaptive_detection import AdaptiveDetection
from magicpointer.cameras import Frame
from magicpointer.cli import main
from magicpointer.ownership import HandOwner
from magicpointer.tracking import Hand, classify
from test_tracking import hand


def sample(label='Right'):
    points = np.asarray(hand(False, [0, 0, 0, 0]))
    return Hand(points[:, :2]*1000+[320, 240], points, label, classify(points))


class AdaptiveDetectionTests(unittest.TestCase):
    def setUp(self):
        self.frames = [Frame(np.zeros((480, 640, 3), np.uint8), 1.),
                       Frame(np.ones((480, 640, 3), np.uint8), 1.005)]

    def test_retries_only_missing_view_on_current_image_and_retains_regular_evidence(self):
        for missing in (0, 1):
            with self.subTest(missing=missing), patch('magicpointer.adaptive_detection.Tracker') as tracker:
                regular, recovered = sample(), sample()
                retry = tracker.return_value
                retry.detect_all.return_value = [recovered]
                experiment = AdaptiveDetection('model.task', 4)
                baseline = [[regular], [regular]]
                baseline[missing] = []
                result, trace = experiment.apply(baseline, self.frames, 1.)
                tracker.assert_called_once_with('model.task', num_hands=4, detection_confidence=.5,
                                                presence_confidence=.5, image_mode=True)
                args = retry.detect_all.call_args.args
                self.assertIs(args[0], self.frames[missing].image)
                self.assertEqual(args[1], self.frames[missing].grabbed_at)
                self.assertIs(result[1-missing], baseline[1-missing])
                self.assertEqual(result[missing], [recovered])
                self.assertEqual(baseline[missing], [])
                self.assertEqual(trace['attempted_view'], missing)
                self.assertEqual(trace['detection_source'][missing], 'recovery')
                experiment.close()
                retry.close.assert_called_once()

    def test_no_retry_for_complete_empty_or_badly_timed_pair(self):
        with patch('magicpointer.adaptive_detection.Tracker') as tracker:
            experiment = AdaptiveDetection('model.task', 4)
            regular = sample()
            for baseline, paired in (([[], []], True), ([[regular], [regular]], True), ([[regular], []], False)):
                result, trace = experiment.apply(baseline, self.frames, 1., paired=paired)
                self.assertEqual(result, baseline)
                self.assertIsNone(trace['attempted_view'])
            tracker.assert_not_called()
            experiment.close()

    def test_retry_rate_limited_without_reusing_old_candidates(self):
        with patch('magicpointer.adaptive_detection.Tracker') as tracker:
            experiment = AdaptiveDetection('model.task', 4)
            tracker.return_value.detect_all.return_value = [sample()]
            baseline = [[sample()], []]
            experiment.apply(baseline, self.frames, 1.)
            result, trace = experiment.apply(baseline, self.frames, 1.1)
            self.assertEqual(result[1], [])
            self.assertEqual(trace['reason'], 'retry cooldown')
            experiment.apply(baseline, self.frames, 1.25)
            self.assertEqual(tracker.return_value.detect_all.call_count, 2)
            tracker.assert_called_once()
            experiment.close()

    def test_empty_retry_and_wrong_handedness_cannot_supply_a_matched_pair(self):
        with patch('magicpointer.adaptive_detection.Tracker') as tracker:
            experiment = AdaptiveDetection('model.task', 4)
            baseline = [[sample()], []]
            tracker.return_value.detect_all.return_value = []
            result, trace = experiment.apply(baseline, self.frames, 1.)
            self.assertEqual(trace['candidate_count'], 0)
            self.assertIsNone(HandOwner(2).select(result, self.frames, 1.).combined)
            tracker.return_value.detect_all.return_value = [sample('Left')]
            result, trace = experiment.apply(baseline, self.frames, 2.)
            self.assertEqual(trace['candidate_count'], 1)
            self.assertIsNone(HandOwner(2).select(result, self.frames, 2.).combined)
            experiment.close()

    def test_low_threshold_inference_failure_releases_created_detector(self):
        with patch('magicpointer.adaptive_detection.Tracker') as tracker:
            experiment = AdaptiveDetection('model.task', 4)
            tracker.return_value.detect_all.side_effect = RuntimeError('inference failed')
            with self.assertRaisesRegex(RuntimeError, 'inference failed'):
                experiment.apply([[sample()], []], self.frames, 1.)
            experiment.close()
            tracker.return_value.close.assert_called_once()

    def test_not_available_for_desktop_control_or_single_camera(self):
        for args in (['run', '--left', '0', '--right', '1'], ['run', '--debug']):
            with redirect_stderr(io.StringIO()) as output, patch('magicpointer.app.Tracker') as tracker:
                self.assertEqual(main([*args, '--adaptive-detection']), 1)
                self.assertIn('two-camera --debug', output.getvalue())
                tracker.assert_not_called()


if __name__ == '__main__':
    unittest.main()
