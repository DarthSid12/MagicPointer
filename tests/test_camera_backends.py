from contextlib import redirect_stderr, redirect_stdout
import io
import unittest
from unittest.mock import MagicMock, call, patch

import cv2
import numpy as np

from magicpointer.cameras import Cameras
from magicpointer.cli import main, parser


def capture():
    cap = MagicMock()
    cap.isOpened.return_value = True
    cap.get.return_value = 30.
    cap.read.return_value = (True, np.zeros((480, 640, 3), dtype=np.uint8))
    cap.retrieve.return_value = cap.read.return_value
    cap.grab.return_value = True
    return cap


class CameraBackendTests(unittest.TestCase):
    def test_pair_uses_one_selected_backend_and_releases_both(self):
        for choice, api in (('default', cv2.CAP_DSHOW), ('dshow', cv2.CAP_DSHOW), ('msmf', cv2.CAP_MSMF)):
            with self.subTest(backend=choice):
                one, two = capture(), capture()
                with patch('magicpointer.cameras.platform.system', return_value='Windows'), \
                        patch('cv2.VideoCapture', side_effect=[one, two]) as opened:
                    with Cameras(1, 2, '30', width=640, height=480, backend=choice) as cameras:
                        self.assertEqual(len(cameras.read()), 2)
                        self.assertEqual([item['backend'] for item in cameras.info],
                                         [choice if choice != 'default' else 'dshow'] * 2)
                        self.assertEqual(cameras.info[0]['width'], 640)
                    self.assertEqual(opened.call_args_list, [call(1, api), call(2, api)])
                for cap in (one, two):
                    cap.release.assert_called_once()
                    cap.set.assert_any_call(cv2.CAP_PROP_FRAME_WIDTH, 640)
                    cap.set.assert_any_call(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                    fourcc_calls = [item for item in cap.set.call_args_list if item.args[0] == cv2.CAP_PROP_FOURCC]
                    self.assertEqual(len(fourcc_calls), 0 if choice == 'msmf' else 1)

    def test_msmf_second_open_failure_cleans_up_and_explains_pair_failure(self):
        one, two = capture(), capture()
        two.isOpened.return_value = False
        with patch('magicpointer.cameras.platform.system', return_value='Windows'), \
                patch('cv2.VideoCapture', side_effect=[one, two]):
            with self.assertRaisesRegex(ValueError, 'Backend: msmf; earlier camera is already open'):
                Cameras(1, 2, backend='msmf')
        one.release.assert_called_once()
        two.release.assert_called_once()

    def test_default_macos_backend_and_explicit_windows_backend_rejection(self):
        with patch('magicpointer.cameras.platform.system', return_value='Darwin'), \
                patch('cv2.VideoCapture', return_value=capture()) as opened:
            with Cameras(0):
                pass
            opened.assert_called_once_with(0, cv2.CAP_ANY)
            opened.reset_mock()
            for choice in ('dshow', 'msmf'):
                with self.assertRaisesRegex(ValueError, 'require Windows'):
                    Cameras(0, backend=choice)
            opened.assert_not_called()

    def test_invalid_capture_settings_rejected_before_open(self):
        with patch('cv2.VideoCapture') as opened:
            for settings in ({'width': 0}, {'height': -1}, {'backend': 'unknown'}):
                with self.assertRaises(ValueError):
                    Cameras(**settings)
            opened.assert_not_called()
        for settings in (['--capture-width', '0'], ['--capture-height', '-1'], ['--camera-backend', 'unknown']):
            with redirect_stderr(io.StringIO()), self.assertRaises(SystemExit):
                parser().parse_args(['run', *settings])

    def test_probe_passes_backend_and_dimensions(self):
        device = MagicMock()
        device.__enter__.return_value.info = [{'index': 0, 'backend': 'msmf'}]
        with patch('magicpointer.cameras.Cameras', return_value=device) as opened, redirect_stdout(io.StringIO()):
            self.assertEqual(main(['cameras', '--limit', '2', '--camera-backend', 'msmf',
                                   '--capture-width', '640', '--capture-height', '480', '--fps', '30']), 0)
        self.assertEqual(opened.call_args_list, [call(i, fps='30', backend='msmf', width=640, height=480) for i in (0, 1)])


if __name__ == '__main__':
    unittest.main()
