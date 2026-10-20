from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

import numpy as np

from magicpointer import app
from magicpointer.cameras import Frame
from magicpointer.cli import parser
from magicpointer.engine import Gesture
from magicpointer.tracking import Hand


def debug_session(poses=None):
    """Exercise actual render/engine/dispatch loop, with synthetic camera input."""
    args = parser().parse_args(['run', '--debug', '--mute'])
    poses = poses or [Gesture.FIST]*12 + [Gesture.POINT]*5 + [Gesture.LEFT]*8 + [Gesture.POINT]*5 + [Gesture.PALM]*8
    pixels = np.array([(900.+(i%4)*25, 650.-(i//4)*30) for i in range(21)])
    camera = MagicMock()
    camera.__enter__.return_value = camera
    camera.info = [{'index': 0, 'width': 1920, 'height': 1080, 'reported_fps': 30}]
    frame = np.full((1080, 1920, 3), (45, 40, 35), dtype=np.uint8)
    tracker = MagicMock()
    index = [0]
    now = [1.]
    displays = []
    audio = MagicMock()

    def tick():
        now[0] += .03
        return now[0]

    def read():
        return [Frame(frame, now[0])]

    def detect(*_):
        pose = poses[min(index[0], len(poses)-1)]
        if isinstance(pose, Hand):
            pose.pixels = pixels.copy()
            return pose
        return Hand(pixels.copy(), [], 'Right', pose)

    def key(_):
        index[0] += 1
        return ord('q') if index[0] >= len(poses) else -1

    camera.read.side_effect = read
    tracker.detect.side_effect = detect
    with patch.object(app, 'Cameras', return_value=camera), patch.object(app, 'Tracker', return_value=tracker), \
            patch.object(app, 'Chimes', return_value=audio), patch.object(app.time, 'monotonic', side_effect=tick), \
            patch.object(app.cv2, 'imshow', side_effect=lambda _, panel: displays.append(panel.copy())), \
            patch.object(app.cv2, 'waitKey', side_effect=key), patch.object(app.cv2, 'getWindowProperty', return_value=1), \
            patch.object(app.cv2, 'destroyAllWindows'), patch('magicpointer.desktop.Desktop') as desktop:
        result = app.run(args)
        desktop.assert_not_called()
    return result, displays, tracker, camera, audio


class AppTests(unittest.TestCase):
    def test_debug_session_renders_states_and_never_constructs_desktop(self):
        result, displays, tracker, camera, audio = debug_session()
        self.assertEqual(result, 0)
        self.assertEqual(displays[0].shape, (540, 960, 3))
        self.assertGreater(len(displays), 30)
        self.assertIn(((True,), {}), [(c.args, c.kwargs) for c in audio.play.call_args_list])
        self.assertIn(((False,), {}), [(c.args, c.kwargs) for c in audio.play.call_args_list])
        tracker.close.assert_called_once()
        camera.__exit__.assert_called_once()
        audio.close.assert_called_once()

    def test_capture_failure_cleans_up_tracker_audio_and_camera(self):
        args = parser().parse_args(['run', '--debug'])
        camera, tracker, audio = MagicMock(), MagicMock(), MagicMock()
        camera.__enter__.return_value = camera
        camera.read.side_effect = ValueError('Camera disconnected')
        with patch.object(app, 'Cameras', return_value=camera), patch.object(app, 'Tracker', return_value=tracker), \
                patch.object(app, 'Chimes', return_value=audio), patch.object(app.cv2, 'destroyAllWindows'):
            with self.assertRaisesRegex(ValueError, 'disconnected'):
                app.run(args)
        tracker.close.assert_called_once()
        audio.close.assert_called_once()
        camera.__exit__.assert_called_once()
