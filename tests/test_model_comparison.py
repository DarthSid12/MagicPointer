import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch
import zipfile

import numpy as np

from magicpointer import comparison_app
from magicpointer.cameras import Frame
from magicpointer.cli import main, parser
from magicpointer.comparison import ComparisonLog
from magicpointer.comparison_models import RTMPoseModel, prediction, wilor_predictions
from magicpointer.worker_client import WiLoRWorker
from test_ownership_fusion import pose


class AdapterTests(unittest.TestCase):
    def test_no_detector_boxes_does_not_invoke_whole_image_pose_fallback(self):
        model = RTMPoseModel.__new__(RTMPoseModel)
        model.detector = MagicMock(return_value=np.empty((0, 4)))
        model.pose = MagicMock()
        model.limit = 4
        self.assertEqual(model.detect(np.zeros((20, 30, 3), np.uint8), 1., 0), [])
        model.pose.assert_not_called()

    def test_rtmpose_preserves_original_pixels_scores_and_absent_3d(self):
        model = RTMPoseModel.__new__(RTMPoseModel)
        model.detector = MagicMock(return_value=np.array([[10, 20, 100, 200]]))
        points, scores = np.arange(42).reshape(1, 21, 2), np.ones((1, 21))*.7
        model.pose = MagicMock(return_value=(points, scores))
        model.limit = 1
        result = model.detect(np.zeros((200, 200, 3), np.uint8), 1., 0)[0]
        self.assertEqual(result['pixels'], points[0].tolist())
        self.assertEqual(result['joint_scores'], scores[0].tolist())
        self.assertIsNone(result['world'])

    def test_wilor_native_projection_and_3d_order_are_preserved(self):
        points, world = np.arange(42).reshape(1, 21, 2), np.arange(63).reshape(1, 21, 3)
        result = wilor_predictions([{'hand_bbox': [1, 2, 30, 40], 'is_right': 0,
            'wilor_preds': {'pred_keypoints_2d': points, 'pred_keypoints_3d': world}}], 1)[0]
        self.assertEqual(result['label'], 'Left')
        self.assertEqual(result['pixels'], points[0].tolist())
        self.assertEqual(result['world'], world[0].tolist())
        with self.assertRaisesRegex(ValueError, 'invalid'):
            prediction(np.full((21, 2), np.nan))


class RecordingTests(unittest.TestCase):
    def test_separate_modes_labels_camera_denominators_and_trial_gaps(self):
        with tempfile.TemporaryDirectory() as directory:
            log = ComparisonLog(directory, {})
            for frame, trial, found, expected in [(0, 1, False, 'present'), (1, 1, False, 'present'),
                                                    (2, 2, False, 'present'), (3, 2, True, 'absent')]:
                log.append({'frame_id': frame, 'trial_id': trial, 'backend': 'rtmpose',
                    'mode': 'single', 'condition': 'far', 'expected': expected,
                    'views': [{'camera': 2, 'hands': [{}] if found else [], 'inference_ms': 12.}]})
            log.close()
            log.close()
            with zipfile.ZipFile(log.path.with_suffix('.zip')) as bundle:
                self.assertEqual(set(bundle.namelist()), {'session.json', 'summary.json', 'samples.jsonl'})
                row = json.loads(bundle.read('summary.json'))['results'][0]
                self.assertEqual(row['expected_present'], 3)
                self.assertEqual(row['found_when_expected'], 0)
                self.assertEqual(row['found_when_absent'], 1)
                self.assertEqual(row['longest_missing_run'], 2)
                self.assertFalse(json.loads(bundle.read('session.json'))['images_recorded'])


class ComparisonAppTests(unittest.TestCase):
    def test_switching_same_frames_capture_options_and_resource_cleanup(self):
        instances, calls, panels = [], [], []
        def factory(name):
            class Fake:
                info = {'name': name}
                def __init__(self, args, views):
                    self.closed = False
                    instances.append(self)
                def detect(self, image, timestamp, view):
                    calls.append((name, view, timestamp, id(image)))
                    value = pose()
                    return [prediction(value.pixels, world=value.world if name != 'rtmpose' else None, label='Right')]
                def close(self):
                    self.closed = True
            return Fake
        factories = {name: factory(name) for name in comparison_app.NAMES}
        camera = MagicMock()
        camera.__enter__.return_value = camera
        camera.info = [{'index': index} for index in (1, 2)]
        images = [np.full((480, 640, 3), color, np.uint8) for color in (25, 70)]
        camera.read.return_value = [Frame(image, 1.+view*.005) for view, image in enumerate(images)]
        with tempfile.TemporaryDirectory() as directory:
            args = parser().parse_args(['compare-models', '--left', '1', '--right', '2',
                '--capture-width', '640', '--capture-height', '480', '--output', directory])
            with patch.dict(comparison_app.FACTORIES, factories), \
                    patch.object(comparison_app, 'MediaPipeModel', factories['mediapipe']), \
                    patch.object(comparison_app, 'Cameras', return_value=camera) as cameras, \
                    patch.object(comparison_app.cv2, 'imshow', side_effect=lambda _, image: panels.append(image.copy())), \
                    patch.object(comparison_app.cv2, 'waitKey', side_effect=[ord('a'), ord('2'), ord('3'), ord('q')]), \
                    patch.object(comparison_app.cv2, 'getWindowProperty', return_value=1), \
                    patch.object(comparison_app.cv2, 'destroyAllWindows'), patch('magicpointer.desktop.Desktop') as desktop:
                self.assertEqual(comparison_app.run(args), 0)
                desktop.assert_not_called()
                cameras.assert_called_once_with(1, 2, '30', width=640, height=480, backend='default')
            with zipfile.ZipFile(next(Path(directory).glob('*.zip'))) as bundle:
                rows = [json.loads(line) for line in bundle.read('samples.jsonl').splitlines()]
        all_rows = [row for row in rows if row['mode'] == 'same-frame']
        self.assertEqual({row['backend'] for row in all_rows}, set(comparison_app.NAMES))
        self.assertEqual(len({row['frame_id'] for row in all_rows}), 1)
        self.assertTrue(all(instance.closed for instance in instances))
        self.assertEqual(panels[1].shape, (830, 1280, 3))
        self.assertTrue(all(call[3] == id(images[call[1]]) for call in calls))

    def test_duplicate_camera_rejected_before_opening_models(self):
        with patch.object(comparison_app, 'run') as run:
            self.assertEqual(main(['compare-models', '--left', '2', '--right', '2']), 1)
            run.assert_not_called()


class WorkerTests(unittest.TestCase):
    def test_actual_subprocess_receives_raw_bgr_bytes_and_closes(self):
        source = '''import json,sys
print('upstream progress output', flush=True)
print('MAGICPOINTER_JSON ' + json.dumps({'info': {'test': True}}), flush=True)
while True:
 header=sys.stdin.buffer.readline()
 if not header: break
 request=json.loads(header)
 size=1
 for dimension in request['shape']: size*=dimension
 data=sys.stdin.buffer.read(size)
 print('MAGICPOINTER_JSON '+json.dumps({'hands':[{'sum':sum(data)}], 'inference_ms':3.}),flush=True)
'''
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory)/'worker.py'
            script.write_text(source)
            original = subprocess.Popen
            args = SimpleNamespace(worker_timeout=5., wilor_python=sys.executable,
                                   model_dir=directory, wilor_device='cpu', max_hands=1)
            with patch('magicpointer.worker_client.subprocess.Popen',
                       side_effect=lambda command, **kwargs: original([sys.executable, '-u', str(script)], **kwargs)):
                worker = WiLoRWorker(args, 1)
                self.assertEqual(worker.detect(np.full((3, 4, 3), 7, np.uint8), 1., 0), [{'sum': 252}])
                self.assertEqual(worker.last_inference_ms, 3.)
                worker.close()
                self.assertEqual(worker.process.poll(), 0)
