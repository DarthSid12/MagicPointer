import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from test_wrist_tilt import camera_pose

spec = importlib.util.spec_from_file_location('replay_wrist_tilt', Path(__file__).resolve().parents[1]/'scripts/replay_wrist_tilt.py')
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


class WristReplayTests(unittest.TestCase):
    def test_open_thumb_down_sequence_replays_to_scroll_after_neutral_capture(self):
        samples = []
        for i in range(60):
            sample = camera_pose(0. if i < 20 else 30., thumb=i >= 20)
            samples.append({'world_landmarks': sample.world, 'handedness': sample.label,
                            'normal_gesture': sample.gesture.value, 'elapsed_seconds': (i+1)*.03,
                            'key': 'neutral' if i == 19 else None, 'valid': i < 20,
                            'phase': 'neutral', 'lines_per_second': 0., 'reason': 'legacy thumb rejection'})
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'recording.zip'
            with zipfile.ZipFile(path, 'w') as archive:
                archive.writestr('session.json', json.dumps({'schema_version': 1, 'kind': 'wrist-tilt',
                                  'dead_zone_degrees': 10., 'full_speed_angle_degrees': 30.}))
                archive.writestr('samples.jsonl', '\n'.join(json.dumps(row) for row in samples))
            report = evaluation.replay(path)
        self.assertTrue(report['neutral_captures'][0]['accepted'])
        self.assertEqual(report['recorded']['down_command_frames'], 0)
        self.assertGreater(report['replayed']['down_command_frames'], 30)
        self.assertGreater(report['replayed']['maximum_down_lines_per_second'], 11.9)
        self.assertNotIn('world_landmarks', json.dumps(report))
