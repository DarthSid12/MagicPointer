import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
import zipfile

from test_tracking import hand

spec = importlib.util.spec_from_file_location('evaluate_gestures', Path(__file__).resolve().parents[1]/'scripts/evaluate_gestures.py')
evaluation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evaluation)


class EvaluationTests(unittest.TestCase):
    def test_replay_reports_improvement_and_dwell_without_exposing_samples(self):
        points = hand(True, [1, 0, 0, 1])  # index + little finger right click
        points[2:5] = [[-.04, -.03, 0.], [-.05, -.055, 0.], [-.06, -.08, 0.]]
        samples = [{'step': 0, 'elapsed_seconds': t, 'detected': True,
                    'predicted': 'unrecognized', 'world_landmarks': points.tolist()} for t in (.03, .13, .23, .33)]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'sample.zip'
            with zipfile.ZipFile(path, 'w') as archive:
                archive.writestr('session.json', json.dumps({'schema_version': 1, 'steps': [{'gesture': 'right click', 'orientation': 'upright'}]}))
                archive.writestr('samples.jsonl', '\n'.join(json.dumps(row) for row in samples))
            report = evaluation.evaluate(path)
        self.assertEqual(report['recorded_correct'], 0)
        self.assertEqual(report['current_correct'], 4)
        self.assertTrue(report['steps'][0]['reaches_engine_dwell'])
        self.assertNotIn('world_landmarks', json.dumps(report))
        self.assertEqual(len(report['source_sha256']), 64)
