"""Data-only measurements for the isolated visual model experiment."""
from collections import defaultdict
from datetime import datetime, timezone
import json
import math
import platform
from pathlib import Path
import statistics
import uuid
import zipfile

from .debug_log import json_value, runtime_info
from . import __version__


class ComparisonLog:
    def __init__(self, output, metadata):
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        self.path = Path(output) / f'model-compare-{stamp}-{uuid.uuid4().hex[:8]}'
        self.path.mkdir(parents=True, exist_ok=False)
        self.metadata = {**runtime_info(), **metadata, 'kind': 'model-comparison',
                         'schema_version': 1, 'images_recorded': False,
                         'desktop_input': False, 'created_utc': stamp,
                         'app_version': __version__, 'os': platform.system(),
                         'python': platform.python_version()}
        self.rows = defaultdict(lambda: {'samples': 0, 'detected': 0,
                                        'expected_present': 0, 'found_when_expected': 0,
                                        'expected_absent': 0, 'found_when_absent': 0,
                                        'longest_missing_run': 0, 'missing_run': 0,
                                        'inference_ms': []})
        self.closed = False
        self.previous = {}
        self._write('session.json', self.metadata)
        self.stream = (self.path / 'samples.jsonl').open('w', encoding='utf-8')

    def _write(self, name, value):
        (self.path / name).write_text(json.dumps(json_value(value), indent=2,
                                               allow_nan=False), encoding='utf-8')

    def append(self, sample):
        self.stream.write(json.dumps(json_value(sample), allow_nan=False) + '\n')
        self.stream.flush()
        # Different execution modes and explicit labels get independent totals.
        for view in sample['views']:
            key = (sample['backend'], view['camera'], sample['mode'], sample['condition'])
            row = self.rows[key]
            found = bool(view['hands'])
            row['samples'] += 1
            row['detected'] += found
            row['inference_ms'].append(view['inference_ms'])
            continuity = (sample['trial_id'], sample['frame_id'])
            previous = self.previous.get(key)
            if previous is None or previous != (continuity[0], continuity[1]-1):
                row['missing_run'] = 0
            self.previous[key] = continuity
            row['missing_run'] = 0 if found else row['missing_run'] + 1
            row['longest_missing_run'] = max(row['longest_missing_run'], row['missing_run'])
            if sample['expected'] == 'present':
                row['expected_present'] += 1
                row['found_when_expected'] += found
            elif sample['expected'] == 'absent':
                row['expected_absent'] += 1
                row['found_when_absent'] += found

    def close(self):
        if self.closed:
            return
        self.stream.close()
        rows = []
        for (backend, camera, mode, condition), values in self.rows.items():
            row = {key: value for key, value in values.items()
                   if key not in ('inference_ms', 'missing_run')}
            timing = sorted(values['inference_ms'])
            rows.append({'backend': backend, 'camera': camera, 'mode': mode,
                         'condition': condition, **row,
                         'median_inference_ms': statistics.median(timing),
                         'p95_inference_ms': timing[math.ceil(.95*len(timing))-1]})
        self._write('summary.json', {'results': rows,
                    'interpretation': 'Presence counts are not joint/gesture accuracy. Labels are user supplied; different live trials are not identical scenes.'})
        self._write('session.json', self.metadata)
        archive = self.path.with_suffix('.zip')
        with zipfile.ZipFile(archive, 'w', zipfile.ZIP_DEFLATED) as bundle:
            for name in ('session.json', 'samples.jsonl', 'summary.json'):
                bundle.write(self.path / name, arcname=name)
        self.closed = True
        print(f'Model comparison saved: {archive}')
