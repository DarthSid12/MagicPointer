"""Continuous local diagnostics for main-app debug sessions."""
from collections import Counter
from dataclasses import asdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
import platform
import time
import uuid
import zipfile

from . import __version__
from .tracking import CLASSIFIER_VERSION, gesture_metrics


def json_value(value):
    """Keep malformed model output inspectable in standards-compliant JSON."""
    if hasattr(value, 'tolist'):
        value = value.tolist()
    if isinstance(value, dict):
        return {key: json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [json_value(item) for item in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def hand_sample(hand):
    if hand is None:
        return None
    return {'gesture_estimate': hand.gesture.value, 'handedness': hand.label,
            'handedness_score': hand.handedness_score, 'pixels': hand.pixels,
            'world_landmarks': hand.world, 'image_landmarks': hand.image_landmarks,
            'metrics': gesture_metrics(hand.world)}


class DebugLog:
    def __init__(self, output, metadata):
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        self.path = Path(output)/f'debug-{stamp}-{uuid.uuid4().hex[:8]}'
        self.path.mkdir(parents=True, exist_ok=False)
        self.metadata = {'schema_version': 1, 'kind': 'debug-run',
                         'created_utc': stamp, 'app_version': __version__,
                         'classifier_version': CLASSIFIER_VERSION,
                         'os': platform.system(), 'python': platform.python_version(),
                         'images_recorded': False, 'desktop_input': False, **metadata}
        self.started = time.monotonic()
        self.counts = Counter()
        self.closed = False
        self._write('session.json', self.metadata)
        self.stream = (self.path/'samples.jsonl').open('w', encoding='utf-8')
        print(f'Debug data recording: {self.path}')

    def _write(self, name, value):
        (self.path/name).write_text(json.dumps(json_value(value), allow_nan=False, indent=2), encoding='utf-8')

    def append(self, frames, hands, observation, engine, events, now, *, reason, paired, verified, state):
        sample = {'elapsed_seconds': now-self.started, 'monotonic_seconds': now,
                  'cameras': [{'grabbed_at': frame.grabbed_at,
                               'image_size': [frame.image.shape[1], frame.image.shape[0]],
                               'hand': hand_sample(hand)} for frame, hand in zip(frames, hands)],
                  'observation': asdict(observation) if observation else None,
                  'reason': reason, 'pair_timing_valid': paired, 'markers_verified': verified,
                  'state': state, 'active': engine.active, 'scrolling': engine.scrolling,
                  'cursor': engine.cursor, 'origin': engine.origin,
                  'scroll': {'neutral_degrees': engine.scroll.neutral,
                             'smoothed_degrees': engine.scroll.filtered,
                             'delta_degrees': engine.scroll.delta,
                             'down_steps_per_second': engine.scroll.speed},
                  'events': [asdict(event) for event in events]}
        self.stream.write(json.dumps(json_value(sample), allow_nan=False)+'\n')
        self.stream.flush()
        self.counts['frames'] += 1
        self.counts['accepted_observations'] += observation is not None
        self.counts['scrolling_frames'] += engine.scrolling
        for event in events:
            self.counts[event.kind+'_events'] += 1

    def close(self):
        if self.closed:
            return
        self.stream.close()
        self._write('session.json', self.metadata)
        self._write('summary.json', dict(self.counts))
        archive = self.path.with_suffix('.zip')
        with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
            for name in ('session.json', 'samples.jsonl', 'summary.json'):
                bundle.write(self.path/name, arcname=name)
        self.closed = True
        print(f'Debug diagnostics saved: {archive}')
