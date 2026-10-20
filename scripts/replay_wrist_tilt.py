"""Replay wrist experiment key events and landmarks without cameras or OS input."""
from collections import Counter
import argparse
import hashlib
import json
from pathlib import Path
import zipfile

from magicpointer.engine import Gesture
from magicpointer.tilt import WristTilt
from magicpointer.tracking import Hand


def summarize(rows):
    speeds = [row['lines_per_second'] for row in rows]
    return {'frames': len(rows), 'valid_frames': sum(row['valid'] for row in rows),
            'live_frames': sum(row['phase'] == 'live' for row in rows),
            'down_command_frames': sum(speed > 0 for speed in speeds),
            'up_command_frames': sum(speed < 0 for speed in speeds),
            'maximum_down_lines_per_second': max([0.] + speeds),
            'maximum_up_lines_per_second': max([0.] + [-speed for speed in speeds]),
            'reasons': dict(Counter(row['reason'] for row in rows))}


def replay(path):
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        metadata = json.loads(archive.read('session.json'))
        samples = [json.loads(line) for line in archive.read('samples.jsonl').splitlines()]
    if metadata.get('schema_version') != 1 or metadata.get('kind') != 'wrist-tilt':
        raise ValueError('Expected a version-1 wrist-tilt diagnostic archive')
    controller = WristTilt(metadata['dead_zone_degrees'], metadata['full_speed_angle_degrees'])
    outputs, keys = [], []
    for sample in samples:
        world = sample['world_landmarks']
        hand = Hand(None, world, sample['handedness'], Gesture(sample['normal_gesture'] or 'unrecognized')) if world is not None else None
        result = controller.update(hand, sample['elapsed_seconds'])
        outputs.append(result)
        if sample['key'] == 'neutral':
            try:
                controller.capture_neutral()
                keys.append({'seconds': sample['elapsed_seconds'], 'accepted': True,
                             'neutral_degrees': controller.neutral})
            except ValueError as error:
                keys.append({'seconds': sample['elapsed_seconds'], 'accepted': False, 'reason': str(error)})
        elif sample['key'] == 'reset':
            controller.reset()
    return {'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'recorded': summarize(samples), 'replayed': summarize(outputs), 'neutral_captures': keys,
            'note': 'Replay of the supplied unlabelled session used for diagnosis. Command counts are simulated output, not intended-direction accuracy. No raw landmarks are included in this report.'}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    text = json.dumps(replay(args.archive), indent=2) + '\n'
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    else:
        print(text)


if __name__ == '__main__':
    main()
