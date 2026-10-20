"""Replay a local diagnostic archive; output aggregates, never raw landmarks."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import zipfile

from magicpointer.tracking import CLASSIFIER_VERSION, classify


def evaluate(path):
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        metadata = json.loads(archive.read('session.json'))
        samples = [json.loads(line) for line in archive.read('samples.jsonl').splitlines()]
    if metadata.get('schema_version') != 1:
        raise ValueError('Unsupported diagnostic schema')
    groups = [[] for _ in metadata['steps']]
    for sample in samples:
        index = sample['step']
        if not isinstance(index, int) or not 0 <= index < len(groups):
            raise ValueError('Invalid sample step')
        groups[index].append(sample)
    rows = []
    for index, (step, group) in enumerate(zip(metadata['steps'], groups)):
        expected = step['gesture']
        old, new = Counter(), Counter()
        start, previous, longest = None, None, 0.
        for sample in group:
            detected = sample['detected']
            old[sample.get('predicted', 'no hand')] += 1
            prediction = classify(sample['world_landmarks']).value if detected else 'no hand'
            new[prediction] += 1
            now = sample['elapsed_seconds']
            if previous is not None and (now < previous or now-previous > .30):
                start = None
            if prediction == expected:
                if start is None:
                    start = now
                longest = max(longest, now-start)
            else:
                start = None
            previous = now
        rows.append({'step': index, 'expected': expected, 'orientation': step['orientation'],
                     'frames': len(group), 'recorded_correct': old[expected], 'current_correct': new[expected],
                     'recorded_predictions': dict(old), 'current_predictions': dict(new),
                     'longest_correct_run_seconds': round(longest, 4),
                     'reaches_engine_dwell': longest >= (.45 if expected == 'fist' else .18)})
    count = len(samples)
    old_total, new_total = (sum(row[key] for row in rows) for key in ('recorded_correct', 'current_correct'))
    return {'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(),
            'classifier_version': CLASSIFIER_VERSION, 'frames': count,
            'recorded_correct': old_total, 'current_correct': new_total,
            'recorded_accuracy': old_total/count if count else None,
            'current_accuracy': new_total/count if count else None,
            'note': 'Replay of the supplied labelled session used to diagnose and tune this correction; not independent accuracy validation.',
            'steps': rows}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    report = evaluate(args.archive)
    text = json.dumps(report, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + '\n')
    else:
        print(text)


if __name__ == '__main__':
    main()
