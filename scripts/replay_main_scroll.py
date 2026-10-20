"""Replay wrist landmarks through main-app scrolling without cameras or OS IO."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import zipfile

from magicpointer.engine import Engine, Gesture, Observation
from magicpointer.tilt import measure_wrist
from magicpointer.tracking import Hand, gesture_metrics


def thumb_summary(path):
    with zipfile.ZipFile(path) as archive:
        rows = [json.loads(line) for line in archive.read('samples.jsonl').splitlines()]
    all_states, live_states = Counter(), Counter()
    for row in rows:
        if row['world_landmarks'] is None:
            continue
        metrics = gesture_metrics(row['world_landmarks'])
        if not metrics['valid']:
            continue
        state = {0: 'tucked_estimate', 1: 'open_estimate', -1: 'uncertain_estimate'}[metrics['thumb']['state']]
        all_states[state] += 1
        if row['phase'] == 'live':
            live_states[state] += 1
    return {'archive': Path(path).name, 'sha256': hashlib.sha256(Path(path).read_bytes()).hexdigest(),
            'detected_estimates': dict(all_states), 'recorded_live_estimates': dict(live_states)}


def replay(path):
    with zipfile.ZipFile(path) as archive:
        rows = [json.loads(line) for line in archive.read('samples.jsonl').splitlines()]
    start = next(i for i, row in enumerate(rows) if row['key'] == 'neutral' and row['neutral_degrees'] is not None)
    engine = Engine()
    # The experiment has no activation or automatic entry. Seed those explicitly;
    # this checks latched scrolling on real landmarks, not gesture entry accuracy.
    now = 0.
    for _ in range(40):
        now += .02
        engine.step(Observation(Gesture.FIST, (0., 0., 2.), (0., 0., 2.)), now)
    neutral = rows[start]['neutral_degrees']
    for _ in range(60):
        now += .02
        engine.step(Observation(Gesture.SCROLL, (0., 0., 2.), (0., 0., 2.), neutral), now)
    origin = rows[start]['elapsed_seconds']
    offset = now
    events = []
    for row in rows[start+1:]:
        world = row['world_landmarks']
        gesture = Gesture(row['normal_gesture'] or Gesture.UNKNOWN)
        hand = Hand(None, world, row['handedness'], gesture) if world is not None else None
        pitch, _ = measure_wrist(hand)
        obs = Observation(gesture, (0., 0., 2.), (0., 0., 2.), pitch) if hand else None
        events.extend(engine.step(obs, offset+row['elapsed_seconds']-origin))
        # Later Space presses belong to the experiment, not the main app.
    scroll = [event.value for event in events if event.kind == 'scroll']
    return {'seeded_activation_and_scroll_entry': True, 'replayed_frames': len(rows)-start-1,
            'up_wheel_steps': sum(max(0, n) for n in scroll),
            'down_wheel_steps': sum(max(0, -n) for n in scroll),
            'click_events': sum(e.kind == 'click' for e in events),
            'pointer_move_events': sum(e.kind == 'move' for e in events),
            'deactivation_events': sum(e.kind == 'deactivate' for e in events)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archives', nargs='+', type=Path)
    parser.add_argument('--replay', type=Path)
    args = parser.parse_args()
    report = {'thumb_estimates': [thumb_summary(path) for path in args.archives],
              'note': 'Landmark estimates cannot establish actual thumb posture. No video or intent labels. No OS input.'}
    if args.replay:
        report['main_app_replay'] = replay(args.replay)
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
