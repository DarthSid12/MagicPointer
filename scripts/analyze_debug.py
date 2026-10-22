"""Summarize data-only debug archives and replay the owner-pause control policy."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import statistics
import zipfile

from magicpointer.engine import Engine, Gesture, Observation


def observation(row):
    value = row.get('observation')
    if not value:
        return None
    return Observation(Gesture(value['gesture']), tuple(value['wrist']), tuple(value['tip']),
                       value.get('scroll_pitch'), value.get('point_shape', False))


def event_counts(events):
    return dict(Counter(event.kind + (':'+str(event.value) if event.kind == 'deactivate' else '') for event in events))


def replay_owned(rows, metadata):
    start = next((i for i, row in enumerate(rows) if any(event['kind'] == 'activate' for event in row['events'])), None)
    if start is None:
        return {'note': 'No recorded activation to seed a comparable owned segment.'}
    initial = observation(rows[start])
    now = rows[start]['monotonic_seconds']
    output = {}
    for name, retain in [('legacy_deactivate', False), ('retain_owner_and_pause', True)]:
        engine = Engine(width=metadata['workspace_width'], height=metadata['workspace_height'],
                        mirror_x=metadata['mirror_x'], dwell=metadata['dwell_seconds'],
                        activation_dwell=metadata['activation_dwell_seconds'], loss_timeout=metadata['loss_timeout_seconds'],
                        retain_activation_on_loss=retain)
        # Both engines start from exactly the same recorded activation. Activation
        # grace is deliberately disabled: only the loss/recovery policy differs.
        fist = Observation(Gesture.FIST, initial.wrist, initial.tip)
        engine.step(fist, now-metadata['activation_dwell_seconds']-.01)
        events = engine.step(fist, now)
        paused = 0
        for row in rows[start+1:]:
            obs = observation(row)
            events.extend(engine.step(obs, row['monotonic_seconds']))
            paused += engine.paused
        output[name] = {'events': event_counts(events), 'paused_frames': paused}
    return {'first_recorded_activation_seconds': rows[start]['elapsed_seconds'], 'frames': len(rows)-start,
            'activation_grace_unchanged': True, 'activation_grace_seconds': 0., 'policies': output,
            'note': 'Same recorded selected-observation stream; no model inference, identity reassociation or OS input. This checks control logic, not gesture-intent accuracy.'}


def recorded_hands(camera):
    return camera.get('hands', [camera['hand']] if camera.get('hand') else [])


def analyze(path):
    path = Path(path)
    with zipfile.ZipFile(path) as archive:
        metadata = json.loads(archive.read('session.json'))
        rows = [json.loads(line) for line in archive.read('samples.jsonl').splitlines()]
    if metadata.get('kind') != 'debug-run' or metadata.get('schema_version') not in (1, 2):
        raise ValueError('Expected a version 1 or 2 debug-run archive')
    dt = [b['monotonic_seconds']-a['monotonic_seconds'] for a, b in zip(rows, rows[1:])]
    owned = [row for row in rows if row.get('selection', {}).get('owner_ids')]
    cameras = []
    for index in range(len(metadata['camera_indices'])):
        missing_despite_detection = 0
        for row in owned:
            hands = recorded_hands(row['cameras'][index])
            if hands and not any(hand.get('track_id') == row.get('selection', {}).get('owner_ids')[index] for hand in hands):
                missing_despite_detection += 1
        cameras.append({'index': metadata['camera_indices'][index],
                        'detected_hand_counts': dict(Counter(len(recorded_hands(row['cameras'][index])) for row in rows)),
                        'owner_missing_despite_other_detections': missing_despite_detection,
                        'raw_pose_estimates': dict(Counter(hand['gesture_estimate'] for row in rows for hand in recorded_hands(row['cameras'][index])))})
    result = {'source_sha256': hashlib.sha256(path.read_bytes()).hexdigest(), 'frames': len(rows),
            'duration_seconds': rows[-1]['elapsed_seconds'] if rows else 0.,
            'median_interval_ms': statistics.median(dt)*1000 if dt else None,
            'recorded_processing_fps': (len(dt)/(rows[-1]['monotonic_seconds']-rows[0]['monotonic_seconds'])) if dt else None,
            'pair_timing_rejected_frames': sum(not row['pair_timing_valid'] for row in rows),
            'status_counts': dict(Counter(row['reason'] for row in rows)), 'owned_frames': len(owned),
            'events': dict(Counter(event['kind']+(':'+str(event['value']) if event['kind']=='deactivate' else '') for row in rows for event in row['events'])),
            'shared_estimates': dict(Counter(row['selection']['combined_hand']['gesture_estimate'] if row.get('selection', {}).get('combined_hand') else 'unavailable' for row in rows)),
            'cameras': cameras, 'owned_control_replay': replay_owned(rows, metadata),
            'limitations': 'No images or intended-gesture labels. Detection counts and estimated poses are not accuracy measurements. No raw landmarks included in this summary.'}
    if metadata.get('adaptive_detection', {}).get('enabled'):
        attempts = [row for row in rows if row.get('adaptive_detection', {}).get('attempted_view') is not None]
        candidate_rows = [row for row in attempts if row['adaptive_detection']['candidate_count'] > 0]
        result['adaptive_detection'] = {
            'configuration': metadata['adaptive_detection'],
            'attempts': len(attempts), 'frames_with_candidates': len(candidate_rows),
            'candidate_frames_with_observation': sum(row['observation'] is not None for row in candidate_rows),
            'candidate_frame_statuses': dict(Counter(row['reason'] for row in candidate_rows)),
            'per_camera': [{'index': camera_index,
                'baseline_detected_frames': sum(bool(row['cameras'][i]['baseline_hands']) for row in rows),
                'effective_detected_frames': sum(bool(recorded_hands(row['cameras'][i])) for row in rows),
                'attempts': sum(row['adaptive_detection']['attempted_view'] == i for row in attempts)}
                for i, camera_index in enumerate(metadata['camera_indices'])],
            'note': 'Effective hands include lower-threshold candidates. Additional detections/observations do not establish correct hands or gestures.'}
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('archive', type=Path)
    parser.add_argument('--output', type=Path)
    args = parser.parse_args()
    result = json.dumps(analyze(args.archive), indent=2)+'\n'
    if args.output:
        args.output.write_text(result)
    else:
        print(result)


if __name__ == '__main__':
    main()
