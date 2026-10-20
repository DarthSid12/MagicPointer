"""Guided labelled landmark capture; never constructs a pointer controller."""
from collections import Counter
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import time
import uuid
import zipfile

from . import __version__
from .cameras import Cameras
from .engine import Gesture
from .tracking import CLASSIFIER_VERSION, Tracker, gesture_metrics

POSES = [
    (Gesture.FIST, "Close all fingers into a fist; tuck thumb in"),
    (Gesture.POINT, "Extend index ONLY; curl middle/ring/little; tuck thumb in"),
    (Gesture.LEFT, "Extend index + thumb; curl middle/ring/little"),
    (Gesture.SCROLL, "Extend index + middle; curl ring/little; tuck thumb in"),
    (Gesture.RIGHT, "Extend index + middle + thumb; curl ring/little"),
    (Gesture.PALM, "Open ALL five fingers into a natural palm"),
]
ANGLES = [
    ("upright", "Palm faces camera; index direction UP"),
    ("roll_left", "Palm faces camera; rotate wrist so fingers point LEFT"),
    ("roll_right", "Palm faces camera; rotate wrist so fingers point RIGHT"),
    ("yaw_left", "Turn palm 30-45 degrees LEFT; keep pose visible"),
    ("yaw_right", "Turn palm 30-45 degrees RIGHT; keep pose visible"),
]
STEPS = [{"gesture": gesture.value, "pose": pose, "orientation": orientation, "angle": instruction}
         for gesture, pose in POSES for orientation, instruction in ANGLES]


def record(hand, step, elapsed, frame):
    result = {"step": step, "elapsed_seconds": elapsed, "detected": hand is not None,
              "image_size": [frame.shape[1], frame.shape[0]]}
    if hand is not None:
        result.update({"predicted": hand.gesture.value, "handedness": hand.label,
                       "handedness_score": hand.handedness_score,
                       "pixels": hand.pixels.tolist(), "world_landmarks": hand.world,
                       "image_landmarks": hand.image_landmarks,
                       "metrics": gesture_metrics(hand.world)})
    return result


class Session:
    def __init__(self, output, metadata):
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        self.path = Path(output) / f"gesture-{stamp}-{uuid.uuid4().hex[:8]}"
        self.path.mkdir(parents=True, exist_ok=False)
        self.metadata = {"schema_version": 1, "app_version": __version__, "created_utc": stamp,
                         "classifier_version": CLASSIFIER_VERSION,
                         "steps": STEPS, "state_legend": {"extended": 1, "curled": 0, "uncertain": -1},
                         "notes": "Handedness score is not landmark or gesture confidence.",
                         **metadata}
        (self.path / 'session.json').write_text(json.dumps(self.metadata, indent=2, allow_nan=False))
        self.stream = (self.path / 'samples.jsonl').open('w')
        self.counts = [{"frames": 0, "detected": 0, "matches": 0, "predictions": Counter(),
                        "finger_states": {name: Counter() for name in ('thumb', 'index', 'middle', 'ring', 'little')}}
                       for _ in STEPS]
        self.finished, self.skipped = set(), set()
        self.closed = False

    def append(self, sample):
        encoded = json.dumps(sample, allow_nan=False)
        self.stream.write(encoded + '\n')
        self.stream.flush()
        row = self.counts[sample['step']]
        row['frames'] += 1
        if sample['detected']:
            row['detected'] += 1
            row['predictions'][sample['predicted']] += 1
            row['matches'] += sample['predicted'] == STEPS[sample['step']]['gesture']
            metrics = sample['metrics']
            if metrics['valid']:
                for name in row['finger_states']:
                    source = metrics['thumb'] if name == 'thumb' else metrics['fingers'][name]
                    row['finger_states'][name][str(source['state'])] += 1

    def close(self):
        if self.closed:
            return
        self.stream.close()
        rows = []
        for index, (step, count) in enumerate(zip(STEPS, self.counts)):
            rows.append({**step, **count, "complete": index in self.finished, "skipped": index in self.skipped,
                         "detection_rate": count['detected']/count['frames'] if count['frames'] else None,
                         "match_rate_on_detected": count['matches']/count['detected'] if count['detected'] else None})
        (self.path / 'summary.json').write_text(json.dumps({"steps": rows}, indent=2))
        archive = self.path.with_suffix('.zip')
        with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
            for name in ('session.json', 'samples.jsonl', 'summary.json'):
                bundle.write(self.path/name, arcname=name)
        self.closed = True
        print(f"\nSaved {self.path}\nShare this diagnostic archive: {archive}")


def run(args):
    import cv2
    from .app import draw_hand
    tracker, session = None, None
    window = 'MagicPointer guided diagnostics - NO desktop input'
    try:
        tracker = Tracker(args.model)
        with Cameras(args.camera, fps=args.fps) as cameras:
            session = Session(args.output, {"camera": cameras.info, "requested_fps": args.fps,
                              "seconds_per_step": args.seconds, "os": platform.system(),
                              "python": platform.python_version(), "mediapipe": version('mediapipe')})
            print("30 poses: six gestures at five angles. SPACE records, S skips, Q saves and exits.")
            print("No video/images recorded; landmarks stay local. Keep one hand in view.")
            index, started, announced = 0, None, None
            while index < len(STEPS):
                frame = cameras.read()[0]
                hand = tracker.detect(frame.image, frame.grabbed_at)
                now = time.monotonic()
                step = STEPS[index]
                if announced != index:
                    print(f"\n{index+1}/30: {step['gesture']} / {step['orientation']}\n{step['pose']}\n{step['angle']}\nHold the pose, then press SPACE.")
                    announced = index
                recording = started is not None
                if recording:
                    session.append(record(hand, index, now-started, frame.image))
                panel = frame.image.copy()
                color = (40, 90, 255) if recording else (50, 220, 120)
                draw_hand(panel, hand, color)
                panel = cv2.resize(panel, (1100, 619))
                lines = [f"{index+1}/30  Expected: {step['gesture']}  Angle: {step['orientation']}",
                         step['pose'], step['angle'],
                         f"Detected: {hand.gesture.value if hand else 'NO HAND'}",
                         f"RECORDING {max(0, args.seconds-(now-started)):.1f}s - hold steady" if recording else 'READY - pose first, then SPACE to record',
                         'SPACE record | S skip | Q / Esc save and exit | no pointer input']
                if hand is not None:
                    metrics = gesture_metrics(hand.world)
                    if metrics['valid']:
                        states = metrics['signature']
                        lines.append(f"thumb/index/middle/ring/little: {states}  (1 open, 0 curled, -1 uncertain)")
                cv2.rectangle(panel, (0, 0), (1100, 220), (25, 25, 25), -1)
                for line_index, line in enumerate(lines):
                    cv2.putText(panel, line, (12, 25+line_index*29), cv2.FONT_HERSHEY_SIMPLEX, .60, color, 1, cv2.LINE_AA)
                cv2.imshow(window, panel)
                key = cv2.waitKey(1) & 255
                if key in (27, ord('q')) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    break
                if recording:
                    if now-started >= args.seconds:
                        session.finished.add(index)
                        index, started = index+1, None
                elif key == ord(' '):
                    started = time.monotonic()
                elif key == ord('s'):
                    session.skipped.add(index)
                    index += 1
            return 0
    finally:
        try:
            if session is not None:
                session.close()
        finally:
            if tracker is not None:
                tracker.close()
            cv2.destroyAllWindows()
