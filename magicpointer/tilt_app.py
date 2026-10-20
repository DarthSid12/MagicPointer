"""Single-camera tilt experiment; renders a virtual page and records local data."""
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import time
import uuid
import zipfile

import cv2
import numpy as np

from .cameras import Cameras
from .tilt import Tilt
from .tracking import CLASSIFIER_VERSION, Tracker


class Log:
    def __init__(self, output, metadata):
        stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
        self.path = Path(output)/f'tilt-{stamp}-{uuid.uuid4().hex[:8]}'
        self.path.mkdir(parents=True, exist_ok=False)
        (self.path/'session.json').write_text(json.dumps({
            'schema_version': 1, 'kind': 'finger-tilt', 'created_utc': stamp,
            'classifier_version': CLASSIFIER_VERSION, 'smoothing_seconds': .10,
            'loss_timeout_seconds': .30, 'max_lines_per_second': 12,
            'note': 'Palm-relative pitch; preview only. No video, audio, or OS input.', **metadata}, indent=2))
        self.stream = (self.path/'samples.jsonl').open('w')
        self.counts = Counter()
        self.closed = False

    def append(self, result, hand, elapsed, key, controller):
        points = None
        if hand is not None:
            # Invalid model outputs remain inspectable without invalid JSON.
            points = [[float(x) if np.isfinite(x) else None for x in point] for point in hand.world]
        sample = {**result, 'elapsed_seconds': elapsed, 'key': key,
                  'normal_gesture': hand.gesture.value if hand else None,
                  'handedness': hand.label if hand else None, 'world_landmarks': points,
                  'neutral_degrees': controller.neutral, 'down_span_degrees': controller.span,
                  'down_sign': controller.direction}
        self.stream.write(json.dumps(sample, allow_nan=False)+'\n')
        self.stream.flush()
        self.counts['frames'] += 1
        self.counts['valid_shape_frames'] += bool(result['valid'])
        self.counts['live_frames'] += result['phase'] == 'live'
        self.counts['normal_scroll_frames'] += hand is not None and hand.gesture.value == 'scroll'

    def close(self):
        if self.closed:
            return
        self.stream.close()
        (self.path/'summary.json').write_text(json.dumps(dict(self.counts), indent=2))
        with zipfile.ZipFile(self.path.with_suffix('.zip'), 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
            for name in ('session.json', 'samples.jsonl', 'summary.json'):
                bundle.write(self.path/name, arcname=name)
        self.closed = True
        print(f'Tilt diagnostics saved: {self.path.with_suffix(".zip")}')


def render(image, hand, result, offset, message):
    from .app import draw_hand
    panel = np.full((720, 1200, 3), 25, dtype=np.uint8)
    camera = image.copy()
    color = (80, 220, 100) if result['valid'] else (0, 170, 255)
    draw_hand(camera, hand, color)
    panel[220:670, :800] = cv2.resize(camera, (800, 450))
    pitch = result['raw_degrees']
    delta = result['delta_degrees']
    speed = result['lines_per_second']
    instruction = {'neutral': 'Extend index + middle; hold comfortable neutral; press SPACE',
                   'down': 'Tilt fingers in your DOWN direction, hold steady, then press D',
                   'live': 'Tilt down/up for continuous scroll; return to neutral to stop'}[result['phase']]
    movement = 'STOP' if abs(speed) < .01 else ('DOWN' if speed > 0 else 'UP')
    lines = ['FINGER TILT EXPERIMENT / SIMULATED SCROLL ONLY', instruction,
             f"{result['reason']} | Normal classifier: {hand.gesture.value if hand else 'no hand'}",
             f"Pitch: {pitch:+.1f} deg" if pitch is not None else 'Pitch: unavailable',
             f"From neutral: {delta:+.1f} deg | {movement} {abs(speed):.1f} lines/s" if delta is not None else 'Capture neutral and down range to start',
             message, 'SPACE neutral | D down range | R reset | Q / Esc save and exit']
    for i, line in enumerate(lines):
        cv2.putText(panel, line, (14, 27+28*i), cv2.FONT_HERSHEY_SIMPLEX, .63, color, 1, cv2.LINE_AA)
    # The page moves under a fixed virtual cursor; no desktop adapter is imported.
    cv2.rectangle(panel, (820, 220), (1180, 670), (220, 220, 220), 1)
    cv2.putText(panel, 'SIMULATED DOCUMENT', (835, 248), cv2.FONT_HERSHEY_SIMPLEX, .55, (220, 220, 220), 1)
    first = int(offset//34)
    for row in range(15):
        y = int(275+row*34-offset % 34)
        if 275 <= y <= 650:
            cv2.putText(panel, f'Line {first+row+1:04d}  Sample content', (835, y), cv2.FONT_HERSHEY_SIMPLEX, .48, (180, 180, 180), 1)
    cv2.drawMarker(panel, (1000, 430), (70, 210, 255), cv2.MARKER_CROSS, 20, 2)
    cv2.line(panel, (850, 690), (1150, 690), (100, 100, 100), 3)
    cv2.circle(panel, (1000+int(speed/12*150), 690), 7, color, -1)
    return panel


def run(args):
    controller = Tilt(args.dead_zone)
    tracker = log = None
    window = 'MagicPointer finger tilt - no desktop input'
    try:
        tracker = Tracker(args.model)
        with Cameras(args.camera, fps=args.fps) as cameras:
            log = Log(args.output, {'camera': cameras.info, 'dead_zone_degrees': args.dead_zone})
            print('Tilt experiment: SPACE neutral, D down range, R reset, Q save/exit. No OS input.')
            print('Keep your palm steady and tilt the extended fingers at their base knuckles.')
            start = previous = time.monotonic()
            offset, message, message_until = 1000., 'Hold each calibration pose steady for half a second', 0.
            while True:
                frame = cameras.read()[0]
                hand = tracker.detect(frame.image, frame.grabbed_at)
                now = time.monotonic()
                result = controller.update(hand, now)
                offset = max(0., min(50000., offset + result['lines_per_second']*34*min(now-previous, .1)))
                previous = now
                panel = render(frame.image, hand, result, offset, message if now < message_until else '')
                cv2.imshow(window, panel)
                key = cv2.waitKey(1) & 255
                action = None
                try:
                    if key == ord(' '):
                        action = 'neutral'
                        controller.capture_neutral()
                        message = 'Neutral captured. Tilt fingers down, hold, then press D.'
                    elif key == ord('d'):
                        action = 'down'
                        controller.capture_down()
                        message = 'Range captured. Try neutral, down, up; then move/rotate the whole hand.'
                    elif key == ord('r'):
                        action = 'reset'
                        controller.reset()
                        message = 'Reset. Capture neutral again with SPACE.'
                    if action:
                        message_until = now + 5.
                except ValueError as error:
                    message, message_until = str(error), now+5.
                log.append(result, hand, now-start, action, controller)
                if key in (ord('q'), 27) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    return 0
    finally:
        try:
            if log:
                log.close()
        finally:
            if tracker:
                tracker.close()
            cv2.destroyAllWindows()
