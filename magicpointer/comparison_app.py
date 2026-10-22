"""Switchable visual-only model trials, with optional identical-frame comparison."""
import hashlib
import io
from pathlib import Path
import time
import urllib.request
import zipfile

import cv2
import numpy as np

from .cameras import Cameras
from .comparison import ComparisonLog
from .comparison_models import MediaPipeModel, RTMPoseModel, RTM_MODELS
from .worker_client import WiLoRWorker

NAMES = ('mediapipe', 'rtmpose', 'wilor')
FACTORIES = {'mediapipe': MediaPipeModel, 'rtmpose': RTMPoseModel, 'wilor': WiLoRWorker}
EDGES = [(0, start) for start in (1, 5, 9, 13, 17)] + [
    (start+i, start+i+1) for start in (1, 5, 9, 13, 17) for i in range(3)]
# Filled from the versioned official archives; model files are also fingerprinted
# in every session. No third-party extraction routine or archive paths are used.
ARCHIVE_SHA256 = {
    'detector': '9c0370a43c02b2fe42b4382aba7383d97cfa3ed35623b655cac4f0c25cfde402',
    'pose': '45c9e1aa21b8fe33859973c49bce4647645d312e7a226e398a42f9137d44a56d',
}


def prepare_rtm(directory):
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    for name, url in RTM_MODELS.items():
        print(f'Downloading official RTMPose {name}...')
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read(100*1024*1024)
        if hashlib.sha256(data).hexdigest() != ARCHIVE_SHA256[name]:
            raise ValueError(f'Official {name} archive checksum mismatch')
        with zipfile.ZipFile(io.BytesIO(data)) as bundle:
            names = [item for item in bundle.namelist() if item.endswith('.onnx')]
            if len(names) != 1:
                raise ValueError('Expected exactly one ONNX file in official model archive')
            payload = bundle.read(names[0])
        target = directory / f'{name}.onnx'
        temporary = target.with_suffix('.download')
        try:
            temporary.write_bytes(payload)
            temporary.replace(target)
        finally:
            temporary.unlink(missing_ok=True)


def draw_panel(frames, results, mode, expected, condition, fps, cameras, gestures=None):
    names = list(results)
    cell_width = 1280 // len(frames)
    cell_height = 230 if len(names) > 1 else 480
    panel = np.full((140+cell_height*len(names), 1280, 3), 24, np.uint8)
    lines = [f'MODEL COMPARISON / NO DESKTOP INPUT / {mode.upper()} / {fps:.1f} loop FPS',
             '1 MediaPipe | 2 RTMPose-Hand | 3 WiLoR | A identical-frame comparison | Q/Esc exit',
             f'H hand present | N no hands | U unknown: {expected} | D near/far: {condition}',
             'REC data only | Inspect false hands, missing joints and jitter; counts alone are not accuracy']
    for index, line in enumerate(lines):
        cv2.putText(panel, line, (15, 26+index*28), cv2.FONT_HERSHEY_SIMPLEX, .56, (220, 235, 225), 1)
    for row, name in enumerate(names):
        for column, frame in enumerate(frames):
            image = frame.image.copy()
            view = results[name][column]
            for hand in view['hands']:
                points = np.clip(np.asarray(hand['pixels']), [-image.shape[1], -image.shape[0]],
                                 [2*image.shape[1], 2*image.shape[0]])
                scores = hand['joint_scores']
                valid = np.ones(21, bool) if scores is None else np.asarray(scores) >= .3
                for a, b in EDGES:
                    if valid[a] and valid[b]:
                        cv2.line(image, tuple(np.rint(points[a]).astype(int)), tuple(np.rint(points[b]).astype(int)), (80, 240, 160), 2)
                for point, good in zip(points, valid):
                    if good:
                        cv2.circle(image, tuple(np.rint(point).astype(int)), 4, (80, 240, 160), -1)
            scale = min(cell_width/image.shape[1], (cell_height-35)/image.shape[0])
            preview = cv2.resize(image, (round(image.shape[1]*scale), round(image.shape[0]*scale)))
            x, y = column*cell_width, 140+row*cell_height
            gesture = (gestures or {}).get(name, {}).get('gesture') or ('2D joints' if name == 'rtmpose' else 'select fist')
            title = f'{name} / cam {cameras[column]} / {len(view["hands"])} hands / {view["inference_ms"]:.0f} ms / {gesture}'
            cv2.putText(panel, title, (x+10, y+23), cv2.FONT_HERSHEY_SIMPLEX, .62, (110, 235, 185), 1)
            panel[y+35:y+35+preview.shape[0], x:x+preview.shape[1]] = preview
    return panel


def run(args):
    models, log = {}, None
    views = 1 if args.right is None else 2
    enabled = [name for name in NAMES if name in args.models]
    try:
        if args.prepare:
            if 'mediapipe' in enabled:
                from .model import download
                download(args.model)
            if 'rtmpose' in enabled:
                prepare_rtm(args.model_dir)
        for name in enabled:
            print(f'Loading {name}; first WiLoR load may download large model assets...', flush=True)
            models[name] = FACTORIES[name](args, views)
        if args.prepare:
            image = np.zeros((480, 640, 3), np.uint8)
            for name, model in models.items():
                print(f'{name}: blank-frame hands={len(model.detect(image, time.monotonic(), 0))}; {model.info}')
            return 0
        with Cameras(args.left, args.right, args.fps, width=args.capture_width,
                     height=args.capture_height, backend=args.camera_backend) as cameras:
            indices = [item['index'] for item in cameras.info]
            log = ComparisonLog(args.output, {'cameras': cameras.info,
                'models': {name: model.info for name, model in models.items()},
                'filters': False, 'adaptive_thresholds': False,
                'max_hands': args.max_hands, 'termination': 'normal'})
            print(f'Data-only recording: {log.path}')
            active, mode = enabled[0], 'single'
            from .ownership import HandOwner
            from .tracking import Hand
            from .engine import Gesture
            owners = {name: HandOwner(views) for name in enabled if name != 'rtmpose'}
            expected, condition, trial = 'unknown', 'near', 1
            counter, fps, started = 0, 0., time.monotonic()
            window = 'MagicPointer model comparison'
            while True:
                frames = cameras.read()
                names = enabled if mode == 'same-frame' else [active]
                # Rotate evaluation order, so one backend is not always last.
                offset = counter % len(names)
                order = names[offset:] + names[:offset]
                results = {}
                gestures = {}
                for name in order:
                    model = models[name]
                    samples = []
                    for view, frame in enumerate(frames):
                        begin = time.perf_counter()
                        hands = model.detect(frame.image, frame.grabbed_at, view)
                        total = (time.perf_counter()-begin)*1000
                        samples.append({'camera': indices[view], 'hands': hands,
                            'grabbed_at': frame.grabbed_at,
                            'image_size': list(frame.image.shape[1::-1]),
                            'inference_ms': getattr(model, 'last_inference_ms', total),
                            'inference_and_transport_ms': total})
                    if name in owners:
                        owner = owners[name]
                        detections = [[Hand(np.asarray(hand['pixels']), hand['world'],
                                            hand['label'], Gesture.UNKNOWN) for hand in view['hands']]
                                      for view in samples]
                        selected = owner.select(detections, frames, time.monotonic())
                        if owner.owner is None and selected.key and selected.progress >= 1.:
                            owner.lock(selected.key)
                        if owner.release_ready:
                            owner.release()
                        gestures[name] = {'gesture': selected.combined.gesture.value if selected.combined else None,
                                          'reason': selected.reason, 'owner_ids': owner.owner,
                                          'evidence': owner.diagnostics,
                                          'world': selected.combined.world if selected.combined else None}
                    log.append({'frame_id': counter, 'trial_id': trial, 'backend': name,
                        'mode': mode, 'expected': expected, 'condition': condition,
                        'evaluation_order': order, 'views': samples,
                        'shared_gesture': gestures.get(name)})
                    results[name] = samples
                panel = draw_panel(frames, {name: results[name] for name in names}, mode,
                                   expected, condition, fps, indices, gestures)
                cv2.imshow(window, panel)
                key = cv2.waitKey(1) & 255
                if key in (27, ord('q')) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    return 0
                old = (active, mode, expected, condition)
                if key in (ord('1'), ord('2'), ord('3')):
                    selected = NAMES[key-ord('1')]
                    if selected in enabled:
                        active, mode = selected, 'single'
                    else:
                        print(f'{selected} not enabled; restart with --models including {selected}')
                elif key == ord('a'):
                    mode = 'single' if mode == 'same-frame' else 'same-frame'
                elif key in (ord('h'), ord('n'), ord('u')):
                    expected = {ord('h'): 'present', ord('n'): 'absent', ord('u'): 'unknown'}[key]
                elif key == ord('d'):
                    condition = 'far' if condition == 'near' else 'near'
                if old != (active, mode, expected, condition):
                    trial += 1
                    for owner in owners.values():
                        owner.release()
                    if old[:2] != (active, mode) and 'mediapipe' in models:
                        # A long gap in VIDEO input must not reuse stale ROI tracking.
                        models['mediapipe'].close()
                        models['mediapipe'] = MediaPipeModel(args, views)
                counter += 1
                now = time.monotonic()
                fps = counter / max(.001, now-started)
    except BaseException as error:
        if log:
            log.metadata.update(termination=type(error).__name__, error=str(error))
        raise
    finally:
        try:
            if log:
                log.close()
        finally:
            for model in models.values():
                model.close()
            cv2.destroyAllWindows()
