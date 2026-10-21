import time

import cv2
import numpy as np

from .audio import Chimes
from .cameras import Cameras
from .engine import Engine, Gesture, Observation
from .debug_log import DebugLog
from .stereo import Stereo
from .tracking import Tracker, distance, gesture_metrics
from .tilt import measure_wrist, direction_pitch

COLORS = {"inactive": (150, 150, 150), "active": (70, 210, 70),
          "click": (220, 120, 240), "scroll": (240, 180, 30), "unavailable": (0, 160, 255)}


def draw_hand(image, hand, color):
    if hand is None:
        return
    points = hand.pixels.astype(int)
    x, y = points.min(axis=0) - 15
    x2, y2 = points.max(axis=0) + 15
    cv2.rectangle(image, (int(x), int(y)), (int(x2), int(y2)), color, 3)
    for start in (1, 5, 9, 13, 17):
        cv2.line(image, tuple(points[0]), tuple(points[start]), color, 2)
        for index in range(start, start+3):
            cv2.line(image, tuple(points[index]), tuple(points[index+1]), color, 2)
    for point in points:
        cv2.circle(image, tuple(point), 4, color, -1)


def debug_views(frames, hands, indices, color):
    """Both raw views with independent hand overlays; never records images."""
    if len(frames) == 1:
        panel = frames[0].image.copy()
        draw_hand(panel, hands[0], color)
        return cv2.resize(panel, (960, 540))
    panel = np.full((760, 1280, 3), 25, dtype=np.uint8)
    for column, (frame, hand, index) in enumerate(zip(frames, hands, indices)):
        image = frame.image.copy()
        draw_hand(image, hand, color)
        height, width = image.shape[:2]
        scale = min(640/width, 360/height)
        resized = cv2.resize(image, (max(1, round(width*scale)), max(1, round(height*scale))))
        h, w = resized.shape[:2]
        x, y = column*640+(640-w)//2, 190+(360-h)//2
        panel[y:y+h, x:x+w] = resized
        side = 'LEFT / primary' if column == 0 else 'RIGHT / secondary'
        label = f'{side}: camera {index} | {width}x{height}'
        pose = f'Pose estimate: {hand.gesture.value} | Hand: {hand.label}' if hand else 'NO HAND'
        for line, text in enumerate((label, pose)):
            cv2.putText(panel, text, (12+column*640, 577+line*25),
                        cv2.FONT_HERSHEY_SIMPLEX, .57, color, 1, cv2.LINE_AA)
        if hand:
            metrics = gesture_metrics(hand.world)
            if metrics['valid']:
                cv2.putText(panel, f"Thumb/index/middle/ring/little estimates: {metrics['signature']}",
                            (12+column*640, 627), cv2.FONT_HERSHEY_SIMPLEX, .48, color, 1, cv2.LINE_AA)
    return panel


def index_only(hand):
    metrics = gesture_metrics(hand.world)
    return metrics['valid'] and [metrics['fingers'][name]['state'] for name in
                                 ('index', 'middle', 'ring', 'little')] == [1, 0, 0, 0]


def observe(left, right=None, *, stereo=None, scrolling=False, mono_scale=1.):
    """Keep stereo association/depth checks; ignore thumb only in latched mode."""
    if left is None:
        return None, "No reliable hand"
    pitch, _ = measure_wrist(left)
    point = index_only(left)
    gesture = left.gesture
    if stereo is None:
        points = left.pixels / mono_scale
        return Observation(gesture, (*points[0], 0.), (*points[8], 0.), pitch, point), "Tracking"
    if right is None or left.label != right.label or (not scrolling and left.gesture != right.gesture):
        return None, "Hands/gestures disagree between cameras"
    right_pitch, _ = measure_wrist(right)
    point = point and index_only(right)
    # Thumb landmarks are neither a scrolling signal nor needed for metric
    # position. Poor thumb estimates must not invalidate the entire stereo pair.
    indices = [0] + list(range(5, 21)) if scrolling or gesture == Gesture.SCROLL else list(range(21))
    points = stereo.reconstruct(left.pixels[indices], right.pixels[indices])
    if points is None or np.linalg.norm(points[0]) > 3.6576:
        return None, "Depth unreliable or beyond 12-foot target range"
    indexed = dict(zip(indices, points))
    if pitch is not None and right_pitch is not None:
        pitch, _ = direction_pitch([indexed[i] for i in (5, 8, 9, 12)])
    else:
        pitch = None
    if scrolling:
        if Gesture.PALM in (left.gesture, right.gesture):
            gesture = Gesture.PALM
        elif ((gesture == Gesture.POINT and not point) or
              (gesture == Gesture.FIST and right.gesture != Gesture.FIST)):
            gesture = Gesture.UNKNOWN
    return Observation(gesture, tuple(indexed[0]), tuple(indexed[8]), pitch, point), "Tracking"


def run(args):
    if not args.debug and args.right is None:
        raise ValueError("Real pointer control needs two cameras. For one-camera preview use run --debug")
    two_cameras = args.right is not None
    stereo, calibration_issue = None, None
    if two_cameras:
        try:
            stereo = Stereo.load(args.calibration)
            if stereo.data.get("camera_indices") != [args.left, args.right]:
                raise ValueError("Camera indices/order differ from calibration; select the calibrated pair or recalibrate")
        except ValueError as error:
            if not args.debug:
                raise
            stereo, calibration_issue = None, str(error)
    preview_only = two_cameras and stereo is None
    engine = Engine(width=args.workspace_width * (10 if stereo is None else 1),
                    height=args.workspace_height * (10 if stereo is None else 1), mirror_x=args.mirror_x)
    trackers, audio, desktop = [], None, None
    log = None
    window = "MagicPointer - Q / Escape stops"
    current_label = None
    mono_scale = 1.
    verified, verification_count = not two_cameras, 0
    next_marker_check = 0.
    previous_wrist, previous_wrist_time = None, None
    try:
        if args.debug:
            log = DebugLog(args.output, {'camera_indices': [args.left] + ([args.right] if two_cameras else []),
                           'requested_fps': args.fps, 'stereo': stereo is not None,
                           'preview_only': preview_only, 'calibration_issue': calibration_issue,
                           'workspace_width': engine.width, 'workspace_height': engine.height,
                           'mirror_x': engine.mirror_x, 'coordinate_units': None if preview_only else ('metres' if stereo else 'activation_palm_sizes'),
                           'dwell_seconds': engine.dwell, 'activation_dwell_seconds': engine.activation_dwell,
                           'loss_timeout_seconds': engine.loss_timeout,
                           'termination': 'normal'})
        # Verify model/dependencies before opening camera devices.
        trackers.append(Tracker(args.model))
        if two_cameras:
            trackers.append(Tracker(args.model))
        if not args.debug:
            from .desktop import Desktop
            desktop = Desktop()
        audio = Chimes(args.mute)
        with Cameras(args.left, args.right, args.fps) as cameras:
            if stereo and args.debug and any((camera['width'], camera['height']) != stereo.size for camera in cameras.info):
                stereo = None
                preview_only = True
                calibration_issue = "Camera resolution differs from calibration; recalibrate at this resolution"
                verified = False
                if log:
                    log.metadata.update(stereo=False, preview_only=True, calibration_issue=calibration_issue, coordinate_units=None)
            if log:
                log.metadata["cameras"] = cameras.info
            if preview_only:
                print("TWO-CAMERA PREVIEW: no depth or joint pointer simulation until calibration is valid.")
                print("Calibration:", calibration_issue)
            print("Camera negotiated settings:", cameras.info)
            print("DEBUG: preview only; no OS pointer input" if args.debug else "STEREO CONTROL: Q/Escape or close window to stop")
            print("Hold fist 0.45 s to activate; hold gestures 0.18 s. Return to index-only between clicks.")
            print("Two fingers toward screen: hold steady to set scroll neutral; tilt up/down. Index-only exits scrolling.")
            start, count, measured_fps = time.monotonic(), 0, 0.
            last_action, action_until = "", 0.
            while True:
                frames = cameras.read()
                now = time.monotonic()
                paired = (abs(frames[0].grabbed_at-frames[1].grabbed_at) <= .04) if two_cameras else True
                if stereo:
                    paired = stereo.check_frames(frames)
                marker_message = "Mono preview: depth is not measured"
                if stereo and (not verified or now >= next_marker_check):
                    good, marker_message = stereo.check_markers([f.image for f in frames], require_visible=not verified)
                    if not verified:
                        verification_count = verification_count + 1 if good and paired else 0
                        verified = verification_count >= 3
                    elif not good:
                        raise ValueError(marker_message)
                    next_marker_check = now + 1.
                hands = [tracker.detect(frame.image, frame.grabbed_at) for tracker, frame in zip(trackers, frames)]
                observation, reason = None, "No reliable hand"
                left = hands[0]
                if preview_only:
                    reason = "Preview only: calibrate this camera pair to enable depth and virtual pointer"
                elif not verified:
                    reason = marker_message
                elif not paired:
                    reason = "Camera pair timing mismatch"
                elif left is not None:
                    if stereo is None and not engine.active:
                        mono_scale = max(10., float(np.linalg.norm(left.pixels[0]-left.pixels[9])))
                    observation, reason = observe(left, hands[1] if two_cameras else None,
                                                  stereo=stereo, scrolling=engine.scrolling,
                                                  mono_scale=mono_scale)
                    if engine.active and left.label != current_label:
                        observation, reason = None, "Controlling hand changed"
                # Reject implausible same-label hand switches in metric stereo mode.
                if observation is not None and stereo and engine.active and previous_wrist is not None:
                    elapsed = now - previous_wrist_time
                    if elapsed > 0 and distance(observation.wrist, previous_wrist) / elapsed > 4.:
                        observation, reason = None, "Hand moved too abruptly; reacquire"
                if observation is not None:
                    previous_wrist, previous_wrist_time = observation.wrist, now
                engine_now = time.monotonic()
                events = engine.step(observation, engine_now)
                for event in events:
                    if event.kind == "activate":
                        current_label = left.label
                        audio.play(True)
                    elif event.kind == "deactivate":
                        current_label, previous_wrist = None, None
                        audio.play(False)
                    elif event.kind == "move" and desktop:
                        desktop.move(event.value)
                    elif event.kind == "click":
                        last_action, action_until = f"{event.value} click", now + .5
                        if desktop:
                            desktop.click(event.value)
                    elif event.kind == "scroll":
                        last_action, action_until = f"scroll {event.value:+d}", now + .3
                        if desktop:
                            desktop.scroll(event.value)
                state = "inactive" if not engine.active else "active"
                if observation is None:
                    state = "unavailable"
                elif engine.scrolling:
                    state = "scroll"
                elif now < action_until:
                    state = "click"
                if log:
                    log.append(frames, hands, observation, engine, events, engine_now,
                               reason=reason, paired=paired, verified=verified, state=state)
                color = COLORS[state]
                if args.debug:
                    panel = debug_views(frames, hands, [args.left, args.right], color)
                else:
                    panel = np.full((540, 960, 3), (25, 25, 25), dtype=np.uint8)
                # Primary-camera raw view is intentionally labelled: controls mirror X by default.
                title = "DEBUG / NO DESKTOP INPUT" if args.debug else "STEREO / DESKTOP CONTROL"
                if args.debug and two_cameras:
                    title = "TWO-CAMERA DEBUG / NO OS INPUT / " + ("PREVIEW ONLY" if preview_only else "CALIBRATED STEREO")
                lines = [title,
                         f"State: {state} | Pose estimate: {left.gesture.value if left else 'none'}" +
                         (" (clicks ignored)" if engine.scrolling else ""),
                         reason if observation is None else (last_action if now < action_until else "Tracking"),
                         f"Processing: {measured_fps:.1f} FPS | Q / Esc / close window to stop"]
                if engine.scrolling:
                    scroll = engine.scroll
                    feedback = "Hold two fingers steady: setting scroll neutral"
                    if scroll.neutral is not None:
                        feedback = (f"Wrist scroll: {scroll.delta:+.1f} deg | {scroll.speed:+.1f}/s (down +)"
                                    if scroll.delta is not None else "Scroll paused: restore two-finger pose")
                    lines.append(feedback + " | Index-only exits")
                if args.debug and two_cameras:
                    skew = abs(frames[0].grabbed_at-frames[1].grabbed_at)*1000
                    depth = f"Wrist range: {np.linalg.norm(observation.wrist):.2f} m" if observation and stereo else "Depth unavailable"
                    lines.append(f"Pair grab skew: {skew:.1f} ms ({'OK' if paired else 'over 40 ms'}) | {depth}")
                if log:
                    lines.append("REC: diagnostic data (auto-saved on exit)")
                for i, line in enumerate(lines):
                    cv2.putText(panel, line, (15, 27+i*26), cv2.FONT_HERSHEY_SIMPLEX, .58, (0, 0, 0), 4)
                    cv2.putText(panel, line, (15, 27+i*26), cv2.FONT_HERSHEY_SIMPLEX, .58, color, 1)
                for i, (name, value) in enumerate(COLORS.items()):
                    cv2.putText(panel, name, (15+i*185, panel.shape[0]-18), cv2.FONT_HERSHEY_SIMPLEX, .5, value, 1)
                # Small virtual screen shows motion even when using just one camera.
                cursor_x, cursor_y = (1050, 30) if args.debug and len(frames) == 2 else (730, 120)
                cv2.rectangle(panel, (cursor_x, cursor_y), (cursor_x+215, cursor_y+135), (220, 220, 220), 1)
                if engine.active:
                    cv2.circle(panel, (cursor_x+round(engine.cursor[0]*215), cursor_y+round(engine.cursor[1]*135)), 5, color, -1)
                cv2.imshow(window, panel)
                key = cv2.waitKey(1) & 255
                if key in (27, ord('q')) or cv2.getWindowProperty(window, cv2.WND_PROP_VISIBLE) < 1:
                    return 0
                count += 1
                if now - start >= 1:
                    measured_fps, start, count = count/(now-start), now, 0
    except BaseException as error:
        if log:
            log.metadata['termination'] = type(error).__name__
            log.metadata['error'] = str(error)
        raise
    finally:
        try:
            if log:
                log.close()
        finally:
            if audio:
                if engine.active:
                    audio.play(False)
                audio.close()
            for tracker in trackers:
                tracker.close()
            cv2.destroyAllWindows()
